#!/usr/bin/env python3
"""Register the cropped Pieris eye label to the full TIFF stack by CT edges."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np


def open_label(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        header = handle.read(352)
    endian = "<" if struct.unpack("<i", header[:4])[0] == 348 else ">"
    dimensions = struct.unpack(endian + "8h", header[40:56])
    shape = tuple(int(x) for x in dimensions[1 : dimensions[0] + 1])
    offset = int(struct.unpack(endian + "f", header[108:112])[0])
    return np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=shape, order="F")


def evenly_sample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, count).round().astype(int)
    return points[indices]


def sample_value_points(label: np.ndarray, value: int, count: int) -> np.ndarray:
    """Sample a material approximately uniformly across slices without a 3-D copy."""
    per_slice = max(1, int(np.ceil(count / label.shape[2])))
    selected = []
    for z in range(label.shape[2]):
        xy = np.argwhere(label[:, :, z] == value)
        if not len(xy):
            continue
        xy = evenly_sample(xy, per_slice)
        selected.append(np.column_stack((xy, np.full(len(xy), z, dtype=int))))
    return evenly_sample(np.vstack(selected), count)


def write_nifti_uint8(path: Path, data: np.ndarray) -> None:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 2)
    struct.pack_into("<h", header, 72, 8)
    struct.pack_into("<8f", header, 76, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, 1.0, 0.0, 0.0, 1.0)
    struct.pack_into("<4f", header, 296, 0.0, 1.0, 0.0, 1.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, 1.0, 1.0)
    header[344:348] = b"n+1\0"
    with path.open("wb") as handle:
        handle.write(header)
    output = np.memmap(
        path,
        dtype=np.uint8,
        mode="r+",
        offset=352,
        shape=data.shape,
        order="F",
    )
    output[:] = data
    output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-value", type=int, default=2)
    parser.add_argument("--eye-value", type=int, default=1)
    args = parser.parse_args()

    volume = np.load(args.volume, mmap_mode="r")
    label = open_label(args.label)
    if volume.shape[:2] != label.shape[:2]:
        raise ValueError(f"XY mismatch: CT {volume.shape}, label {label.shape}")
    maximum_offset = volume.shape[2] - label.shape[2]
    if maximum_offset < 0:
        raise ValueError("Label is deeper than the CT volume")

    surface = np.argwhere(label == args.surface_value)
    surface = surface[
        (surface[:, 0] > 0)
        & (surface[:, 0] < volume.shape[0] - 1)
        & (surface[:, 1] > 0)
        & (surface[:, 1] < volume.shape[1] - 1)
        & (surface[:, 2] > 0)
        & (surface[:, 2] < label.shape[2] - 1)
    ]
    surface = evenly_sample(surface, 50000)

    inside = sample_value_points(label, args.eye_value, 30000)
    outside = sample_value_points(label, 0, 30000)

    rows = []
    for offset in range(maximum_offset + 1):
        x, y, z = surface.T
        z = z + offset
        gx = volume[x + 1, y, z].astype(float) - volume[x - 1, y, z].astype(float)
        gy = volume[x, y + 1, z].astype(float) - volume[x, y - 1, z].astype(float)
        gz = volume[x, y, z + 1].astype(float) - volume[x, y, z - 1].astype(float)
        gradient = np.sqrt(gx * gx + gy * gy + gz * gz)

        ix, iy, iz = inside.T
        ox, oy, oz = outside.T
        inside_values = volume[ix, iy, iz + offset].astype(float)
        outside_values = volume[ox, oy, oz + offset].astype(float)
        rows.append(
            {
                "z_offset": offset,
                "surface_gradient_mean": float(np.mean(gradient)),
                "surface_gradient_median": float(np.median(gradient)),
                "eye_median": float(np.median(inside_values)),
                "background_median": float(np.median(outside_values)),
                "eye_minus_background": float(
                    np.median(inside_values) - np.median(outside_values)
                ),
            }
        )

    gradient_values = np.asarray([row["surface_gradient_mean"] for row in rows])
    contrast_values = np.asarray([row["eye_minus_background"] for row in rows])
    robust_z = lambda values: (values - np.median(values)) / (
        np.median(np.abs(values - np.median(values))) + 1e-6
    )
    combined = robust_z(gradient_values) + 0.25 * robust_z(contrast_values)
    for row, score in zip(rows, combined):
        row["registration_score"] = float(score)
    best_offset = int(rows[int(np.argmax(combined))]["z_offset"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "offset_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    full_label = np.lib.format.open_memmap(
        args.output_dir / "registered_label.npy",
        mode="w+",
        dtype=np.uint8,
        shape=volume.shape,
        fortran_order=True,
    )
    full_label[:] = 0
    mapped = np.zeros(label.shape, dtype=np.uint8)
    mapped[label == args.eye_value] = 3
    mapped[label == args.surface_value] = 7
    full_label[:, :, best_offset : best_offset + label.shape[2]] = mapped
    full_label.flush()
    write_nifti_uint8(args.output_dir / "registered_label.nii", full_label)

    summary = {
        "method": "maximise labelled-surface CT gradient with eye/background contrast tie-breaker",
        "best_z_offset": best_offset,
        "label_shape": list(label.shape),
        "ct_shape": list(volume.shape),
        "surface_value_mapped_to_7": args.surface_value,
        "eye_volume_value_mapped_to_3": args.eye_value,
        "top_candidates": sorted(rows, key=lambda row: row["registration_score"], reverse=True)[:10],
    }
    (args.output_dir / "registration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


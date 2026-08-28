#!/usr/bin/env python3
"""Build a blinded local-CT pack for independent boundary annotation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from annotation_core import PACK_VERSION, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrrd", required=True, type=Path)
    parser.add_argument("--centers", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--edge-table", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--spacing-um", type=float, default=3.7)
    parser.add_argument("--surface-threshold", type=float, default=50.0)
    parser.add_argument("--accepted-per-block", type=int, default=5)
    parser.add_argument("--failed-controls", type=int, default=5)
    parser.add_argument("--seed", default="asaphus-boundary-v1")
    parser.add_argument("--uv-extent-vox", type=float, default=10.0)
    parser.add_argument("--uv-step-vox", type=float, default=0.5)
    parser.add_argument("--depth-range-vox", type=float, nargs=2,
                        default=[-2.0, 25.0])
    parser.add_argument("--depth-step-vox", type=float, default=0.5)
    return parser.parse_args()


def read_nrrd(path: Path) -> tuple[np.ndarray, dict]:
    with path.open("rb") as stream:
        lines = []
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in NRRD header")
            if line in (b"\n", b"\r\n"):
                body_offset = stream.tell()
                break
            lines.append(line.decode("ascii").rstrip())
    header = {}
    for line in lines:
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    sizes = tuple(map(int, header["sizes"].split()))
    with path.open("rb") as stream:
        stream.seek(body_offset)
        encoding = header.get("encoding", "raw").lower()
        if encoding in {"gzip", "gz"}:
            with gzip.GzipFile(fileobj=stream, mode="rb") as zipped:
                raw = zipped.read()
        elif encoding == "raw":
            raw = stream.read()
        else:
            raise ValueError(f"Unsupported NRRD encoding: {encoding}")
    if header.get("type", "").lower() not in {
        "uchar", "unsigned char", "uint8", "uint8_t"
    }:
        raise ValueError("This pack builder currently expects a uint8 NRRD")
    volume = np.frombuffer(raw, dtype=np.uint8).reshape(sizes, order="F")
    return volume, header


def stable_rank(seed: str, facet_id: int) -> str:
    return hashlib.sha256(f"{seed}:{facet_id}".encode()).hexdigest()


def choose_cases(
    samples: pd.DataFrame,
    edge_table: pd.DataFrame,
    accepted_per_block: int,
    failed_controls: int,
    seed: str,
) -> list[tuple[int, str, int | None]]:
    accepted = samples[["facet_id", "cv_block"]].drop_duplicates()
    selected: list[tuple[int, str, int | None]] = []
    for block, group in accepted.groupby("cv_block", sort=True):
        ids = sorted(group["facet_id"].astype(int),
                     key=lambda value: stable_rank(seed, value))
        if len(ids) < accepted_per_block:
            raise ValueError(f"Block {block} has only {len(ids)} accepted facets")
        selected.extend((fid, "accepted", int(block))
                        for fid in ids[:accepted_per_block])
    failed = edge_table.loc[~edge_table["high_quality"].astype(bool),
                            "facet_id"].astype(int).tolist()
    failed = sorted(failed, key=lambda value: stable_rank(seed + ":failed", value))
    selected.extend((fid, "failed_qc_control", None)
                    for fid in failed[:failed_controls])
    return sorted(selected, key=lambda row: stable_rank(seed + ":blind", row[0]))


def subvoxel_surface(volume: np.ndarray, threshold: float) -> np.ndarray:
    smooth = ndi.gaussian_filter1d(volume.astype(np.float32), sigma=1.0, axis=0)
    above = smooth >= threshold
    reverse = np.argmax(above[::-1], axis=0)
    any_above = above.any(axis=0)
    x0 = (smooth.shape[0] - 1 - reverse).astype(np.int32)
    surface = np.full(any_above.shape, np.nan, dtype=np.float32)
    yy, zz = np.where(any_above)
    xx = x0[yy, zz]
    valid = xx < smooth.shape[0] - 1
    yv, zv, xv = yy[valid], zz[valid], xx[valid]
    value0, value1 = smooth[xv, yv, zv], smooth[xv + 1, yv, zv]
    denominator = value0 - value1
    fraction = np.divide(value0 - threshold, denominator,
                         out=np.zeros_like(value0),
                         where=np.abs(denominator) > 1e-6)
    surface[yv, zv] = xv + fraction
    surface[yy[~valid], zz[~valid]] = xx[~valid]
    return surface


def local_volume(
    volume: np.ndarray,
    point: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    uv_extent: float,
    uv_step: float,
    depth_range: tuple[float, float],
    depth_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = point
    yi, zi = int(round(y)), int(round(z))
    normal = np.array([1.0, -fy[yi, zi], -fz[yi, zi]], float)
    normal /= np.linalg.norm(normal)
    tangent1 = np.array([fy[yi, zi], 1.0, 0.0], float)
    tangent1 -= normal * np.dot(tangent1, normal)
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    tangent2 /= np.linalg.norm(tangent2)
    u = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
    v = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
    depth = np.arange(depth_range[0], depth_range[1] + 1e-9, depth_step)
    uu, vv, dd = np.meshgrid(u, v, depth, indexing="ij")
    xyz = (
        point[:, None, None, None]
        + tangent1[:, None, None, None] * uu[None]
        + tangent2[:, None, None, None] * vv[None]
        - normal[:, None, None, None] * dd[None]
    )
    values = ndi.map_coordinates(
        volume.astype(np.float32), xyz.reshape(3, -1), order=1, mode="nearest"
    ).reshape(uu.shape)
    return values, u, v, depth


def main() -> None:
    args = parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    case_folder = args.out / "cases"
    case_folder.mkdir()

    volume, header = read_nrrd(args.nrrd)
    centers = pd.read_csv(args.centers).set_index("facet_id")
    samples = pd.read_csv(args.samples)
    edge_table = pd.read_csv(args.edge_table)
    chosen = choose_cases(samples, edge_table, args.accepted_per_block,
                          args.failed_controls, args.seed)
    missing = [fid for fid, _, _ in chosen if fid not in centers.index]
    if missing:
        raise ValueError(f"Centres missing facet IDs: {missing}")

    surface = subvoxel_surface(volume, args.surface_threshold)
    valid = np.isfinite(surface)
    nearest = ndi.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = surface[tuple(nearest)]
    smooth = ndi.gaussian_filter(filled, sigma=2.0)
    fy, fz = np.gradient(smooth)

    public_cases = []
    private_rows = []
    for number, (facet_id, source_group, block) in enumerate(chosen, start=1):
        case_id = f"F{number:03d}"
        center = centers.loc[facet_id]
        point = center[["x_vox", "y_vox", "z_vox"]].to_numpy(float)
        intensity, u, v, depth = local_volume(
            volume, point, fy, fz, args.uv_extent_vox, args.uv_step_vox,
            tuple(args.depth_range_vox), args.depth_step_vox,
        )
        filename = f"cases/{case_id}.npz"
        np.savez_compressed(args.out / filename, intensity=intensity,
                            u_vox=u, v_vox=v, depth_vox=depth)
        public_cases.append({
            "case_id": case_id,
            "file": filename,
            "shape": list(intensity.shape),
        })
        private_rows.append({
            "case_id": case_id,
            "facet_id": facet_id,
            "source_group": source_group,
            "cv_block": "" if block is None else block,
            "x_vox": point[0], "y_vox": point[1], "z_vox": point[2],
        })

    dataset_hash = sha256(args.nrrd)
    pack_id = hashlib.sha256(
        (dataset_hash + ":" + args.seed).encode()
    ).hexdigest()[:16]
    manifest = {
        "pack_version": PACK_VERSION,
        "pack_id": pack_id,
        "source_dataset_sha256": dataset_hash,
        "voxel_spacing_um": args.spacing_um,
        "source_shape": list(volume.shape),
        "source_nrrd_header": {
            key: header[key] for key in ("type", "dimension", "sizes", "encoding")
            if key in header
        },
        "blinding": (
            "Case IDs are randomized. Original facet IDs, spatial blocks, "
            "QC status and algorithm-derived boundary depths are absent."
        ),
        "instruction": (
            "Trace only a boundary that is visible in the raw CT sections. "
            "Use uncertain or no visible boundary when appropriate."
        ),
        "cases": public_cases,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    private_path = args.out.parent / f"{args.out.name}_PRIVATE_KEY.csv"
    pd.DataFrame(private_rows).to_csv(private_path, index=False)
    print(f"Created {len(public_cases)} blinded cases in {args.out}")
    print(f"Keep private and do not give annotators: {private_path}")


if __name__ == "__main__":
    main()


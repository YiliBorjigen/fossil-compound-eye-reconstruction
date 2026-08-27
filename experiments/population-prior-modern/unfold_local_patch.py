#!/usr/bin/env python3
"""Unfold a local compound-eye patch along the labelled corneal surface."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


def open_label(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        header = handle.read(352)
    little = struct.unpack("<i", header[:4])[0] == 348
    endian = "<" if little else ">"
    dim = struct.unpack(endian + "8h", header[40:56])
    shape = tuple(int(value) for value in dim[1 : dim[0] + 1])
    return np.memmap(path, dtype=np.uint8, mode="r", offset=352, shape=shape, order="F")


def collect_surface_points(
    labels: np.ndarray, target: np.ndarray, search_radius: int
) -> np.ndarray:
    points: list[np.ndarray] = []
    z0 = max(0, int(target[2] - search_radius))
    z1 = min(labels.shape[2], int(target[2] + search_radius + 1))
    for z in range(z0, z1):
        xy = np.argwhere(labels[:, :, z] == 7)
        if not xy.size:
            continue
        keep = (
            (np.abs(xy[:, 0] - target[0]) <= search_radius)
            & (np.abs(xy[:, 1] - target[1]) <= search_radius)
        )
        if np.any(keep):
            points.append(np.column_stack((xy[keep], np.full(np.count_nonzero(keep), z))))
    if not points:
        raise ValueError("No labelled corneal surface found near the requested seed")
    return np.vstack(points).astype(np.float64)


def quadratic_design(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(u), u, v, u * u, u * v, v * v))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", nargs=3, type=float, default=(800, 700, 225))
    parser.add_argument("--half-width", type=float, default=70.0)
    parser.add_argument("--depth", type=int, default=60)
    args = parser.parse_args()

    volume = np.load(args.volume, mmap_mode="r")
    labels = open_label(args.label)
    target = np.asarray(args.seed, dtype=np.float64)

    surface = collect_surface_points(labels, target, search_radius=130)
    seed = surface[np.argmin(np.sum((surface - target) ** 2, axis=1))]
    local = surface[np.linalg.norm(surface - seed, axis=1) < args.half_width + 25]

    centre = local.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(local - centre, rowvar=False))
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    tangent_u, tangent_v, base_normal = eigenvectors.T

    local_coords = (local - seed) @ eigenvectors
    u, v, w = local_coords.T
    fit_keep = (np.abs(u) <= args.half_width + 10) & (np.abs(v) <= args.half_width + 10)
    design = quadratic_design(u[fit_keep], v[fit_keep])
    coefficient, *_ = np.linalg.lstsq(design, w[fit_keep], rcond=None)

    grid_axis = np.arange(-args.half_width, args.half_width + 0.001, 1.0)
    grid_u, grid_v = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_design = quadratic_design(grid_u.ravel(), grid_v.ravel())
    grid_w = (grid_design @ coefficient).reshape(grid_u.shape)
    surface_xyz = (
        seed
        + grid_u[..., None] * tangent_u
        + grid_v[..., None] * tangent_v
        + grid_w[..., None] * base_normal
    )

    dw_du = coefficient[1] + 2 * coefficient[3] * grid_u + coefficient[4] * grid_v
    dw_dv = coefficient[2] + coefficient[4] * grid_u + 2 * coefficient[5] * grid_v
    normals = (
        base_normal
        - dw_du[..., None] * tangent_u
        - dw_dv[..., None] * tangent_v
    )
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    # Select the direction that reaches the published cone-layer label (3).
    probe = np.arange(1, min(args.depth, 45) + 1, dtype=np.float64)
    centre_index = len(grid_axis) // 2
    centre_surface = surface_xyz[centre_index, centre_index]
    centre_normal = normals[centre_index, centre_index]
    scores = []
    for sign in (-1.0, 1.0):
        query = centre_surface[:, None] + sign * centre_normal[:, None] * probe
        sampled = map_coordinates(labels, query, order=0, mode="constant", cval=0, prefilter=False)
        scores.append(int(np.count_nonzero(sampled == 3)))
    inward_sign = (-1.0, 1.0)[int(scores[1] > scores[0])]
    inward = inward_sign * normals

    depths = np.arange(args.depth + 1, dtype=np.float64)
    shape = grid_u.shape + (len(depths),)
    unfolded_data = np.empty(shape, dtype=np.uint8)
    unfolded_labels = np.empty(shape, dtype=np.uint8)
    for index, depth in enumerate(depths):
        query = surface_xyz + depth * inward
        coordinates = np.moveaxis(query, -1, 0)
        unfolded_data[:, :, index] = np.clip(
            np.rint(map_coordinates(volume, coordinates, order=1, mode="constant", cval=0)),
            0,
            255,
        ).astype(np.uint8)
        unfolded_labels[:, :, index] = map_coordinates(
            labels, coordinates, order=0, mode="constant", cval=0, prefilter=False
        ).astype(np.uint8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "unfolded_intensity.npy", unfolded_data)
    np.save(args.output_dir / "unfolded_labels.npy", unfolded_labels)
    np.save(args.output_dir / "surface_xyz.npy", surface_xyz.astype(np.float32))
    np.save(args.output_dir / "inward_normals.npy", inward.astype(np.float32))

    coverage = np.mean(unfolded_labels == 3, axis=(0, 1))
    best_depth = int(np.argmax(coverage))
    metadata = {
        "requested_seed_xyz": target.tolist(),
        "actual_seed_xyz": seed.tolist(),
        "patch_shape_uvd": list(unfolded_data.shape),
        "inward_direction_scores": {"negative": scores[0], "positive": scores[1]},
        "inward_sign": inward_sign,
        "cone_layer_peak_depth_voxels": best_depth,
        "cone_layer_peak_coverage": float(coverage[best_depth]),
        "quadratic_surface_rmse_voxels": float(
            np.sqrt(np.mean((design @ coefficient - w[fit_keep]) ** 2))
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    selected = sorted(set([5, 10, 15, 20, 25, 30, 35, 40, 50, best_depth]))
    selected = [depth for depth in selected if depth <= args.depth]
    columns = 5
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for axis, depth in zip(axes, selected):
        image = unfolded_data[:, :, depth]
        lo, hi = np.percentile(image, (1, 99))
        axis.imshow(np.clip((image - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
        axis.contour(unfolded_labels[:, :, depth] == 3, levels=[0.5], colors="red", linewidths=0.5)
        axis.set_title(f"depth {depth} voxels")
        axis.axis("off")
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output_dir / "unfolded_depth_series.png", bbox_inches="tight")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

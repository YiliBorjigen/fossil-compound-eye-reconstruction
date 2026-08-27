#!/usr/bin/env python3
"""Choose three Pieris patches from surface geometry only."""

from __future__ import annotations

import argparse
import json
import struct
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


def open_label(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        header = handle.read(352)
    dimensions = struct.unpack("<8h", header[40:56])
    shape = tuple(int(x) for x in dimensions[1 : dimensions[0] + 1])
    return np.memmap(path, dtype=np.uint8, mode="r", offset=352, shape=shape, order="F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    volume = np.load(args.volume, mmap_mode="r")
    label = open_label(args.label)
    points = np.argwhere(label == 7).astype(float)
    margin = np.asarray((80, 80, 70))
    safe = np.all((points >= margin) & (points < np.asarray(label.shape) - margin), axis=1)
    safe_points = points[safe]
    if len(safe_points) < 1000:
        raise ValueError("Insufficient safely bounded corneal surface")

    sample_index = np.linspace(0, len(safe_points) - 1, min(100000, len(safe_points))).round().astype(int)
    sample = safe_points[sample_index]
    model = KMeans(n_clusters=9, random_state=45, n_init=30).fit(sample)
    tree = cKDTree(safe_points)
    _, medoid_indices = tree.query(model.cluster_centers_, k=1)
    candidates = safe_points[medoid_indices]

    best_triplet = None
    best_score = None
    eye_centre = np.mean(safe_points, axis=0)
    for indices in combinations(range(len(candidates)), 3):
        chosen = candidates[list(indices)]
        distances = np.linalg.norm(chosen[:, None, :] - chosen[None, :, :], axis=2)
        pairwise = distances[np.triu_indices(3, 1)]
        centre_penalty = 0.05 * np.mean(np.linalg.norm(chosen - eye_centre, axis=1))
        score = float(np.min(pairwise) - centre_penalty)
        if best_score is None or score > best_score:
            best_score = score
            best_triplet = indices
    assert best_triplet is not None
    seeds = candidates[list(best_triplet)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "selection": "three maximally separated medoids from nine surface-only KMeans regions",
        "random_state": 45,
        "safe_margin_xyz_voxels": margin.tolist(),
        "surface_points": int(len(points)),
        "safe_surface_points": int(len(safe_points)),
        "seeds_xyz": seeds.tolist(),
        "minimum_seed_separation_voxels": float(
            np.min(np.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=2) + np.eye(3) * 1e9)
        ),
    }
    (args.output_dir / "seeds.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    display = points[:: max(1, len(points) // 20000)]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    projections = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    for axis, (a, b, xlabel, ylabel) in zip(axes, projections):
        axis.scatter(display[:, a], display[:, b], s=0.5, c="0.65", alpha=0.5)
        axis.scatter(seeds[:, a], seeds[:, b], s=80, c=["#0072B2", "#D55E00", "#009E73"], edgecolor="black")
        for index, seed in enumerate(seeds, start=1):
            axis.annotate(str(index), (seed[a], seed[b]), xytext=(5, 5), textcoords="offset points")
        axis.set_xlabel(f"{xlabel} (voxels)")
        axis.set_ylabel(f"{ylabel} (voxels)")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle("Pieris patch selection from the registered corneal surface")
    figure.tight_layout()
    figure.savefig(args.output_dir / "surface_seed_selection.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


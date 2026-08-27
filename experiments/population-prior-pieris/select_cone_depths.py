#!/usr/bin/env python3
"""Select a cone cross-section from periodic CT texture, before reconstruction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def load_frozen(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_population_prior", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-script", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--minimum-depth", type=int, default=15)
    parser.add_argument("--maximum-depth", type=int, default=35)
    args = parser.parse_args()

    frozen = load_frozen(args.frozen_script)
    data = np.load(args.patch_dir / "unfolded_intensity.npy")
    labels = np.load(args.patch_dir / "unfolded_labels.npy")
    broad_roi = binary_erosion(np.any(labels[:, :, :45] == 3, axis=2), iterations=6)
    outer_rows = []
    for outer_depth in range(0, 11):
        centres = frozen.detect_internal_centres(data[:, :, outer_depth], broad_roi)
        if len(centres) < 30:
            continue
        nearest = cKDTree(centres).query(centres, k=2)[0][:, 1]
        median_spacing = float(np.median(nearest))
        if not 5.0 <= median_spacing <= 18.0:
            continue
        outer_rows.append(
            {
                "outer_depth": outer_depth,
                "outer_centres": len(centres),
                "median_nearest_neighbour_voxels": median_spacing,
                "nearest_neighbour_cv": float(np.std(nearest) / np.mean(nearest)),
            }
        )
    if not outer_rows:
        raise ValueError("No shallow corneal lattice passed the fixed checks")
    selected_outer = min(outer_rows, key=lambda row: row["nearest_neighbour_cv"])

    rows = []
    for detection_depth in range(args.minimum_depth, args.maximum_depth + 1):
        roi = binary_erosion(labels[:, :, detection_depth] == 3, iterations=6)
        if np.count_nonzero(roi) < 1000:
            continue
        centres = frozen.detect_internal_centres(data[:, :, detection_depth], roi)
        if len(centres) < 30:
            continue
        nearest = cKDTree(centres).query(centres, k=2)[0][:, 1]
        median_spacing = float(np.median(nearest))
        nn_cv = float(np.std(nearest) / np.mean(nearest))
        rows.append(
            {
                "detection_depth": detection_depth,
                "cone_depth_for_frozen_script": detection_depth + 2,
                "internal_centres": len(centres),
                "median_nearest_neighbour_voxels": median_spacing,
                "nearest_neighbour_cv": nn_cv,
            }
        )
    if not rows:
        raise ValueError("No candidate cone depth passed the fixed texture checks")

    with (args.patch_dir / "cone_depth_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plausible = [
        row
        for row in rows
        if 5.0 <= row["median_nearest_neighbour_voxels"] <= 18.0
    ]
    if not plausible:
        raise ValueError("No depth satisfied the pre-specified lattice plausibility range")
    selected = min(plausible, key=lambda row: row["nearest_neighbour_cv"])

    metadata_path = args.patch_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["label_peak_depth_original"] = metadata["cone_layer_peak_depth_voxels"]
    metadata["outer_lattice_depth_voxels"] = selected_outer["outer_depth"]
    metadata["cone_layer_peak_depth_voxels"] = selected["cone_depth_for_frozen_script"]
    metadata["cone_depth_selection"] = {
        "status": "specimen adapter fixed before held-out scoring",
        "criterion": "minimum nearest-neighbour CV with count and spacing plausibility gates",
        "selected_outer": selected_outer,
        "selected": selected,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["cone_depth_selection"], indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Freeze a minimal Patch 4 spatial orientation field from Patches 1--3 only.

The primary field is inverse-distance-squared interpolation of the three
regional median cone-skew vectors in scanner coordinates.  This bounded model
was chosen because only three training regions are available.  A two-coordinate
affine field and an eye-wide constant field are stored as secondary controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_cross_region_axis_transfer import eye_axes, fit_constant_skew, load_tracks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actual_seed(patch_dir: Path) -> np.ndarray:
    metadata = json.loads((patch_dir / "metadata.json").read_text())
    return np.asarray(metadata["actual_seed_xyz"], dtype=float)


def inverse_distance_squared(
    training_seeds: np.ndarray,
    training_values: np.ndarray,
    query_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(training_seeds - query_seed, axis=1)
    if np.any(distances <= 0):
        index = int(np.argmin(distances))
        weights = np.zeros(len(distances), dtype=float)
        weights[index] = 1.0
    else:
        weights = distances ** -2
        weights /= np.sum(weights)
    return weights @ training_values, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-label", type=Path, required=True)
    for number in (1, 2, 3):
        parser.add_argument(f"--patch-{number}-dir", type=Path, required=True)
        parser.add_argument(
            f"--patch-{number}-annotations", type=Path, required=True
        )
    parser.add_argument("--patch-4-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    axes = eye_axes(args.registered_label)
    patch_1 = load_tracks(
        "patch_1",
        args.patch_1_dir,
        args.patch_1_annotations,
        axes,
        selected_ids={1, 2, 3, 4, 5},
        depth_window=(34, 50),
    )
    patch_2 = load_tracks(
        "patch_2",
        args.patch_2_dir,
        args.patch_2_annotations,
        axes,
        selected_ids={1, 2, 3, 4, 5},
        depth_window=(30, 60),
    )
    patch_3 = load_tracks(
        "patch_3",
        args.patch_3_dir,
        args.patch_3_annotations,
        axes,
        selected_ids=set(range(2, 13)),
        minimum_nodes=6,
    )
    region_tracks = [patch_1, patch_2, patch_3]
    if tuple(map(len, region_tracks)) != (5, 5, 11):
        raise RuntimeError(
            f"Unexpected training counts: {tuple(map(len, region_tracks))}"
        )

    patch_dirs = [args.patch_1_dir, args.patch_2_dir, args.patch_3_dir]
    training_seeds = np.asarray([actual_seed(path) for path in patch_dirs])
    query_seed = actual_seed(args.patch_4_dir)
    regional_medians = np.asarray(
        [np.median([track.skew for track in tracks], axis=0) for tracks in region_tracks]
    )

    primary_prediction, primary_weights = inverse_distance_squared(
        training_seeds, regional_medians, query_seed
    )

    # Secondary affine surface field.  The first two whole-eye PCA coordinates
    # provide a deterministic 2-D chart; three region medians determine it.
    training_coordinates = training_seeds @ axes[:2].T
    query_coordinate = query_seed @ axes[:2].T
    affine_design = np.column_stack((np.ones(3), training_coordinates))
    affine_coefficients = np.linalg.solve(affine_design, regional_medians)
    affine_prediction = np.r_[1.0, query_coordinate] @ affine_coefficients
    constant_prediction = fit_constant_skew(patch_1 + patch_2 + patch_3)

    leave_one_region_out = []
    for held_out in range(3):
        keep = np.arange(3) != held_out
        predicted, weights = inverse_distance_squared(
            training_seeds[keep], regional_medians[keep], training_seeds[held_out]
        )
        leave_one_region_out.append(
            {
                "held_out_region": held_out + 1,
                "predicted_skew": predicted.tolist(),
                "observed_regional_median_skew": regional_medians[held_out].tolist(),
                "euclidean_skew_error": float(
                    np.linalg.norm(predicted - regional_medians[held_out])
                ),
                "weights_on_remaining_regions": weights.tolist(),
            }
        )

    annotation_paths = [
        args.patch_1_annotations,
        args.patch_2_annotations,
        args.patch_3_annotations,
    ]
    payload = {
        "status": "frozen before reading or scoring Patch 4 annotations",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(args.protocol),
        "registered_label_sha256": sha256(args.registered_label),
        "coordinate_system": "whole-eye PCA tangent frame",
        "voxel_size_micrometres": 1.08,
        "training": [
            {
                "region": index + 1,
                "seed_xyz": training_seeds[index].tolist(),
                "tracks": len(region_tracks[index]),
                "regional_median_skew": regional_medians[index].tolist(),
                "annotations_sha256": sha256(annotation_paths[index]),
            }
            for index in range(3)
        ],
        "patch_4_seed_xyz": query_seed.tolist(),
        "primary_model": {
            "name": "inverse-distance-squared regional-median field",
            "distance": "3-D Euclidean scanner-coordinate distance between frozen surface seeds",
            "exponent": 2,
            "weights_patch_1_to_3": primary_weights.tolist(),
            "predicted_skew": primary_prediction.tolist(),
        },
        "secondary_affine_model": {
            "name": "affine field in first two whole-eye PCA coordinates",
            "training_coordinates": training_coordinates.tolist(),
            "patch_4_coordinate": query_coordinate.tolist(),
            "coefficients": affine_coefficients.tolist(),
            "predicted_skew": affine_prediction.tolist(),
        },
        "secondary_constant_model": {
            "name": "median skew over all 21 training tracks",
            "predicted_skew": constant_prediction.tolist(),
        },
        "training_only_leave_one_region_out": leave_one_region_out,
        "claim_limit": (
            "This is a same-specimen three-region interpolation model. Patch 4 is a "
            "regional test, not independent biological validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

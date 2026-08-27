#!/usr/bin/env python3
"""Experiment 51: quantify the visual consequence of cone-axis tilt in Pieris.

The primary analysis is the prospectively held-out Patch 4 comparison between
surface normals and the spatial field frozen in Experiment 50.  Patches 1--3
are included only as leave-one-region-out diagnostics using the predictions
already stored in the frozen model.  No direction model is re-fitted here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import binomtest

from run_cross_region_axis_transfer import eye_axes, load_tracks


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.degrees(np.arccos(np.clip(np.dot(unit(left), unit(right)), -1.0, 1.0)))
    )


def direction(track, skew: np.ndarray) -> np.ndarray:
    return unit(
        track.normal + skew[0] * track.tangent_1 + skew[1] * track.tangent_2
    )


def pairwise_angles(vectors: np.ndarray) -> np.ndarray:
    dots = np.clip(vectors @ vectors.T, -1.0, 1.0)
    upper = np.triu_indices(len(vectors), 1)
    return np.degrees(np.arccos(dots[upper]))


def tangent_hull_area(vectors: np.ndarray) -> float:
    """Small-field tangent-plane convex-hull area, approximately steradians."""
    centre = unit(np.sum(vectors, axis=0))
    reference = np.asarray((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, centre))) > 0.8:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent_1 = unit(reference - np.dot(reference, centre) * centre)
    tangent_2 = unit(np.cross(centre, tangent_1))
    denominators = vectors @ centre
    if np.any(denominators <= 0):
        return float("nan")
    coordinates = np.column_stack(
        ((vectors @ tangent_1) / denominators, (vectors @ tangent_2) / denominators)
    )
    if len(coordinates) < 3:
        return float("nan")
    return float(ConvexHull(coordinates).volume)


def nearest_surface_pairs(tracks) -> list[tuple[int, int]]:
    anchors = np.asarray(
        [track.xyz[0] - track.depths[0] * track.normal for track in tracks]
    )
    distances = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return sorted(
        {tuple(sorted((index, int(np.argmin(distances[index]))))) for index in range(len(tracks))}
    )


def pair_values(vectors: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(
        [angle_degrees(vectors[left], vectors[right]) for left, right in pairs]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-label", type=Path, required=True)
    for number in (1, 2, 3, 4):
        parser.add_argument(f"--patch-{number}-dir", type=Path, required=True)
        parser.add_argument(
            f"--patch-{number}-annotations", type=Path, required=True
        )
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_model.read_text())
    axes = eye_axes(args.registered_label)
    regions = {
        "patch_1": load_tracks(
            "patch_1",
            args.patch_1_dir,
            args.patch_1_annotations,
            axes,
            selected_ids={1, 2, 3, 4, 5},
            depth_window=(34, 50),
        ),
        "patch_2": load_tracks(
            "patch_2",
            args.patch_2_dir,
            args.patch_2_annotations,
            axes,
            selected_ids={1, 2, 3, 4, 5},
            depth_window=(30, 60),
        ),
        "patch_3": load_tracks(
            "patch_3",
            args.patch_3_dir,
            args.patch_3_annotations,
            axes,
            selected_ids=set(range(2, 13)),
            minimum_nodes=6,
        ),
        "patch_4": load_tracks(
            "patch_4",
            args.patch_4_dir,
            args.patch_4_annotations,
            axes,
            selected_ids={1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15},
            minimum_nodes=6,
        ),
    }
    expected = {"patch_1": 5, "patch_2": 5, "patch_3": 11, "patch_4": 13}
    if {key: len(value) for key, value in regions.items()} != expected:
        raise RuntimeError(f"Unexpected track counts: { {key: len(value) for key, value in regions.items()} }")

    predictions = {
        f"patch_{row['held_out_region']}": np.asarray(row["predicted_skew"], float)
        for row in frozen["training_only_leave_one_region_out"]
    }
    predictions["patch_4"] = np.asarray(
        frozen["primary_model"]["predicted_skew"], float
    )

    per_track = []
    regional = []
    region_arrays = {}
    for region, tracks in regions.items():
        predicted_skew = predictions[region]
        normal_axes = np.asarray([unit(track.normal) for track in tracks])
        observed_axes = np.asarray([direction(track, track.skew) for track in tracks])
        predicted_axes = np.asarray(
            [direction(track, predicted_skew) for track in tracks]
        )
        normal_errors = np.asarray(
            [angle_degrees(normal_axes[i], observed_axes[i]) for i in range(len(tracks))]
        )
        predicted_errors = np.asarray(
            [angle_degrees(predicted_axes[i], observed_axes[i]) for i in range(len(tracks))]
        )
        for index, track in enumerate(tracks):
            per_track.append(
                {
                    "region": region,
                    "cone_id": track.cone_id,
                    "normal_axis_error_degrees": normal_errors[index],
                    "predicted_axis_error_degrees": predicted_errors[index],
                    "predicted_beats_normal": bool(
                        predicted_errors[index] < normal_errors[index]
                    ),
                    "observed_skew_1": float(track.skew[0]),
                    "observed_skew_2": float(track.skew[1]),
                    "predicted_skew_1": float(predicted_skew[0]),
                    "predicted_skew_2": float(predicted_skew[1]),
                }
            )

        pairs = nearest_surface_pairs(tracks)
        normal_neighbours = pair_values(normal_axes, pairs)
        observed_neighbours = pair_values(observed_axes, pairs)
        predicted_neighbours = pair_values(predicted_axes, pairs)
        normal_pairwise = pairwise_angles(normal_axes)
        observed_pairwise = pairwise_angles(observed_axes)
        predicted_pairwise = pairwise_angles(predicted_axes)
        regional.append(
            {
                "region": region,
                "tracks": len(tracks),
                "prediction_status": (
                    "prospective frozen test" if region == "patch_4" else "training-only leave-one-region-out diagnostic"
                ),
                "median_normal_axis_error_degrees": float(np.median(normal_errors)),
                "median_predicted_axis_error_degrees": float(np.median(predicted_errors)),
                "predicted_wins": int(np.sum(predicted_errors < normal_errors)),
                "normal_local_span_degrees": float(np.max(normal_pairwise)),
                "observed_local_span_degrees": float(np.max(observed_pairwise)),
                "predicted_local_span_degrees": float(np.max(predicted_pairwise)),
                "normal_pairwise_geometry_rmse_degrees": float(
                    np.sqrt(np.mean((normal_pairwise - observed_pairwise) ** 2))
                ),
                "predicted_pairwise_geometry_rmse_degrees": float(
                    np.sqrt(np.mean((predicted_pairwise - observed_pairwise) ** 2))
                ),
                "normal_tangent_hull_area_sr_approx": tangent_hull_area(normal_axes),
                "observed_tangent_hull_area_sr_approx": tangent_hull_area(observed_axes),
                "predicted_tangent_hull_area_sr_approx": tangent_hull_area(predicted_axes),
                "median_normal_nearest_traced_angle_degrees": float(np.median(normal_neighbours)),
                "median_observed_nearest_traced_angle_degrees": float(np.median(observed_neighbours)),
                "median_predicted_nearest_traced_angle_degrees": float(np.median(predicted_neighbours)),
            }
        )
        region_arrays[region] = (normal_errors, predicted_errors, normal_axes, observed_axes, predicted_axes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_track_axis_error.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_track[0]))
        writer.writeheader()
        writer.writerows(per_track)
    with (args.output_dir / "regional_functional_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regional[0]))
        writer.writeheader()
        writer.writerows(regional)

    patch_4_normal, patch_4_predicted, *_ = region_arrays["patch_4"]
    patch_4_wins = int(np.sum(patch_4_predicted < patch_4_normal))
    all_normal = np.concatenate([value[0] for value in region_arrays.values()])
    all_predicted = np.concatenate([value[1] for value in region_arrays.values()])
    summary = {
        "status": "functional consequence of manual and frozen-predicted visual axes",
        "primary_test": {
            "region": "patch_4",
            "status": "prospective frozen regional test from Experiment 50",
            "tracks": len(patch_4_normal),
            "median_surface_normal_direction_error_degrees": float(np.median(patch_4_normal)),
            "median_frozen_field_direction_error_degrees": float(np.median(patch_4_predicted)),
            "median_paired_improvement_degrees": float(
                np.median(patch_4_normal - patch_4_predicted)
            ),
            "frozen_field_wins": patch_4_wins,
            "two_sided_exact_sign_test_p": float(
                binomtest(patch_4_wins, len(patch_4_normal), 0.5).pvalue
            ),
        },
        "secondary_all_region_diagnostic": {
            "tracks": len(all_normal),
            "median_surface_normal_direction_error_degrees": float(np.median(all_normal)),
            "median_region_holdout_prediction_error_degrees": float(np.median(all_predicted)),
            "region_holdout_wins": int(np.sum(all_predicted < all_normal)),
        },
        "regional_metrics": regional,
        "interpretation": (
            "Surface normals materially misstate local visual-axis direction in the traced Pieris regions. "
            "The frozen field corrects most of that error in prospective Patch 4, but leave-one-region-out "
            "performance is not reliable across the whole eye."
        ),
        "claim_limits": [
            "The manual paths are not author-provided anatomical cone segmentations.",
            "The local span and hull metrics describe sparse traced regions, not the full-eye field of view.",
            "Nearest traced-neighbour angles are not equivalent to true adjacent-facet interommatidial angles.",
            "All data come from one specimen and one annotator.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, figure_axes = plt.subplots(2, 2, figsize=(11.8, 9.2), dpi=180)
    colours = {"patch_1": "#0072B2", "patch_2": "#D55E00", "patch_3": "#009E73", "patch_4": "#CC79A7"}
    axis = figure_axes[0, 0]
    offset = 0
    for region in regions:
        normal_errors, predicted_errors, *_ = region_arrays[region]
        for left, right in zip(normal_errors, predicted_errors):
            axis.plot((0, 1), (left, right), color=colours[region], alpha=0.45, linewidth=0.8)
        axis.scatter(np.zeros(len(normal_errors)), normal_errors, s=20, color=colours[region])
        axis.scatter(np.ones(len(predicted_errors)), predicted_errors, s=20, color=colours[region], label=region.replace("_", " "))
        offset += len(normal_errors)
    axis.set_xticks((0, 1), ("Surface normal", "Region-held-out field"))
    axis.set_ylabel("Visual-axis direction error (degrees)")
    axis.set_title("A  Normal versus predicted visual direction")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[-4:], labels[-4:], frameon=False, fontsize=8)

    axis = figure_axes[0, 1]
    patch_4_tracks = regions["patch_4"]
    observed_skew = np.asarray([track.skew for track in patch_4_tracks])
    predicted_skew = predictions["patch_4"]
    for observed in observed_skew:
        axis.plot((0, observed[0]), (0, observed[1]), color="#56B4E9", alpha=0.55, linewidth=1.0)
    axis.scatter(observed_skew[:, 0], observed_skew[:, 1], color="#56B4E9", label="Measured Patch 4")
    axis.scatter(*predicted_skew, marker="*", s=140, color="#CC79A7", label="Frozen prediction")
    axis.scatter(0, 0, marker="x", s=65, color="#555555", label="Surface normal")
    axis.set_xlabel("Eye-frame skew component 1")
    axis.set_ylabel("Eye-frame skew component 2")
    axis.set_title("B  Prospective Patch 4 direction")
    axis.legend(frameon=False, fontsize=8)

    axis = figure_axes[1, 0]
    x = np.arange(4)
    width = 0.24
    normal_span = [row["normal_local_span_degrees"] for row in regional]
    observed_span = [row["observed_local_span_degrees"] for row in regional]
    predicted_span = [row["predicted_local_span_degrees"] for row in regional]
    axis.bar(x - width, normal_span, width, label="Surface normals", color="#777777")
    axis.bar(x, observed_span, width, label="Measured axes", color="#56B4E9")
    axis.bar(x + width, predicted_span, width, label="Predicted field", color="#CC79A7")
    axis.set_xticks(x, [f"Patch {index}" for index in range(1, 5)])
    axis.set_ylabel("Maximum sampled angular span (degrees)")
    axis.set_title("C  Sparse local field envelope")
    axis.legend(frameon=False, fontsize=8)

    axis = figure_axes[1, 1]
    normal_geometry = [row["normal_pairwise_geometry_rmse_degrees"] for row in regional]
    predicted_geometry = [row["predicted_pairwise_geometry_rmse_degrees"] for row in regional]
    axis.bar(x - width / 2, normal_geometry, width, label="Surface normals", color="#777777")
    axis.bar(x + width / 2, predicted_geometry, width, label="Predicted field", color="#CC79A7")
    axis.set_xticks(x, [f"Patch {index}" for index in range(1, 5)])
    axis.set_ylabel("Pairwise angular-geometry RMSE (degrees)")
    axis.set_title("D  Error in sampled visual-field geometry")
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_51_functional_axis_consequence.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

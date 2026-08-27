#!/usr/bin/env python3
"""Experiment 45: adapted blind transfer in one Pieris patch.

This is an exploratory pilot. Outer facet centres are detected at the corneal
surface (depth 0). Internal cone centres and held-out target voxels are used
only as ground truth. For every test block, centre mapping and population
templates are learned exclusively from the other spatial blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    center_of_mass,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
    shift,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


def detect_outer_centres(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    interior_distance = distance_transform_edt((image > 240) & roi)
    plateau = (
        (interior_distance >= maximum_filter(interior_distance, size=11))
        & (interior_distance >= 1.5)
        & roi
    )
    components, count = label(plateau)
    return np.asarray(
        center_of_mass(interior_distance, components, range(1, count + 1)),
        dtype=np.float64,
    )


def detect_internal_centres(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image.astype(np.float64), sigma=2.0)
    candidates = (
        (smooth >= maximum_filter(smooth, size=11))
        & (smooth > np.percentile(smooth[roi], 50))
        & roi
    )
    return np.argwhere(candidates).astype(np.float64)


def associate_lattices(
    outer: np.ndarray, inner: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    best: tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for row_shift in np.arange(-5.0, 5.01, 0.5):
        for column_shift in np.arange(-5.0, 5.01, 0.5):
            translated = outer + (row_shift, column_shift)
            cost = np.linalg.norm(translated[:, None, :] - inner[None, :, :], axis=2)
            outer_index, inner_index = linear_sum_assignment(cost)
            distances = cost[outer_index, inner_index]
            score = float(np.median(np.sort(distances)[: min(40, len(distances))]))
            if best is None or score < best[0]:
                best = (score, row_shift, column_shift, distances, outer_index, inner_index)
    assert best is not None
    _, row_shift, column_shift, distances, outer_index, inner_index = best
    keep = distances <= 4.0
    return (
        outer[outer_index[keep]],
        inner[inner_index[keep]],
        distances[keep],
        np.asarray((row_shift, column_shift), dtype=np.float64),
    )


def extract_patch(
    volume: np.ndarray, centre: np.ndarray, radius: int, depth_start: int, depth_stop: int
) -> np.ndarray:
    row, column = np.rint(centre).astype(int)
    return volume[
        row - radius : row + radius + 1,
        column - radius : column + radius + 1,
        depth_start:depth_stop,
    ].astype(np.float64)


def fit_translation(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return np.median(inner - outer, axis=0)


def fit_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(outer)), outer))
    coefficient, *_ = np.linalg.lstsq(design, inner, rcond=None)
    return coefficient


def predict_affine(coefficient: np.ndarray, outer: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(outer)), outer))
    return design @ coefficient


def select_mapping_on_training(outer: np.ndarray, inner: np.ndarray) -> str:
    translation_errors = []
    affine_errors = []
    for held_out in range(len(outer)):
        keep = np.arange(len(outer)) != held_out
        translation = fit_translation(outer[keep], inner[keep])
        affine = fit_affine(outer[keep], inner[keep])
        translation_errors.append(
            np.linalg.norm(outer[held_out] + translation - inner[held_out])
        )
        affine_errors.append(
            np.linalg.norm(predict_affine(affine, outer[held_out : held_out + 1])[0] - inner[held_out])
        )
    return "translation" if np.median(translation_errors) <= np.median(affine_errors) else "affine"


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed_flat = observed.ravel() - np.mean(observed)
    predicted_flat = predicted.ravel() - np.mean(predicted)
    denominator = np.linalg.norm(observed_flat) * np.linalg.norm(predicted_flat)
    return float(observed_flat @ predicted_flat / denominator) if denominator else 0.0


def radialise_template(template: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(template.shape[:2])
    centre = (np.asarray(template.shape[:2]) - 1) / 2
    radius = np.rint(np.sqrt((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2)).astype(int)
    result = np.empty_like(template)
    for radial_bin in np.unique(radius):
        mask = radius == radial_bin
        result[mask, :] = np.median(template[mask, :], axis=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-micrometres", type=float, default=1.08)
    parser.add_argument("--outer-depth", type=int, required=True)
    parser.add_argument(
        "--mapping-rule", choices=("affine", "training-loo"), default="affine"
    )
    parser.add_argument("--evaluation-radius", type=int, default=6)
    args = parser.parse_args()

    data = np.load(args.patch_dir / "unfolded_intensity.npy")
    anatomical_labels = np.load(args.patch_dir / "unfolded_labels.npy")
    metadata = json.loads((args.patch_dir / "metadata.json").read_text(encoding="utf-8"))
    cone_depth = int(metadata["cone_layer_peak_depth_voxels"])
    layer_start = max(0, cone_depth - 6)
    layer_stop = min(data.shape[2], cone_depth + 12)
    cone_field = np.mean(
        anatomical_labels[:, :, layer_start:layer_stop] == 3, axis=2
    ) > 0.55
    roi = binary_erosion(cone_field, iterations=6)
    if np.count_nonzero(roi) < 1000:
        raise ValueError("The local patch does not contain a sufficiently large cone field")

    outer_all = detect_internal_centres(data[:, :, args.outer_depth], roi)
    internal_detection_depth = max(0, cone_depth - 2)
    inner_all = detect_internal_centres(data[:, :, internal_detection_depth], roi)
    outer, inner, association_distance, exploratory_shift = associate_lattices(
        outer_all, inner_all
    )

    radius = 6
    depth_start = max(0, cone_depth - 8)
    depth_stop = min(data.shape[2], cone_depth + 16)
    safe = (
        (inner[:, 0] >= radius)
        & (inner[:, 0] < data.shape[0] - radius)
        & (inner[:, 1] >= radius)
        & (inner[:, 1] < data.shape[1] - radius)
    )
    outer, inner, association_distance = outer[safe], inner[safe], association_distance[safe]
    patches = np.stack(
        [extract_patch(data, point, radius, depth_start, depth_stop) for point in inner]
    )

    block = KMeans(n_clusters=5, random_state=44, n_init=20).fit_predict(outer)
    spatial_tree = cKDTree(outer)
    circular = (
        (np.indices((2 * radius + 1, 2 * radius + 1))[0] - radius) ** 2
        + (np.indices((2 * radius + 1, 2 * radius + 1))[1] - radius) ** 2
        <= args.evaluation_radius**2
    )
    evaluation_mask = np.repeat(circular[:, :, None], depth_stop - depth_start, axis=2)

    records: list[dict[str, float | int | str]] = []
    predictions: dict[tuple[int, str], np.ndarray] = {}
    centre_predictions: dict[int, np.ndarray] = {}

    for fold in range(5):
        test_indices = np.flatnonzero(block == fold)
        train_indices = np.flatnonzero(block != fold)
        translation = fit_translation(outer[train_indices], inner[train_indices])
        affine = fit_affine(outer[train_indices], inner[train_indices])
        predicted_translation = outer[test_indices] + translation
        predicted_affine = predict_affine(affine, outer[test_indices])
        mapping_used = (
            select_mapping_on_training(outer[train_indices], inner[train_indices])
            if args.mapping_rule == "training-loo"
            else "affine"
        )

        global_template = np.median(patches[train_indices], axis=0)
        axisymmetric_template = radialise_template(global_template)
        border_mask = ~circular
        background_profile = np.median(
            patches[train_indices][:, border_mask, :], axis=(0, 1)
        )
        background = np.broadcast_to(
            background_profile, (2 * radius + 1, 2 * radius + 1, len(background_profile))
        ).copy()

        for local_index, sample_index in enumerate(test_indices):
            observed = patches[sample_index]
            predicted_centre = (
                predicted_translation[local_index]
                if mapping_used == "translation"
                else predicted_affine[local_index]
            )
            centre_predictions[int(sample_index)] = predicted_centre
            displacement = np.r_[predicted_centre - inner[sample_index], 0.0]

            train_distance = np.linalg.norm(
                outer[train_indices] - outer[sample_index], axis=1
            )
            nearest_order = np.argsort(train_distance)
            nearest_template = patches[train_indices[nearest_order[0]]]
            local_template = np.median(
                patches[train_indices[nearest_order[: min(6, len(nearest_order))]]], axis=0
            )
            method_images = {
                "background": background,
                "global_population": shift(
                    global_template,
                    displacement,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                ),
                "axisymmetric_population": shift(
                    axisymmetric_template,
                    displacement,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                ),
                "nearest_cone": shift(
                    nearest_template,
                    displacement,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                ),
                "local_population": shift(
                    local_template,
                    displacement,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                ),
            }
            dynamic_range = np.percentile(observed[evaluation_mask], 95) - np.percentile(
                observed[evaluation_mask], 5
            )
            for method, predicted in method_images.items():
                error = np.abs(observed[evaluation_mask] - predicted[evaluation_mask])
                record = {
                    "sample": int(sample_index),
                    "fold": int(fold),
                    "method": method,
                    "mapping_used": mapping_used,
                    "mae_intensity": float(np.mean(error)),
                    "normalised_mae": float(np.mean(error) / max(dynamic_range, 1.0)),
                    "correlation": correlation(
                        observed[evaluation_mask], predicted[evaluation_mask]
                    ),
                    "centre_error_voxels_affine": float(
                        np.linalg.norm(predicted_centre - inner[sample_index])
                    ),
                    "centre_error_voxels_translation": float(
                        np.linalg.norm(
                            predicted_translation[local_index] - inner[sample_index]
                        )
                    ),
                    "centre_error_voxels_outer_only": float(
                        np.linalg.norm(outer[sample_index] - inner[sample_index])
                    ),
                }
                records.append(record)
                predictions[(int(sample_index), method)] = predicted

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with (args.output_dir / "per_cone_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    methods = [
        "background",
        "axisymmetric_population",
        "global_population",
        "nearest_cone",
        "local_population",
    ]
    summary: dict[str, object] = {
        "status": "independent-specimen transfer with a pre-scoring Pieris imaging adapter",
        "outer_detection": "local maxima in the shallow corneal lattice",
        "outer_detection_depth_voxels": args.outer_depth,
        "mapping_rule_used_for_reconstruction": args.mapping_rule,
        "evaluation_radius_voxels": args.evaluation_radius,
        "outer_centres_detected": int(len(outer_all)),
        "internal_centres_detected_for_ground_truth": int(len(inner_all)),
        "matched_cones": int(len(outer)),
        "exploratory_global_shift_row_column": exploratory_shift.tolist(),
        "association_distance_voxels_median": float(np.median(association_distance)),
        "cone_layer_peak_depth_voxels": cone_depth,
        "internal_ground_truth_detection_depth_voxels": internal_detection_depth,
        "reconstruction_depth_range_voxels": [depth_start, depth_stop],
        "voxel_size_micrometres": args.voxel_size_micrometres,
        "voxel_calibration_note": (
            "Uses the 1.08 micrometre Pieris acquisition voxel size from the "
            "MorphoSource manifest; the supplied NIfTI label header is unitless."
        ),
        "centre_mapping": {},
        "reconstruction": {},
    }
    sample_rows = [record for record in records if record["method"] == "background"]
    for key in (
        "centre_error_voxels_outer_only",
        "centre_error_voxels_translation",
        "centre_error_voxels_affine",
    ):
        values = np.asarray([float(record[key]) for record in sample_rows])
        summary["centre_mapping"][key] = {
            "median_voxels": float(np.median(values)),
            "median_micrometres": float(args.voxel_size_micrometres * np.median(values)),
            "p90_voxels": float(np.percentile(values, 90)),
        }
    for method in methods:
        subset = [record for record in records if record["method"] == method]
        summary["reconstruction"][method] = {
            "median_mae_intensity": float(
                np.median([float(record["mae_intensity"]) for record in subset])
            ),
            "median_normalised_mae": float(
                np.median([float(record["normalised_mae"]) for record in subset])
            ),
            "median_correlation": float(
                np.median([float(record["correlation"]) for record in subset])
            ),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    representative = []
    centre_errors = np.asarray(
        [
            float(record["centre_error_voxels_affine"])
            for record in records
            if record["method"] == "background"
        ]
    )
    samples = np.asarray(
        [int(record["sample"]) for record in records if record["method"] == "background"]
    )
    for quantile in (0.2, 0.5, 0.8):
        target = np.quantile(centre_errors, quantile)
        representative.append(int(samples[np.argmin(np.abs(centre_errors - target))]))

    figure, axes = plt.subplots(len(representative), 6, figsize=(14, 7), dpi=170)
    display_depth = internal_detection_depth - depth_start
    titles = [
        "Observed (hidden)",
        "Background",
        "Axisymmetric",
        "Global",
        "Nearest",
        "Local population",
    ]
    for row, sample_index in enumerate(representative):
        images = [
            patches[sample_index],
            predictions[(sample_index, "background")],
            predictions[(sample_index, "axisymmetric_population")],
            predictions[(sample_index, "global_population")],
            predictions[(sample_index, "nearest_cone")],
            predictions[(sample_index, "local_population")],
        ]
        lo, hi = np.percentile(images[0][:, :, display_depth], (2, 98))
        for column, image in enumerate(images):
            axes[row, column].imshow(
                np.clip((image[:, :, display_depth] - lo) / (hi - lo + 1e-6), 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=9)
        axes[row, 0].set_ylabel(f"cone {sample_index}", fontsize=8)
    figure.suptitle("Experiment 45: Pieris spatial-block transfer", fontsize=12)
    figure.tight_layout()
    figure.savefig(args.output_dir / "blind_reconstruction_examples.png", bbox_inches="tight")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=170)
    for position, metric in enumerate(("normalised_mae", "correlation")):
        values = [
            [float(record[metric]) for record in records if record["method"] == method]
            for method in methods
        ]
        axes[position].boxplot(
            values,
            tick_labels=["Background", "Axisymmetric", "Global", "Nearest", "Local"],
        )
        axes[position].set_ylabel(metric.replace("_", " "))
        axes[position].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "blind_reconstruction_metrics.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


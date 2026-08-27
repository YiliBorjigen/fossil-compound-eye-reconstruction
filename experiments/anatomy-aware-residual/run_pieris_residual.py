#!/usr/bin/env python3
"""Experiment 46: anatomy-aware residual reconstruction in Pieris.

This is method development after the failed independent transfer in Experiment
45. It is not an independent validation. Hyperparameters express anatomical
choices made before scoring this script: a four-voxel hidden core, a surviving
five-to-six-voxel annulus, and orientation modulo the hexagonal lattice's 60
degree symmetry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    affine_transform,
    binary_erosion,
    gaussian_filter,
    maximum_filter,
    shift,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


METHODS = (
    "local_depth_background",
    "raw_population",
    "axisymmetric_residual",
    "anatomy_residual",
    "oracle_centred_residual",
)


def detect_centres(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image.astype(np.float64), sigma=2.0)
    candidates = (
        (smooth >= maximum_filter(smooth, size=11))
        & (smooth > np.percentile(smooth[roi], 50))
        & roi
    )
    return np.argwhere(candidates).astype(np.float64)


def associate_lattices(
    outer: np.ndarray, inner: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = None
    for row_shift in np.arange(-5.0, 5.01, 0.5):
        for column_shift in np.arange(-5.0, 5.01, 0.5):
            translated = outer + (row_shift, column_shift)
            cost = np.linalg.norm(translated[:, None, :] - inner[None, :, :], axis=2)
            outer_index, inner_index = linear_sum_assignment(cost)
            distances = cost[outer_index, inner_index]
            score = float(np.median(np.sort(distances)[: min(40, len(distances))]))
            if best is None or score < best[0]:
                best = (score, distances, outer_index, inner_index)
    if best is None:
        raise RuntimeError("No lattice association was found")
    _, distances, outer_index, inner_index = best
    keep = distances <= 4.0
    return outer[outer_index[keep]], inner[inner_index[keep]], distances[keep]


def extract_patch(
    volume: np.ndarray,
    centre: np.ndarray,
    radius: int,
    depth_start: int,
    depth_stop: int,
) -> np.ndarray:
    row, column = np.rint(centre).astype(int)
    return volume[
        row - radius : row + radius + 1,
        column - radius : column + radius + 1,
        depth_start:depth_stop,
    ].astype(np.float64)


def fit_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(outer)), outer))
    coefficient, *_ = np.linalg.lstsq(design, inner, rcond=None)
    return coefficient


def predict_affine(coefficient: np.ndarray, outer: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(outer)), outer)) @ coefficient


def local_frames(field: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return local spacing and lattice angle modulo 60 degrees."""
    tree = cKDTree(field)
    spacings = []
    angles = []
    for point in points:
        _, indices = tree.query(point, k=min(8, len(field)))
        neighbours = field[np.atleast_1d(indices)]
        vectors = neighbours - point
        distances = np.linalg.norm(vectors, axis=1)
        keep = distances > 1e-6
        vectors = vectors[keep][:6]
        distances = distances[keep][:6]
        spacings.append(float(np.median(distances)))
        theta = np.arctan2(vectors[:, 0], vectors[:, 1])
        angles.append(float(np.angle(np.mean(np.exp(6j * theta))) / 6.0))
    return np.asarray(spacings), np.asarray(angles)


def warp_xy(volume: np.ndarray, angle: float, scale: float, to_canonical: bool) -> np.ndarray:
    """Rotate and scale around the patch centre without changing depth."""
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    xy = scale * rotation if to_canonical else rotation.T / scale
    matrix = np.eye(3)
    matrix[:2, :2] = xy
    centre = (np.asarray(volume.shape, dtype=float) - 1.0) / 2.0
    offset = centre - matrix @ centre
    return affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        output_shape=volume.shape,
        order=1,
        mode="nearest",
        prefilter=False,
    )


def depth_profile(volume: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    return np.median(volume[spatial_mask, :], axis=0)


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    left = observed.ravel() - np.mean(observed)
    right = predicted.ravel() - np.mean(predicted)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def radialise(template: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(template.shape[:2])
    centre = (np.asarray(template.shape[:2]) - 1) / 2
    radius = np.rint(np.hypot(rows - centre[0], columns - centre[1])).astype(int)
    result = np.empty_like(template)
    for value in np.unique(radius):
        mask = radius == value
        result[mask, :] = np.median(template[mask, :], axis=0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-depth", type=int, required=True)
    parser.add_argument("--voxel-size-micrometres", type=float, default=1.08)
    args = parser.parse_args()

    data = np.load(args.patch_dir / "unfolded_intensity.npy")
    labels = np.load(args.patch_dir / "unfolded_labels.npy")
    metadata = json.loads((args.patch_dir / "metadata.json").read_text())
    cone_depth = int(metadata["cone_layer_peak_depth_voxels"])
    layer_start = max(0, cone_depth - 6)
    layer_stop = min(data.shape[2], cone_depth + 12)
    cone_field = np.mean(labels[:, :, layer_start:layer_stop] == 3, axis=2) > 0.55
    roi = binary_erosion(cone_field, iterations=6)

    outer_all = detect_centres(data[:, :, args.outer_depth], roi)
    inner_all = detect_centres(data[:, :, max(0, cone_depth - 2)], roi)
    outer, inner, association_distance = associate_lattices(outer_all, inner_all)

    radius = 6
    depth_start = max(0, cone_depth - 8)
    depth_stop = min(data.shape[2], cone_depth + 16)
    safe = (
        (inner[:, 0] >= radius)
        & (inner[:, 0] < data.shape[0] - radius)
        & (inner[:, 1] >= radius)
        & (inner[:, 1] < data.shape[1] - radius)
    )
    outer, inner = outer[safe], inner[safe]
    association_distance = association_distance[safe]
    patches = np.stack(
        [extract_patch(data, point, radius, depth_start, depth_stop) for point in inner]
    )
    spacing, angle = local_frames(outer_all, outer)
    block = KMeans(n_clusters=5, random_state=46, n_init=20).fit_predict(outer)

    rows, columns = np.indices((2 * radius + 1, 2 * radius + 1))
    radial_distance = np.hypot(rows - radius, columns - radius)
    hidden_core = radial_distance <= 4.0
    training_annulus = (radial_distance >= 5.0) & (radial_distance <= 6.0)
    evaluation_mask = np.repeat(hidden_core[:, :, None], depth_stop - depth_start, axis=2)

    records: list[dict[str, float | int | str]] = []
    example_images: dict[tuple[int, str], np.ndarray] = {}
    for fold in range(5):
        test_indices = np.flatnonzero(block == fold)
        train_indices = np.flatnonzero(block != fold)
        coefficient = fit_affine(outer[train_indices], inner[train_indices])
        predicted_centres = predict_affine(coefficient, outer[test_indices])
        reference_spacing = float(np.median(spacing[train_indices]))

        normalised_residuals = []
        raw_training = []
        for index in train_indices:
            profile = depth_profile(patches[index], training_annulus)
            contrast = max(
                float(np.percentile(patches[index][training_annulus, :], 95)
                      - np.percentile(patches[index][training_annulus, :], 5)),
                1.0,
            )
            residual = (patches[index] - profile[None, None, :]) / contrast
            normalised_residuals.append(
                warp_xy(residual, angle[index], spacing[index] / reference_spacing, True)
            )
            raw_training.append(patches[index])
        canonical_template = np.median(np.stack(normalised_residuals), axis=0)
        axisymmetric_template = radialise(canonical_template)
        raw_template = np.median(np.stack(raw_training), axis=0)

        for local_index, sample_index in enumerate(test_indices):
            observed = patches[sample_index]
            displacement_xy = predicted_centres[local_index] - inner[sample_index]
            predicted_centre_in_patch = np.asarray((radius, radius)) + displacement_xy
            predicted_radius = np.hypot(
                rows - predicted_centre_in_patch[0],
                columns - predicted_centre_in_patch[1],
            )
            available_annulus = (
                (predicted_radius >= 5.0)
                & (predicted_radius <= 6.0)
                & (~hidden_core)
            )
            if np.count_nonzero(available_annulus) < 10:
                available_annulus = training_annulus & (~hidden_core)
            local_profile = depth_profile(observed, available_annulus)
            local_contrast = max(
                float(np.percentile(observed[available_annulus, :], 95)
                      - np.percentile(observed[available_annulus, :], 5)),
                1.0,
            )
            background = np.broadcast_to(local_profile, observed.shape).copy()

            anatomy_residual = warp_xy(
                canonical_template,
                angle[sample_index],
                spacing[sample_index] / reference_spacing,
                False,
            )
            axisymmetric_residual = warp_xy(
                axisymmetric_template,
                0.0,
                spacing[sample_index] / reference_spacing,
                False,
            )
            displacement = np.r_[displacement_xy, 0.0]
            method_images = {
                "local_depth_background": background,
                "raw_population": shift(
                    raw_template, displacement, order=1, mode="nearest", prefilter=False
                ),
                "axisymmetric_residual": background
                + local_contrast
                * shift(axisymmetric_residual, displacement, order=1, mode="nearest", prefilter=False),
                "anatomy_residual": background
                + local_contrast
                * shift(anatomy_residual, displacement, order=1, mode="nearest", prefilter=False),
                "oracle_centred_residual": background + local_contrast * anatomy_residual,
            }
            dynamic_range = max(
                float(np.percentile(observed[evaluation_mask], 95)
                      - np.percentile(observed[evaluation_mask], 5)),
                1.0,
            )
            for method, predicted in method_images.items():
                error = np.abs(observed[evaluation_mask] - predicted[evaluation_mask])
                records.append(
                    {
                        "sample": int(sample_index),
                        "fold": int(fold),
                        "method": method,
                        "normalised_mae": float(np.mean(error) / dynamic_range),
                        "correlation": correlation(observed[evaluation_mask], predicted[evaluation_mask]),
                        "centre_error_voxels": float(np.linalg.norm(displacement_xy)),
                        "local_spacing_voxels": float(spacing[sample_index]),
                        "lattice_angle_radians": float(angle[sample_index]),
                    }
                )
                example_images[(int(sample_index), method)] = predicted

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_cone_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, object] = {
        "status": "post-transfer method development; not independent validation",
        "matched_cones": int(len(outer)),
        "folds": 5,
        "hidden_core_radius_voxels": 4,
        "available_annulus_voxels": [5, 6],
        "association_distance_median_voxels": float(np.median(association_distance)),
        "centre_error_median_voxels": float(
            np.median([r["centre_error_voxels"] for r in records if r["method"] == METHODS[0]])
        ),
        "voxel_size_micrometres": args.voxel_size_micrometres,
        "methods": {},
    }
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        summary["methods"][method] = {
            "median_normalised_mae": float(np.median([r["normalised_mae"] for r in subset])),
            "median_correlation": float(np.median([r["correlation"] for r in subset])),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    method_labels = ["Background", "Raw", "Axis residual", "Anatomy residual", "Oracle centre"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    values = [
        [r["normalised_mae"] for r in records if r["method"] == method]
        for method in METHODS
    ]
    axes[0].boxplot(values, tick_labels=method_labels, showfliers=False)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Normalised MAE in hidden core")
    axes[0].set_title("A  Anatomy-aware residual test")

    background = {
        int(r["sample"]): float(r["normalised_mae"])
        for r in records if r["method"] == "local_depth_background"
    }
    anatomy = {
        int(r["sample"]): float(r["normalised_mae"])
        for r in records if r["method"] == "anatomy_residual"
    }
    centres = {
        int(r["sample"]): float(r["centre_error_voxels"])
        for r in records if r["method"] == "local_depth_background"
    }
    samples = sorted(background)
    axes[1].scatter(
        [centres[sample] for sample in samples],
        [anatomy[sample] - background[sample] for sample in samples],
        s=24,
        alpha=0.75,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Predicted centre error (voxels)")
    axes[1].set_ylabel("Anatomy residual − background MAE")
    axes[1].set_title("B  Registration remains a separate error")
    figure.tight_layout()
    figure.savefig(args.output_dir / "anatomy_residual_metrics.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

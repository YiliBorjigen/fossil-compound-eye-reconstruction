#!/usr/bin/env python3
"""Audit normal estimation and hypothetical deformation in the Asaphus crop.

The affine transforms below are sensitivity scenarios, not recovered
retrodeformations.  With one cropped eye and no independent strain marker, the
original biological geometry is not identifiable from this volume alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from run_visual_field_sensitivity import (
    CENTRAL_T,
    THRESHOLDS,
    adjacent_pairs,
    angular_separation,
    gnomonic_coordinates,
    read_nrrd,
    spherical_hull_area,
    subvoxel_surface,
    tangent_frame,
    unit,
)


SURFACE_SMOOTHING_SIGMAS = (1.0, 2.0, 4.0, 8.0, 12.0, 15.0)
VOLUME_AXIS_SIGMAS = (0.5, 1.0, 1.5, 2.0)
SAMPLE_OFFSETS = (
    (-0.5, -0.5),
    (-0.5, 0.0),
    (-0.5, 0.5),
    (0.0, -0.5),
    (0.0, 0.5),
    (0.5, -0.5),
    (0.5, 0.0),
    (0.5, 0.5),
)


def load_baseline(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    facet_ids = np.asarray(table["facet_id"], int)
    points = np.column_stack((table["x_vox"], table["y_vox"], table["z_vox"]))
    normals = unit(
        np.column_stack((table["normal_x"], table["normal_y"], table["normal_z"]))
    )
    return facet_ids, points, normals


def normals_from_filled_surface(
    filled_surface: np.ndarray,
    points: np.ndarray,
    smoothing_sigma: float,
    offset_yz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    smooth_surface = ndi.gaussian_filter(filled_surface, sigma=smoothing_sigma)
    gradient_y, gradient_z = np.gradient(smooth_surface)
    coordinates = np.vstack(
        (
            points[:, 1] + offset_yz[0],
            points[:, 2] + offset_yz[1],
        )
    )
    sampled_y = ndi.map_coordinates(
        gradient_y, coordinates, order=1, mode="nearest"
    )
    sampled_z = ndi.map_coordinates(
        gradient_z, coordinates, order=1, mode="nearest"
    )
    return unit(np.column_stack((np.ones(len(points)), -sampled_y, -sampled_z)))


def angular_metrics(
    points: np.ndarray, normals: np.ndarray, pairs: np.ndarray
) -> dict[str, float | bool]:
    pairwise = angular_separation(normals[:, None, :], normals[None, :, :])
    neighbour_angles = angular_separation(normals[pairs[:, 0]], normals[pairs[:, 1]])
    mean_axis = unit(np.sum(normals, axis=0))
    tangent_1, tangent_2 = tangent_frame(mean_axis)
    hemisphere_valid = bool(np.all(normals @ mean_axis > 0))
    if hemisphere_valid:
        coordinates = gnomonic_coordinates(normals, mean_axis, tangent_1, tangent_2)
        solid_angle, _ = spherical_hull_area(normals, coordinates, mean_axis)
    else:
        # A non-hemispheric result is itself evidence that this extraction
        # setting is unstable.  Keep the angular metrics and record the hull as
        # undefined instead of silently removing the offending facet.
        solid_angle = float("nan")
    return {
        "maximum_pairwise_span_degrees": float(np.max(pairwise)),
        "median_adjacent_normal_angle_degrees": float(np.median(neighbour_angles)),
        "p95_adjacent_normal_angle_degrees": float(np.percentile(neighbour_angles, 95)),
        "spherical_hull_solid_angle_sr": float(solid_angle),
        "single_hemisphere": hemisphere_valid,
    }


def comparison_metrics(
    candidate: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    changes = angular_separation(candidate, baseline)
    return {
        "median_normal_change_degrees": float(np.median(changes)),
        "p95_normal_change_degrees": float(np.percentile(changes, 95)),
        "maximum_normal_change_degrees": float(np.max(changes)),
    }


def transform_geometry(
    points: np.ndarray, normals: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.mean(points, axis=0)
    transformed_points = centre + (points - centre) @ matrix.T
    transformed_normals = unit(normals @ np.linalg.inv(matrix))
    return transformed_points, transformed_normals


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _, points, baseline_normals = load_baseline(args.baseline_csv)
    pairs = adjacent_pairs(points)
    baseline_metrics = angular_metrics(points, baseline_normals, pairs)

    volume = read_nrrd(args.input)
    nominal_smoothed = ndi.gaussian_filter1d(
        volume.astype(np.float32), sigma=1.0, axis=0
    )

    parameter_rows = []
    nominal_filled = None
    nominal_subpixel_normals = None
    for threshold in THRESHOLDS:
        _, filled = subvoxel_surface(nominal_smoothed, threshold)
        if threshold == CENTRAL_T:
            nominal_filled = filled
        for surface_sigma in SURFACE_SMOOTHING_SIGMAS:
            normals = normals_from_filled_surface(
                filled, points, smoothing_sigma=surface_sigma
            )
            if threshold == CENTRAL_T and surface_sigma == 2.0:
                nominal_subpixel_normals = normals
            parameter_rows.append(
                {
                    "audit": "threshold_and_surface_smoothing",
                    "volume_axis_sigma_vox": 1.0,
                    "surface_threshold": threshold,
                    "surface_smoothing_sigma_vox": surface_sigma,
                    **comparison_metrics(normals, baseline_normals),
                    **angular_metrics(points, normals, pairs),
                }
            )
    if nominal_filled is None or nominal_subpixel_normals is None:
        raise RuntimeError("Nominal surface was not constructed")

    offset_rows = []
    for offset_y, offset_z in SAMPLE_OFFSETS:
        normals = normals_from_filled_surface(
            nominal_filled,
            points,
            smoothing_sigma=2.0,
            offset_yz=(offset_y, offset_z),
        )
        offset_rows.append(
            {
                "offset_y_vox": offset_y,
                "offset_z_vox": offset_z,
                **comparison_metrics(normals, nominal_subpixel_normals),
                **angular_metrics(points, normals, pairs),
            }
        )

    del nominal_smoothed
    for volume_sigma in VOLUME_AXIS_SIGMAS:
        if volume_sigma == 1.0:
            continue
        smoothed = ndi.gaussian_filter1d(
            volume.astype(np.float32), sigma=volume_sigma, axis=0
        )
        _, filled = subvoxel_surface(smoothed, CENTRAL_T)
        normals = normals_from_filled_surface(filled, points, smoothing_sigma=2.0)
        parameter_rows.append(
            {
                "audit": "volume_axis_smoothing",
                "volume_axis_sigma_vox": volume_sigma,
                "surface_threshold": CENTRAL_T,
                "surface_smoothing_sigma_vox": 2.0,
                **comparison_metrics(normals, baseline_normals),
                **angular_metrics(points, normals, pairs),
            }
        )
        del smoothed

    centred = points - np.mean(points, axis=0)
    _, _, principal_rows = np.linalg.svd(centred, full_matrices=False)
    principal_basis = principal_rows.T
    deformation_rows = []
    for axis_index in range(3):
        for factor in (0.8, 0.9, 1.1, 1.2):
            principal_matrix = np.eye(3)
            principal_matrix[axis_index, axis_index] = factor
            matrix = principal_basis @ principal_matrix @ principal_basis.T
            transformed_points, transformed_normals = transform_geometry(
                points, baseline_normals, matrix
            )
            deformation_rows.append(
                {
                    "scenario": "principal_axis_scaling",
                    "axis_from": axis_index + 1,
                    "axis_towards": "",
                    "magnitude": factor - 1.0,
                    "determinant": float(np.linalg.det(matrix)),
                    **comparison_metrics(transformed_normals, baseline_normals),
                    **angular_metrics(transformed_points, transformed_normals, pairs),
                }
            )
    for axis_from in range(3):
        for axis_towards in range(3):
            if axis_from == axis_towards:
                continue
            for shear in (-0.2, -0.1, 0.1, 0.2):
                principal_matrix = np.eye(3)
                principal_matrix[axis_from, axis_towards] = shear
                matrix = principal_basis @ principal_matrix @ principal_basis.T
                transformed_points, transformed_normals = transform_geometry(
                    points, baseline_normals, matrix
                )
                deformation_rows.append(
                    {
                        "scenario": "principal_axis_shear",
                        "axis_from": axis_from + 1,
                        "axis_towards": axis_towards + 1,
                        "magnitude": shear,
                        "determinant": float(np.linalg.det(matrix)),
                        **comparison_metrics(transformed_normals, baseline_normals),
                        **angular_metrics(transformed_points, transformed_normals, pairs),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "normal_parameter_sweep.csv", parameter_rows)
    write_rows(args.output_dir / "sampling_offset_audit.csv", offset_rows)
    write_rows(args.output_dir / "affine_deformation_sensitivity.csv", deformation_rows)

    threshold_nominal = [
        row
        for row in parameter_rows
        if row["audit"] == "threshold_and_surface_smoothing"
        and row["surface_smoothing_sigma_vox"] == 2.0
    ]
    smoothing_nominal = [
        row
        for row in parameter_rows
        if row["audit"] == "threshold_and_surface_smoothing"
        and row["surface_threshold"] == CENTRAL_T
    ]
    deformation_10 = [
        row for row in deformation_rows if np.isclose(abs(row["magnitude"]), 0.1)
    ]
    deformation_20 = [
        row for row in deformation_rows if np.isclose(abs(row["magnitude"]), 0.2)
    ]

    def range_of(rows: list[dict], key: str) -> list[float]:
        values = [float(row[key]) for row in rows]
        return [float(np.min(values)), float(np.max(values))]

    summary = {
        "status": "preserved-surface normal and hypothetical affine-deformation sensitivity audit",
        "facets": len(points),
        "baseline": baseline_metrics,
        "normal_estimation": {
            "subpixel_sampling_change_from_experiment_52_degrees": comparison_metrics(
                nominal_subpixel_normals, baseline_normals
            ),
            "threshold_45_to_70_at_surface_sigma_2": {
                "maximum_span_range_degrees": range_of(
                    threshold_nominal, "maximum_pairwise_span_degrees"
                ),
                "median_neighbour_angle_range_degrees": range_of(
                    threshold_nominal, "median_adjacent_normal_angle_degrees"
                ),
                "largest_p95_per_facet_change_degrees": float(
                    max(row["p95_normal_change_degrees"] for row in threshold_nominal)
                ),
            },
            "surface_smoothing_sigma_1_to_15_at_threshold_50": {
                "maximum_span_range_degrees": range_of(
                    smoothing_nominal, "maximum_pairwise_span_degrees"
                ),
                "median_neighbour_angle_range_degrees": range_of(
                    smoothing_nominal, "median_adjacent_normal_angle_degrees"
                ),
                "largest_p95_per_facet_change_degrees": float(
                    max(row["p95_normal_change_degrees"] for row in smoothing_nominal)
                ),
            },
            "half_voxel_sampling_offsets": {
                "maximum_span_range_degrees": range_of(
                    offset_rows, "maximum_pairwise_span_degrees"
                ),
                "largest_p95_change_from_subpixel_nominal_degrees": float(
                    max(row["p95_normal_change_degrees"] for row in offset_rows)
                ),
            },
        },
        "hypothetical_affine_retrodeformation": {
            "ten_percent_scaling_or_shear": {
                "maximum_span_range_degrees": range_of(
                    deformation_10, "maximum_pairwise_span_degrees"
                ),
                "median_neighbour_angle_range_degrees": range_of(
                    deformation_10, "median_adjacent_normal_angle_degrees"
                ),
                "largest_p95_normal_change_degrees": float(
                    max(row["p95_normal_change_degrees"] for row in deformation_10)
                ),
            },
            "twenty_percent_scaling_or_shear": {
                "maximum_span_range_degrees": range_of(
                    deformation_20, "maximum_pairwise_span_degrees"
                ),
                "median_neighbour_angle_range_degrees": range_of(
                    deformation_20, "median_adjacent_normal_angle_degrees"
                ),
                "largest_p95_normal_change_degrees": float(
                    max(row["p95_normal_change_degrees"] for row in deformation_20)
                ),
            },
        },
        "interpretation": (
            "The output quantifies how the preserved-state surface-normal geometry changes under "
            "measurement choices and transparent affine strain scenarios. It does not identify the "
            "specimen's actual taphonomic strain or recover its living geometry."
        ),
        "claim_limits": [
            "The specimen-specific deformation field is unknown.",
            "The 10% and 20% affine scenarios are sensitivity tests, not estimated geological strain.",
            "Only a cropped field from one side of one specimen is available, so bilateral retrodeformation is impossible here.",
            "Surface normals remain geometric normals rather than validated optical axes.",
            "No refractive, birefringent, rhabdom-length, sensitivity or acuity model is evaluated.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), dpi=180)
    axis = axes[0, 0]
    for threshold in THRESHOLDS:
        rows = [
            row
            for row in parameter_rows
            if row["audit"] == "threshold_and_surface_smoothing"
            and row["surface_threshold"] == threshold
        ]
        axis.plot(
            [row["surface_smoothing_sigma_vox"] for row in rows],
            [
                row["maximum_pairwise_span_degrees"]
                if row["single_hemisphere"]
                else np.nan
                for row in rows
            ],
            marker="o",
            markersize=3,
            label=f"T={threshold}",
        )
    axis.axhline(
        baseline_metrics["maximum_pairwise_span_degrees"],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel("Surface smoothing sigma (voxels)")
    axis.set_ylabel("Maximum normal span (degrees)")
    axis.set_title("A  Normal span versus extraction choices")
    axis.text(
        0.02,
        0.04,
        "T=70, sigma=1 failed the hemisphere check",
        transform=axis.transAxes,
        fontsize=8,
        color="#7A3E00",
    )
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[0, 1]
    rows = sorted(
        smoothing_nominal, key=lambda row: row["surface_smoothing_sigma_vox"]
    )
    axis.plot(
        [row["surface_smoothing_sigma_vox"] for row in rows],
        [row["median_adjacent_normal_angle_degrees"] for row in rows],
        marker="o",
        color="#009E73",
    )
    axis.axhline(
        baseline_metrics["median_adjacent_normal_angle_degrees"],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel("Surface smoothing sigma (voxels)")
    axis.set_ylabel("Median adjacent-normal angle (degrees)")
    axis.set_title("B  Neighbour angle is smoothing-dependent")

    axis = axes[1, 0]
    scales = [
        row for row in deformation_rows if row["scenario"] == "principal_axis_scaling"
    ]
    for axis_index in range(1, 4):
        rows = sorted(
            [row for row in scales if row["axis_from"] == axis_index],
            key=lambda row: row["magnitude"],
        )
        axis.plot(
            [100.0 * row["magnitude"] for row in rows],
            [row["maximum_pairwise_span_degrees"] for row in rows],
            marker="o",
            label=f"Principal axis {axis_index}",
        )
    axis.axhline(
        baseline_metrics["maximum_pairwise_span_degrees"],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel("Hypothetical principal-axis scaling (%)")
    axis.set_ylabel("Maximum normal span (degrees)")
    axis.set_title("C  Affine scaling sensitivity")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    labels = ("10% strain", "20% strain")
    span_ranges = (
        range_of(deformation_10, "maximum_pairwise_span_degrees"),
        range_of(deformation_20, "maximum_pairwise_span_degrees"),
    )
    for index, values in enumerate(span_ranges):
        axis.plot((index, index), values, color="#D55E00", linewidth=5, alpha=0.7)
        axis.scatter((index, index), values, color="#D55E00", zorder=3)
    axis.axhline(
        baseline_metrics["maximum_pairwise_span_degrees"],
        color="black",
        linestyle="--",
        label="Preserved-state baseline",
    )
    axis.set_xticks((0, 1), labels)
    axis.set_ylabel("Maximum normal span (degrees)")
    axis.set_title("D  Range across affine scenarios")
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(
        args.output_dir / "experiment_53_preservation_geometry_audit.png",
        bbox_inches="tight",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

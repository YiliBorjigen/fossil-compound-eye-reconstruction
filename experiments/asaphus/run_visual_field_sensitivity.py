#!/usr/bin/env python3
"""Measure preserved-state angular geometry of the Asaphus facet surface.

This experiment deliberately uses only the preserved outer surface.  Its
surface normals are geometric measurements of the fossil as preserved, not
anatomically verified optical axes or a reconstruction of the living field of
view.  The sensitivity table shows how an assumed internal-axis departure
would widen the angular bounds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay, cKDTree


SPACING_UM = 3.7
SURFACE_SIGMA_X = 1.0
CURVATURE_SIGMA = 15.0
PATCH_Y = (100, 450)
PATCH_Z = (300, 480)
PEAK_MIN_DISTANCE = 7
PEAK_RELIEF_THRESHOLD = 1.2
CENTRAL_T = 50
THRESHOLDS = (45, 50, 55, 60, 65, 70)
PERSIST_DISTANCE = 4.0
PERSIST_MIN_HITS = 4


def read_nrrd(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if line in (b"\n", b"\r\n"):
                body_offset = handle.tell()
                break
            if not line:
                raise ValueError("NRRD header has no terminating blank line")
            header_lines.append(line.decode("ascii").rstrip())
    header = {}
    for line in header_lines:
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    if header.get("type") not in {"uchar", "uint8", "unsigned char"}:
        raise ValueError(f"Expected uint8 NRRD, found {header.get('type')}")
    if header.get("encoding", "raw").lower() not in {"gzip", "gz"}:
        raise ValueError(f"Expected gzip NRRD, found {header.get('encoding')}")
    sizes = tuple(map(int, header["sizes"].split()))
    with path.open("rb") as handle:
        handle.seek(body_offset)
        with gzip.GzipFile(fileobj=handle, mode="rb") as compressed:
            raw = compressed.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(sizes, order="F")


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, float)
    if vectors.ndim == 1:
        return vectors / np.linalg.norm(vectors)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def angular_separation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = unit(left)
    right = unit(right)
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)))


def tangent_frame(mean_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((0.0, 1.0, 0.0))
    if abs(float(np.dot(reference, mean_axis))) > 0.8:
        reference = np.asarray((0.0, 0.0, 1.0))
    tangent_1 = unit(reference - np.dot(reference, mean_axis) * mean_axis)
    tangent_2 = unit(np.cross(mean_axis, tangent_1))
    return tangent_1, tangent_2


def gnomonic_coordinates(
    vectors: np.ndarray, mean_axis: np.ndarray, tangent_1: np.ndarray, tangent_2: np.ndarray
) -> np.ndarray:
    denominator = vectors @ mean_axis
    if np.any(denominator <= 0):
        raise ValueError("Visual directions do not lie within one hemisphere")
    return np.column_stack(
        ((vectors @ tangent_1) / denominator, (vectors @ tangent_2) / denominator)
    )


def triangle_solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    numerator = abs(float(np.dot(a, np.cross(b, c))))
    denominator = 1.0 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * float(np.arctan2(numerator, denominator))


def spherical_hull_area(
    vectors: np.ndarray, coordinates: np.ndarray, mean_axis: np.ndarray
) -> tuple[float, np.ndarray]:
    hull = ConvexHull(coordinates)
    boundary = vectors[hull.vertices]
    area = 0.0
    for index in range(len(boundary)):
        area += triangle_solid_angle(
            mean_axis, boundary[index], boundary[(index + 1) % len(boundary)]
        )
    return area, hull.vertices


def subvoxel_surface(smoothed: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    above = smoothed >= threshold
    reverse_index = np.argmax(above[::-1, :, :], axis=0)
    any_above = above.any(axis=0)
    integer_surface = (smoothed.shape[0] - 1 - reverse_index).astype(np.int32)
    surface = np.full(any_above.shape, np.nan, dtype=np.float32)
    ys, zs = np.where(any_above)
    xs = integer_surface[ys, zs]
    valid = xs < smoothed.shape[0] - 1
    yv, zv, xv = ys[valid], zs[valid], xs[valid]
    value_0 = smoothed[xv, yv, zv]
    value_1 = smoothed[xv + 1, yv, zv]
    denominator = value_0 - value_1
    fraction = np.divide(
        value_0 - threshold,
        denominator,
        out=np.zeros_like(value_0),
        where=np.abs(denominator) > 1e-6,
    )
    surface[yv, zv] = xv + fraction
    surface[ys[~valid], zs[~valid]] = xs[~valid]

    finite = np.isfinite(surface)
    nearest = ndi.distance_transform_edt(
        ~finite, return_distances=False, return_indices=True
    )
    filled = surface[tuple(nearest)]
    return surface, filled


def surface_and_points(smoothed: np.ndarray, threshold: float):
    surface, filled = subvoxel_surface(smoothed, threshold)
    relief = filled - ndi.gaussian_filter(filled, sigma=CURVATURE_SIGMA)
    patch = relief[PATCH_Y[0] : PATCH_Y[1], PATCH_Z[0] : PATCH_Z[1]]
    # Equivalent to the square-footprint peak search used in Experiment 36,
    # without requiring a separate image-processing package.
    local_maximum = ndi.maximum_filter(
        patch, size=2 * PEAK_MIN_DISTANCE + 1, mode="nearest"
    )
    peak_candidates = np.argwhere(
        (patch == local_maximum) & (patch > PEAK_RELIEF_THRESHOLD)
    )
    order = np.argsort(-patch[peak_candidates[:, 0], peak_candidates[:, 1]])
    accepted = []
    for candidate in peak_candidates[order]:
        if not accepted or all(
            np.max(np.abs(candidate - existing)) >= PEAK_MIN_DISTANCE
            for existing in accepted
        ):
            accepted.append(candidate)
    peaks = np.asarray(accepted, dtype=int)
    yy = peaks[:, 0] + PATCH_Y[0]
    zz = peaks[:, 1] + PATCH_Z[0]
    xx = surface[yy, zz]
    good = np.isfinite(xx)
    points = np.column_stack((xx[good], yy[good], zz[good]))

    smooth_surface = ndi.gaussian_filter(filled, sigma=2)
    gradient_y, gradient_z = np.gradient(smooth_surface)
    return points, gradient_y, gradient_z


def normals_at(points: np.ndarray, gradient_y: np.ndarray, gradient_z: np.ndarray):
    yi = np.rint(points[:, 1]).astype(int)
    zi = np.rint(points[:, 2]).astype(int)
    return unit(
        np.column_stack(
            (
                np.ones(len(points)),
                -gradient_y[yi, zi],
                -gradient_z[yi, zi],
            )
        )
    )


def adjacent_pairs(points: np.ndarray) -> np.ndarray:
    surface_coordinates = points[:, 1:3]
    triangulation = Delaunay(surface_coordinates)
    pairs = set()
    for triangle in triangulation.simplices:
        for left, right in ((0, 1), (1, 2), (2, 0)):
            pairs.add(tuple(sorted((int(triangle[left]), int(triangle[right])))))
    pairs = np.asarray(sorted(pairs), dtype=int)
    distances = np.linalg.norm(
        surface_coordinates[pairs[:, 0]] - surface_coordinates[pairs[:, 1]], axis=1
    )
    nearest = cKDTree(surface_coordinates).query(surface_coordinates, k=2)[0][:, 1]
    return pairs[distances <= 1.6 * np.median(nearest)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    volume = read_nrrd(args.input)
    smoothed = ndi.gaussian_filter1d(
        volume.astype(np.float32), sigma=SURFACE_SIGMA_X, axis=0
    )

    threshold_data = {}
    for threshold in THRESHOLDS:
        points, gradient_y, gradient_z = surface_and_points(smoothed, threshold)
        threshold_data[threshold] = {
            "points": points,
            "normals": normals_at(points, gradient_y, gradient_z),
        }

    base_points = threshold_data[CENTRAL_T]["points"]
    persistence_hits = np.zeros(len(base_points), dtype=int)
    for threshold in THRESHOLDS:
        if threshold == CENTRAL_T:
            continue
        distance, _ = cKDTree(threshold_data[threshold]["points"]).query(base_points)
        persistence_hits += distance <= PERSIST_DISTANCE
    robust = persistence_hits >= PERSIST_MIN_HITS
    points = base_points[robust]
    normals = threshold_data[CENTRAL_T]["normals"][robust]
    facet_ids = np.where(robust)[0]

    threshold_rows = []
    threshold_angles = []
    for threshold in THRESHOLDS:
        if threshold == CENTRAL_T:
            continue
        distance, matched = cKDTree(threshold_data[threshold]["points"]).query(points)
        valid = distance <= PERSIST_DISTANCE
        angles = angular_separation(
            normals[valid], threshold_data[threshold]["normals"][matched[valid]]
        )
        threshold_angles.extend(angles.tolist())
        threshold_rows.append(
            {
                "threshold": threshold,
                "matched_facets": int(np.sum(valid)),
                "median_normal_change_degrees": float(np.median(angles)),
                "p95_normal_change_degrees": float(np.percentile(angles, 95)),
            }
        )

    mean_axis = unit(np.sum(normals, axis=0))
    tangent_1, tangent_2 = tangent_frame(mean_axis)
    coordinates = gnomonic_coordinates(normals, mean_axis, tangent_1, tangent_2)
    solid_angle_sr, hull_indices = spherical_hull_area(normals, coordinates, mean_axis)
    pairwise = angular_separation(normals[:, None, :], normals[None, :, :])
    maximum_span = float(np.max(pairwise))
    off_axis = angular_separation(normals, np.repeat(mean_axis[None, :], len(normals), axis=0))

    pairs = adjacent_pairs(points)
    neighbour_angles = angular_separation(normals[pairs[:, 0]], normals[pairs[:, 1]])
    neighbour_distances_um = (
        np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1) * SPACING_UM
    )

    uncertainty_rows = []
    for bound in (0, 5, 10, 15, 20):
        uncertainty_rows.append(
            {
                "axis_departure_bound_degrees": bound,
                "guaranteed_minimum_possible_maximum_span_degrees": max(
                    0.0, maximum_span - 2.0 * bound
                ),
                "guaranteed_maximum_possible_maximum_span_degrees": min(
                    180.0, maximum_span + 2.0 * bound
                ),
                "mean_normal_direction_uncertainty_degrees": bound,
                "individual_axis_uncertainty_degrees": bound,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    facet_rows = []
    for index in range(len(points)):
        facet_rows.append(
            {
                "facet_id": int(facet_ids[index]),
                "x_vox": float(points[index, 0]),
                "y_vox": float(points[index, 1]),
                "z_vox": float(points[index, 2]),
                "normal_x": float(normals[index, 0]),
                "normal_y": float(normals[index, 1]),
                "normal_z": float(normals[index, 2]),
                "persistence_hits_of_5": int(persistence_hits[facet_ids[index]]),
            }
        )
    with (args.output_dir / "facet_normal_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(facet_rows[0]))
        writer.writeheader()
        writer.writerows(facet_rows)
    with (args.output_dir / "threshold_normal_stability.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with (args.output_dir / "axis_uncertainty_bounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(uncertainty_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(uncertainty_rows)

    summary = {
        "status": "preserved-state outer-surface geometry with anatomical-axis sensitivity",
        "facets": len(points),
        "surface_threshold": CENTRAL_T,
        "spacing_um": SPACING_UM,
        "surface_normal_baseline": {
            "maximum_pairwise_angular_span_degrees": maximum_span,
            "maximum_off_mean_axis_degrees": float(np.max(off_axis)),
            "spherical_convex_hull_solid_angle_sr": solid_angle_sr,
            "spherical_convex_hull_fraction_of_full_sphere": solid_angle_sr / (4.0 * np.pi),
            "adjacent_surface_pairs": len(pairs),
            "median_surface_normal_neighbour_angle_degrees": float(
                np.median(neighbour_angles)
            ),
            "median_adjacent_facet_spacing_um": float(np.median(neighbour_distances_um)),
        },
        "measurement_stability": {
            "matched_normal_comparisons": len(threshold_angles),
            "median_change_across_neighbouring_thresholds_degrees": float(
                np.median(threshold_angles)
            ),
            "p95_change_across_neighbouring_thresholds_degrees": float(
                np.percentile(threshold_angles, 95)
            ),
        },
        "interpretation": (
            "The outer surface defines a reproducible preserved-state normal baseline. "
            "It is not retrodeformed, and Experiment 51 shows that surface normals need not equal "
            "internal optical axes. The uncertainty table therefore bounds angular geometry without "
            "transferring Pieris anatomy to Asaphus."
        ),
        "claim_limits": [
            "Surface normals are not anatomically verified optical axes.",
            "The analysed crop is part of one Asaphus specimen, not the complete eye or an independent sample.",
            "The solid angle is the convex envelope of sampled surface normals, not a validated biological field of view.",
            "The bounded-axis sensitivity calculation is geometric and does not assert a Pieris-like tilt in Asaphus.",
            "No taphonomic retrodeformation is estimated from this unilateral crop.",
            "No refraction, birefringence, sensitivity or acuity is modelled.",
        ],
        "axis_uncertainty_bounds": uncertainty_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 9.0), dpi=180)
    axis = axes[0, 0]
    scatter = axis.scatter(
        points[:, 2], points[:, 1], c=off_axis, cmap="viridis", s=25, edgecolor="none"
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xlabel("Volume z (voxels)")
    axis.set_ylabel("Volume y (voxels)")
    axis.set_title("A  Robust fossil facets")
    colourbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("Normal angle from mean (degrees)")

    axis = axes[0, 1]
    angular_coordinates = np.degrees(np.arctan(coordinates))
    axis.scatter(angular_coordinates[:, 0], angular_coordinates[:, 1], s=22, color="#0072B2")
    boundary = np.r_[hull_indices, hull_indices[0]]
    axis.plot(
        angular_coordinates[boundary, 0], angular_coordinates[boundary, 1],
        color="#D55E00", linewidth=1.5,
    )
    axis.set_aspect("equal")
    axis.set_xlabel("Tangent normal angle 1 (degrees)")
    axis.set_ylabel("Tangent normal angle 2 (degrees)")
    axis.set_title("B  Preserved surface-normal envelope")

    axis = axes[1, 0]
    axis.hist(neighbour_angles, bins=18, color="#009E73", alpha=0.85)
    axis.axvline(np.median(neighbour_angles), color="black", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Adjacent surface-normal separation (degrees)")
    axis.set_ylabel("Facet pairs")
    axis.set_title("C  Geometric neighbour-angle baseline")

    axis = axes[1, 1]
    bounds = np.asarray([row["axis_departure_bound_degrees"] for row in uncertainty_rows])
    lower = np.asarray(
        [row["guaranteed_minimum_possible_maximum_span_degrees"] for row in uncertainty_rows]
    )
    upper = np.asarray(
        [row["guaranteed_maximum_possible_maximum_span_degrees"] for row in uncertainty_rows]
    )
    axis.fill_between(bounds, lower, upper, color="#CC79A7", alpha=0.3, label="Guaranteed range")
    axis.plot(bounds, lower, color="#CC79A7")
    axis.plot(bounds, upper, color="#CC79A7")
    axis.axhline(maximum_span, color="black", linestyle="--", label="Surface-normal baseline")
    axis.set_xlabel("Allowed optical-axis departure (degrees)")
    axis.set_ylabel("Possible maximum axis span (degrees)")
    axis.set_title("D  Sensitivity to unknown internal axes")
    axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(
        args.output_dir / "experiment_52_asaphus_visual_field_sensitivity.png",
        bbox_inches="tight",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

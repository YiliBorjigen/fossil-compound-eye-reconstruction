#!/usr/bin/env python3
"""Outcome-blind robust distal-core selection for Experiment 64.

The selector in this module operates only on one physical-coordinate point
cloud.  It does not use voxel adjacency, mesh connectivity, an anatomical
axis, an eye frame, a target surface, or a fitted prediction.  The same
operator can therefore be applied before sealing either voxel-derived or
mesh-derived distal observations.

The frozen algorithm is deliberately global and rigid-motion equivariant:

1. estimate a deterministic geometric median of all points;
2. retain the tie-inclusive inner q90 of three-dimensional radii for
   estimating a tangent plane by PCA;
3. represent that plane by its sign-invariant orthogonal projector; and
4. compute the tie-inclusive inner q90 of lateral radii over **all** input
   points and return its intersection with the Euclidean q90.

The two q90 masks each retain at least 30 of 33 inputs, so their intersection
retains at least 27.  A later tie-inclusive q90 Stage-2 fit therefore has at
least 25 points.  This support arithmetic is checked from the configuration
rather than assumed.

The returned boolean mask is aligned to the caller's original input.  The
returned retained points are additionally sorted in canonical lexicographic
``(x, y, z)`` order so a permutation of an identical input point set produces
byte-identical selected-point output.  Rigid-motion equivariance applies to
the selected *set*; canonical output order can naturally change after a
rotation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


ROBUST_DISTAL_CORE_SCHEMA_VERSION = (
    "experiment64.robust-distal-core.geometric-median-q90-intersection.v1"
)

# Keep every algorithm choice JSON-native so adapters can bind the complete
# operator into their own sealed-artifact configuration hash.
ROBUST_CORE_CONFIG: dict[str, Any] = {
    "method": "geometric_median_euclidean_q90_intersect_trimmed_pca_lateral_q90_v1",
    "geometric_median_relative_tolerance": 1.0e-12,
    "geometric_median_max_iterations": 512,
    "geometric_median_zero_distance_rule": "modified_weiszfeld_vardi_zhang",
    "pca_trim_quantile": 0.90,
    "core_lateral_quantile": 0.90,
    "downstream_fit_quantile": 0.90,
    "quantile_tie_relative_tolerance": 1.0e-12,
    "pca_minimum_normal_gap_ratio": 1.0e-6,
    "minimum_input_points": 33,
    "minimum_retained_points": 27,
    "minimum_downstream_fit_points": 25,
    "numerical_epsilon": 1.0e-12,
    "pca_normal_sign_policy": "projector_only_sign_irrelevant",
    "canonical_output_order": "lexicographic_xyz",
}

# Backward-readable name for callers that conventionally request a module's
# defaults.  Experiment 64 adapters should embed ``ROBUST_CORE_CONFIG`` (and
# its canonical hash) verbatim in both source and validation sealed schemas.
DEFAULT_ROBUST_CORE_CONFIG = ROBUST_CORE_CONFIG


class RobustDistalCoreError(ValueError):
    """Raised when a point cloud or robust-core fit violates the contract."""


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"configuration contains a non-JSON value: {type(value).__name__}")


def normalise_robust_core_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and return the fully populated robust-core configuration."""

    merged = dict(ROBUST_CORE_CONFIG)
    if config is not None:
        unknown = sorted(set(config) - set(ROBUST_CORE_CONFIG))
        if unknown:
            raise RobustDistalCoreError(
                "unknown robust-core configuration key(s): " + ", ".join(unknown)
            )
        merged.update(_json_native(config))

    fixed_strings = {
        "method": "geometric_median_euclidean_q90_intersect_trimmed_pca_lateral_q90_v1",
        "geometric_median_zero_distance_rule": "modified_weiszfeld_vardi_zhang",
        "pca_normal_sign_policy": "projector_only_sign_irrelevant",
        "canonical_output_order": "lexicographic_xyz",
    }
    for key, expected in fixed_strings.items():
        if merged[key] != expected:
            raise RobustDistalCoreError(f"{key} must remain frozen at {expected!r}")

    fixed_numeric = {
        "pca_trim_quantile": 0.90,
        "core_lateral_quantile": 0.90,
        "downstream_fit_quantile": 0.90,
        "minimum_input_points": 33,
        "minimum_retained_points": 27,
        "minimum_downstream_fit_points": 25,
    }
    for key, expected in fixed_numeric.items():
        value = merged[key]
        if isinstance(value, bool) or value != expected:
            raise RobustDistalCoreError(f"{key} must remain frozen at {expected!r}")

    integer_keys = (
        "geometric_median_max_iterations",
        "minimum_input_points",
        "minimum_retained_points",
        "minimum_downstream_fit_points",
    )
    for key in integer_keys:
        value = merged[key]
        if isinstance(value, bool):
            raise RobustDistalCoreError(f"{key} must be a positive integer")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RobustDistalCoreError(f"{key} must be a positive integer") from exc
        if integer < 1 or integer != value:
            raise RobustDistalCoreError(f"{key} must be a positive integer")
        merged[key] = integer

    unit_interval_keys = (
        "pca_trim_quantile",
        "core_lateral_quantile",
        "downstream_fit_quantile",
    )
    for key in unit_interval_keys:
        value = float(merged[key])
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise RobustDistalCoreError(f"{key} must be finite and in (0, 1]")
        merged[key] = value

    positive_float_keys = (
        "geometric_median_relative_tolerance",
        "quantile_tie_relative_tolerance",
        "pca_minimum_normal_gap_ratio",
        "numerical_epsilon",
    )
    for key in positive_float_keys:
        value = float(merged[key])
        if not math.isfinite(value) or value <= 0.0:
            raise RobustDistalCoreError(f"{key} must be finite and strictly positive")
        merged[key] = value

    if merged["minimum_retained_points"] > merged["minimum_input_points"]:
        raise RobustDistalCoreError(
            "minimum_retained_points must not exceed minimum_input_points"
        )
    pca_guaranteed = math.ceil(
        merged["pca_trim_quantile"] * merged["minimum_input_points"]
    )
    lateral_guaranteed = math.ceil(
        merged["core_lateral_quantile"] * merged["minimum_input_points"]
    )
    intersection_guaranteed = max(
        0,
        pca_guaranteed + lateral_guaranteed - merged["minimum_input_points"],
    )
    if intersection_guaranteed < merged["minimum_retained_points"]:
        raise RobustDistalCoreError(
            "the intersection of the Euclidean and lateral quantiles cannot guarantee "
            "minimum_retained_points at minimum_input_points"
        )
    downstream_guaranteed = math.ceil(
        merged["downstream_fit_quantile"] * merged["minimum_retained_points"]
    )
    if downstream_guaranteed < merged["minimum_downstream_fit_points"]:
        raise RobustDistalCoreError(
            "the downstream fit quantile cannot guarantee minimum_downstream_fit_points "
            "at minimum_retained_points"
        )
    return merged


def canonical_robust_core_config_json(config: Mapping[str, Any] | None = None) -> str:
    """Return canonical JSON suitable for a sealed-artifact provenance hash."""

    return json.dumps(
        normalise_robust_core_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def robust_core_config_sha256(config: Mapping[str, Any] | None = None) -> str:
    """Return the SHA-256 digest of the fully populated canonical config."""

    payload = canonical_robust_core_config_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_points(points_xyz_um: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz_um)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise RobustDistalCoreError("points_xyz_um must have shape (n, 3)")
    if points.shape[0] < minimum:
        raise RobustDistalCoreError(
            f"points_xyz_um has {points.shape[0]} points; at least {minimum} are required"
        )
    if points.dtype.kind not in "fiu":
        raise RobustDistalCoreError("points_xyz_um must have a real numeric dtype")
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise RobustDistalCoreError("points_xyz_um must contain only finite coordinates")

    # Sorting makes all reductions independent of caller order.  Exact
    # duplicate coordinates are rejected because otherwise repeated vertices
    # would silently act as representation-specific statistical weights.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    canonical = np.ascontiguousarray(points[order])
    if np.any(np.all(np.diff(canonical, axis=0) == 0.0, axis=1)):
        raise RobustDistalCoreError(
            "points_xyz_um contains duplicate coordinates; adapters must deduplicate first"
        )
    return canonical, order


def _geometric_median(
    points: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Modified Weiszfeld geometric median with a deterministic start/rule."""

    estimate = np.mean(points, axis=0, dtype=np.float64)
    initial_distances = np.linalg.norm(points - estimate, axis=1)
    scale = max(float(np.sqrt(np.mean(initial_distances * initial_distances))), np.finfo(float).tiny)
    convergence_tolerance = float(config["geometric_median_relative_tolerance"]) * scale
    zero_tolerance = float(config["numerical_epsilon"]) * scale
    last_step = math.inf
    zero_rule_used = False

    for iteration in range(1, int(config["geometric_median_max_iterations"]) + 1):
        offsets = points - estimate
        distances = np.linalg.norm(offsets, axis=1)
        coincident = distances <= zero_tolerance
        multiplicity = int(np.count_nonzero(coincident))

        if multiplicity:
            zero_rule_used = True
            nonzero = ~coincident
            if not np.any(nonzero):
                candidate = estimate.copy()
            else:
                unit_sum = np.sum(
                    offsets[nonzero] / distances[nonzero, None], axis=0, dtype=np.float64
                )
                residual_norm = float(np.linalg.norm(unit_sum))
                if residual_norm <= multiplicity + float(config["numerical_epsilon"]):
                    objective = float(np.sum(distances, dtype=np.float64))
                    return estimate, {
                        "converged": True,
                        "iterations": iteration,
                        "final_step_um": 0.0,
                        "objective_um": objective,
                        "initial_rms_radius_um": scale,
                        "zero_distance_rule_used": True,
                    }
                inverse = 1.0 / distances[nonzero]
                ordinary_update = np.sum(
                    points[nonzero] * inverse[:, None], axis=0, dtype=np.float64
                ) / float(np.sum(inverse, dtype=np.float64))
                beta = min(1.0, multiplicity / residual_norm)
                candidate = beta * estimate + (1.0 - beta) * ordinary_update
        else:
            inverse = 1.0 / distances
            candidate = np.sum(
                points * inverse[:, None], axis=0, dtype=np.float64
            ) / float(np.sum(inverse, dtype=np.float64))

        if not np.all(np.isfinite(candidate)):
            raise RobustDistalCoreError("geometric-median iteration produced nonfinite values")
        last_step = float(np.linalg.norm(candidate - estimate))
        estimate = candidate
        if last_step <= convergence_tolerance:
            objective = float(
                np.sum(np.linalg.norm(points - estimate, axis=1), dtype=np.float64)
            )
            return estimate, {
                "converged": True,
                "iterations": iteration,
                "final_step_um": last_step,
                "objective_um": objective,
                "initial_rms_radius_um": scale,
                "zero_distance_rule_used": zero_rule_used,
            }

    raise RobustDistalCoreError(
        "geometric median did not converge in "
        f"{config['geometric_median_max_iterations']} iterations "
        f"(last step {last_step:.6g} um)"
    )


def _tie_inclusive_quantile_mask(
    values: np.ndarray,
    quantile: float,
    tie_relative_tolerance: float,
) -> tuple[np.ndarray, float, int, int]:
    """Select the lowest fixed fraction, expanding rather than breaking ties."""

    requested_count = max(1, int(math.ceil(quantile * len(values))))
    threshold = float(np.partition(values, requested_count - 1)[requested_count - 1])
    tie_tolerance = tie_relative_tolerance * max(1.0, abs(threshold))
    mask = values <= threshold + tie_tolerance
    return mask, threshold, requested_count, int(np.count_nonzero(mask))


def _quantile_diagnostics(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 1.0)
    measured = np.quantile(values, levels, method="linear")
    names = ("q00", "q20", "q50", "q80", "q90", "q95", "q99", "q100")
    return {name: float(value) for name, value in zip(names, measured, strict=True)}


def select_robust_distal_core(
    points_xyz_um: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a global robust lateral core from physical ``xyz`` coordinates.

    Parameters
    ----------
    points_xyz_um:
        Unique finite physical coordinates with shape ``(n, 3)``.  Voxel
        adapters must convert source indices to physical coordinates before
        this call, then apply ``retained_mask`` to their unchanged integer
        coordinates.  Mesh adapters must canonicalize and deduplicate vertices
        before this call.
    config:
        Optional overrides of exported configuration values.  Unknown values,
        configurations that weaken the support guarantee, and changes to
        named algorithm policies fail closed.

    Returns
    -------
    dict
        ``retained_mask`` and ``lateral_radius_um`` are aligned to the original
        input order. ``retained_points_xyz_um`` is lexicographically sorted.
        ``tangent_projector`` is a sign-invariant 3x3 plane projector.  The
        JSON-native ``diagnostics`` contains convergence, support, threshold,
        tie-expansion, eigenvalue and radius-quantile records.

    Notes
    -----
    If the normal subspace is not identifiable, the geometric median does not
    converge, or fewer than the configured number of points survive, this
    function raises :class:`RobustDistalCoreError`; it never falls back to a
    coordinate-axis-dependent plane or an untrimmed point set.
    """

    normalized = normalise_robust_core_config(config)
    points, canonical_order = _validate_points(
        points_xyz_um, int(normalized["minimum_input_points"])
    )
    median, median_diagnostics = _geometric_median(points, normalized)

    radial_distances = np.linalg.norm(points - median, axis=1)
    pca_mask, radial_threshold, pca_requested, pca_count = _tie_inclusive_quantile_mask(
        radial_distances,
        float(normalized["pca_trim_quantile"]),
        float(normalized["quantile_tie_relative_tolerance"]),
    )
    if pca_count < 3:
        raise RobustDistalCoreError("trimmed PCA has fewer than three support points")

    pca_points = points[pca_mask]
    pca_centroid = np.mean(pca_points, axis=0, dtype=np.float64)
    centered = pca_points - pca_centroid
    covariance = (centered.T @ centered) / float(pca_count)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
        raise RobustDistalCoreError("trimmed PCA covariance is degenerate")
    normal_gap_ratio = float((eigenvalues[1] - eigenvalues[0]) / eigenvalues[-1])
    if normal_gap_ratio < float(normalized["pca_minimum_normal_gap_ratio"]):
        raise RobustDistalCoreError(
            "trimmed PCA normal subspace is not identifiable: "
            f"gap ratio {normal_gap_ratio:.6g} is below "
            f"{normalized['pca_minimum_normal_gap_ratio']:.6g}"
        )

    normal = eigenvectors[:, 0]
    normal_projector = np.outer(normal, normal)
    tangent_projector = np.eye(3, dtype=np.float64) - normal_projector
    lateral_vectors = (points - median) @ tangent_projector
    lateral_radii = np.linalg.norm(lateral_vectors, axis=1)
    # The lateral selection intentionally ranges over every input point before
    # its intersection with the Euclidean mask used for trimmed PCA.
    lateral_mask, lateral_threshold, lateral_requested, lateral_count = (
        _tie_inclusive_quantile_mask(
            lateral_radii,
            float(normalized["core_lateral_quantile"]),
            float(normalized["quantile_tie_relative_tolerance"]),
        )
    )
    core_mask = pca_mask & lateral_mask
    retained_count = int(np.count_nonzero(core_mask))
    if retained_count < int(normalized["minimum_retained_points"]):
        raise RobustDistalCoreError(
            f"robust distal core retained {retained_count} points; at least "
            f"{normalized['minimum_retained_points']} are required"
        )

    retained_points = np.ascontiguousarray(points[core_mask])
    original_mask = np.zeros(len(points), dtype=np.bool_)
    original_mask[canonical_order] = core_mask
    original_lateral_radii = np.empty(len(points), dtype=np.float64)
    original_lateral_radii[canonical_order] = lateral_radii
    original_radial_distances = np.empty(len(points), dtype=np.float64)
    original_radial_distances[canonical_order] = radial_distances
    original_euclidean_mask = np.zeros(len(points), dtype=np.bool_)
    original_euclidean_mask[canonical_order] = pca_mask
    original_lateral_mask = np.zeros(len(points), dtype=np.bool_)
    original_lateral_mask[canonical_order] = lateral_mask

    rejected_radii = lateral_radii[~core_mask]
    axial_distances = np.linalg.norm((points - pca_centroid) @ normal_projector, axis=1)
    retained_axial_distances = axial_distances[core_mask]
    axial_quantiles = _quantile_diagnostics(axial_distances)
    arithmetic_mean = np.mean(points, axis=0, dtype=np.float64)
    guaranteed_intersection = max(0, pca_requested + lateral_requested - len(points))
    downstream_support_lower_bound = math.ceil(
        float(normalized["downstream_fit_quantile"]) * retained_count
    )
    if downstream_support_lower_bound < int(normalized["minimum_downstream_fit_points"]):
        raise RobustDistalCoreError(
            "robust distal core cannot meet the configured downstream support floor"
        )
    diagnostics: dict[str, Any] = {
        "schema_version": ROBUST_DISTAL_CORE_SCHEMA_VERSION,
        "input_support": int(len(points)),
        "retained_support": retained_count,
        "retained_fraction": float(retained_count / len(points)),
        "input_point_count": int(len(points)),
        "retained_point_count": retained_count,
        "rejected_point_count": int(len(points) - retained_count),
        "geometric_median_converged": bool(median_diagnostics["converged"]),
        "geometric_median_iterations": int(median_diagnostics["iterations"]),
        "geometric_median_final_step_um": float(median_diagnostics["final_step_um"]),
        "geometric_median_objective_um": float(median_diagnostics["objective_um"]),
        "geometric_median_initial_rms_radius_um": float(
            median_diagnostics["initial_rms_radius_um"]
        ),
        "arithmetic_mean_to_geometric_median_um": float(
            np.linalg.norm(arithmetic_mean - median)
        ),
        "geometric_median_zero_distance_rule_used": bool(
            median_diagnostics["zero_distance_rule_used"]
        ),
        "geometric_median_xyz_um": [float(value) for value in median],
        "radial_radius_quantiles_um": _quantile_diagnostics(radial_distances),
        "pca_trim_requested_count": pca_requested,
        "pca_trim_retained_count": pca_count,
        "pca_trim_tie_expansion_count": int(pca_count - pca_requested),
        "pca_radial_threshold_um": radial_threshold,
        "pca_centroid_xyz_um": [float(value) for value in pca_centroid],
        "pca_eigenvalues_um2": [float(value) for value in eigenvalues],
        "pca_normal_gap_ratio": normal_gap_ratio,
        "lateral_radius_quantiles_um": _quantile_diagnostics(lateral_radii),
        "lateral_q90_requested_count": lateral_requested,
        "lateral_q90_retained_count": lateral_count,
        "lateral_q90_tie_expansion_count": int(lateral_count - lateral_requested),
        "intersection_guaranteed_count_from_requested_supports": int(
            guaranteed_intersection
        ),
        "intersection_retained_count": retained_count,
        "euclidean_only_count": int(np.count_nonzero(pca_mask & ~lateral_mask)),
        "lateral_only_count": int(np.count_nonzero(~pca_mask & lateral_mask)),
        "excluded_by_both_count": int(np.count_nonzero(~pca_mask & ~lateral_mask)),
        "downstream_q90_support_lower_bound": int(downstream_support_lower_bound),
        "lateral_core_threshold_um": lateral_threshold,
        "maximum_retained_lateral_radius_um": float(np.max(lateral_radii[core_mask])),
        "minimum_rejected_lateral_radius_um": (
            float(np.min(rejected_radii)) if len(rejected_radii) else None
        ),
        "full_axial_distance_quantiles_um": axial_quantiles,
        "retained_axial_distance_quantiles_um": _quantile_diagnostics(
            retained_axial_distances
        ),
        "single_sheet_p95_axial_over_lateral_q90": float(
            axial_quantiles["q95"] / max(lateral_threshold, float(normalized["numerical_epsilon"]))
        ),
        "single_sheet_p99_axial_over_lateral_q90": float(
            axial_quantiles["q99"] / max(lateral_threshold, float(normalized["numerical_epsilon"]))
        ),
        "config_sha256": robust_core_config_sha256(normalized),
    }

    return {
        "retained_mask": original_mask,
        "euclidean_q90_mask": original_euclidean_mask,
        "lateral_q90_mask": original_lateral_mask,
        "retained_points_xyz_um": retained_points,
        "euclidean_radius_um": original_radial_distances,
        "lateral_radius_um": original_lateral_radii,
        "geometric_median_xyz_um": median,
        "tangent_projector": tangent_projector,
        "normal_projector": normal_projector,
        "config": normalized,
        "config_json": canonical_robust_core_config_json(normalized),
        "config_sha256": robust_core_config_sha256(normalized),
        "diagnostics": diagnostics,
    }


__all__ = (
    "DEFAULT_ROBUST_CORE_CONFIG",
    "ROBUST_CORE_CONFIG",
    "ROBUST_DISTAL_CORE_SCHEMA_VERSION",
    "RobustDistalCoreError",
    "canonical_robust_core_config_json",
    "normalise_robust_core_config",
    "robust_core_config_sha256",
    "select_robust_distal_core",
)

#!/usr/bin/env python3
"""Target-free final-fit and modality-specific diagnostics for Experiment 64.

The functions in this module accept only a sealed distal core and its
distal-only fitted geometry.  They must never be passed a proximal surface,
target, prediction, model, or error.  The adapters share the final-q90 fit
definition, while connectivity is deliberately Maike-only because Arthur's
irregular mesh vertices do not support a voxel-neighbourhood interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


TECHNICAL_COHERENCE_SCHEMA_VERSION = "experiment64.technical-coherence.v1"
TECHNICAL_COHERENCE_CONFIG: dict[str, Any] = {
    "fit_domain": "final_distal_frame_tie_inclusive_q90_lateral_radius",
    "fit_support_minimum": 25,
    "fit_rmse_max_um": 2.5,
    "fit_numerical_tolerance_um": 1.0e-10,
    "fit_point_connectivity": 26,
    "fit_largest_component_fraction_min": 0.99,
    "fit_abs_residual_p99_over_scale_max": 0.75,
    "maike_connectivity_domain": "final_q90_fit_voxels_in_source_zyx_grid",
    "maike_coherence_margin": (
        "min((fit_support-25)/25,(2.5-fit_rmse_um)/2.5,"
        "(fit_26_lcc_fraction-0.99)/0.01,"
        "(0.75-fit_abs_residual_p99_over_scale)/0.75)"
    ),
    "arthur_connectivity_gate": "not_applicable_irregular_surface_mesh_sampling",
    "arthur_coherence_margin": (
        "min((fit_support-25)/25,(2.5-fit_rmse_um)/2.5)"
    ),
}


class TechnicalMetricError(ValueError):
    """Raised when target-free coherence diagnostics cannot be derived."""


def canonical_technical_coherence_config_json(
    config: Mapping[str, Any] | None = None,
) -> str:
    selected = dict(TECHNICAL_COHERENCE_CONFIG)
    if config is not None:
        if set(config) != set(selected) or dict(config) != selected:
            raise TechnicalMetricError("technical coherence configuration must remain frozen")
    return json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False)


def technical_coherence_config_sha256(
    config: Mapping[str, Any] | None = None,
) -> str:
    return hashlib.sha256(
        canonical_technical_coherence_config_json(config).encode("utf-8")
    ).hexdigest()


def _geometry_vector(
    geometry: Mapping[str, Any], *names: str, length: int
) -> np.ndarray:
    for name in names:
        if name in geometry:
            value = np.asarray(geometry[name], dtype=np.float64)
            break
    else:
        raise TechnicalMetricError(f"geometry is missing {'/'.join(names)}")
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise TechnicalMetricError(f"geometry {'/'.join(names)} is invalid")
    return value


def _voxel_26_components(points_zyx: np.ndarray) -> tuple[int, int]:
    """Count deterministic 26-connected components in unique grid voxels."""

    if len(points_zyx) == 0:
        raise TechnicalMetricError("26-connectivity requires at least one fit voxel")
    remaining = {tuple(map(int, point)) for point in points_zyx.tolist()}
    component_sizes: list[int] = []
    offsets = tuple(
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    )
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        size = 0
        while stack:
            z, y, x = stack.pop()
            size += 1
            for dz, dy, dx in offsets:
                neighbor = (z + dz, y + dy, x + dx)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)
    return len(component_sizes), max(component_sizes)


def _base_final_fit_metrics(
    points_xyz_um: Any,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Return common final-q90 diagnostics, aligned fit mask, and fit scale."""

    canonical_technical_coherence_config_json(config)
    cfg = TECHNICAL_COHERENCE_CONFIG
    points = np.asarray(points_xyz_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        raise TechnicalMetricError("sealed distal core must have shape (n,3), n>=2")
    if not np.all(np.isfinite(points)) or len(np.unique(points, axis=0)) != len(points):
        raise TechnicalMetricError("sealed distal core must contain unique finite points")

    origin = _geometry_vector(geometry, "origin", "origin_xyz_um", length=3)
    u = _geometry_vector(geometry, "u", "u_axis_xyz", length=3)
    v = _geometry_vector(geometry, "v", "v_axis_xyz", length=3)
    w = _geometry_vector(geometry, "w", "outward", "outward_axis_xyz", length=3)
    beta = _geometry_vector(
        geometry,
        "quadratic_beta_normalized",
        "distal_coefficients_normalized",
        length=6,
    )
    scale = float(geometry.get("scale_um", geometry.get("distal_scale_um", math.nan)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise TechnicalMetricError("distal scale must be finite and positive")

    relative = points - origin
    x_um = relative @ u
    y_um = relative @ v
    x = x_um / scale
    y = y_um / scale
    z = relative @ w
    # Match distal_only_geometry._fit_with_frame exactly: its tolerance is in
    # physical micrometres, not in normalized lateral coordinates.
    fit_mask = np.hypot(x_um, y_um) <= scale + float(
        cfg["fit_numerical_tolerance_um"]
    )
    fit_support = int(np.count_nonzero(fit_mask))
    if fit_support < int(cfg["fit_support_minimum"]):
        raise TechnicalMetricError(
            f"final q90 fit support {fit_support} is below "
            f"{cfg['fit_support_minimum']}"
        )
    design = np.column_stack(
        (
            np.ones(fit_support),
            x[fit_mask],
            y[fit_mask],
            x[fit_mask] ** 2,
            x[fit_mask] * y[fit_mask],
            y[fit_mask] ** 2,
        )
    )
    absolute_residual = np.abs(z[fit_mask] - design @ beta)
    fit_rmse = float(np.sqrt(np.mean(absolute_residual * absolute_residual)))
    stored_rmse = float(
        geometry.get("quadratic_rmse_um", geometry.get("distal_fit_rmse_um", fit_rmse))
    )
    if not math.isfinite(stored_rmse) or not math.isclose(
        fit_rmse, stored_rmse, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise TechnicalMetricError("recomputed and stored distal fit RMSE differ")

    support_margin = (fit_support - float(cfg["fit_support_minimum"])) / float(
        cfg["fit_support_minimum"]
    )
    rmse_margin = (float(cfg["fit_rmse_max_um"]) - fit_rmse) / float(
        cfg["fit_rmse_max_um"]
    )
    result = {
        "schema_version": TECHNICAL_COHERENCE_SCHEMA_VERSION,
        "config_sha256": technical_coherence_config_sha256(),
        "distal_fit_support": fit_support,
        "distal_fit_rmse_um": fit_rmse,
        "distal_abs_residual_p95_um": float(np.quantile(absolute_residual, 0.95)),
        "distal_abs_residual_p99_um": float(np.quantile(absolute_residual, 0.99)),
        "coherence_support_margin": support_margin,
        "coherence_rmse_margin": rmse_margin,
    }
    return result, fit_mask, scale


def arthur_final_distal_coherence_metrics(
    points_xyz_um: Any,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return mesh-appropriate metrics without a connectivity gate.

    Arthur's supplied surface vertices have irregular and sometimes
    disconnected tessellation sampling, so a radius or voxel-connectivity
    statistic would not measure anatomical cap coherence.
    """

    result, _, _ = _base_final_fit_metrics(points_xyz_um, geometry, config)
    result.update(
        {
            "connectivity_gate_applicable": False,
            "connectivity_gate_reason": (
                "irregular_surface_mesh_sampling"
            ),
            "coherence_margin": float(
                min(
                    result["coherence_support_margin"],
                    result["coherence_rmse_margin"],
                )
            ),
        }
    )
    return result


def maike_final_distal_coherence_metrics(
    points_zyx: Any,
    spacing_zyx_um: Any,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return final-q90 voxel coherence and the two calibrated Maike gates."""

    voxels = np.asarray(points_zyx)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    if (
        voxels.dtype != np.dtype(np.int32)
        or voxels.ndim != 2
        or voxels.shape[1:] != (3,)
        or np.any(voxels < 0)
        or len(np.unique(voxels, axis=0)) != len(voxels)
    ):
        raise TechnicalMetricError(
            "Maike sealed distal core must be unique non-negative int32 ZYX"
        )
    if (
        spacing.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
    ):
        raise TechnicalMetricError(
            "Maike voxel spacing must contain three positive finite values"
        )
    points_xyz_um = voxels[:, ::-1].astype(np.float64) * spacing[::-1]
    result, fit_mask, scale = _base_final_fit_metrics(
        points_xyz_um, geometry, config
    )
    fit_voxels = voxels[fit_mask]
    n_components, largest_support = _voxel_26_components(fit_voxels)
    largest_fraction = float(largest_support / len(fit_voxels))
    p99_over_scale = float(result["distal_abs_residual_p99_um"] / scale)
    lcc_threshold = float(
        TECHNICAL_COHERENCE_CONFIG["fit_largest_component_fraction_min"]
    )
    p99_threshold = float(
        TECHNICAL_COHERENCE_CONFIG["fit_abs_residual_p99_over_scale_max"]
    )
    lcc_margin = (largest_fraction - lcc_threshold) / (1.0 - lcc_threshold)
    p99_margin = (p99_threshold - p99_over_scale) / p99_threshold
    reasons: list[str] = []
    if largest_fraction < lcc_threshold:
        reasons.append("fit_points_26_lcc_fraction_below_minimum")
    if p99_over_scale > p99_threshold:
        reasons.append("fit_abs_residual_p99_over_scale_above_maximum")
    result.update(
        {
            "distal_fit_26_component_count": n_components,
            "distal_fit_26_largest_component_support": largest_support,
            "distal_fit_26_largest_component_fraction": largest_fraction,
            "distal_fit_p99_residual_over_scale": p99_over_scale,
            "coherence_lcc_margin": lcc_margin,
            "coherence_p99_over_scale_margin": p99_margin,
            "coherence_margin": float(
                min(
                    result["coherence_support_margin"],
                    result["coherence_rmse_margin"],
                    lcc_margin,
                    p99_margin,
                )
            ),
            "maike_final_fit_gate_pass": not reasons,
            "maike_final_fit_gate_reasons": reasons,
        }
    )
    return result


def final_distal_coherence_metrics(
    points_xyz_um: Any,
    geometry: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility alias for Arthur's non-connectivity metric contract."""

    return arthur_final_distal_coherence_metrics(
        points_xyz_um, geometry, config
    )


__all__ = (
    "TECHNICAL_COHERENCE_CONFIG",
    "TECHNICAL_COHERENCE_SCHEMA_VERSION",
    "TechnicalMetricError",
    "arthur_final_distal_coherence_metrics",
    "canonical_technical_coherence_config_json",
    "final_distal_coherence_metrics",
    "maike_final_distal_coherence_metrics",
    "technical_coherence_config_sha256",
)

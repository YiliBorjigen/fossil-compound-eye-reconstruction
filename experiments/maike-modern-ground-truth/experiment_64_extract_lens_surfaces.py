#!/usr/bin/env python3
"""Build one leakage-isolated Experiment 64 Maike eye bundle.

Stage 1 retains Experiment 63's oracle centre/axis localisation, then applies
the shared Experiment 64 robust-core selector in physical XYZ coordinates.
Only the selector's retained mask is applied to the unchanged int32 source
ZYX coordinates, and only that core is sealed for Stage 2.

Stage 2 reopens the sealed cores and derives every predictor, QC decision,
frame, and coherence diagnostic without access to a proximal observation.
Fitted proximal outcome tables and arrays are computed afterwards and confined
to ``sealed_outcomes``.  The technical documents bind that directory through
its manifest artifact only; they never copy outcome columns, values, or cohort
counts.  Reviewer instances retain the source segmentation needed for visual
technical QC, but Stage 2 never reads those arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import distal_only_geometry as distal_geometry
import experiment_64_robust_distal_core as robust_distal_core
import experiment_64_technical_metrics as technical_metrics
from extract_lens_surfaces import (
    DEFAULT_PIPELINE_CONFIG as EXPERIMENT63_PIPELINE_CONFIG,
    DEFAULT_THRESHOLD_CONFIG as EXPERIMENT63_THRESHOLD_CONFIG,
    ExtractionError,
    _artifact_entry,
    _atomic_csv,
    _atomic_json,
    _atomic_savez,
    _canonical_geometry_entry,
    _canonical_grid,
    _canonical_json,
    _extract_target,
    _git_identity,
    _load_frame_audit_module,
    _normalise_geometry_result,
    _partition_foreground,
    _quadratic_design,
    _require_exact_staging_files,
    _require_seed_mask_cross_binding,
    _sha256_bytes,
    _validate_mask,
    _validate_seed_provenance,
    deterministic_largest_component,
    load_seeds,
    localize_distal_cap,
    target_component_fraction_passes,
)


TECHNICAL_BUNDLE_SCHEMA = "experiment64.maike-technical-bundle.v1"
BUNDLE_SCHEMA = TECHNICAL_BUNDLE_SCHEMA
INSTANCE_SCHEMA = "experiment64.maike-instance.v1"
SEALED_DISTAL_SCHEMA = "experiment64.maike-sealed-distal-core.v1"
OUTCOME_MANIFEST_SCHEMA = "experiment64.maike-outcomes.v1"
TARGET_SCHEMA = "experiment64.maike-target.v1"
LENS_SCHEMA = TARGET_SCHEMA
FRAME_AUDIT_SCHEMA = "experiment64.distal-frame-audit.v1"
TECHNICAL_BUNDLE_SCHEMA_VERSION = TECHNICAL_BUNDLE_SCHEMA
INSTANCE_SCHEMA_VERSION = INSTANCE_SCHEMA
SEALED_CORE_SCHEMA_VERSION = SEALED_DISTAL_SCHEMA
OUTCOME_MANIFEST_SCHEMA_VERSION = OUTCOME_MANIFEST_SCHEMA
TARGET_SCHEMA_VERSION = TARGET_SCHEMA

ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
ISOLATION_BASIS = "stage2_reads_only_sha256_sealed_robust_distal_core_artifacts"
ORACLE_STAGE1_SCOPE = "oracle_correspondence_and_distal_localization_only"

TECHNICAL_INVENTORY_FIELDS = (
    "eye_id",
    "lens_index",
    "seed_id",
    "species",
    "sex",
    "assignment_status",
    "full_assigned_size",
    "main_component_size",
    "component_removed_size",
    "main_component_fraction",
    "component_sizes_json",
    "component_fraction_gate_pass",
    "raw_distal_support",
    "robust_core_status",
    "robust_core_reasons",
    "robust_core_support",
    "robust_core_retained_fraction",
    "robust_core_diagnostics_json",
    "distal_qc",
    "distal_qc_reasons",
    "central",
    "position_u_um",
    "position_v_um",
    "distal_scale_um",
    "distal_fit_support",
    "distal_fit_rmse_um",
    "distal_abs_residual_p95_um",
    "distal_abs_residual_p99_um",
    "distal_fit_p99_residual_over_scale",
    "distal_fit_26_component_count",
    "distal_fit_26_largest_component_support",
    "distal_fit_26_largest_component_fraction",
    "maike_final_fit_gate_pass",
    "maike_final_fit_gate_reasons",
    "distal_gradient_magnitude",
    "distal_curvature_eigenvalue_1",
    "distal_curvature_eigenvalue_2",
    "distal_normalized_fit_residual",
    "coherence_support_margin",
    "coherence_rmse_margin",
    "coherence_lcc_margin",
    "coherence_p99_over_scale_margin",
    "coherence_margin",
    "instance_relpath",
    "sealed_distal_relpath",
)
TECHNICAL_FIELDS = TECHNICAL_INVENTORY_FIELDS

DISTAL_QC_SAMPLING_FIELDS = (
    "eye_id",
    "lens_index",
    "seed_id",
    "distal_eligible",
    "position_u_um",
    "position_v_um",
    "distal_scale_um",
    "coherence_margin",
    "instance_relpath",
    "sealed_distal_relpath",
)

TARGET_TABLE_FIELDS = (
    "eye_id",
    "lens_index",
    "target_resolvable",
    "target_resolvability_reasons",
    "target_qc",
    "target_qc_reasons",
    "target_depth_um",
    "target_support",
    "target_q05_raw_thickness_um",
    "target_rmse_um",
    "target_c0",
    "target_c1",
    "target_c2",
    "target_c3",
    "target_c4",
    "target_c5",
    "lens_relpath",
)
TARGET_FIELDS = TARGET_TABLE_FIELDS

INSTANCE_KEYS = frozenset(
    {
        "schema_version",
        "lens_index",
        "full_assigned_points_zyx",
        "main_component_points_zyx",
        "raw_distal_points_zyx",
        "component_sizes_descending",
        "spacing_um",
        "seed_source_zyx",
        "oda_axis_source_zyx",
        "config_json",
        "config_sha256",
    }
)
INSTANCE_MEMBERS = INSTANCE_KEYS

SEALED_DISTAL_KEYS = frozenset(
    {
        "schema_version",
        "lens_index",
        "points_zyx",
        "spacing_um",
        "config_json",
        "config_sha256",
        "raw_distal_support",
        "robust_core_config_sha256",
        "robust_core_diagnostics_json",
    }
)
SEALED_CORE_KEYS = SEALED_DISTAL_KEYS
SEALED_CORE_MEMBERS = SEALED_DISTAL_KEYS

TARGET_KEYS = frozenset(
    {
        "schema_version",
        "eye_id",
        "lens_index",
        "proximal_points_xyz_um",
        "canonical_grid_xy",
        "target_smoothed_thickness_um",
        "raw_target_xy_normalized",
        "raw_target_thickness_um",
        "target_coefficients_c0_c5",
        "sealed_distal_relpath",
        "sealed_distal_sha256",
        "technical_config_sha256",
        "outcome_config_json",
        "outcome_config_sha256",
    }
)

# The old geometry implementation is intentionally reused without changing its
# thresholds.  Experiment 64 adds a robust selection before this frozen fit.
DEFAULT_THRESHOLD_CONFIG: dict[str, Any] = dict(EXPERIMENT63_THRESHOLD_CONFIG)
DEFAULT_PIPELINE_CONFIG: dict[str, Any] = dict(EXPERIMENT63_PIPELINE_CONFIG)


def _scalar(array: np.ndarray, name: str) -> Any:
    if array.shape != ():
        raise ExtractionError(f"sealed core field {name!r} must be scalar")
    return array.item()


def _int64_scalar(array: np.ndarray, name: str) -> int:
    value = np.asarray(array)
    if value.dtype != np.dtype(np.int64) or value.shape != ():
        raise ExtractionError(f"field {name!r} must be a scalar int64")
    return int(value.item())


def _require_npz_members(path: Path, expected: frozenset[str]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual = frozenset(archive.files)
    except (OSError, ValueError) as exc:
        raise ExtractionError(f"cannot validate NPZ member inventory {path}: {exc}") from exc
    if actual != expected:
        raise ExtractionError(
            f"NPZ member inventory differs for {path.name}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_manifest_matches(
    root: Path, manifest: Mapping[str, Mapping[str, Any]]
) -> None:
    for relative_path, expected in manifest.items():
        path = root / relative_path
        if not path.is_file() or _artifact_entry(path) != dict(expected):
            raise ExtractionError(f"artifact changed after hashing: {relative_path}")


def _validate_stage1_pair(
    instance_path: Path,
    core_path: Path,
    *,
    lens_index: int,
    raw_distal_support: int,
) -> None:
    """Cross-check the target-free reviewer artifact against the sealed core."""

    _require_npz_members(instance_path, INSTANCE_KEYS)
    _require_npz_members(core_path, SEALED_DISTAL_KEYS)
    with np.load(instance_path, allow_pickle=False) as instance:
        schema = str(_scalar(instance["schema_version"], "schema_version"))
        stored_index = _int64_scalar(instance["lens_index"], "lens_index")
        full = np.asarray(instance["full_assigned_points_zyx"])
        main = np.asarray(instance["main_component_points_zyx"])
        raw = np.asarray(instance["raw_distal_points_zyx"])
        instance_config_sha256 = str(
            _scalar(instance["config_sha256"], "config_sha256")
        )
    with np.load(core_path, allow_pickle=False) as core:
        core_points = np.asarray(core["points_zyx"])
        stored_core_index = _int64_scalar(core["lens_index"], "lens_index")
        stored_raw_support = _int64_scalar(
            core["raw_distal_support"], "raw_distal_support"
        )
        core_config_sha256 = str(_scalar(core["config_sha256"], "config_sha256"))
    if (
        schema != INSTANCE_SCHEMA
        or stored_index != lens_index
        or stored_core_index != lens_index
    ):
        raise ExtractionError("instance schema or lens index differs from its contract")
    for name, points in (("full", full), ("main", main), ("raw distal", raw), ("core", core_points)):
        if points.dtype != np.dtype(np.int32) or points.ndim != 2 or points.shape[1:] != (3,):
            raise ExtractionError(f"{name} source coordinates must be int32 with shape (n,3)")
        if len(np.unique(points, axis=0)) != len(points):
            raise ExtractionError(f"{name} source coordinates must be unique")
    if len(raw) != raw_distal_support or stored_raw_support != raw_distal_support:
        raise ExtractionError("raw distal support differs across technical artifacts")
    full_set = set(map(tuple, full.tolist()))
    main_set = set(map(tuple, main.tolist()))
    raw_set = set(map(tuple, raw.tolist()))
    if not main_set.issubset(full_set):
        raise ExtractionError("main component is not a subset of the assigned instance")
    if not raw_set.issubset(main_set):
        raise ExtractionError("raw distal candidate is not a subset of the main component")
    if not set(map(tuple, core_points.tolist())).issubset(raw_set):
        raise ExtractionError("sealed robust core is not a subset of the raw distal candidate")
    if instance_config_sha256 != core_config_sha256:
        raise ExtractionError("instance and sealed core configurations differ")


def _normalise_configs(
    threshold_config: Mapping[str, Any] | None,
    pipeline_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    thresholds = dict(DEFAULT_THRESHOLD_CONFIG)
    if threshold_config is not None:
        unknown = sorted(set(threshold_config) - set(thresholds))
        if unknown:
            raise ExtractionError(
                "unknown distal geometry configuration key(s): " + ", ".join(unknown)
            )
        thresholds.update(dict(threshold_config))
    # Canonicalisation is also the shared geometry module's complete validator.
    thresholds = json.loads(distal_geometry.canonical_config_json(thresholds))
    frozen_thresholds = json.loads(
        distal_geometry.canonical_config_json(DEFAULT_THRESHOLD_CONFIG)
    )
    if thresholds != frozen_thresholds:
        raise ExtractionError(
            "Experiment 64 distal geometry configuration must remain exactly frozen"
        )

    pipeline = dict(DEFAULT_PIPELINE_CONFIG)
    if pipeline_config is not None:
        unknown = sorted(set(pipeline_config) - set(pipeline))
        if unknown:
            raise ExtractionError(
                "unknown pipeline configuration key(s): " + ", ".join(unknown)
            )
        pipeline.update(dict(pipeline_config))
    try:
        pipeline = json.loads(_canonical_json(pipeline))
        frozen_pipeline = json.loads(_canonical_json(DEFAULT_PIPELINE_CONFIG))
    except (TypeError, ValueError) as exc:
        raise ExtractionError("pipeline configuration must be JSON-native") from exc
    if pipeline != frozen_pipeline:
        raise ExtractionError(
            "Experiment 64 pipeline configuration must remain exactly frozen"
        )

    robust_config = robust_distal_core.normalise_robust_core_config()
    coherence_config = dict(technical_metrics.TECHNICAL_COHERENCE_CONFIG)
    support_floor = int(coherence_config["fit_support_minimum"])
    rmse_ceiling = float(coherence_config["fit_rmse_max_um"])
    if int(thresholds["min_sealed_distal_cap_points"]) != support_floor:
        raise ExtractionError(
            "distal geometry support floor differs from the shared technical metric"
        )
    if not math.isclose(
        float(thresholds["distal_fit_rmse_max_um"]),
        rmse_ceiling,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExtractionError(
            "distal geometry RMSE ceiling differs from the shared technical metric"
        )
    if int(robust_config["minimum_downstream_fit_points"]) < support_floor:
        raise ExtractionError(
            "robust core does not guarantee the shared downstream fit support floor"
        )
    if not math.isclose(
        float(robust_config["downstream_fit_quantile"]),
        float(distal_geometry.SCALE_QUANTILE),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExtractionError(
            "robust-core downstream quantile differs from distal geometry"
        )
    if int(pipeline["target_min_support"]) != support_floor:
        raise ExtractionError("outcome support must remain equal to the frozen support floor")
    if pipeline["fixed_point_policy"] != "monotone_drop_only_no_reentry":
        raise ExtractionError("fixed-point policy must remain drop-only")
    if pipeline["distal_split"] != (
        "largest_26_component_boundary_deterministic_1d_two_means"
    ):
        raise ExtractionError("oracle distal split policy has changed")
    if pipeline["canonical_grid"] != "experiment57_disk_radius_0.65_step_0.13":
        raise ExtractionError("canonical predictor grid has changed")
    if float(pipeline["target_main_component_fraction_min"]) != 0.99:
        raise ExtractionError("component fraction minimum must remain frozen at 0.99")
    spacing = np.asarray(pipeline["original_spacing_um"], dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ExtractionError("original_spacing_um must contain three positive values")
    pipeline["original_spacing_um"] = [float(value) for value in spacing]
    return thresholds, pipeline, robust_config


def _technical_configs(
    thresholds: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    robust_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictor_pipeline_config = {
        key: pipeline[key]
        for key in (
            "fixed_point_policy",
            "original_spacing_um",
            "distal_split",
            "canonical_grid",
        )
    }
    technical_pipeline_config = {
        **predictor_pipeline_config,
        "component_fraction_minimum": float(
            pipeline["target_main_component_fraction_min"]
        ),
    }
    sealed_config = {
        "analysis_scope": ANALYSIS_SCOPE,
        "isolation_basis": ISOLATION_BASIS,
        "predictor_pipeline_config": predictor_pipeline_config,
        "technical_pipeline_config": technical_pipeline_config,
        "robust_core_config": dict(robust_config),
        "robust_core_config_sha256": robust_distal_core.robust_core_config_sha256(
            robust_config
        ),
        "technical_coherence_config": dict(
            technical_metrics.TECHNICAL_COHERENCE_CONFIG
        ),
        "technical_coherence_config_sha256": (
            technical_metrics.technical_coherence_config_sha256()
        ),
        "threshold_config": dict(thresholds),
    }
    return technical_pipeline_config, sealed_config


def _outcome_config(
    pipeline: Mapping[str, Any], technical_config_sha256: str
) -> dict[str, Any]:
    return {
        "technical_config_sha256": technical_config_sha256,
        "component_fraction_gate_source": (
            "technical_inventory.component_fraction_gate_pass"
        ),
        "target_main_component_fraction_min": float(
            pipeline["target_main_component_fraction_min"]
        ),
        "target_min_support": int(pipeline["target_min_support"]),
        "target_fit_rmse_max_um": float(pipeline["target_fit_rmse_max_um"]),
        "target_q05_thickness_min_um_exclusive": float(
            pipeline["target_q05_thickness_min_um_exclusive"]
        ),
        "target_observation_operator": (
            "one_minimum_axial_voxel_per_spanning_lateral_bin"
        ),
        "target_fit_method": "shared_huber_irls_normalized_quadratic",
        "canonical_grid": pipeline["canonical_grid"],
    }


def _robust_core_or_empty(
    raw_points_zyx: np.ndarray,
    spacing_zyx_um: Sequence[float],
    robust_config: Mapping[str, Any],
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    """Select in physical XYZ and apply the mask to unchanged int32 ZYX."""

    points_zyx = np.asarray(raw_points_zyx)
    spacing = np.asarray(spacing_zyx_um, dtype=np.float64)
    if (
        points_zyx.dtype != np.dtype(np.int32)
        or points_zyx.ndim != 2
        or points_zyx.shape[1:] != (3,)
        or np.any(points_zyx < 0)
        or len(np.unique(points_zyx, axis=0)) != len(points_zyx)
    ):
        raise ExtractionError(
            "raw localized distal points must be unique non-negative int32 ZYX"
        )
    if len(points_zyx):
        order = np.lexsort(
            (points_zyx[:, 2], points_zyx[:, 1], points_zyx[:, 0])
        )
        if not np.array_equal(order, np.arange(len(points_zyx))):
            raise ExtractionError("raw localized distal points must be sorted in ZYX order")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ExtractionError("voxel spacing must contain three positive finite values")
    points_xyz = points_zyx[:, ::-1].astype(np.float64) * spacing[::-1]
    try:
        selection = robust_distal_core.select_robust_distal_core(
            points_xyz, config=robust_config
        )
    except robust_distal_core.RobustDistalCoreError as exc:
        reason = str(exc)
        return (
            np.empty((0, 3), dtype=np.int32),
            "ineligible",
            [reason],
            {
                "status": "ineligible",
                "input_support": int(len(points_zyx)),
                "reason": reason,
                "config_sha256": (
                    robust_distal_core.robust_core_config_sha256(robust_config)
                ),
            },
        )
    retained_mask = np.asarray(selection["retained_mask"])
    if retained_mask.dtype != np.dtype(np.bool_) or retained_mask.shape != (
        len(points_zyx),
    ):
        raise ExtractionError(
            "robust-core retained_mask is not aligned to localized ZYX input"
        )
    core_zyx = points_zyx[retained_mask].astype(np.int32, copy=True)
    selected_xyz = core_zyx[:, ::-1].astype(np.float64) * spacing[::-1]
    canonical_selected_xyz = np.asarray(
        selection["retained_points_xyz_um"], dtype=np.float64
    )
    if set(map(tuple, selected_xyz.tolist())) != set(
        map(tuple, canonical_selected_xyz.tolist())
    ):
        raise ExtractionError(
            "robust-core physical output differs from its retained input mask"
        )
    if selection["config_sha256"] != (
        robust_distal_core.robust_core_config_sha256(robust_config)
    ):
        raise ExtractionError("robust-core selector used a different configuration")
    diagnostics = {**dict(selection["diagnostics"]), "status": "pass"}
    return core_zyx, "pass", [], diagnostics


def _sealed_core_payload(
    lens_index: int,
    points_zyx: np.ndarray,
    spacing_um: Sequence[float],
    sealed_config: Mapping[str, Any],
    *,
    raw_distal_support: int,
    robust_core_diagnostics: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    points = np.asarray(points_zyx)
    if points.dtype != np.dtype(np.int32) or points.ndim != 2 or points.shape[1:] != (3,):
        raise ExtractionError("sealed robust core must be int32 with shape (n,3)")
    config_json = _canonical_json(sealed_config)
    return {
        "schema_version": np.asarray(SEALED_DISTAL_SCHEMA),
        "lens_index": np.asarray(lens_index, dtype=np.int64),
        "points_zyx": points,
        "spacing_um": np.asarray(spacing_um, dtype=np.float64),
        "config_json": np.asarray(config_json),
        "config_sha256": np.asarray(_sha256_bytes(config_json.encode("utf-8"))),
        "raw_distal_support": np.asarray(raw_distal_support, dtype=np.int64),
        "robust_core_config_sha256": np.asarray(
            robust_distal_core.robust_core_config_sha256(
                sealed_config["robust_core_config"]
            )
        ),
        "robust_core_diagnostics_json": np.asarray(
            _canonical_json(robust_core_diagnostics)
        ),
    }


# Backward-readable local helper name used by small producer checks.
_sealed_payload = _sealed_core_payload


def load_sealed_distal_core(
    path: str | Path,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly validate and reopen one Experiment 64 Stage-2 core."""

    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".npz" or not artifact_path.is_file():
        raise ExtractionError(f"sealed distal core does not exist: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            names = frozenset(archive.files)
            if names != SEALED_DISTAL_KEYS:
                raise ExtractionError(
                    "sealed distal core keys differ from contract; "
                    f"missing={sorted(SEALED_DISTAL_KEYS - names)}, "
                    f"extra={sorted(names - SEALED_DISTAL_KEYS)}"
                )
            schema = str(_scalar(archive["schema_version"], "schema_version"))
            lens_index = _int64_scalar(archive["lens_index"], "lens_index")
            points_zyx = np.asarray(archive["points_zyx"])
            spacing_um = np.asarray(archive["spacing_um"])
            config_json = str(_scalar(archive["config_json"], "config_json"))
            stored_config_sha256 = str(
                _scalar(archive["config_sha256"], "config_sha256")
            )
            raw_distal_support = _int64_scalar(
                archive["raw_distal_support"], "raw_distal_support"
            )
            stored_robust_config_sha256 = str(
                _scalar(
                    archive["robust_core_config_sha256"],
                    "robust_core_config_sha256",
                )
            )
            robust_diagnostics_json = str(
                _scalar(
                    archive["robust_core_diagnostics_json"],
                    "robust_core_diagnostics_json",
                )
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ExtractionError):
            raise
        raise ExtractionError(f"cannot read sealed distal core {artifact_path}: {exc}") from exc

    if schema != SEALED_DISTAL_SCHEMA:
        raise ExtractionError(f"unexpected sealed distal core schema {schema!r}")
    if lens_index < 0:
        raise ExtractionError("sealed core lens_index must be a non-negative integer")
    if raw_distal_support < 0:
        raise ExtractionError("sealed core raw_distal_support must be a non-negative integer")
    if points_zyx.dtype != np.dtype(np.int32) or points_zyx.ndim != 2 or points_zyx.shape[1:] != (3,):
        raise ExtractionError("sealed core points_zyx must be int32 with shape (n,3)")
    if np.any(points_zyx < 0):
        raise ExtractionError("sealed core points_zyx must be non-negative")
    if len(np.unique(points_zyx, axis=0)) != len(points_zyx):
        raise ExtractionError("sealed core points_zyx must be unique")
    if len(points_zyx) > raw_distal_support:
        raise ExtractionError("sealed core support exceeds raw localized support")
    if len(points_zyx):
        order = np.lexsort((points_zyx[:, 2], points_zyx[:, 1], points_zyx[:, 0]))
        if not np.array_equal(order, np.arange(len(points_zyx))):
            raise ExtractionError("sealed core points_zyx must retain canonical ZYX order")
    if spacing_um.dtype != np.dtype(np.float64) or spacing_um.shape != (3,):
        raise ExtractionError("sealed core spacing_um must be float64 with shape (3,)")
    if not np.all(np.isfinite(spacing_um)) or np.any(spacing_um <= 0.0):
        raise ExtractionError("sealed core spacing_um must be finite and positive")
    calculated_hash = _sha256_bytes(config_json.encode("utf-8"))
    if stored_config_sha256 != calculated_hash:
        raise ExtractionError("sealed core config_sha256 does not match config_json")
    if expected_config_sha256 is not None and stored_config_sha256 != expected_config_sha256:
        raise ExtractionError("sealed core configuration hash differs from the producer")
    try:
        parsed_config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ExtractionError("sealed core config_json is invalid") from exc
    if not isinstance(parsed_config, dict) or _canonical_json(parsed_config) != config_json:
        raise ExtractionError("sealed core config_json must be a canonical object")
    required_config_keys = {
        "analysis_scope",
        "isolation_basis",
        "predictor_pipeline_config",
        "technical_pipeline_config",
        "robust_core_config",
        "robust_core_config_sha256",
        "technical_coherence_config",
        "technical_coherence_config_sha256",
        "threshold_config",
    }
    if set(parsed_config) != required_config_keys:
        raise ExtractionError("sealed core configuration has unexpected keys")
    if parsed_config["analysis_scope"] != ANALYSIS_SCOPE:
        raise ExtractionError("sealed core analysis scope differs from Experiment 64")
    if parsed_config["isolation_basis"] != ISOLATION_BASIS:
        raise ExtractionError("sealed core isolation basis differs from Experiment 64")
    predictor_config = parsed_config["predictor_pipeline_config"]
    expected_predictor_keys = {
        "fixed_point_policy",
        "original_spacing_um",
        "distal_split",
        "canonical_grid",
    }
    if not isinstance(predictor_config, dict) or set(predictor_config) != expected_predictor_keys:
        raise ExtractionError("sealed predictor pipeline configuration has unexpected keys")
    if predictor_config["fixed_point_policy"] != "monotone_drop_only_no_reentry":
        raise ExtractionError("sealed predictor fixed-point policy is not drop-only")
    if predictor_config["distal_split"] != (
        "largest_26_component_boundary_deterministic_1d_two_means"
    ):
        raise ExtractionError("sealed predictor oracle split policy differs")
    if predictor_config["canonical_grid"] != "experiment57_disk_radius_0.65_step_0.13":
        raise ExtractionError("sealed predictor canonical grid differs")
    predictor_spacing = np.asarray(
        predictor_config["original_spacing_um"], dtype=np.float64
    )
    if (
        predictor_spacing.shape != (3,)
        or not np.all(np.isfinite(predictor_spacing))
        or np.any(predictor_spacing <= 0.0)
        or not np.array_equal(predictor_spacing, spacing_um)
    ):
        raise ExtractionError("sealed predictor spacing differs from spacing_um")
    technical_pipeline_config = parsed_config["technical_pipeline_config"]
    if (
        not isinstance(technical_pipeline_config, dict)
        or set(technical_pipeline_config)
        != expected_predictor_keys | {"component_fraction_minimum"}
        or any(
            technical_pipeline_config[key] != predictor_config[key]
            for key in expected_predictor_keys
        )
        or float(technical_pipeline_config["component_fraction_minimum"]) != 0.99
    ):
        raise ExtractionError("sealed technical pipeline configuration differs")
    thresholds = json.loads(
        distal_geometry.canonical_config_json(parsed_config["threshold_config"])
    )
    normalized_robust = robust_distal_core.normalise_robust_core_config(
        parsed_config["robust_core_config"]
    )
    if normalized_robust != parsed_config["robust_core_config"]:
        raise ExtractionError("sealed robust-core configuration is not canonical")
    if parsed_config["robust_core_config_sha256"] != (
        robust_distal_core.robust_core_config_sha256(normalized_robust)
    ):
        raise ExtractionError("sealed robust-core configuration hash differs")
    if stored_robust_config_sha256 != parsed_config["robust_core_config_sha256"]:
        raise ExtractionError("sealed core robust-core hash member differs from config")
    try:
        robust_diagnostics = json.loads(robust_diagnostics_json)
    except json.JSONDecodeError as exc:
        raise ExtractionError("sealed robust-core diagnostics JSON is invalid") from exc
    if (
        not isinstance(robust_diagnostics, dict)
        or _canonical_json(robust_diagnostics) != robust_diagnostics_json
    ):
        raise ExtractionError("sealed robust-core diagnostics must be a canonical object")
    if int(robust_diagnostics.get("input_support", -1)) != raw_distal_support:
        raise ExtractionError("sealed robust-core diagnostics raw support differs")
    if robust_diagnostics.get("config_sha256") != stored_robust_config_sha256:
        raise ExtractionError("sealed robust-core diagnostics config hash differs")
    minimum_retained = int(normalized_robust["minimum_retained_points"])
    if len(points_zyx):
        if len(points_zyx) < minimum_retained:
            raise ExtractionError("nonempty sealed robust core is below its support floor")
        if int(robust_diagnostics.get("retained_support", -1)) != len(points_zyx):
            raise ExtractionError("sealed robust-core diagnostics retained support differs")
        if robust_diagnostics.get("status") != "pass":
            raise ExtractionError("nonempty sealed robust core is not marked pass")
    elif robust_diagnostics.get("status") != "ineligible":
        raise ExtractionError("empty sealed robust core lacks ineligible diagnostics")
    technical_metrics.canonical_technical_coherence_config_json(
        parsed_config["technical_coherence_config"]
    )
    if parsed_config["technical_coherence_config_sha256"] != (
        technical_metrics.technical_coherence_config_sha256()
    ):
        raise ExtractionError("sealed technical-coherence configuration hash differs")
    points_xyz_um = points_zyx[:, ::-1].astype(np.float64) * spacing_um[::-1]
    minimum = int(thresholds["min_sealed_distal_cap_points"])
    return {
        "schema_version": schema,
        "lens_index": lens_index,
        "lens_id": lens_index,
        "points_zyx": points_zyx.copy(),
        "spacing_um": spacing_um.copy(),
        "points_xyz_um": points_xyz_um,
        "raw_distal_support": raw_distal_support,
        "robust_core_config_sha256": stored_robust_config_sha256,
        "robust_core_diagnostics": robust_diagnostics,
        "config": thresholds,
        "sealed_config": parsed_config,
        "config_json": config_json,
        "config_sha256": stored_config_sha256,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _artifact_entry(artifact_path)["sha256"],
        # The Experiment 63 frame-audit API accepts this capability marker;
        # schema validation has already happened above under the v64 contract.
        "sealed_distal_artifact": True,
        "stage1_eligible": bool(len(points_zyx) >= minimum),
    }


def _input_bindings(
    *,
    mask_path: Path,
    mask_provenance_path: Path,
    seeds_path: Path,
    seed_provenance_path: Path,
) -> dict[str, dict[str, Any]]:
    return {
        role: {"name": path.name, **_artifact_entry(path)}
        for role, path in (
            ("mask_npy", mask_path),
            ("mask_provenance", mask_provenance_path),
            ("seed_csv", seeds_path),
            ("seed_provenance", seed_provenance_path),
        )
    }


def _implementation_bindings(directory: Path) -> dict[str, dict[str, Any]]:
    names = (
        "prepare_maike_masks.py",
        "map_oda_to_source.py",
        "extract_lens_surfaces.py",
        "distal_only_geometry.py",
        "audit_distal_frame_stability.py",
        "experiment_64_robust_distal_core.py",
        "experiment_64_technical_metrics.py",
        "experiment_64_extract_lens_surfaces.py",
    )
    return {name: _artifact_entry(directory / name) for name in names}


def _empty_target(reason: str) -> dict[str, Any]:
    return {
        "target_resolvable": False,
        "target_qc": False,
        "target_resolvability_reasons": [reason],
        "target_qc_reasons": [reason],
        "proximal_points_xyz_um": np.empty((0, 3), dtype=np.float64),
        "raw_target_xy_normalized": np.empty((0, 2), dtype=np.float64),
        "raw_target_thickness_um": np.empty((0,), dtype=np.float64),
        "target_coefficients": np.full(6, np.nan, dtype=np.float64),
        "target_depth_um": math.nan,
        "target_support": 0,
        "target_q05_raw_thickness_um": math.nan,
        "target_rmse_um": math.nan,
    }


def _empty_coherence() -> dict[str, Any]:
    return {
        "distal_fit_support": 0,
        "distal_fit_rmse_um": math.nan,
        "distal_abs_residual_p95_um": math.nan,
        "distal_abs_residual_p99_um": math.nan,
        "coherence_support_margin": math.nan,
        "coherence_rmse_margin": math.nan,
        "coherence_lcc_margin": math.nan,
        "coherence_p99_over_scale_margin": math.nan,
        "coherence_margin": math.nan,
        "distal_fit_p99_residual_over_scale": math.nan,
        "distal_fit_26_component_count": 0,
        "distal_fit_26_largest_component_support": 0,
        "distal_fit_26_largest_component_fraction": math.nan,
        "maike_final_fit_gate_pass": False,
        "maike_final_fit_gate_reasons": ["base_distal_qc_failed"],
    }


def _maike_final_fit_metrics(
    record: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the two calibrated voxel-only gates on final q90 fit points."""

    points_zyx = np.asarray(record["points_zyx"])
    spacing = np.asarray(record["spacing_um"], dtype=np.float64)
    result = technical_metrics.maike_final_distal_coherence_metrics(
        points_zyx, spacing, geometry
    )
    if result.get("config_sha256") != (
        technical_metrics.technical_coherence_config_sha256()
    ):
        raise ExtractionError("shared Maike coherence metrics used a different config")
    reasons = sorted(set(map(str, result["maike_final_fit_gate_reasons"])))
    if bool(result["maike_final_fit_gate_pass"]) != (not reasons):
        raise ExtractionError("shared Maike coherence gate pass/reasons disagree")
    result["maike_final_fit_gate_reasons"] = reasons
    return result


def _run_maike_combined_fixed_point(
    records: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[
    set[int],
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """Continue base QC with calibrated Maike gates, without lens re-entry."""

    base_result = distal_geometry.run_monotone_fixed_point(
        records, config=thresholds
    )
    base_eligible, base_fixed, base_qc = _normalise_geometry_result(
        base_result,
        len(records),
        int(thresholds["fixed_point_max_iterations"]),
    )
    active = sorted(base_eligible)
    qc_by_index = {int(index): dict(value) for index, value in base_qc.items()}
    gate_by_index = {index: _empty_coherence() for index in range(len(records))}
    iteration_diagnostics: list[dict[str, Any]] = []
    eligible_counts = list(base_fixed["eligible_counts"])
    if len(eligible_counts) >= 2 and eligible_counts[-1] == eligible_counts[-2]:
        eligible_counts.pop()

    converged = False
    final_geometry_by_index: dict[int, dict[str, Any]] = {}
    for iteration in range(1, int(thresholds["fixed_point_max_iterations"]) + 1):
        geometry_result = distal_geometry.derive_distal_only_eye_geometry(
            [records[index] for index in active], config=thresholds
        )
        keep: list[int] = []
        dropped: list[int] = []
        reason_counts: dict[str, int] = {}
        iteration_geometry: dict[int, dict[str, Any]] = {}
        for index, raw_geometry in zip(
            active, geometry_result["per_lens"], strict=True
        ):
            geometry = _canonical_geometry_entry(raw_geometry)
            iteration_geometry[index] = geometry
            base_reasons = sorted(
                set(map(str, raw_geometry.get("distal_qc_reasons", [])))
            )
            if base_reasons:
                gate = _empty_coherence()
                combined_reasons = base_reasons
            else:
                gate = _maike_final_fit_metrics(records[index], geometry)
                combined_reasons = list(gate["maike_final_fit_gate_reasons"])
            gate_by_index[index] = gate
            qc_by_index[index] = {
                **dict(qc_by_index.get(index, {})),
                **geometry,
                "reasons": combined_reasons,
                "distal_qc_reasons": combined_reasons,
            }
            if combined_reasons:
                dropped.append(index)
                for reason in combined_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                keep.append(index)
        eligible_counts.append(len(keep))
        iteration_diagnostics.append(
            {
                "iteration": iteration,
                "start_count": len(active),
                "kept_count": len(keep),
                "dropped_count": len(dropped),
                "dropped_lens_indices": dropped,
                "reason_counts": dict(sorted(reason_counts.items())),
            }
        )
        if not dropped:
            converged = True
            active = keep
            final_geometry_by_index = iteration_geometry
            break
        active = keep
        if len(active) < 4:
            raise ExtractionError(
                "Maike final-fit gates left fewer than four distal lenses"
            )
    if not converged:
        raise ExtractionError("Maike combined distal-QC fixed point did not converge")
    if len(eligible_counts) - 1 > int(thresholds["fixed_point_max_iterations"]):
        raise ExtractionError("Maike combined distal-QC fixed point exceeded its limit")

    active_set = set(active)
    for index in range(len(records)):
        if index not in base_eligible:
            gate_by_index[index] = _empty_coherence()
        if index not in active_set:
            row = dict(qc_by_index.get(index, {}))
            reasons = sorted(
                set(
                    map(
                        str,
                        row.get("reasons", row.get("distal_qc_reasons", [])),
                    )
                )
            )
            row["reasons"] = reasons
            row["distal_qc_reasons"] = reasons
            qc_by_index[index] = row

    fixed_metadata = {
        "converged": True,
        "max_iterations": int(thresholds["fixed_point_max_iterations"]),
        "iterations": len(eligible_counts) - 1,
        "eligible_counts": eligible_counts,
        "readded_count": 0,
        "base_distal_qc": base_fixed,
        "technical_coherence_config_sha256": (
            technical_metrics.technical_coherence_config_sha256()
        ),
        "maike_final_fit_gate_iterations": iteration_diagnostics,
    }
    return (
        active_set,
        fixed_metadata,
        qc_by_index,
        final_geometry_by_index,
        gate_by_index,
    )


def build_eye_bundle(
    *,
    mask_path: Path,
    mask_provenance_path: Path,
    seeds_path: Path,
    seed_provenance_path: Path,
    eye_id: str,
    species: str,
    sex: str,
    output_path: Path,
    repository_root: Path | None = None,
    threshold_config: Mapping[str, Any] | None = None,
    pipeline_config: Mapping[str, Any] | None = None,
) -> Path:
    """Build and atomically publish a complete Experiment 64 eye bundle."""

    if not eye_id or not species:
        raise ExtractionError("eye_id and species must be non-empty")
    if sex not in {"F", "M"}:
        raise ExtractionError("sex must be 'F' or 'M'")
    thresholds, pipeline, robust_config = _normalise_configs(
        threshold_config, pipeline_config
    )
    technical_pipeline_config, sealed_config = _technical_configs(
        thresholds, pipeline, robust_config
    )
    technical_config_json = _canonical_json(sealed_config)
    technical_config_sha256 = _sha256_bytes(technical_config_json.encode("utf-8"))
    outcome_config = _outcome_config(pipeline, technical_config_sha256)
    outcome_config_json = _canonical_json(outcome_config)
    outcome_config_sha256 = _sha256_bytes(outcome_config_json.encode("utf-8"))

    # Bind every source before its first parser/validator read, then verify
    # again immediately after parsing to close a read-before-hash TOCTOU gap.
    input_hashes = _input_bindings(
        mask_path=mask_path,
        mask_provenance_path=mask_provenance_path,
        seeds_path=seeds_path,
        seed_provenance_path=seed_provenance_path,
    )
    mask, mask_provenance = _validate_mask(mask_path, mask_provenance_path)
    seed_provenance = _validate_seed_provenance(
        seed_provenance_path, seeds_path, eye_id=eye_id
    )
    _require_seed_mask_cross_binding(
        seed_provenance,
        mask_path=mask_path,
        mask_provenance_path=mask_provenance_path,
    )
    seeds = load_seeds(seeds_path)
    if mask_provenance.get("eye_id") not in (None, eye_id):
        raise ExtractionError("mask provenance eye_id differs from the requested eye")
    if seed_provenance.get("eye_id") not in (None, eye_id):
        raise ExtractionError("seed provenance eye_id differs from the requested eye")
    for count_key in ("n_expected", "n_rows"):
        recorded_count = seed_provenance.get(count_key)
        if recorded_count is not None:
            if isinstance(recorded_count, bool) or not isinstance(recorded_count, int):
                raise ExtractionError(f"seed provenance {count_key} must be an integer")
            if recorded_count != len(seeds):
                raise ExtractionError(f"seed provenance {count_key} differs from the CSV")
    candidate_count = seed_provenance.get("candidate_seeds_per_voxel", 1)
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != 1
    ):
        raise ExtractionError("seed provenance changed candidate_seeds_per_voxel")
    if _input_bindings(
        mask_path=mask_path,
        mask_provenance_path=mask_provenance_path,
        seeds_path=seeds_path,
        seed_provenance_path=seed_provenance_path,
    ) != input_hashes:
        raise ExtractionError("an input artifact changed while it was being parsed")

    repository_root = repository_root or Path(__file__).resolve().parents[2]
    git_identity = _git_identity(repository_root)
    if git_identity.get("dirty") is not False:
        raise ExtractionError("Experiment 64 production requires an exactly clean Git tree")
    if output_path.exists():
        raise ExtractionError(f"output already exists: {output_path}")

    implementation_directory = Path(__file__).resolve().parent
    implementation_hashes = _implementation_bindings(implementation_directory)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging.", dir=output_path.parent)
    )
    try:
        (staging / "instances").mkdir()
        (staging / "sealed_distal").mkdir()
        (staging / "sealed_outcomes").mkdir()
        instances, partition = _partition_foreground(mask, seeds)
        spacing = np.asarray(pipeline["original_spacing_um"], dtype=np.float64)

        stage1_rows: list[dict[str, Any]] = []
        components: dict[int, np.ndarray] = {}
        sealed_paths: list[Path] = []
        for seed, assigned in zip(seeds, instances, strict=True):
            component, component_sizes = deterministic_largest_component(assigned)
            components[seed.lens_index] = component
            fraction = float(len(component) / len(assigned)) if len(assigned) else 0.0
            localized, split = localize_distal_cap(
                component, seed.centre_zyx, seed.outward_axis_zyx
            )
            localized = np.unique(
                np.asarray(localized, dtype=np.int32), axis=0
            ).astype(np.int32, copy=False)
            (
                core_zyx,
                robust_status,
                robust_reasons,
                robust_diagnostics,
            ) = _robust_core_or_empty(localized, spacing, robust_config)

            instance_relpath = f"instances/lens_{seed.lens_index:06d}.npz"
            sealed_relpath = f"sealed_distal/lens_{seed.lens_index:06d}.npz"
            _atomic_savez(
                staging / instance_relpath,
                schema_version=np.asarray(INSTANCE_SCHEMA),
                lens_index=np.asarray(seed.lens_index, dtype=np.int64),
                full_assigned_points_zyx=np.asarray(assigned, dtype=np.int32),
                main_component_points_zyx=np.asarray(component, dtype=np.int32),
                raw_distal_points_zyx=localized,
                component_sizes_descending=np.asarray(component_sizes, dtype=np.int64),
                spacing_um=spacing,
                seed_source_zyx=np.asarray(seed.centre_zyx, dtype=np.float64),
                oda_axis_source_zyx=np.asarray(seed.outward_axis_zyx, dtype=np.float64),
                config_json=np.asarray(technical_config_json),
                config_sha256=np.asarray(technical_config_sha256),
            )
            _atomic_savez(
                staging / sealed_relpath,
                **_sealed_core_payload(
                    seed.lens_index,
                    core_zyx,
                    spacing,
                    sealed_config,
                    raw_distal_support=len(localized),
                    robust_core_diagnostics=robust_diagnostics,
                ),
            )
            _validate_stage1_pair(
                staging / instance_relpath,
                staging / sealed_relpath,
                lens_index=seed.lens_index,
                raw_distal_support=len(localized),
            )
            sealed_paths.append(staging / sealed_relpath)
            stage1_rows.append(
                {
                    "lens_index": seed.lens_index,
                    "seed_id": seed.seed_id,
                    "assignment_status": "ok" if len(assigned) else "empty_assignment",
                    "full_assigned_size": int(len(assigned)),
                    "main_component_size": int(len(component)),
                    "component_removed_size": int(len(assigned) - len(component)),
                    "main_component_fraction": fraction,
                    "component_sizes_json": _canonical_json(component_sizes),
                    "component_fraction_gate_pass": target_component_fraction_passes(
                        len(component),
                        len(assigned),
                        float(pipeline["target_main_component_fraction_min"]),
                    ),
                    "raw_distal_support": int(len(localized)),
                    "oracle_localization_status": str(split.get("status", "unknown")),
                    "robust_core_status": robust_status,
                    "robust_core_reasons": robust_reasons,
                    "robust_core_support": int(len(core_zyx)),
                    "robust_core_retained_fraction": (
                        float(len(core_zyx) / len(localized)) if len(localized) else 0.0
                    ),
                    "robust_core_diagnostics": robust_diagnostics,
                    "instance_relpath": instance_relpath,
                    "sealed_distal_relpath": sealed_relpath,
                }
            )

        # Stage 2 is passed only records reconstructed from the just-sealed
        # integer cores.  Neither the assigned volumes nor the oracle axes are
        # arguments to any geometry, frame, QC, or coherence operation.
        del instances
        sealed_stage1_manifest = {
            path.relative_to(staging).as_posix(): _artifact_entry(path)
            for path in sorted(sealed_paths)
        }
        records: list[dict[str, Any]] = []
        for sealed_path, stage1_row in zip(sealed_paths, stage1_rows, strict=True):
            relative = sealed_path.relative_to(staging).as_posix()
            if _artifact_entry(sealed_path) != sealed_stage1_manifest[relative]:
                raise ExtractionError(
                    f"sealed distal core changed before Stage 2: {relative}"
                )
            loaded = load_sealed_distal_core(
                sealed_path,
                expected_config_sha256=technical_config_sha256,
            )
            if int(loaded["lens_index"]) != int(stage1_row["lens_index"]):
                raise ExtractionError("sealed core ordering differs from Stage 1")
            if int(loaded["raw_distal_support"]) != int(
                stage1_row["raw_distal_support"]
            ):
                raise ExtractionError("sealed core raw support differs from inventory")
            if len(loaded["points_zyx"]) != int(stage1_row["robust_core_support"]):
                raise ExtractionError("sealed core retained support differs from inventory")
            records.append(loaded)
        record_by_index = {int(record["lens_index"]): record for record in records}
        if len(record_by_index) != len(seeds):
            raise ExtractionError("sealed distal core lens indices are not unique")

        (
            eligible,
            fixed_point_metadata,
            per_lens_qc,
            per_lens_geometry,
            per_lens_gate,
        ) = _run_maike_combined_fixed_point(records, thresholds)

        audit_module = _load_frame_audit_module()
        audit_records = [
            {**record, "stage1_eligible": int(record["lens_index"]) in eligible}
            for record in records
        ]
        frame_audit = dict(
            audit_module.run_frame_stability_audit(
                audit_records, eye_id=eye_id, config=thresholds
            )
        )
        if frame_audit.get("schema_version") != "experiment63.distal-frame-audit.v1":
            raise ExtractionError("shared frame audit returned an unexpected schema")
        if frame_audit.get("eye_id") != eye_id:
            raise ExtractionError("shared frame audit returned the wrong eye_id")
        if not bool(frame_audit.get("gate_passed")):
            raise ExtractionError("distal frame stability gate failed")
        frame_audit["schema_version"] = FRAME_AUDIT_SCHEMA
        frame_audit["input_type"] = "sealed_robust_distal_core_artifacts_only"
        frame_audit["fixed_point"] = fixed_point_metadata
        frame_audit["technical_coherence_config_sha256"] = (
            technical_metrics.technical_coherence_config_sha256()
        )
        _atomic_json(staging / "distal_frame_audit.json", frame_audit)

        technical_rows: list[dict[str, Any]] = []
        sampling_rows: list[dict[str, Any]] = []
        for base in stage1_rows:
            index = int(base["lens_index"])
            geom = dict(per_lens_geometry.get(index, {}))
            qc = dict(per_lens_qc.get(index, {}))
            is_eligible = index in eligible
            coherence = dict(per_lens_gate[index])
            if is_eligible:
                if not geom:
                    raise ExtractionError(
                        f"eligible lens {index} has no final distal geometry"
                    )
                if int(coherence["distal_fit_support"]) < int(
                    technical_metrics.TECHNICAL_COHERENCE_CONFIG[
                        "fit_support_minimum"
                    ]
                ):
                    raise ExtractionError(
                        f"eligible lens {index} violates final fit support guarantee"
                    )
                if not bool(coherence["maike_final_fit_gate_pass"]):
                    raise ExtractionError(
                        f"eligible lens {index} failed a Maike final-fit gate"
                    )
                coherence_margin = float(coherence["coherence_margin"])
                if (
                    not math.isfinite(coherence_margin)
                    or coherence_margin < -1.0e-12
                ):
                    raise ExtractionError(
                        f"eligible lens {index} has an invalid coherence margin"
                    )
            distal_reasons = qc.get(
                "reasons", qc.get("distal_qc_reasons", [])
            )
            row = {
                "eye_id": eye_id,
                "lens_index": index,
                "seed_id": base["seed_id"],
                "species": species,
                "sex": sex,
                "assignment_status": base["assignment_status"],
                "full_assigned_size": base["full_assigned_size"],
                "main_component_size": base["main_component_size"],
                "component_removed_size": base["component_removed_size"],
                "main_component_fraction": base["main_component_fraction"],
                "component_sizes_json": base["component_sizes_json"],
                "component_fraction_gate_pass": base[
                    "component_fraction_gate_pass"
                ],
                "raw_distal_support": base["raw_distal_support"],
                "robust_core_status": base["robust_core_status"],
                "robust_core_reasons": "|".join(base["robust_core_reasons"]),
                "robust_core_support": base["robust_core_support"],
                "robust_core_retained_fraction": base[
                    "robust_core_retained_fraction"
                ],
                "robust_core_diagnostics_json": _canonical_json(
                    base["robust_core_diagnostics"]
                ),
                "distal_qc": bool(is_eligible),
                "distal_qc_reasons": "|".join(map(str, distal_reasons)),
                "central": bool(geom.get("central", False)),
                "position_u_um": geom.get("position_u_um", math.nan),
                "position_v_um": geom.get("position_v_um", math.nan),
                "distal_scale_um": geom.get(
                    "distal_scale_um", qc.get("distal_scale_um", math.nan)
                ),
                "distal_fit_support": coherence["distal_fit_support"],
                "distal_fit_rmse_um": coherence["distal_fit_rmse_um"],
                "distal_abs_residual_p95_um": coherence[
                    "distal_abs_residual_p95_um"
                ],
                "distal_abs_residual_p99_um": coherence[
                    "distal_abs_residual_p99_um"
                ],
                "distal_fit_p99_residual_over_scale": coherence[
                    "distal_fit_p99_residual_over_scale"
                ],
                "distal_fit_26_component_count": coherence[
                    "distal_fit_26_component_count"
                ],
                "distal_fit_26_largest_component_support": coherence[
                    "distal_fit_26_largest_component_support"
                ],
                "distal_fit_26_largest_component_fraction": coherence[
                    "distal_fit_26_largest_component_fraction"
                ],
                "maike_final_fit_gate_pass": bool(
                    coherence["maike_final_fit_gate_pass"]
                ),
                "maike_final_fit_gate_reasons": "|".join(
                    sorted(set(coherence["maike_final_fit_gate_reasons"]))
                ),
                "distal_gradient_magnitude": geom.get(
                    "distal_gradient_magnitude",
                    qc.get("distal_gradient_magnitude", math.nan),
                ),
                "distal_curvature_eigenvalue_1": geom.get(
                    "distal_curvature_eigenvalue_1",
                    qc.get("distal_curvature_eigenvalue_1", math.nan),
                ),
                "distal_curvature_eigenvalue_2": geom.get(
                    "distal_curvature_eigenvalue_2",
                    qc.get("distal_curvature_eigenvalue_2", math.nan),
                ),
                "distal_normalized_fit_residual": geom.get(
                    "distal_normalized_fit_residual",
                    qc.get("distal_normalized_fit_residual", math.nan),
                ),
                "coherence_support_margin": coherence[
                    "coherence_support_margin"
                ],
                "coherence_rmse_margin": coherence["coherence_rmse_margin"],
                "coherence_lcc_margin": coherence["coherence_lcc_margin"],
                "coherence_p99_over_scale_margin": coherence[
                    "coherence_p99_over_scale_margin"
                ],
                "coherence_margin": coherence["coherence_margin"],
                "instance_relpath": base["instance_relpath"],
                "sealed_distal_relpath": base["sealed_distal_relpath"],
            }
            technical_rows.append(row)
            sampling_rows.append(
                {
                    "eye_id": eye_id,
                    "lens_index": index,
                    "seed_id": base["seed_id"],
                    "distal_eligible": bool(is_eligible),
                    "position_u_um": row["position_u_um"] if is_eligible else "",
                    "position_v_um": row["position_v_um"] if is_eligible else "",
                    "distal_scale_um": row["distal_scale_um"] if is_eligible else "",
                    "coherence_margin": row["coherence_margin"] if is_eligible else "",
                    "instance_relpath": base["instance_relpath"],
                    "sealed_distal_relpath": base["sealed_distal_relpath"],
                }
            )

        _atomic_csv(
            staging / "technical_inventory.csv",
            technical_rows,
            TECHNICAL_INVENTORY_FIELDS,
        )
        _atomic_csv(
            staging / "distal_qc_sampling.csv",
            sampling_rows,
            DISTAL_QC_SAMPLING_FIELDS,
        )

        # Outcome stage.  These objects are deliberately created only after
        # every technical row and frame/coherence metric is final.
        target_rows: list[dict[str, Any]] = []
        canonical_grid = _canonical_grid()
        runtime_config = {**thresholds, **pipeline}
        for base in stage1_rows:
            index = int(base["lens_index"])
            geom = dict(per_lens_geometry.get(index, {}))
            core_xyz = np.asarray(
                record_by_index[index]["points_xyz_um"], dtype=np.float64
            )
            target = (
                _extract_target(
                    components[index],
                    core_xyz,
                    geom,
                    runtime_config,
                    robust_quadratic_fit=distal_geometry.fit_robust_quadratic,
                )
                if geom
                else _empty_target("no_distal_geometry")
            )
            if not bool(base["component_fraction_gate_pass"]):
                target["target_resolvable"] = False
                target["target_qc"] = False
                target["target_resolvability_reasons"] = sorted(
                    set(target.get("target_resolvability_reasons", []))
                    | {"main_component_fraction_below_minimum"}
                )
                target["target_qc_reasons"] = sorted(
                    set(target.get("target_qc_reasons", []))
                    | {"main_component_fraction_below_minimum"}
                )
            q05_raw = float(target["target_q05_raw_thickness_um"])
            target_rmse = float(target["target_rmse_um"])
            expected_target_qc = bool(
                target["target_resolvable"]
                and math.isfinite(q05_raw)
                and q05_raw
                > float(pipeline["target_q05_thickness_min_um_exclusive"])
                and math.isfinite(target_rmse)
                and target_rmse <= float(pipeline["target_fit_rmse_max_um"])
            )
            if bool(target["target_qc"]) != expected_target_qc:
                raise ExtractionError(f"target_qc equivalence failed for lens {index}")

            coefficients = np.asarray(
                target["target_coefficients"], dtype=np.float64
            )
            smoothed = _quadratic_design(canonical_grid) @ coefficients
            lens_relpath = f"sealed_outcomes/lens_{index:06d}.npz"
            sealed_relpath = str(base["sealed_distal_relpath"])
            _atomic_savez(
                staging / lens_relpath,
                schema_version=np.asarray(TARGET_SCHEMA),
                eye_id=np.asarray(eye_id),
                lens_index=np.asarray(index, dtype=np.int64),
                proximal_points_xyz_um=np.asarray(
                    target["proximal_points_xyz_um"], dtype=np.float64
                ),
                canonical_grid_xy=canonical_grid,
                target_smoothed_thickness_um=smoothed.astype(np.float64),
                raw_target_xy_normalized=np.asarray(
                    target["raw_target_xy_normalized"], dtype=np.float64
                ),
                raw_target_thickness_um=np.asarray(
                    target["raw_target_thickness_um"], dtype=np.float64
                ),
                target_coefficients_c0_c5=coefficients,
                sealed_distal_relpath=np.asarray(sealed_relpath),
                sealed_distal_sha256=np.asarray(
                    sealed_stage1_manifest[sealed_relpath]["sha256"]
                ),
                technical_config_sha256=np.asarray(technical_config_sha256),
                outcome_config_json=np.asarray(outcome_config_json),
                outcome_config_sha256=np.asarray(outcome_config_sha256),
            )
            _require_npz_members(staging / lens_relpath, TARGET_KEYS)
            target_row: dict[str, Any] = {
                "eye_id": eye_id,
                "lens_index": index,
                "target_resolvable": bool(target["target_resolvable"]),
                "target_resolvability_reasons": "|".join(
                    map(str, target.get("target_resolvability_reasons", []))
                ),
                "target_qc": bool(target["target_qc"]),
                "target_qc_reasons": "|".join(
                    map(str, target.get("target_qc_reasons", []))
                ),
                "target_depth_um": target["target_depth_um"],
                "target_support": int(target["target_support"]),
                "target_q05_raw_thickness_um": target[
                    "target_q05_raw_thickness_um"
                ],
                "target_rmse_um": target["target_rmse_um"],
                "lens_relpath": lens_relpath,
            }
            for coefficient_index in range(6):
                target_row[f"target_c{coefficient_index}"] = coefficients[
                    coefficient_index
                ]
            target_rows.append(target_row)

        _atomic_csv(
            staging / "sealed_outcomes/target_table.csv",
            target_rows,
            TARGET_TABLE_FIELDS,
        )

        technical_artifact_paths = {
            "technical_inventory.csv",
            "distal_qc_sampling.csv",
            "distal_frame_audit.json",
            *{f"instances/lens_{index:06d}.npz" for index in range(len(seeds))},
            *{
                f"sealed_distal/lens_{index:06d}.npz"
                for index in range(len(seeds))
            },
        }
        outcome_payload_paths = {
            "sealed_outcomes/target_table.csv",
            *{
                f"sealed_outcomes/lens_{index:06d}.npz"
                for index in range(len(seeds))
            },
        }
        _require_exact_staging_files(
            staging, technical_artifact_paths | outcome_payload_paths
        )
        technical_manifest = {
            relative: _artifact_entry(staging / relative)
            for relative in sorted(technical_artifact_paths)
        }
        outcome_payload_manifest = {
            relative: _artifact_entry(staging / relative)
            for relative in sorted(outcome_payload_paths)
        }
        technical_inventory_binding = {
            "relative_path": "technical_inventory.csv",
            **_artifact_entry(staging / "technical_inventory.csv"),
        }
        outcome_manifest = {
            "schema_version": OUTCOME_MANIFEST_SCHEMA,
            "status": "complete",
            "experiment": 64,
            "eye_id": eye_id,
            "species": species,
            "sex": sex,
            "analysis_scope": ANALYSIS_SCOPE,
            "technical_inventory_binding": technical_inventory_binding,
            "technical_config_sha256": technical_config_sha256,
            "threshold_config_sha256": _sha256_bytes(
                distal_geometry.canonical_config_json(thresholds).encode("utf-8")
            ),
            "robust_core_config_sha256": (
                robust_distal_core.robust_core_config_sha256(robust_config)
            ),
            "target_config": outcome_config,
            "target_config_sha256": outcome_config_sha256,
            "n_expected": len(seeds),
            "n_rows": len(target_rows),
            "counts": {
                "target_resolvable": sum(
                    bool(row["target_resolvable"]) for row in target_rows
                ),
                "target_qc": sum(bool(row["target_qc"]) for row in target_rows),
            },
            "contiguous_indices": True,
            "index_range": [0, len(seeds) - 1],
            "target_cohort_definitions": {
                "target_resolvable": (
                    "final distal frame available; component fraction gate passes; "
                    "at least 25 spanning lateral bins; finite full-rank fit"
                ),
                "target_qc": (
                    "target_resolvable and raw thickness q05 > 0 and outcome RMSE <= 2.5 um"
                ),
            },
            "input_hashes": input_hashes,
            "git": git_identity,
            "producer_code_sha256": implementation_hashes,
            "output_manifest": outcome_payload_manifest,
        }
        _atomic_json(staging / "sealed_outcomes/manifest.json", outcome_manifest)
        outcome_manifest_binding = {
            "relative_path": "sealed_outcomes/manifest.json",
            **_artifact_entry(staging / "sealed_outcomes/manifest.json"),
        }

        technical_counts = {
            "instances": len(technical_rows),
            "robust_core": sum(
                row["robust_core_status"] == "pass" for row in technical_rows
            ),
            "base_distal_qc": sum(
                row["maike_final_fit_gate_reasons"] != "base_distal_qc_failed"
                for row in technical_rows
            ),
            "distal_qc": sum(bool(row["distal_qc"]) for row in technical_rows),
            "maike_final_fit_connectivity_excluded": sum(
                "fit_points_26_lcc_fraction_below_minimum"
                in row["maike_final_fit_gate_reasons"].split("|")
                for row in technical_rows
            ),
            "maike_final_fit_residual_excluded": sum(
                "fit_abs_residual_p99_over_scale_above_maximum"
                in row["maike_final_fit_gate_reasons"].split("|")
                for row in technical_rows
            ),
            "maike_final_fit_both_excluded": sum(
                {
                    "fit_points_26_lcc_fraction_below_minimum",
                    "fit_abs_residual_p99_over_scale_above_maximum",
                }.issubset(set(row["maike_final_fit_gate_reasons"].split("|")))
                for row in technical_rows
            ),
        }
        common = {
            "schema_version": TECHNICAL_BUNDLE_SCHEMA,
            "status": "complete",
            "experiment": 64,
            "eye_id": eye_id,
            "species": species,
            "sex": sex,
            "biological_independence": {
                "independent_unit": "animal",
                "animal_id": eye_id,
                "one_eye_per_animal_in_validation": True,
                "source_basis": "one supplied eye stack per uniquely named fly",
            },
            "analysis_scope": ANALYSIS_SCOPE,
            "isolation_basis": ISOLATION_BASIS,
            "oracle_stage1_scope": ORACLE_STAGE1_SCOPE,
            "threshold_config": thresholds,
            "threshold_config_sha256": _sha256_bytes(
                distal_geometry.canonical_config_json(thresholds).encode("utf-8")
            ),
            "predictor_pipeline_config": sealed_config[
                "predictor_pipeline_config"
            ],
            "predictor_pipeline_config_sha256": _sha256_bytes(
                _canonical_json(
                    sealed_config["predictor_pipeline_config"]
                ).encode("utf-8")
            ),
            "technical_pipeline_config": technical_pipeline_config,
            "robust_core_config": robust_config,
            "robust_core_config_sha256": (
                robust_distal_core.robust_core_config_sha256(robust_config)
            ),
            "technical_coherence_config": dict(
                technical_metrics.TECHNICAL_COHERENCE_CONFIG
            ),
            "technical_coherence_config_sha256": (
                technical_metrics.technical_coherence_config_sha256()
            ),
            "technical_config_sha256": technical_config_sha256,
            "sealed_config_sha256": technical_config_sha256,
            "n_expected": len(seeds),
            "n_rows": len(technical_rows),
            "technical_counts": technical_counts,
            "contiguous_indices": True,
            "index_range": [0, len(seeds) - 1],
            "instance_segmentation_validated": False,
            "partition_evidence": partition,
            "fixed_point": fixed_point_metadata,
            "sealed_distal_stage1_manifest": sealed_stage1_manifest,
            "input_hashes": input_hashes,
            "git": git_identity,
            "producer_code_sha256": implementation_hashes,
            "technical_output_manifest": technical_manifest,
            # This is the technical side's only binding to sealed outcomes.
            "sealed_outcome_manifest_binding": outcome_manifest_binding,
        }
        technical_provenance = {
            **common,
            "document_role": "technical_provenance",
            "input_validation": {
                "mask_provenance_schema": mask_provenance.get("schema_version"),
                "seed_provenance_schema": seed_provenance.get("schema_version"),
                "seed_mask_cross_binding_validated": True,
            },
        }
        technical_completion = {
            **common,
            "document_role": "technical_completion",
        }
        _atomic_json(
            staging / "technical_provenance.json", technical_provenance
        )
        _atomic_json(
            staging / "technical_completion.json", technical_completion
        )

        expected_final_paths = (
            technical_artifact_paths
            | outcome_payload_paths
            | {
                "sealed_outcomes/manifest.json",
                "technical_provenance.json",
                "technical_completion.json",
            }
        )
        _require_exact_staging_files(staging, expected_final_paths)
        _require_manifest_matches(staging, technical_manifest)
        _require_manifest_matches(staging, outcome_payload_manifest)
        if _artifact_entry(staging / "sealed_outcomes/manifest.json") != {
            key: value
            for key, value in outcome_manifest_binding.items()
            if key != "relative_path"
        }:
            raise ExtractionError("sealed outcome manifest changed after binding")
        if _implementation_bindings(implementation_directory) != implementation_hashes:
            raise ExtractionError("producer implementation changed during extraction")
        current_inputs = _input_bindings(
            mask_path=mask_path,
            mask_provenance_path=mask_provenance_path,
            seeds_path=seeds_path,
            seed_provenance_path=seed_provenance_path,
        )
        if current_inputs != input_hashes:
            raise ExtractionError("an input artifact changed during extraction")
        if _git_identity(repository_root) != git_identity:
            raise ExtractionError("clean Git identity changed during extraction")
        os.replace(staging, output_path)
        return output_path
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="prepared uint8 NPY mask")
    parser.add_argument(
        "--provenance", type=Path, required=True, help="validated mask provenance JSON"
    )
    parser.add_argument("--seeds", type=Path, required=True, help="mapped ODA seed CSV")
    parser.add_argument("--seed-provenance", type=Path, required=True)
    parser.add_argument("--eye-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--sex", required=True, choices=["F", "M"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    result = build_eye_bundle(
        mask_path=arguments.mask,
        mask_provenance_path=arguments.provenance,
        seeds_path=arguments.seeds,
        seed_provenance_path=arguments.seed_provenance,
        eye_id=arguments.eye_id,
        species=arguments.species,
        sex=arguments.sex,
        output_path=arguments.output,
        repository_root=arguments.repository_root,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

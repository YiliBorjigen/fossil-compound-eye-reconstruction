#!/usr/bin/env python3
"""Prepare the robust-core Arthur source bundle for Experiment 64.

This is a separately numbered successor to Experiment 63.  It leaves the
Experiment 63 producer untouched, applies the same target-blind robust-core
operator used by the Maike adapter before sealing, and separates technical
predictor artifacts from hash-sealed target artifacts.  The latter are built
atomically but must not be opened until all twelve Experiment 64 Maike visual
QC attestations have passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
ARTHUR_DIRECTORY = REPOSITORY_ROOT / "experiments" / "arthur-modern-ground-truth"
for directory in (HERE, ARTHUR_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import distal_only_geometry as distal_geometry  # noqa: E402
import extract_lens_surfaces as maike63  # noqa: E402
import prepare_arthur_source_table as source63  # noqa: E402
from experiment_64_robust_distal_core import (  # noqa: E402
    RobustDistalCoreError,
    canonical_robust_core_config_json,
    normalise_robust_core_config,
    robust_core_config_sha256,
    select_robust_distal_core,
)
from experiment_64_technical_metrics import (  # noqa: E402
    TECHNICAL_COHERENCE_CONFIG,
    TechnicalMetricError,
    arthur_final_distal_coherence_metrics,
    technical_coherence_config_sha256,
)


TECHNICAL_BUNDLE_SCHEMA = "experiment64.arthur-technical-source.v1"
SEALED_DISTAL_SCHEMA = "experiment64.arthur-sealed-distal-core.v1"
OUTCOME_MANIFEST_SCHEMA = "experiment64.arthur-outcomes.v1"
TARGET_SCHEMA = "experiment64.arthur-target.v1"
FRAME_AUDIT_SCHEMA = "experiment64.distal-frame-audit.v1"
ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
ISOLATION_BASIS = "stage2_reads_only_sha256_sealed_robust_distal_core_artifacts"
ORACLE_STAGE1_SCOPE = "oracle_correspondence_and_distal_localization_only"

TECHNICAL_FIELDS = (
    "volume",
    "animal_id",
    "eye_id",
    "source_eye_unit",
    "lens_index",
    "species",
    "sex",
    "age_days_min",
    "age_days_max",
    "raw_pixel_pitch_um",
    "oracle_stage1_eligible",
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
    "distal_gradient_magnitude",
    "distal_curvature_eigenvalue_1",
    "distal_curvature_eigenvalue_2",
    "distal_normalized_fit_residual",
    "coherence_support_margin",
    "coherence_rmse_margin",
    "coherence_margin",
    "sealed_distal_relpath",
)

TARGET_FIELDS = (
    "volume",
    "eye_id",
    "lens_index",
    "target_resolvable",
    "target_resolvability_reasons",
    "target_qc",
    "target_qc_reasons",
    "target_depth_um",
    "target_q05_raw_thickness_um",
    "target_support",
    "target_rmse_um",
    "target_c0",
    "target_c1",
    "target_c2",
    "target_c3",
    "target_c4",
    "target_c5",
    "lens_relpath",
)

TARGET_KEYS = frozenset(
    {
        "schema_version",
        "volume",
        "eye_id",
        "lens_index",
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

CODE_PATHS = (
    "experiments/maike-modern-ground-truth/experiment_64_prepare_arthur_source_table.py",
    "experiments/maike-modern-ground-truth/experiment_64_robust_distal_core.py",
    "experiments/maike-modern-ground-truth/experiment_64_technical_metrics.py",
    "experiments/maike-modern-ground-truth/prepare_arthur_source_table.py",
    "experiments/maike-modern-ground-truth/extract_lens_surfaces.py",
    "experiments/maike-modern-ground-truth/distal_only_geometry.py",
    "experiments/maike-modern-ground-truth/audit_distal_frame_stability.py",
    "experiments/arthur-modern-ground-truth/experiment_57_outer_only_validation.py",
    "experiments/arthur-modern-ground-truth/experiment_58_cross_volume_confirmation.py",
)


class Experiment64SourceError(RuntimeError):
    """Raised when an Experiment 64 source invariant fails."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    result = maike63._artifact_entry(path)
    if relative_path is not None:
        return {"relative_path": relative_path, **result}
    return result


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            "true"
                            if isinstance(value, (bool, np.bool_)) and bool(value)
                            else "false"
                            if isinstance(value, (bool, np.bool_))
                            else value
                        )
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sealed_config(
    threshold_config: Mapping[str, Any], robust_config: Mapping[str, Any]
) -> dict[str, Any]:
    modality_specific_qc = {
        "observation_modality": "irregular_supplied_surface_mesh_vertices",
        "connectivity_gate": "not_applicable_irregular_surface_mesh_sampling",
        "residual_tail_gate": "not_applied_p95_and_p99_recorded_descriptively",
        "automatic_gate_sequence": [
            "robust_core_minimum_input_points",
            "robust_core_trimmed_pca_normal_gap",
            "robust_core_minimum_retained_points",
            "distal_scale_range_um",
            "distal_quadratic_rmse_max_um",
            "final_q90_fit_support_minimum",
        ],
        "robust_core_minimum_input_points": int(
            robust_config["minimum_input_points"]
        ),
        "robust_core_pca_minimum_normal_gap_ratio": float(
            robust_config["pca_minimum_normal_gap_ratio"]
        ),
        "robust_core_minimum_retained_points": int(
            robust_config["minimum_retained_points"]
        ),
        "distal_scale_min_um": float(threshold_config["distal_scale_min_um"]),
        "distal_scale_max_um": float(threshold_config["distal_scale_max_um"]),
        "distal_fit_rmse_max_um": float(
            threshold_config["distal_fit_rmse_max_um"]
        ),
        "final_q90_fit_support_minimum": int(
            threshold_config["min_sealed_distal_cap_points"]
        ),
    }
    return {
        "analysis_scope": ANALYSIS_SCOPE,
        "isolation_basis": ISOLATION_BASIS,
        "predictor_pipeline_config": {
            "fixed_point_policy": "monotone_drop_only_no_reentry",
            "distal_input": "oracle_localized_unique_mesh_vertices",
            "robust_core_method": robust_config["method"],
            "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
        },
        "robust_core_config": dict(robust_config),
        "robust_core_config_sha256": robust_core_config_sha256(robust_config),
        "technical_coherence_config": dict(TECHNICAL_COHERENCE_CONFIG),
        "technical_coherence_config_sha256": technical_coherence_config_sha256(),
        "modality_specific_qc": modality_specific_qc,
        "threshold_config": dict(threshold_config),
    }


def _outcome_config(technical_config_sha256: str) -> dict[str, Any]:
    return {
        "technical_config_sha256": technical_config_sha256,
        **dict(source63.TARGET_CONFIG),
    }


def _require_npz_members(path: Path, expected: frozenset[str]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual = frozenset(archive.files)
    except (OSError, ValueError) as exc:
        raise Experiment64SourceError(
            f"cannot validate NPZ member inventory {path}: {exc}"
        ) from exc
    if actual != expected:
        raise Experiment64SourceError(
            f"NPZ member inventory differs for {path.name}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _robust_core_or_empty(
    raw_points_xyz_um: np.ndarray,
    robust_config: Mapping[str, Any],
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    points = np.unique(np.asarray(raw_points_xyz_um, dtype=np.float64), axis=0)
    try:
        selected = select_robust_distal_core(points, config=robust_config)
    except RobustDistalCoreError as exc:
        return (
            np.empty((0, 3), dtype=np.float64),
            "ineligible",
            [str(exc)],
            {
                "status": "ineligible",
                "input_support": int(len(points)),
                "reason": str(exc),
                "config_sha256": robust_core_config_sha256(robust_config),
            },
        )
    core = np.asarray(selected["retained_points_xyz_um"], dtype=np.float64)
    return core, "pass", [], {**dict(selected["diagnostics"]), "status": "pass"}


def _write_sealed_core(
    path: Path,
    *,
    volume: str,
    eye_id: int,
    lens_index: int,
    raw_support: int,
    points_xyz_um: np.ndarray,
    diagnostics: Mapping[str, Any],
    sealed_config: Mapping[str, Any],
) -> dict[str, Any]:
    config_json = _canonical_json(sealed_config)
    config_sha = _sha256_bytes(config_json.encode("utf-8"))
    diagnostic_json = _canonical_json(diagnostics)
    maike63._atomic_savez(
        path,
        schema_version=np.asarray(SEALED_DISTAL_SCHEMA),
        volume=np.asarray(volume),
        eye_id=np.asarray(eye_id, dtype=np.int64),
        lens_index=np.asarray(lens_index, dtype=np.int64),
        points_xyz_um=np.asarray(points_xyz_um, dtype=np.float64),
        raw_distal_support=np.asarray(raw_support, dtype=np.int64),
        robust_core_config_sha256=np.asarray(
            robust_core_config_sha256(sealed_config["robust_core_config"])
        ),
        robust_core_diagnostics_json=np.asarray(diagnostic_json),
        config_json=np.asarray(config_json),
        config_sha256=np.asarray(config_sha),
    )
    return load_sealed_core(
        path,
        expected_volume=volume,
        expected_eye_id=eye_id,
        expected_lens_index=lens_index,
        expected_sealed_config=sealed_config,
    )


def _scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise Experiment64SourceError(f"sealed core {name} must be scalar")
    return value.item()


def _int64_scalar(array: np.ndarray, name: str) -> int:
    value = np.asarray(array)
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise Experiment64SourceError(f"sealed core {name} must be scalar int64")
    return int(value.item())


def load_sealed_core(
    path: Path,
    *,
    expected_volume: str,
    expected_eye_id: int,
    expected_lens_index: int,
    expected_sealed_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and load one Arthur robust core without any target input."""

    expected_members = {
        "schema_version",
        "volume",
        "eye_id",
        "lens_index",
        "points_xyz_um",
        "raw_distal_support",
        "robust_core_config_sha256",
        "robust_core_diagnostics_json",
        "config_json",
        "config_sha256",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_members:
                raise Experiment64SourceError("Arthur sealed-core members differ from contract")
            schema = str(_scalar(archive["schema_version"], "schema_version"))
            volume = str(_scalar(archive["volume"], "volume"))
            eye_id = _int64_scalar(archive["eye_id"], "eye_id")
            lens_index = _int64_scalar(archive["lens_index"], "lens_index")
            points = np.asarray(archive["points_xyz_um"])
            raw_support = _int64_scalar(
                archive["raw_distal_support"], "raw_distal_support"
            )
            core_sha = str(
                _scalar(archive["robust_core_config_sha256"], "robust_core_config_sha256")
            )
            diagnostic_json = str(
                _scalar(archive["robust_core_diagnostics_json"], "robust_core_diagnostics_json")
            )
            config_json = str(_scalar(archive["config_json"], "config_json"))
            config_sha = str(_scalar(archive["config_sha256"], "config_sha256"))
    except (OSError, ValueError) as exc:
        if isinstance(exc, Experiment64SourceError):
            raise
        raise Experiment64SourceError(f"cannot read Arthur sealed core {path}: {exc}") from exc

    if (
        schema != SEALED_DISTAL_SCHEMA
        or volume != expected_volume
        or eye_id != expected_eye_id
        or lens_index != expected_lens_index
    ):
        raise Experiment64SourceError("Arthur sealed-core identity differs from contract")
    if points.dtype != np.dtype("float64") or points.ndim != 2 or points.shape[1:] != (3,):
        raise Experiment64SourceError("Arthur sealed-core points must be float64[N,3]")
    if not np.all(np.isfinite(points)):
        raise Experiment64SourceError("Arthur sealed-core points are nonfinite")
    if len(points) and not np.array_equal(points, np.unique(points, axis=0)):
        raise Experiment64SourceError("Arthur sealed-core points are not unique canonical XYZ")
    if raw_support < len(points) or raw_support < 0:
        raise Experiment64SourceError("Arthur sealed-core support counts are impossible")
    expected_json = _canonical_json(expected_sealed_config)
    if config_json != expected_json or config_sha != _sha256_bytes(config_json.encode("utf-8")):
        raise Experiment64SourceError("Arthur sealed-core configuration binding differs")
    robust_config = expected_sealed_config["robust_core_config"]
    if core_sha != robust_core_config_sha256(robust_config):
        raise Experiment64SourceError("Arthur sealed-core robust config hash differs")
    try:
        diagnostics = json.loads(diagnostic_json)
    except json.JSONDecodeError as exc:
        raise Experiment64SourceError("Arthur sealed-core diagnostics are invalid JSON") from exc
    if _canonical_json(diagnostics) != diagnostic_json:
        raise Experiment64SourceError("Arthur sealed-core diagnostics are noncanonical")
    if int(diagnostics.get("input_support", -1)) != raw_support:
        raise Experiment64SourceError("Arthur sealed-core diagnostics input support differs")
    if diagnostics.get("config_sha256") != core_sha:
        raise Experiment64SourceError("Arthur sealed-core diagnostics config hash differs")
    if len(points):
        if diagnostics.get("status") != "pass":
            raise Experiment64SourceError("nonempty Arthur core is not marked pass")
        if int(diagnostics.get("retained_support", -1)) != len(points):
            raise Experiment64SourceError("Arthur sealed-core retained support differs")
        if raw_support < int(robust_config["minimum_input_points"]):
            raise Experiment64SourceError("Arthur core input support is below robust minimum")
        if len(points) < int(robust_config["minimum_retained_points"]):
            raise Experiment64SourceError("Arthur retained core is below robust minimum")
    elif diagnostics.get("status") != "ineligible":
        raise Experiment64SourceError("empty Arthur sealed core lacks an ineligible record")

    return {
        "schema_version": SEALED_DISTAL_SCHEMA,
        "volume": volume,
        "eye_id": eye_id,
        "lens_index": lens_index,
        "points_xyz_um": points.copy(),
        "config": dict(expected_sealed_config["threshold_config"]),
        "config_json": config_json,
        "config_sha256": config_sha,
        "artifact_path": str(path.resolve()),
        "artifact_sha256": maike63.sha256_file(path),
        "sealed_distal_artifact": True,
        "stage1_eligible": len(points)
        >= int(
            expected_sealed_config["threshold_config"]["min_sealed_distal_cap_points"]
        ),
    }


def _fixed_point_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(result.get("converged")) or int(result.get("readded_count", -1)) != 0:
        raise Experiment64SourceError("Arthur robust-core fixed point did not converge cleanly")
    reasons: Counter[str] = Counter()
    for row in result.get("per_lens", []):
        reasons.update(map(str, row.get("drop_reasons", [])))
    return {
        "converged": True,
        "max_iterations": int(result["max_iterations"]),
        "iterations": int(result["iterations"]),
        "eligible_counts": [int(value) for value in result["eligible_counts"]],
        "readded_count": 0,
        "initial_count": len(result["initial_record_indices"]),
        "eligible_count": len(result["eligible_indices"]),
        "excluded_count": len(result["initial_record_indices"])
        - len(result["eligible_indices"]),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "iteration_diagnostics": result["iteration_diagnostics"],
    }


def _empty_geometry_values() -> dict[str, Any]:
    names = (
        "position_u_um",
        "position_v_um",
        "distal_scale_um",
        "distal_fit_support",
        "distal_fit_rmse_um",
        "distal_abs_residual_p95_um",
        "distal_abs_residual_p99_um",
        "distal_gradient_magnitude",
        "distal_curvature_eigenvalue_1",
        "distal_curvature_eigenvalue_2",
        "distal_normalized_fit_residual",
        "coherence_support_margin",
        "coherence_rmse_margin",
        "coherence_margin",
    )
    result = {name: math.nan for name in names}
    result["central"] = False
    return result


def _geometry_values(
    points_xyz_um: np.ndarray, geometry: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        coherence = arthur_final_distal_coherence_metrics(
            points_xyz_um, geometry
        )
    except TechnicalMetricError as exc:
        raise Experiment64SourceError(f"cannot derive final distal coherence: {exc}") from exc
    position = np.asarray(geometry["position_2d_um"], dtype=np.float64)
    curvature = np.asarray(geometry["curvature_eigenvalues"], dtype=np.float64)
    return {
        "central": bool(geometry["central"]),
        "position_u_um": float(position[0]),
        "position_v_um": float(position[1]),
        "distal_scale_um": float(geometry["scale_um"]),
        "distal_fit_support": int(coherence["distal_fit_support"]),
        "distal_fit_rmse_um": float(coherence["distal_fit_rmse_um"]),
        "distal_abs_residual_p95_um": float(coherence["distal_abs_residual_p95_um"]),
        "distal_abs_residual_p99_um": float(coherence["distal_abs_residual_p99_um"]),
        "distal_gradient_magnitude": float(geometry["gradient_magnitude"]),
        "distal_curvature_eigenvalue_1": float(curvature[0]),
        "distal_curvature_eigenvalue_2": float(curvature[1]),
        "distal_normalized_fit_residual": float(geometry["normalised_rmse"]),
        "coherence_support_margin": float(coherence["coherence_support_margin"]),
        "coherence_rmse_margin": float(coherence["coherence_rmse_margin"]),
        "coherence_margin": float(coherence["coherence_margin"]),
    }


def _empty_target() -> dict[str, Any]:
    return {
        "target_resolvable": False,
        "target_resolvability_reasons": ["distal_qc_failed"],
        "target_qc": False,
        "target_qc_reasons": ["distal_qc_failed"],
        "target_depth_um": math.nan,
        "target_q05_raw_thickness_um": math.nan,
        "target_support": 0,
        "target_rmse_um": math.nan,
        "target_coefficients": np.full(6, np.nan, dtype=np.float64),
        "canonical_grid_xy": source63.CANONICAL_GRID_XY.copy(),
        "target_smoothed_thickness_um": np.full(81, np.nan, dtype=np.float64),
        "raw_target_xy_normalized": np.empty((0, 2), dtype=np.float64),
        "raw_target_thickness_um": np.empty(0, dtype=np.float64),
    }


def _process_eye(
    *,
    volume: str,
    eye_id: int,
    stage1_records: Sequence[Mapping[str, Any]],
    staging: Path,
    threshold_config: Mapping[str, Any],
    robust_config: Mapping[str, Any],
    technical_config_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    unit = f"{volume}:eye_{eye_id}"
    sealed_config = _sealed_config(threshold_config, robust_config)
    sealed_config_json = _canonical_json(sealed_config)
    if _sha256_bytes(sealed_config_json.encode("utf-8")) != technical_config_sha256:
        raise Experiment64SourceError("Arthur technical configuration binding differs")
    distal_records: list[dict[str, Any]] = []
    record_metadata: dict[int, dict[str, Any]] = {}
    proximal: dict[int, np.ndarray] = {}
    technical_artifacts: dict[str, Any] = {}

    for record in stage1_records:
        lens_index = int(record["landmark_id"])
        raw = np.unique(np.asarray(record["outer"], dtype=np.float64), axis=0)
        core, status, reasons, diagnostics = _robust_core_or_empty(raw, robust_config)
        relpath = f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        path = staging / relpath
        loaded = _write_sealed_core(
            path,
            volume=volume,
            eye_id=eye_id,
            lens_index=lens_index,
            raw_support=len(raw),
            points_xyz_um=core,
            diagnostics=diagnostics,
            sealed_config=sealed_config,
        )
        distal_records.append(loaded)
        record_metadata[lens_index] = {
            "raw_support": int(len(raw)),
            "core_status": status,
            "core_reasons": reasons,
            "core_support": int(len(core)),
            "core_fraction": float(len(core) / len(raw)) if len(raw) else 0.0,
            "core_diagnostics": diagnostics,
        }
        proximal[lens_index] = np.unique(np.asarray(record["inner"], dtype=np.float64), axis=0)
        technical_artifacts[relpath] = _binding(path)

    fixed = distal_geometry.run_monotone_fixed_point(distal_records, config=threshold_config)
    evidence = _fixed_point_evidence(fixed)
    eligible = {int(value) for value in fixed["eligible_indices"]}
    geometry_by_lens = {
        int(record["lens_index"]): geometry
        for record, geometry in zip(
            fixed["eligible_records"], fixed["geometry"]["per_lens"], strict=True
        )
    }
    fixed_by_lens = {int(row["lens_index"]): row for row in fixed["per_lens"]}
    distal_by_lens = {int(row["lens_index"]): row for row in distal_records}

    audit = source63.audit_frame_stability(
        distal_records, eye_id=unit, config=threshold_config
    )
    if not bool(audit.get("gate_passed")):
        raise Experiment64SourceError(f"distal frame stability gate failed for {unit}")
    audit["schema_version"] = FRAME_AUDIT_SCHEMA
    audit["input_type"] = "sealed_robust_distal_core_artifacts_only"
    audit["fixed_point"] = evidence
    for entry in audit["input_artifacts"]:
        index = int(entry["lens_index"])
        entry["relative_path"] = (
            f"sealed_distal/{volume}/eye_{eye_id}/lens_{index:06d}.npz"
        )
    audit_relpath = f"distal_frame_audits/{volume}_eye_{eye_id}.json"
    audit_path = staging / audit_relpath
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    maike63._atomic_json(audit_path, audit)
    technical_artifacts[audit_relpath] = _binding(audit_path)

    technical_rows: list[dict[str, Any]] = []
    pending_outcomes: list[dict[str, Any]] = []
    for record in stage1_records:
        lens_index = int(record["landmark_id"])
        metadata = record_metadata[lens_index]
        geometry = geometry_by_lens.get(lens_index)
        fixed_row = fixed_by_lens[lens_index]
        geometry_values = (
            _geometry_values(distal_by_lens[lens_index]["points_xyz_um"], geometry)
            if geometry is not None
            else _empty_geometry_values()
        )
        relpath = f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        technical_rows.append(
            {
                "volume": volume,
                "animal_id": f"Arthur_{volume}",
                "eye_id": eye_id,
                "source_eye_unit": unit,
                "lens_index": lens_index,
                "species": "Drosophila melanogaster",
                "sex": "female",
                "age_days_min": 6,
                "age_days_max": 7,
                "raw_pixel_pitch_um": (
                    source63.RAW_PIXEL_PITCH_UM[volume]
                    if source63.RAW_PIXEL_PITCH_UM[volume] is not None
                    else math.nan
                ),
                "oracle_stage1_eligible": True,
                "raw_distal_support": metadata["raw_support"],
                "robust_core_status": metadata["core_status"],
                "robust_core_reasons": "|".join(metadata["core_reasons"]),
                "robust_core_support": metadata["core_support"],
                "robust_core_retained_fraction": metadata["core_fraction"],
                "robust_core_diagnostics_json": _canonical_json(
                    metadata["core_diagnostics"]
                ),
                "distal_qc": lens_index in eligible,
                "distal_qc_reasons": "|".join(fixed_row.get("drop_reasons", [])),
                **geometry_values,
                "sealed_distal_relpath": relpath,
            }
        )

        pending_outcomes.append(
            {
                "volume": volume,
                "eye_id": eye_id,
                "lens_index": lens_index,
                "proximal_points_xyz_um": proximal[lens_index],
                "geometry": geometry,
            }
        )

    return technical_rows, pending_outcomes, evidence, audit_relpath, technical_artifacts


def _write_pending_targets(
    *,
    pending_outcomes: Sequence[Mapping[str, Any]],
    staging: Path,
    threshold_config: Mapping[str, Any],
    robust_config: Mapping[str, Any],
    technical_config_sha256: str,
    outcome_config_json: str,
    outcome_config_sha256: str,
) -> list[dict[str, Any]]:
    """Fit and seal outcomes only after every Arthur technical unit passed."""

    sealed_config = _sealed_config(threshold_config, robust_config)
    if _sha256_bytes(_canonical_json(sealed_config).encode("utf-8")) != (
        technical_config_sha256
    ):
        raise Experiment64SourceError("Arthur outcome technical binding differs")
    target_rows: list[dict[str, Any]] = []
    for context in pending_outcomes:
        volume = str(context["volume"])
        eye_id = int(context["eye_id"])
        lens_index = int(context["lens_index"])
        sealed_relpath = (
            f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        )
        sealed = load_sealed_core(
            staging / sealed_relpath,
            expected_volume=volume,
            expected_eye_id=eye_id,
            expected_lens_index=lens_index,
            expected_sealed_config=sealed_config,
        )
        geometry = context["geometry"]
        target = (
            source63.fit_source_target(
                np.asarray(sealed["points_xyz_um"], dtype=np.float64),
                np.asarray(context["proximal_points_xyz_um"], dtype=np.float64),
                geometry,
                threshold_config,
            )
            if geometry is not None
            else _empty_target()
        )
        coefficients = np.asarray(target["target_coefficients"], dtype=np.float64)
        target_relpath = (
            f"sealed_outcomes/lenses/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        )
        target_path = staging / target_relpath
        maike63._atomic_savez(
            target_path,
            schema_version=np.asarray(TARGET_SCHEMA),
            volume=np.asarray(volume),
            eye_id=np.asarray(eye_id, dtype=np.int64),
            lens_index=np.asarray(lens_index, dtype=np.int64),
            canonical_grid_xy=np.asarray(
                target["canonical_grid_xy"], dtype=np.float64
            ),
            target_smoothed_thickness_um=np.asarray(
                target["target_smoothed_thickness_um"], dtype=np.float64
            ),
            raw_target_xy_normalized=np.asarray(
                target["raw_target_xy_normalized"], dtype=np.float64
            ),
            raw_target_thickness_um=np.asarray(
                target["raw_target_thickness_um"], dtype=np.float64
            ),
            target_coefficients_c0_c5=coefficients,
            sealed_distal_relpath=np.asarray(sealed_relpath),
            sealed_distal_sha256=np.asarray(sealed["artifact_sha256"]),
            technical_config_sha256=np.asarray(technical_config_sha256),
            outcome_config_json=np.asarray(outcome_config_json),
            outcome_config_sha256=np.asarray(outcome_config_sha256),
        )
        _require_npz_members(target_path, TARGET_KEYS)
        target_row: dict[str, Any] = {
            "volume": volume,
            "eye_id": eye_id,
            "lens_index": lens_index,
            "target_resolvable": bool(target["target_resolvable"]),
            "target_resolvability_reasons": "|".join(
                target["target_resolvability_reasons"]
            ),
            "target_qc": bool(target["target_qc"]),
            "target_qc_reasons": "|".join(target["target_qc_reasons"]),
            "target_depth_um": target["target_depth_um"],
            "target_q05_raw_thickness_um": target[
                "target_q05_raw_thickness_um"
            ],
            "target_support": int(target["target_support"]),
            "target_rmse_um": target["target_rmse_um"],
            "lens_relpath": target_relpath,
        }
        for coefficient_index in range(6):
            target_row[f"target_c{coefficient_index}"] = float(
                coefficients[coefficient_index]
            )
        target_rows.append(target_row)
    return target_rows


def _technical_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    robust_success = sum(row["robust_core_status"] == "pass" for row in rows)
    distal_success = sum(bool(row["distal_qc"]) for row in rows)
    return {
        "rows": len(rows),
        "oracle_stage1": len(rows),
        "robust_core": robust_success,
        "robust_core_input_failures": len(rows) - robust_success,
        "distal_qc": distal_success,
        "distal_qc_failures_after_core": robust_success - distal_success,
    }


def _target_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "target_resolvable": sum(bool(row["target_resolvable"]) for row in rows),
        "target_qc": sum(bool(row["target_qc"]) for row in rows),
    }


def build_source_bundle(
    *,
    manifest_path: Path,
    eyemap_root: Path,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    allow_dirty_diagnostic: bool = False,
) -> Path:
    """Build one atomic source bundle while keeping target artifacts sealed."""

    repository_root = repository_root.resolve()
    eyemap_root = eyemap_root.resolve()
    git = source63.git_identity(repository_root)
    if git["dirty"] and not allow_dirty_diagnostic:
        raise Experiment64SourceError(
            "repository is dirty; commit the Experiment 64 producer before building"
        )
    eyemap_git = source63.git_identity(eyemap_root)
    if eyemap_git != {"commit": source63.EYEMAP_COMMIT, "dirty": False}:
        raise Experiment64SourceError(
            f"eyemap_T4 must be clean at {source63.EYEMAP_COMMIT}"
        )
    manifest_binding = {"name": manifest_path.name, **_binding(manifest_path)}
    manifest = source63.load_manifest(manifest_path)
    if {"name": manifest_path.name, **_binding(manifest_path)} != manifest_binding:
        raise Experiment64SourceError("Arthur input manifest changed while being parsed")
    if output_directory.exists():
        raise Experiment64SourceError(f"refusing to overwrite {output_directory}")

    robust_config = normalise_robust_core_config()
    threshold_config = dict(distal_geometry.DEFAULT_CONFIG)
    sealed_config = _sealed_config(threshold_config, robust_config)
    technical_config_json = _canonical_json(sealed_config)
    technical_config_sha256 = _sha256_bytes(
        technical_config_json.encode("utf-8")
    )
    outcome_config = _outcome_config(technical_config_sha256)
    outcome_config_json = _canonical_json(outcome_config)
    outcome_config_sha256 = _sha256_bytes(outcome_config_json.encode("utf-8"))
    code_hashes = {
        relpath: _binding(repository_root / relpath) for relpath in CODE_PATHS
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging.", dir=output_directory.parent
        )
    )
    try:
        technical_rows: list[dict[str, Any]] = []
        pending_outcomes: list[dict[str, Any]] = []
        technical_artifacts: dict[str, Any] = {}
        fixed_points: dict[str, Any] = {}
        frame_audits: dict[str, Any] = {}
        stage1_diagnostics: dict[str, Any] = {}
        input_files: dict[str, Any] = {}
        source_paths: dict[str, dict[str, Path]] = {}

        for manifest_row in manifest:
            volume = str(manifest_row["volume"])
            lens_mesh = Path(manifest_row["lens_mesh"])
            tip_mesh = Path(manifest_row["tip_mesh"])
            rdata = Path(manifest_row["rdata"])
            source_paths[volume] = {
                "lens_mesh": lens_mesh,
                "tip_mesh": tip_mesh,
                "rdata": rdata,
            }
            bound_sources = {
                role: {"name": path.name, **_binding(path)}
                for role, path in source_paths[volume].items()
            }
            if bound_sources != source63.EXPECTED_INPUT_IDENTITIES[volume]:
                raise Experiment64SourceError(
                    f"{volume} source identities changed after manifest validation"
                )
            _, lens_surfaces = source63.parse_wrl_surfaces(lens_mesh)
            _, tip_surfaces = source63.parse_wrl_surfaces(tip_mesh)
            if len(tip_surfaces) != 2:
                raise Experiment64SourceError(f"{volume} tip mesh does not contain two eyes")
            lens_positions, tip_positions = source63.raw_positions_from_rdata(rdata)
            stage1, diagnostics = source63.prepare_oracle_split_records(
                lens_surfaces, lens_positions, tip_positions
            )
            if {
                role: {"name": path.name, **_binding(path)}
                for role, path in source_paths[volume].items()
            } != bound_sources:
                raise Experiment64SourceError(
                    f"{volume} source input changed while being parsed"
                )
            if diagnostics["n_landmarks"] != source63.EXPECTED_LANDMARK_COUNTS[volume]:
                raise Experiment64SourceError(f"{volume} landmark count changed")
            if len(stage1) != source63.EXPECTED_STAGE1_COUNTS[volume]:
                raise Experiment64SourceError(f"{volume} oracle Stage-1 count changed")
            stage1_diagnostics[volume] = diagnostics
            input_files[volume] = bound_sources
            for eye_id in (0, 1):
                eye_records = [row for row in stage1 if int(row["eye"]) == eye_id]
                technical, pending, fixed, audit_relpath, artifacts = _process_eye(
                    volume=volume,
                    eye_id=eye_id,
                    stage1_records=eye_records,
                    staging=staging,
                    threshold_config=threshold_config,
                    robust_config=robust_config,
                    technical_config_sha256=technical_config_sha256,
                )
                unit = f"{volume}:eye_{eye_id}"
                technical_rows.extend(technical)
                pending_outcomes.extend(pending)
                fixed_points[unit] = fixed
                frame_audits[unit] = {
                    "relative_path": audit_relpath,
                    **_binding(staging / audit_relpath),
                }
                technical_artifacts.update(artifacts)

        order_key = lambda row: (
            source63.VOLUMES.index(str(row["volume"])),
            int(row["eye_id"]),
            int(row["lens_index"]),
        )
        technical_rows.sort(key=order_key)
        if len(technical_rows) != source63.EXPECTED_SOURCE_COUNTS["stage1"]:
            raise Experiment64SourceError("complete Arthur Stage-1 row count changed")
        if len(pending_outcomes) != len(technical_rows):
            raise Experiment64SourceError("technical/pending-outcome counts differ")

        technical_table_relpath = "arthur_technical_table.csv"
        technical_table_path = staging / technical_table_relpath
        _atomic_csv(technical_table_path, technical_rows, TECHNICAL_FIELDS)
        technical_artifacts[technical_table_relpath] = _binding(technical_table_path)

        # No target fit or outcome artifact is created until all six Arthur
        # eyes have completed robust-core Stage 2, frame audit, and technical
        # table publication in the staging tree.
        target_rows = _write_pending_targets(
            pending_outcomes=pending_outcomes,
            staging=staging,
            threshold_config=threshold_config,
            robust_config=robust_config,
            technical_config_sha256=technical_config_sha256,
            outcome_config_json=outcome_config_json,
            outcome_config_sha256=outcome_config_sha256,
        )
        target_rows.sort(key=order_key)
        if len(target_rows) != len(technical_rows):
            raise Experiment64SourceError("technical/target row counts differ")

        target_table_relpath = "sealed_outcomes/target_table.csv"
        target_table_path = staging / target_table_relpath
        _atomic_csv(target_table_path, target_rows, TARGET_FIELDS)
        outcome_artifacts = {target_table_relpath: _binding(target_table_path)}
        for row in target_rows:
            relpath = str(row["lens_relpath"])
            outcome_artifacts[relpath] = _binding(staging / relpath)

        outcome_manifest = {
            "schema_version": OUTCOME_MANIFEST_SCHEMA,
            "status": "diagnostic" if git["dirty"] else "complete",
            "experiment": 64,
            "analysis_scope": ANALYSIS_SCOPE,
            "technical_table_binding": {
                "relative_path": technical_table_relpath,
                **_binding(technical_table_path),
            },
            "technical_config_sha256": technical_config_sha256,
            "threshold_config_sha256": _sha256_bytes(
                distal_geometry.canonical_config_json(threshold_config).encode("utf-8")
            ),
            "robust_core_config_sha256": robust_core_config_sha256(robust_config),
            "target_config": outcome_config,
            "target_config_sha256": outcome_config_sha256,
            "counts": _target_counts(target_rows),
            "git": git,
            "producer_code_sha256": code_hashes,
            "output_manifest": outcome_artifacts,
        }
        outcome_manifest_relpath = "sealed_outcomes/manifest.json"
        outcome_manifest_path = staging / outcome_manifest_relpath
        maike63._atomic_json(outcome_manifest_path, outcome_manifest)

        technical_counts = _technical_counts(technical_rows)
        common = {
            "schema_version": TECHNICAL_BUNDLE_SCHEMA,
            "status": "diagnostic" if git["dirty"] else "complete",
            "experiment": 64,
            "analysis_scope": ANALYSIS_SCOPE,
            "isolation_basis": ISOLATION_BASIS,
            "oracle_stage1_scope": ORACLE_STAGE1_SCOPE,
            "threshold_config": threshold_config,
            "threshold_config_sha256": _sha256_bytes(
                distal_geometry.canonical_config_json(threshold_config).encode("utf-8")
            ),
            "predictor_pipeline_config": sealed_config["predictor_pipeline_config"],
            "predictor_pipeline_config_sha256": _sha256_bytes(
                _canonical_json(sealed_config["predictor_pipeline_config"]).encode("utf-8")
            ),
            "robust_core_config": robust_config,
            "robust_core_config_sha256": robust_core_config_sha256(robust_config),
            "technical_coherence_config": dict(TECHNICAL_COHERENCE_CONFIG),
            "technical_coherence_config_sha256": technical_coherence_config_sha256(),
            "modality_specific_qc": sealed_config["modality_specific_qc"],
            "technical_config_sha256": technical_config_sha256,
            "sealed_config_sha256": _sha256_bytes(
                _canonical_json(sealed_config).encode("utf-8")
            ),
            "n_rows": len(technical_rows),
            "technical_counts": technical_counts,
            "technical_counts_by_volume": {
                volume: _technical_counts(
                    [row for row in technical_rows if row["volume"] == volume]
                )
                for volume in source63.VOLUMES
            },
            "fixed_points": fixed_points,
            "frame_audits": frame_audits,
            "input_files": input_files,
            "input_manifest": manifest_binding,
            "git": git,
            "producer_code_sha256": code_hashes,
            "eyemap": eyemap_git,
            "biological_independence": {
                "independent_unit": "whole_head_scan_animal",
                "n_independent_animals": 3,
                "eyes_per_animal": 2,
                "nesting": "bilateral_eyes_nested_within_animal",
                "animals": [f"Arthur_{volume}" for volume in source63.VOLUMES],
            },
            "technical_output_manifest": technical_artifacts,
            "sealed_outcome_manifest_binding": {
                "relative_path": outcome_manifest_relpath,
                **_binding(outcome_manifest_path),
            },
        }
        completion = {**common, "document_role": "technical_completion"}
        provenance = {
            **common,
            "document_role": "technical_provenance",
            "source_stage1_diagnostics": stage1_diagnostics,
        }
        maike63._atomic_json(staging / "technical_completion.json", completion)
        maike63._atomic_json(staging / "technical_provenance.json", provenance)

        hashes_after = {
            relpath: _binding(repository_root / relpath) for relpath in CODE_PATHS
        }
        if hashes_after != code_hashes:
            raise Experiment64SourceError("producer implementation changed during build")
        if {"name": manifest_path.name, **_binding(manifest_path)} != manifest_binding:
            raise Experiment64SourceError("Arthur input manifest changed during build")
        input_files_after = {
            volume: {
                role: {"name": path.name, **_binding(path)}
                for role, path in paths.items()
            }
            for volume, paths in source_paths.items()
        }
        if input_files_after != input_files:
            raise Experiment64SourceError("an Arthur source input changed during build")
        if source63.git_identity(repository_root) != git:
            raise Experiment64SourceError("repository Git identity changed during build")
        if source63.git_identity(eyemap_root) != eyemap_git:
            raise Experiment64SourceError("eyemap_T4 Git identity changed during build")
        actual_files = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        }
        expected_files = {
            *technical_artifacts,
            *outcome_artifacts,
            outcome_manifest_relpath,
            "technical_completion.json",
            "technical_provenance.json",
        }
        if actual_files != expected_files:
            raise Experiment64SourceError(
                "Arthur bundle inventory differs from contract; "
                f"missing={sorted(expected_files - actual_files)[:5]}, "
                f"extra={sorted(actual_files - expected_files)[:5]}"
            )
        for label, manifest_entries in (
            ("technical", technical_artifacts),
            ("outcome", outcome_artifacts),
        ):
            for relpath, expected_binding in manifest_entries.items():
                if _binding(staging / relpath) != expected_binding:
                    raise Experiment64SourceError(
                        f"Arthur {label} artifact changed after hashing: {relpath}"
                    )
        if {
            "relative_path": outcome_manifest_relpath,
            **_binding(outcome_manifest_path),
        } != common["sealed_outcome_manifest_binding"]:
            raise Experiment64SourceError(
                "Arthur sealed outcome manifest changed after technical binding"
            )
        os.replace(staging, output_directory)
        return output_directory
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eyemap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--allow-dirty-diagnostic",
        action="store_true",
        help="write a backend-rejected diagnostic bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        output = build_source_bundle(
            manifest_path=arguments.manifest,
            eyemap_root=arguments.eyemap_root,
            output_directory=arguments.output,
            repository_root=arguments.repository_root,
            allow_dirty_diagnostic=arguments.allow_dirty_diagnostic,
        )
    except (Experiment64SourceError, source63.SourcePreparationError) as exc:
        print(f"Experiment 64 Arthur source preparation failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

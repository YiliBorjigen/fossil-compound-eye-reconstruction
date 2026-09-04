#!/usr/bin/env python3
"""Sealed-first, two-pass primary backend for Experiment 64.

Pass 1 routes all artifact access through a guarded reader that rejects
outcome paths before opening them.  It checks the clean repository identity,
the exact shared robust-core configuration, the sealed-outcome-free technical
inventories from the Arthur and Maike bundles, and every disjoint visual-QC
sample/review/attestation. Target-bearing
manifests, tables and NPZ files are represented only by deferred hash bindings
during this pass.  This is an in-process access contract, not an operating-
system sandbox or a claim that complete component geometry is proximal-blind.

Pass 2 accepts only an in-process clearance object produced by a successful
Pass 1.  It then opens the deferred outcomes, joins them to the already
validated predictors, and delegates the unchanged ridge fitting and primary
81-point scoring calculations to the frozen Experiment 63 implementation.
Experiment 63's orchestration and files are not modified.

Experiment 64 is a post-QC, model/error-sequestered evaluation on animals
whose input morphology informed preprocessing development.  Its fixed
10-of-12 rule is descriptive; this module never labels the result pristine
external confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import experiment_63_primary_backend as experiment63
import experiment_64_prepare_arthur_source_table as arthur64
import experiment_64_robust_distal_core as robust64
import experiment_64_technical_metrics as technical64


PASS1_SCHEMA = "experiment64.technical-pass1-clearance.v1"
RUN_SCHEMA = "experiment64.primary-run.v1"
OUTCOME_ATTEMPT_SCHEMA = "experiment64.outcome-attempt.v1"
ANALYSIS_LABEL = "post_qc_model_error_sequestered_evaluation"
RUN_STATUS = "sealed_first_experiment64_model_outcome_run"
INFERENCE_LABEL = "post_qc_evaluation_not_pristine_external_confirmation"
OUTCOME_MANIFEST_RELPATH = "sealed_outcomes/manifest.json"
REPRESENTATION_CAVEAT = (
    "Arthur source observations are irregular mesh vertices whereas Maike "
    "observations are voxel-boundary samples; using the same physical-coordinate "
    "operator does not remove this acquisition and representation shift."
)

EXPECTED_EYES = dict(experiment63.EXPECTED_EYES)
EXPECTED_TOTAL = int(experiment63.EXPECTED_TOTAL)
EXPECTED_ARTHUR_ROWS = int(sum(experiment63.ARTHUR_STAGE1_COUNTS.values()))
ROBUST_CORE_CONFIG_SHA256 = robust64.robust_core_config_sha256()

MAIKE_TECHNICAL_SCHEMA = "experiment64.maike-technical-bundle.v1"
MAIKE_INSTANCE_SCHEMA = "experiment64.maike-instance.v1"
MAIKE_CORE_SCHEMA = "experiment64.maike-sealed-distal-core.v1"
MAIKE_OUTCOME_SCHEMA = "experiment64.maike-outcomes.v1"
MAIKE_TARGET_SCHEMA = "experiment64.maike-target.v1"
FRAME_AUDIT_SCHEMA = "experiment64.distal-frame-audit.v1"

# The producers export the authoritative tuples.  The Maike adapter is loaded
# lazily because it is an independent executable and can be absent while this
# backend module is imported for leakage-barrier tests.
ARTHUR_TECHNICAL_FIELDS = tuple(arthur64.TECHNICAL_FIELDS)
ARTHUR_TARGET_FIELDS = tuple(arthur64.TARGET_FIELDS)

MAIKE_SAMPLING_FIELDS = (
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
MAIKE_TARGET_FIELDS = (
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

TECHNICAL_NUMERIC_FIELDS = (
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
    "distal_fit_p99_residual_over_scale",
    "distal_fit_26_component_count",
    "distal_fit_26_largest_component_support",
    "distal_fit_26_largest_component_fraction",
    "coherence_support_margin",
    "coherence_rmse_margin",
    "coherence_lcc_margin",
    "coherence_p99_over_scale_margin",
    "coherence_margin",
)

PASS1_FORBIDDEN_SEGMENTS = frozenset({"sealed_outcomes"})
PASS1_FORBIDDEN_FILENAME_TOKENS = (
    "target_table",
    "prediction",
    "model_output",
    "per_lens_metrics",
    "lens_summary",
)


class ContractError(RuntimeError):
    """Raised when an Experiment 64 frozen contract is violated."""


def _fail(message: str) -> None:
    raise ContractError(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Value is not canonical-JSON encodable: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    _fail(f"{label} is not a strict boolean: {value!r}")


def _strict_int(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        _fail(f"{label} is a boolean, not an integer")
    try:
        numeric = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{label} is not an integer: {value!r}") from exc
    if not math.isfinite(numeric) or numeric != integer:
        _fail(f"{label} is not an exact integer: {value!r}")
    if minimum is not None and integer < minimum:
        _fail(f"{label} must be >= {minimum}; got {integer}")
    return integer


def _strict_float(value: Any, label: str, *, finite: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc
    if finite and not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    expected_set = set(expected)
    if set(value) != expected_set:
        _fail(
            f"{label} keys differ from contract; expected {sorted(expected_set)}, "
            f"got {sorted(value)}"
        )
    return value


def _require_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    missing = sorted(set(expected) - set(value))
    if missing:
        _fail(f"{label} is missing required keys: {missing}")
    return value


def _safe_pure_relpath(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        _fail(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"Unsafe relative path in {label}: {value!r}")
    return path


def _pass1_path_allowed(relpath: PurePosixPath, purpose: str) -> None:
    lowered_parts = tuple(part.lower() for part in relpath.parts)
    if PASS1_FORBIDDEN_SEGMENTS.intersection(lowered_parts):
        _fail(f"Pass 1 refused outcome path {relpath} for {purpose}")
    name = lowered_parts[-1]
    if any(token in name for token in PASS1_FORBIDDEN_FILENAME_TOKENS):
        _fail(f"Pass 1 refused outcome-like path {relpath} for {purpose}")


@dataclass(frozen=True)
class ReadEvent:
    phase: str
    purpose: str
    path: str


@dataclass
class GuardedArtifactReader:
    """One auditable read boundary for both execution phases.

    Tests can inject ``byte_opener`` to record every physical read.  All
    backend JSON, CSV, render and NPZ access is routed through this class.
    """

    byte_opener: Callable[[Path], bytes] = lambda path: path.read_bytes()
    events: list[ReadEvent] = field(default_factory=list)

    def resolve(self, root: Path, relpath: str, *, phase: str, purpose: str) -> Path:
        pure = _safe_pure_relpath(relpath, f"{purpose}.relative_path")
        if phase == "pass1":
            _pass1_path_allowed(pure, purpose)
        if root.is_symlink():
            _fail(f"Symlinked artifact root is forbidden: {root}")
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"Missing artifact root {root}: {exc}") from exc
        cursor = resolved_root
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                _fail(f"Symlinked artifact path is forbidden: {cursor}")
        try:
            resolved = cursor.resolve(strict=True)
        except OSError as exc:
            raise ContractError(f"Missing {purpose}: {cursor}") from exc
        if resolved_root != resolved and resolved_root not in resolved.parents:
            _fail(f"{purpose} escapes its declared root: {relpath!r}")
        if not resolved.is_file():
            _fail(f"{purpose} is not a regular file: {resolved}")
        return resolved

    def read(self, root: Path, relpath: str, *, phase: str, purpose: str) -> bytes:
        path = self.resolve(root, relpath, phase=phase, purpose=purpose)
        self.events.append(ReadEvent(phase=phase, purpose=purpose, path=str(path)))
        try:
            return self.byte_opener(path)
        except OSError as exc:
            raise ContractError(f"Cannot read {purpose} {path}: {exc}") from exc

    def read_bound(
        self,
        root: Path,
        relpath: str,
        binding: Any,
        *,
        phase: str,
        purpose: str,
    ) -> bytes:
        binding = _exact_keys(binding, {"sha256", "size_bytes"}, f"{purpose}.binding")
        expected_hash = _strict_sha256(binding["sha256"], f"{purpose}.sha256")
        expected_size = _strict_int(binding["size_bytes"], f"{purpose}.size_bytes", 0)
        payload = self.read(root, relpath, phase=phase, purpose=purpose)
        if len(payload) != expected_size:
            _fail(f"Size mismatch for {purpose}: {relpath}")
        if _sha256_bytes(payload) != expected_hash:
            _fail(f"SHA-256 mismatch for {purpose}: {relpath}")
        return payload


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value


def _csv_table(payload: bytes, label: str) -> pd.DataFrame:
    try:
        # Both producers serialize Python float64 values with a shortest
        # round-trippable decimal representation.  Pandas' default fast parser
        # can shift such values by one ULP, which would make the exact CSV/NPZ
        # cross-binding below reject an untampered producer artifact.
        return pd.read_csv(
            io.BytesIO(payload),
            keep_default_na=True,
            float_precision="round_trip",
        )
    except Exception as exc:
        raise ContractError(f"{label} is not a readable CSV: {exc}") from exc


def _npz_arrays(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ContractError(f"{label} is not a safe NPZ archive: {exc}") from exc


def _scalar_text(array: np.ndarray, label: str) -> str:
    if array.shape != () or array.dtype.kind != "U":
        _fail(f"{label} must be a Unicode scalar")
    return str(array.item())


def _scalar_int64(array: np.ndarray, label: str) -> int:
    if array.shape != () or array.dtype != np.dtype("int64"):
        _fail(f"{label} must be an int64 scalar")
    return int(array.item())


@dataclass(frozen=True)
class DeferredArtifact:
    root: Path
    relative_path: str
    sha256: str
    size_bytes: int
    owner: str

    def binding(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


def _deferred_binding(root: Path, value: Any, label: str) -> DeferredArtifact:
    value = _exact_keys(value, {"relative_path", "sha256", "size_bytes"}, label)
    pure = _safe_pure_relpath(value["relative_path"], f"{label}.relative_path")
    if pure.as_posix() != OUTCOME_MANIFEST_RELPATH:
        _fail(f"{label} must bind exactly {OUTCOME_MANIFEST_RELPATH}")
    return DeferredArtifact(
        root=root.resolve(),
        relative_path=pure.as_posix(),
        sha256=_strict_sha256(value["sha256"], f"{label}.sha256"),
        size_bytes=_strict_int(value["size_bytes"], f"{label}.size_bytes", 0),
        owner=label,
    )


def _validate_git_record(record: Any, expected_commit: str, label: str) -> None:
    record = _require_keys(record, {"commit", "dirty"}, label)
    if record["commit"] != expected_commit:
        _fail(f"{label} was produced at a different commit")
    if _strict_bool(record["dirty"], f"{label}.dirty"):
        _fail(f"{label} was produced from a dirty worktree")


def _validate_robust_core_binding(document: Mapping[str, Any], label: str) -> None:
    required = _require_keys(
        document,
        {"robust_core_config", "robust_core_config_sha256"},
        label,
    )
    config = required["robust_core_config"]
    if not isinstance(config, Mapping):
        _fail(f"{label}.robust_core_config must be an object")
    try:
        normalized = robust64.normalise_robust_core_config(config)
    except robust64.RobustDistalCoreError as exc:
        raise ContractError(f"{label} has invalid robust-core configuration: {exc}") from exc
    if normalized != robust64.ROBUST_CORE_CONFIG:
        _fail(f"{label} robust-core configuration differs from the frozen operator")
    observed_hash = _strict_sha256(
        required["robust_core_config_sha256"],
        f"{label}.robust_core_config_sha256",
    )
    if observed_hash != ROBUST_CORE_CONFIG_SHA256:
        _fail(f"{label} robust-core hash differs from the frozen operator")
    if robust64.robust_core_config_sha256(config) != observed_hash:
        _fail(f"{label} robust-core configuration/hash binding is invalid")
    if int(normalized["minimum_input_points"]) < 33:
        _fail(f"{label} robust core permits fewer than 33 input points")
    if int(normalized["minimum_retained_points"]) < 27:
        _fail(f"{label} robust core permits fewer than 27 retained points")
    if int(normalized["minimum_downstream_fit_points"]) < 25:
        _fail(f"{label} robust core permits fewer than 25 Stage-2 fit points")


def _validate_technical_coherence_binding(document: Mapping[str, Any], label: str) -> None:
    document = _require_keys(
        document,
        {"technical_coherence_config", "technical_coherence_config_sha256"},
        label,
    )
    if document["technical_coherence_config"] != technical64.TECHNICAL_COHERENCE_CONFIG:
        _fail(f"{label} technical-coherence configuration differs from the frozen contract")
    expected = technical64.technical_coherence_config_sha256()
    if _strict_sha256(
        document["technical_coherence_config_sha256"],
        f"{label}.technical_coherence_config_sha256",
    ) != expected:
        _fail(f"{label} technical-coherence configuration hash mismatch")


def _validate_common_technical_config_hashes(
    document: Mapping[str, Any], label: str
) -> None:
    document = _require_keys(
        document,
        {
            "threshold_config",
            "threshold_config_sha256",
            "predictor_pipeline_config",
            "predictor_pipeline_config_sha256",
            "technical_config_sha256",
            "sealed_config_sha256",
        },
        label,
    )
    if document["threshold_config"] != experiment63.EXPECTED_THRESHOLD_CONFIG:
        _fail(f"{label} changed the frozen distal-geometry thresholds")
    threshold_json = experiment63.canonical_json(document["threshold_config"])
    if _strict_sha256(
        document["threshold_config_sha256"], f"{label}.threshold_config_sha256"
    ) != _sha256_bytes(threshold_json.encode("utf-8")):
        _fail(f"{label} threshold configuration hash is invalid")
    predictor = document["predictor_pipeline_config"]
    if not isinstance(predictor, Mapping):
        _fail(f"{label}.predictor_pipeline_config must be an object")
    if _strict_sha256(
        document["predictor_pipeline_config_sha256"],
        f"{label}.predictor_pipeline_config_sha256",
    ) != _sha256_bytes(_canonical_json(predictor).encode("utf-8")):
        _fail(f"{label} predictor-pipeline configuration hash is invalid")
    technical_hash = _strict_sha256(
        document["technical_config_sha256"], f"{label}.technical_config_sha256"
    )
    if _strict_sha256(
        document["sealed_config_sha256"], f"{label}.sealed_config_sha256"
    ) != technical_hash:
        _fail(f"{label} technical/sealed configuration hashes differ")


def _validated_manifest_bindings(value: Any, label: str, *, phase: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{label} must be a nonempty object")
    result: dict[str, dict[str, Any]] = {}
    for relpath, binding in value.items():
        pure = _safe_pure_relpath(relpath, f"{label}.path")
        if phase == "pass1":
            _pass1_path_allowed(pure, label)
        binding = _exact_keys(binding, {"sha256", "size_bytes"}, f"{label}[{relpath!r}]")
        result[pure.as_posix()] = {
            "sha256": _strict_sha256(binding["sha256"], f"{label}[{relpath!r}].sha256"),
            "size_bytes": _strict_int(binding["size_bytes"], f"{label}[{relpath!r}].size_bytes", 0),
        }
    return result


def _documents_agree(completion: Mapping[str, Any], provenance: Mapping[str, Any], label: str) -> None:
    for key in completion:
        if key == "document_role":
            continue
        if key not in provenance or provenance[key] != completion[key]:
            _fail(f"{label} completion/provenance disagree at {key!r}")


def _read_bound_json(
    reader: GuardedArtifactReader,
    root: Path,
    relpath: str,
    binding: Mapping[str, Any],
    *,
    phase: str,
    purpose: str,
) -> dict[str, Any]:
    return _json_object(
        reader.read_bound(root, relpath, binding, phase=phase, purpose=purpose),
        purpose,
    )


def _read_document(reader: GuardedArtifactReader, root: Path, name: str, purpose: str) -> tuple[dict[str, Any], bytes]:
    payload = reader.read(root, name, phase="pass1", purpose=purpose)
    return _json_object(payload, purpose), payload


def _validate_repository_code_bindings(
    reader: GuardedArtifactReader,
    repository_root: Path,
    value: Any,
    *,
    label: str,
    maike_basenames: bool,
) -> None:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{label} must be a nonempty implementation-binding object")
    seen_robust_core = False
    for raw_name, binding in sorted(value.items()):
        if not isinstance(raw_name, str):
            _fail(f"{label} implementation names must be strings")
        relpath = (
            f"experiments/maike-modern-ground-truth/{raw_name}"
            if maike_basenames
            else raw_name
        )
        if PurePosixPath(relpath).name == "experiment_64_robust_distal_core.py":
            seen_robust_core = True
        reader.read_bound(
            repository_root,
            relpath,
            binding,
            phase="pass1",
            purpose=f"{label} implementation {raw_name}",
        )
    if not seen_robust_core:
        _fail(f"{label} does not bind the shared robust-core implementation")


def _validate_table_header(table: pd.DataFrame, expected: Sequence[str], label: str) -> None:
    if tuple(table.columns) != tuple(expected):
        _fail(
            f"{label} header differs from producer contract; expected {list(expected)}, "
            f"got {list(table.columns)}"
        )


def _validate_technical_rows(
    table: pd.DataFrame,
    *,
    label: str,
    eligible_column: str,
    expected_rows: int,
) -> pd.DataFrame:
    if len(table) != expected_rows:
        _fail(f"{label} has {len(table)} rows; expected {expected_rows}")
    if eligible_column not in table:
        _fail(f"{label} lacks {eligible_column}")
    result = table.copy()
    result[eligible_column] = [
        _strict_bool(value, f"{label}.{eligible_column}") for value in result[eligible_column]
    ]
    if "central" in result:
        result["central"] = [_strict_bool(value, f"{label}.central") for value in result["central"]]
    if "lens_index" not in result:
        _fail(f"{label} lacks lens_index")
    result["lens_index"] = [
        _strict_int(value, f"{label}.lens_index", 0) for value in result["lens_index"]
    ]
    if result["lens_index"].duplicated().any() and "volume" not in result:
        _fail(f"{label} contains duplicate lens indices")
    for row_number, row in result.iterrows():
        raw = _strict_int(row["raw_distal_support"], f"{label}[{row_number}].raw_distal_support", 0)
        core = _strict_int(row["robust_core_support"], f"{label}[{row_number}].robust_core_support", 0)
        status = str(row["robust_core_status"])
        fraction = _strict_float(
            row["robust_core_retained_fraction"],
            f"{label}[{row_number}].robust_core_retained_fraction",
        )
        expected_fraction = core / raw if raw else 0.0
        if not math.isclose(fraction, expected_fraction, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail(f"{label}[{row_number}] robust-core retained fraction is inconsistent")
        if status == "pass":
            if raw < 33 or core < 27 or core > raw:
                _fail(f"{label}[{row_number}] passing robust core violates support floors")
        elif status == "ineligible":
            if core != 0:
                _fail(f"{label}[{row_number}] ineligible robust core must be empty")
        else:
            _fail(f"{label}[{row_number}] has invalid robust_core_status {status!r}")
        if bool(result.at[row_number, eligible_column]) and status != "pass":
            _fail(f"{label}[{row_number}] is distal eligible without a passing robust core")

    eligible = result[eligible_column].to_numpy(bool)
    common_numeric_fields = (
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
        "coherence_margin",
    )
    for column in common_numeric_fields:
        if column not in result:
            _fail(f"{label} lacks technical field {column}")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    # Arthur records three additional modality-appropriate diagnostics.  The
    # Maike producer instead records exact 26-neighbour q90-fit connectivity
    # and a residual-p99/scale gate; its voxel connectivity is intentionally
    # not imposed on irregular Arthur mesh samples.
    optional_numeric = [
        column for column in TECHNICAL_NUMERIC_FIELDS if column in result
    ]
    values = result.loc[eligible, optional_numeric].to_numpy(float)
    if not np.all(np.isfinite(values)):
        _fail(f"{label} has nonfinite technical values in eligible rows")
    support = result.loc[eligible, "distal_fit_support"].to_numpy(float)
    if np.any(support < 25) or np.any(support != np.floor(support)):
        _fail(f"{label} has an eligible row with post-Stage-2 fit support below 25")
    core_support = pd.to_numeric(result.loc[eligible, "robust_core_support"], errors="coerce").to_numpy(float)
    raw_support = pd.to_numeric(result.loc[eligible, "raw_distal_support"], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(raw_support)) or np.any(raw_support < 33) or np.any(raw_support != np.floor(raw_support)):
        _fail(f"{label} has an eligible row with robust-core input support below 33")
    if np.any(~np.isfinite(core_support)) or np.any(core_support < 27) or np.any(core_support != np.floor(core_support)):
        _fail(f"{label} has an eligible row with robust-core retained support below 27")
    if np.any(support > core_support) or np.any(core_support > raw_support):
        _fail(f"{label} has impossible input/core/final-fit support ordering")
    scales = result.loc[eligible, "distal_scale_um"].to_numpy(float)
    if np.any((scales < 3.0) | (scales > 13.0)):
        _fail(f"{label} has eligible distal scales outside [3,13] um")
    rmse = result.loc[eligible, "distal_fit_rmse_um"].to_numpy(float)
    if np.any(rmse < 0.0) or np.any(rmse > 2.5):
        _fail(f"{label} has eligible final-fit RMSE outside [0,2.5] um")
    residuals = result.loc[eligible, "distal_normalized_fit_residual"].to_numpy(float)
    gradients = result.loc[eligible, "distal_gradient_magnitude"].to_numpy(float)
    if np.any(residuals < 0.0) or np.any(gradients < 0.0):
        _fail(f"{label} has negative shape magnitude/residual")
    eig1 = result.loc[eligible, "distal_curvature_eigenvalue_1"].to_numpy(float)
    eig2 = result.loc[eligible, "distal_curvature_eigenvalue_2"].to_numpy(float)
    if np.any(eig1 > eig2 + 1.0e-12):
        _fail(f"{label} curvature eigenvalues are not ordered")
    maike_gate_fields = {
        "distal_fit_p99_residual_over_scale",
        "distal_fit_26_component_count",
        "distal_fit_26_largest_component_support",
        "distal_fit_26_largest_component_fraction",
        "maike_final_fit_gate_pass",
        "maike_final_fit_gate_reasons",
    }
    if maike_gate_fields.intersection(result.columns):
        missing = sorted(maike_gate_fields - set(result.columns))
        if missing:
            _fail(f"{label} has an incomplete Maike final-fit gate: {missing}")
        result["maike_final_fit_gate_pass"] = [
            _strict_bool(value, f"{label}.maike_final_fit_gate_pass")
            for value in result["maike_final_fit_gate_pass"]
        ]
        for column in (
            "distal_fit_p99_residual_over_scale",
            "distal_fit_26_component_count",
            "distal_fit_26_largest_component_support",
            "distal_fit_26_largest_component_fraction",
        ):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if not result.loc[eligible, "maike_final_fit_gate_pass"].all():
            _fail(f"{label} retains a row that failed the Maike final-fit gate")
        lcc = result.loc[eligible, "distal_fit_26_largest_component_fraction"].to_numpy(float)
        p99_scaled = result.loc[eligible, "distal_fit_p99_residual_over_scale"].to_numpy(float)
        if np.any(~np.isfinite(lcc)) or np.any(lcc < 0.99):
            _fail(f"{label} retains a row below the frozen q90-fit 26-LCC fraction 0.99")
        if np.any(~np.isfinite(p99_scaled)) or np.any(p99_scaled > 0.75):
            _fail(f"{label} retains a row above the frozen q90-fit p99/scale limit 0.75")
        lcc_support = result.loc[eligible, "distal_fit_26_largest_component_support"].to_numpy(float)
        fit_support = result.loc[eligible, "distal_fit_support"].to_numpy(float)
        if np.any(lcc_support > fit_support) or np.any(lcc_support < 0):
            _fail(f"{label} has impossible q90-fit connected-component support")
        expected_gate = (
            (fit_support >= 25)
            & (result.loc[eligible, "distal_fit_rmse_um"].to_numpy(float) <= 2.5)
            & (lcc >= 0.99)
            & (p99_scaled <= 0.75)
        )
        if not np.all(expected_gate):
            _fail(f"{label} has an eligible row inconsistent with the exact Maike gate")
        expected_margins = {
            "coherence_support_margin": (fit_support - 25.0) / 25.0,
            "coherence_rmse_margin": (
                2.5 - result.loc[eligible, "distal_fit_rmse_um"].to_numpy(float)
            )
            / 2.5,
            "coherence_lcc_margin": (lcc - 0.99) / 0.01,
            "coherence_p99_over_scale_margin": (0.75 - p99_scaled) / 0.75,
        }
        for column, expected_values in expected_margins.items():
            actual_values = result.loc[eligible, column].to_numpy(float)
            if not np.allclose(actual_values, expected_values, rtol=1.0e-12, atol=1.0e-12):
                _fail(f"{label} {column} differs from its frozen definition")
        expected_coherence = np.min(np.vstack(list(expected_margins.values())), axis=0)
        if not np.allclose(
            result.loc[eligible, "coherence_margin"].to_numpy(float),
            expected_coherence,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            _fail(f"{label} coherence_margin differs from the four frozen Maike margins")
    elif {"coherence_support_margin", "coherence_rmse_margin"}.issubset(result.columns):
        support_margin = (support - 25.0) / 25.0
        rmse_margin = (2.5 - rmse) / 2.5
        if not np.allclose(
            result.loc[eligible, "coherence_support_margin"].to_numpy(float),
            support_margin,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            _fail(f"{label} Arthur support margin differs from its frozen definition")
        if not np.allclose(
            result.loc[eligible, "coherence_rmse_margin"].to_numpy(float),
            rmse_margin,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            _fail(f"{label} Arthur RMSE margin differs from its frozen definition")
        if not np.allclose(
            result.loc[eligible, "coherence_margin"].to_numpy(float),
            np.minimum(support_margin, rmse_margin),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            _fail(f"{label} Arthur coherence margin differs from its two frozen margins")
    return result


def _reason_tokens(value: Any, label: str) -> tuple[str, ...]:
    """Return one producer reason cell as a validated, duplicate-free tuple."""

    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value))):
        return ()
    if not isinstance(value, str):
        _fail(f"{label} must be an empty cell or a pipe-delimited string")
    if value == "":
        return ()
    tokens = tuple(value.split("|"))
    if any(not token or token.strip() != token for token in tokens):
        _fail(f"{label} contains an empty or whitespace-padded reason")
    if len(tokens) != len(set(tokens)):
        _fail(f"{label} contains duplicate reasons")
    return tokens


def _reason_counts(table: pd.DataFrame, column: str, label: str) -> dict[str, int]:
    if column not in table:
        _fail(f"{label} lacks reason field {column}")
    counts: dict[str, int] = {}
    for row_number, value in enumerate(table[column]):
        for reason in _reason_tokens(value, f"{label}[{row_number}].{column}"):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _status_counts(table: pd.DataFrame, column: str, label: str) -> dict[str, int]:
    if column not in table:
        _fail(f"{label} lacks status field {column}")
    counts: dict[str, int] = {}
    for row_number, value in enumerate(table[column]):
        if not isinstance(value, str) or not value or value.strip() != value:
            _fail(f"{label}[{row_number}].{column} is not a canonical status string")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _strict_count_mapping(value: Any, expected: Mapping[str, int], label: str) -> None:
    value = _exact_keys(value, expected, label)
    observed = {
        key: _strict_int(value[key], f"{label}.{key}", 0) for key in expected
    }
    if observed != dict(expected):
        _fail(f"{label} differs from counts recomputed from the technical table")


def _arthur_producer_counts(table: pd.DataFrame) -> dict[str, int]:
    for row_number, row in table.iterrows():
        if not _strict_bool(
            row["oracle_stage1_eligible"],
            f"Arthur technical table[{row_number}].oracle_stage1_eligible",
        ):
            _fail(f"Arthur technical table[{row_number}] changed the oracle Stage-1 cohort")
        robust_reasons = _reason_tokens(
            row["robust_core_reasons"],
            f"Arthur technical table[{row_number}].robust_core_reasons",
        )
        distal_reasons = _reason_tokens(
            row["distal_qc_reasons"],
            f"Arthur technical table[{row_number}].distal_qc_reasons",
        )
        robust_pass = str(row["robust_core_status"]) == "pass"
        distal_pass = _strict_bool(
            row["distal_qc"], f"Arthur technical table[{row_number}].distal_qc"
        )
        if robust_pass != (not robust_reasons):
            _fail(f"Arthur technical table[{row_number}] robust pass/reasons disagree")
        if distal_pass != (not distal_reasons):
            _fail(f"Arthur technical table[{row_number}] distal-QC pass/reasons disagree")
        if distal_pass and not robust_pass:
            _fail(f"Arthur technical table[{row_number}] passes distal QC without a robust core")
    robust_retained = int((table["robust_core_status"] == "pass").sum())
    distal_retained = sum(
        _strict_bool(value, "Arthur technical table.distal_qc")
        for value in table["distal_qc"]
    )
    return {
        "rows": len(table),
        "oracle_stage1": len(table),
        "robust_core": robust_retained,
        "robust_core_input_failures": len(table) - robust_retained,
        "distal_qc": distal_retained,
        "distal_qc_failures_after_core": robust_retained - distal_retained,
    }


def _arthur_technical_exclusion_report(table: pd.DataFrame) -> dict[str, Any]:
    counts = _arthur_producer_counts(table)
    return {
        "fixed_source_denominator": counts["rows"],
        "oracle_stage1_retained": counts["oracle_stage1"],
        "robust_core_retained": counts["robust_core"],
        "robust_core_excluded": counts["robust_core_input_failures"],
        "robust_core_exclusion_reason_counts": _reason_counts(
            table, "robust_core_reasons", "Arthur technical table"
        ),
        "distal_qc_retained": counts["distal_qc"],
        "distal_qc_excluded": counts["rows"] - counts["distal_qc"],
        "distal_qc_excluded_after_robust_core": counts[
            "distal_qc_failures_after_core"
        ],
        "distal_qc_exclusion_reason_counts": _reason_counts(
            table, "distal_qc_reasons", "Arthur technical table"
        ),
    }


def _maike_producer_counts(table: pd.DataFrame, label: str) -> dict[str, int]:
    final_reasons: list[tuple[str, ...]] = []
    robust_retained = 0
    distal_retained = 0
    allowed_evaluated_reasons = {
        "fit_points_26_lcc_fraction_below_minimum",
        "fit_abs_residual_p99_over_scale_above_maximum",
    }
    for row_number, row in table.iterrows():
        full_assigned = _strict_int(
            row["full_assigned_size"],
            f"{label}[{row_number}].full_assigned_size",
            0,
        )
        expected_assignment_status = "ok" if full_assigned > 0 else "empty_assignment"
        if row["assignment_status"] != expected_assignment_status:
            _fail(
                f"{label}[{row_number}] assignment_status is inconsistent with "
                "full_assigned_size"
            )
        robust_reasons = _reason_tokens(
            row["robust_core_reasons"], f"{label}[{row_number}].robust_core_reasons"
        )
        distal_reasons = _reason_tokens(
            row["distal_qc_reasons"], f"{label}[{row_number}].distal_qc_reasons"
        )
        gate_reasons = _reason_tokens(
            row["maike_final_fit_gate_reasons"],
            f"{label}[{row_number}].maike_final_fit_gate_reasons",
        )
        robust_pass = str(row["robust_core_status"]) == "pass"
        distal_pass = _strict_bool(row["distal_qc"], f"{label}[{row_number}].distal_qc")
        final_gate_pass = _strict_bool(
            row["maike_final_fit_gate_pass"],
            f"{label}[{row_number}].maike_final_fit_gate_pass",
        )
        if robust_pass != (not robust_reasons):
            _fail(f"{label}[{row_number}] robust pass/reasons disagree")
        if distal_pass != (not distal_reasons):
            _fail(f"{label}[{row_number}] distal-QC pass/reasons disagree")
        if final_gate_pass != (not gate_reasons):
            _fail(f"{label}[{row_number}] final-fit pass/reasons disagree")
        if distal_pass != final_gate_pass:
            _fail(f"{label}[{row_number}] distal-QC and final-fit pass flags disagree")
        if gate_reasons == ("base_distal_qc_failed",):
            if distal_pass:
                _fail(f"{label}[{row_number}] passes despite base distal-QC failure")
        elif not set(gate_reasons).issubset(allowed_evaluated_reasons):
            _fail(f"{label}[{row_number}] has an unrecognized final-fit exclusion reason")
        robust_retained += int(robust_pass)
        distal_retained += int(distal_pass)
        final_reasons.append(gate_reasons)
    base_retained = sum(reasons != ("base_distal_qc_failed",) for reasons in final_reasons)
    connectivity_reason = "fit_points_26_lcc_fraction_below_minimum"
    residual_reason = "fit_abs_residual_p99_over_scale_above_maximum"
    connectivity_excluded = sum(
        connectivity_reason in reasons for reasons in final_reasons
    )
    residual_excluded = sum(residual_reason in reasons for reasons in final_reasons)
    both_excluded = sum(
        {connectivity_reason, residual_reason}.issubset(reasons)
        for reasons in final_reasons
    )
    final_fit_excluded = connectivity_excluded + residual_excluded - both_excluded
    if base_retained != distal_retained + final_fit_excluded:
        _fail(f"{label} has inconsistent base/final distal-QC exclusion partitions")
    return {
        "instances": len(table),
        "robust_core": robust_retained,
        "base_distal_qc": base_retained,
        "distal_qc": distal_retained,
        "maike_final_fit_connectivity_excluded": connectivity_excluded,
        "maike_final_fit_residual_excluded": residual_excluded,
        "maike_final_fit_both_excluded": both_excluded,
    }


def _maike_technical_exclusion_report(
    table: pd.DataFrame, label: str
) -> dict[str, Any]:
    counts = _maike_producer_counts(table, label)
    component_retained = sum(
        _strict_bool(value, f"{label}.component_fraction_gate_pass")
        for value in table["component_fraction_gate_pass"]
    )
    return {
        "fixed_oda_denominator": counts["instances"],
        "assignment_status_counts": _status_counts(table, "assignment_status", label),
        "component_fraction_gate_retained": component_retained,
        "component_fraction_gate_failed": counts["instances"] - component_retained,
        "robust_core_retained": counts["robust_core"],
        "robust_core_excluded": counts["instances"] - counts["robust_core"],
        "robust_core_exclusion_reason_counts": _reason_counts(
            table, "robust_core_reasons", label
        ),
        "base_distal_qc_retained": counts["base_distal_qc"],
        "base_distal_qc_excluded": counts["instances"] - counts["base_distal_qc"],
        "maike_final_fit_evaluated": counts["base_distal_qc"],
        "maike_final_fit_retained": counts["distal_qc"],
        "maike_final_fit_connectivity_excluded": counts[
            "maike_final_fit_connectivity_excluded"
        ],
        "maike_final_fit_residual_excluded": counts[
            "maike_final_fit_residual_excluded"
        ],
        "maike_final_fit_both_excluded": counts["maike_final_fit_both_excluded"],
        "maike_final_fit_excluded": (
            counts["maike_final_fit_connectivity_excluded"]
            + counts["maike_final_fit_residual_excluded"]
            - counts["maike_final_fit_both_excluded"]
        ),
        "maike_final_fit_reason_counts": _reason_counts(
            table, "maike_final_fit_gate_reasons", label
        ),
        "distal_qc_retained": counts["distal_qc"],
        "distal_qc_excluded": counts["instances"] - counts["distal_qc"],
        "distal_qc_exclusion_reason_counts": _reason_counts(
            table, "distal_qc_reasons", label
        ),
    }


def _validate_core_npz(
    arrays: Mapping[str, np.ndarray],
    *,
    label: str,
    schema: str,
    lens_index: int,
    expected_core_support: int,
    expected_raw_support: int,
    arthur: bool,
    expected_technical_config_sha256: str,
    expected_volume: str | None = None,
    expected_eye_id: int | None = None,
    expected_robust_status: str | None = None,
    expected_diagnostics_json: str | None = None,
) -> None:
    identity_members = {"volume", "eye_id"} if arthur else set()
    point_member = "points_xyz_um" if arthur else "points_zyx"
    expected_members = {
        "schema_version",
        "lens_index",
        point_member,
        "raw_distal_support",
        "robust_core_config_sha256",
        "robust_core_diagnostics_json",
        "config_json",
        "config_sha256",
        *identity_members,
    }
    if not arthur:
        expected_members.add("spacing_um")
    if set(arrays) != expected_members:
        _fail(f"{label} NPZ members differ from the sealed-core contract")
    if _scalar_text(arrays["schema_version"], f"{label}.schema_version") != schema:
        _fail(f"{label} has the wrong sealed-core schema")
    if _scalar_int64(arrays["lens_index"], f"{label}.lens_index") != lens_index:
        _fail(f"{label} lens identity mismatch")
    if arthur:
        if expected_volume is None or expected_eye_id is None:
            _fail(f"{label} Arthur identity expectation is incomplete")
        if _scalar_text(arrays["volume"], f"{label}.volume") != expected_volume:
            _fail(f"{label} volume identity mismatch")
        if _scalar_int64(arrays["eye_id"], f"{label}.eye_id") != expected_eye_id:
            _fail(f"{label} eye identity mismatch")
    if _scalar_int64(arrays["raw_distal_support"], f"{label}.raw_distal_support") != expected_raw_support:
        _fail(f"{label} raw support differs from its technical row")
    if _scalar_text(
        arrays["robust_core_config_sha256"], f"{label}.robust_core_config_sha256"
    ) != ROBUST_CORE_CONFIG_SHA256:
        _fail(f"{label} robust-core hash mismatch")
    points = arrays[point_member]
    expected_dtype = np.dtype("float64") if arthur else np.dtype("int32")
    if points.dtype != expected_dtype or points.ndim != 2 or points.shape[1:] != (3,):
        _fail(f"{label}.{point_member} has wrong dtype or shape")
    if len(points) != expected_core_support:
        _fail(f"{label} core support differs from its technical row")
    if expected_raw_support < expected_core_support:
        _fail(f"{label} raw support is smaller than its robust core")
    if not np.all(np.isfinite(points)) or len(np.unique(points, axis=0)) != len(points):
        _fail(f"{label} sealed core has nonfinite or duplicate points")
    if not arthur:
        if np.any(points < 0):
            _fail(f"{label} sealed ZYX core contains negative indices")
        spacing = arrays["spacing_um"]
        if (
            spacing.dtype != np.dtype("float64")
            or spacing.shape != (3,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0.0)
        ):
            _fail(f"{label}.spacing_um must contain three positive float64 values")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    if not np.array_equal(order, np.arange(len(points))):
        _fail(f"{label} sealed core is not lexicographically sorted")
    config_text = _scalar_text(arrays["config_json"], f"{label}.config_json")
    config_hash = _scalar_text(arrays["config_sha256"], f"{label}.config_sha256")
    if _sha256_bytes(config_text.encode("utf-8")) != config_hash:
        _fail(f"{label} sealed configuration hash mismatch")
    if config_hash != expected_technical_config_sha256:
        _fail(f"{label} sealed configuration differs from its technical manifest")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} sealed configuration is invalid JSON") from exc
    if _canonical_json(config) != config_text:
        _fail(f"{label} sealed configuration is not canonical JSON")
    _validate_robust_core_binding(config, f"{label}.config")
    diagnostics_text = _scalar_text(
        arrays["robust_core_diagnostics_json"],
        f"{label}.robust_core_diagnostics_json",
    )
    try:
        diagnostics = json.loads(diagnostics_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} robust-core diagnostics are invalid JSON") from exc
    if not isinstance(diagnostics, Mapping):
        _fail(f"{label} robust-core diagnostics must be an object")
    if _canonical_json(diagnostics) != diagnostics_text:
        _fail(f"{label} robust-core diagnostics are not canonical JSON")
    if expected_diagnostics_json is not None and diagnostics_text != expected_diagnostics_json:
        _fail(f"{label} robust-core diagnostics differ from its technical row")
    if _strict_int(
        diagnostics.get("input_support"), f"{label}.input_support", 0
    ) != expected_raw_support:
        _fail(f"{label} robust-core diagnostics input support mismatch")
    if diagnostics.get("config_sha256") != ROBUST_CORE_CONFIG_SHA256:
        _fail(f"{label} robust-core diagnostics have wrong config hash")
    if expected_core_support:
        if (
            expected_robust_status != "pass"
            or diagnostics.get("status") != "pass"
            or expected_core_support < 27
            or expected_raw_support < 33
        ):
            _fail(f"{label} nonempty robust core violates status/support floors")
    elif expected_robust_status != "ineligible" or diagnostics.get("status") != "ineligible":
        _fail(f"{label} empty robust core lacks exact ineligible status")
    if expected_core_support >= 27:
        if diagnostics.get("schema_version") != robust64.ROBUST_DISTAL_CORE_SCHEMA_VERSION:
            _fail(f"{label} robust-core diagnostics have wrong schema")
        if diagnostics.get("config_sha256") != ROBUST_CORE_CONFIG_SHA256:
            _fail(f"{label} robust-core diagnostics have wrong config hash")
        reported = diagnostics.get("retained_support", diagnostics.get("output_support"))
        if reported is None or _strict_int(reported, f"{label}.retained_support", 0) != expected_core_support:
            _fail(f"{label} robust-core diagnostics support mismatch")
        if diagnostics.get("geometric_median_converged") is not True:
            _fail(f"{label} geometric median did not converge")
        gap = _strict_float(
            diagnostics.get("pca_normal_gap_ratio"), f"{label}.pca_normal_gap_ratio"
        )
        if gap < float(robust64.ROBUST_CORE_CONFIG["pca_minimum_normal_gap_ratio"]):
            _fail(f"{label} robust-core PCA normal gap is below the frozen floor")
        downstream = _strict_int(
            diagnostics.get("downstream_q90_support_lower_bound"),
            f"{label}.downstream_q90_support_lower_bound",
            0,
        )
        if downstream < 25:
            _fail(f"{label} robust core cannot guarantee 25 post-Stage-2 fit points")


def _validate_frame_audit(document: Mapping[str, Any], label: str) -> None:
    if document.get("schema_version") != FRAME_AUDIT_SCHEMA:
        _fail(f"{label} has wrong frame-audit schema")
    if _strict_bool(document.get("gate_passed"), f"{label}.gate_passed") is not True:
        _fail(f"{label} did not pass the frame-stability gate")
    fixed = _require_keys(document.get("fixed_point"), {"converged", "readded_count"}, f"{label}.fixed_point")
    if not _strict_bool(fixed["converged"], f"{label}.fixed_point.converged"):
        _fail(f"{label} fixed point did not converge")
    if _strict_int(fixed["readded_count"], f"{label}.fixed_point.readded_count", 0) != 0:
        _fail(f"{label} fixed point re-added a lens")


ARTHUR_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "analysis_scope",
        "isolation_basis",
        "oracle_stage1_scope",
        "threshold_config",
        "threshold_config_sha256",
        "predictor_pipeline_config",
        "predictor_pipeline_config_sha256",
        "robust_core_config",
        "robust_core_config_sha256",
        "technical_coherence_config",
        "technical_coherence_config_sha256",
        "modality_specific_qc",
        "technical_config_sha256",
        "sealed_config_sha256",
        "n_rows",
        "technical_counts",
        "technical_counts_by_volume",
        "fixed_points",
        "frame_audits",
        "input_files",
        "input_manifest",
        "git",
        "producer_code_sha256",
        "eyemap",
        "biological_independence",
        "technical_output_manifest",
        "sealed_outcome_manifest_binding",
        "document_role",
    }
)

MAIKE_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "eye_id",
        "species",
        "sex",
        "biological_independence",
        "analysis_scope",
        "isolation_basis",
        "oracle_stage1_scope",
        "threshold_config",
        "threshold_config_sha256",
        "predictor_pipeline_config",
        "predictor_pipeline_config_sha256",
        "technical_pipeline_config",
        "robust_core_config",
        "robust_core_config_sha256",
        "technical_coherence_config",
        "technical_coherence_config_sha256",
        "technical_config_sha256",
        "sealed_config_sha256",
        "n_expected",
        "n_rows",
        "technical_counts",
        "contiguous_indices",
        "index_range",
        "instance_segmentation_validated",
        "partition_evidence",
        "fixed_point",
        "sealed_distal_stage1_manifest",
        "input_hashes",
        "git",
        "producer_code_sha256",
        "technical_output_manifest",
        "sealed_outcome_manifest_binding",
        "document_role",
    }
)

ARTHUR_OUTCOME_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "analysis_scope",
        "technical_table_binding",
        "technical_config_sha256",
        "threshold_config_sha256",
        "robust_core_config_sha256",
        "target_config",
        "target_config_sha256",
        "counts",
        "git",
        "producer_code_sha256",
        "output_manifest",
    }
)

MAIKE_OUTCOME_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "experiment",
        "eye_id",
        "species",
        "sex",
        "analysis_scope",
        "technical_inventory_binding",
        "technical_config_sha256",
        "threshold_config_sha256",
        "robust_core_config_sha256",
        "target_config",
        "target_config_sha256",
        "n_expected",
        "n_rows",
        "counts",
        "contiguous_indices",
        "index_range",
        "target_cohort_definitions",
        "input_hashes",
        "git",
        "producer_code_sha256",
        "output_manifest",
    }
)


@dataclass(frozen=True)
class TechnicalBundle:
    kind: str
    unit_id: str
    root: Path
    table: pd.DataFrame
    completion_sha256: str
    provenance_sha256: str
    technical_manifest: Mapping[str, Mapping[str, Any]]
    outcome_manifest: DeferredArtifact
    qc_binding_sha256: str | None = None
    technical_config_sha256: str | None = None
    technical_exclusions: Mapping[str, Any] = field(default_factory=dict)


def _read_verified_technical_artifact(
    reader: GuardedArtifactReader,
    root: Path,
    manifest: Mapping[str, Mapping[str, Any]],
    relpath: str,
    purpose: str,
) -> bytes:
    if relpath not in manifest:
        _fail(f"{purpose} is absent from technical_output_manifest: {relpath}")
    return reader.read_bound(root, relpath, manifest[relpath], phase="pass1", purpose=purpose)


def _load_arthur_technical_bundle(
    reader: GuardedArtifactReader,
    root: Path,
    expected_commit: str,
    repository_root: Path,
) -> TechnicalBundle:
    completion, completion_bytes = _read_document(
        reader, root, "technical_completion.json", "Arthur technical completion"
    )
    provenance, provenance_bytes = _read_document(
        reader, root, "technical_provenance.json", "Arthur technical provenance"
    )
    _exact_keys(completion, ARTHUR_COMPLETION_KEYS, "Arthur technical completion")
    _exact_keys(
        provenance,
        ARTHUR_COMPLETION_KEYS | {"source_stage1_diagnostics"},
        "Arthur technical provenance",
    )
    _documents_agree(completion, provenance, "Arthur technical documents")
    for name, document in (("completion", completion), ("provenance", provenance)):
        label = f"Arthur technical {name}"
        if document["schema_version"] != arthur64.TECHNICAL_BUNDLE_SCHEMA:
            _fail(f"{label} has wrong schema")
        if document["status"] != "complete" or document["experiment"] != 64:
            _fail(f"{label} is not a clean completed Experiment 64 bundle")
        if document["analysis_scope"] != arthur64.ANALYSIS_SCOPE:
            _fail(f"{label} has wrong analysis scope")
        if document["document_role"] != f"technical_{name}":
            _fail(f"{label} has wrong document role")
        _validate_git_record(document["git"], expected_commit, f"{label}.git")
        _validate_robust_core_binding(document, label)
        _validate_technical_coherence_binding(document, label)
        _validate_common_technical_config_hashes(document, label)
    if _strict_int(completion["n_rows"], "Arthur technical completion.n_rows", 1) != EXPECTED_ARTHUR_ROWS:
        _fail("Arthur technical row count differs from the frozen source count")
    expected_arthur_sealed_config = arthur64._sealed_config(
        experiment63.EXPECTED_THRESHOLD_CONFIG,
        robust64.ROBUST_CORE_CONFIG,
    )
    expected_arthur_technical_hash = _sha256_bytes(
        _canonical_json(expected_arthur_sealed_config).encode("utf-8")
    )
    if completion["technical_config_sha256"] != expected_arthur_technical_hash:
        _fail("Arthur technical configuration differs from the frozen adapter contract")
    if completion["modality_specific_qc"] != expected_arthur_sealed_config["modality_specific_qc"]:
        _fail("Arthur modality-specific QC disclosure differs from the frozen mesh contract")
    _validate_repository_code_bindings(
        reader,
        repository_root,
        completion["producer_code_sha256"],
        label="Arthur producer",
        maike_basenames=False,
    )
    manifest = _validated_manifest_bindings(
        completion["technical_output_manifest"],
        "Arthur technical_output_manifest",
        phase="pass1",
    )
    table_relpath = "arthur_technical_table.csv"
    table = _csv_table(
        _read_verified_technical_artifact(reader, root, manifest, table_relpath, "Arthur technical table"),
        "Arthur technical table",
    )
    _validate_table_header(table, ARTHUR_TECHNICAL_FIELDS, "Arthur technical table")
    table = _validate_technical_rows(
        table,
        label="Arthur technical table",
        eligible_column="distal_qc",
        expected_rows=EXPECTED_ARTHUR_ROWS,
    )
    arthur_counts = _arthur_producer_counts(table)
    _strict_count_mapping(
        completion["technical_counts"],
        arthur_counts,
        "Arthur technical completion.technical_counts",
    )
    expected_counts_by_volume = {
        volume: _arthur_producer_counts(table.loc[table["volume"].astype(str) == volume])
        for volume in experiment63.ARTHUR_VOLUMES
    }
    counts_by_volume = _exact_keys(
        completion["technical_counts_by_volume"],
        expected_counts_by_volume,
        "Arthur technical completion.technical_counts_by_volume",
    )
    for volume, expected_counts in expected_counts_by_volume.items():
        _strict_count_mapping(
            counts_by_volume[volume],
            expected_counts,
            f"Arthur technical completion.technical_counts_by_volume.{volume}",
        )
    units = set(table["source_eye_unit"].astype(str))
    if len(units) != 6:
        _fail("Arthur technical table does not contain exactly six eye-within-volume units")
    if set(table["volume"].astype(str)) != set(experiment63.ARTHUR_VOLUMES):
        _fail("Arthur technical table does not cover the three frozen volumes")
    if set(completion["frame_audits"]) != units:
        _fail("Arthur frame-audit units differ from the technical table")

    expected_manifest_paths = {table_relpath}
    for row_number, row in table.iterrows():
        relpath = str(row["sealed_distal_relpath"])
        expected_manifest_paths.add(relpath)
        arrays = _npz_arrays(
            _read_verified_technical_artifact(
                reader,
                root,
                manifest,
                relpath,
                f"Arthur sealed distal core row {row_number}",
            ),
            f"Arthur sealed distal core row {row_number}",
        )
        _validate_core_npz(
            arrays,
            label=f"Arthur sealed distal core row {row_number}",
            schema=arthur64.SEALED_DISTAL_SCHEMA,
            lens_index=int(row["lens_index"]),
            expected_core_support=_strict_int(row["robust_core_support"], f"Arthur row {row_number}.robust_core_support", 0),
            expected_raw_support=_strict_int(row["raw_distal_support"], f"Arthur row {row_number}.raw_distal_support", 0),
            arthur=True,
            expected_technical_config_sha256=str(completion["technical_config_sha256"]),
            expected_volume=str(row["volume"]),
            expected_eye_id=_strict_int(row["eye_id"], f"Arthur row {row_number}.eye_id", 0),
            expected_robust_status=str(row["robust_core_status"]),
            expected_diagnostics_json=str(row["robust_core_diagnostics_json"]),
        )
    for unit, binding in completion["frame_audits"].items():
        binding = _exact_keys(binding, {"relative_path", "sha256", "size_bytes"}, f"Arthur frame audit {unit}")
        relpath = _safe_pure_relpath(binding["relative_path"], f"Arthur frame audit {unit}.relative_path").as_posix()
        expected_manifest_paths.add(relpath)
        compact = {"sha256": binding["sha256"], "size_bytes": binding["size_bytes"]}
        if manifest.get(relpath) != compact:
            _fail(f"Arthur frame audit {unit} binding disagrees with technical manifest")
        audit = _read_bound_json(
            reader,
            root,
            relpath,
            compact,
            phase="pass1",
            purpose=f"Arthur frame audit {unit}",
        )
        _validate_frame_audit(audit, f"Arthur frame audit {unit}")
    if set(manifest) != expected_manifest_paths:
        _fail(
            "Arthur technical_output_manifest is not the exact "
            "sealed-outcome-free technical inventory"
        )

    # This computation uses only distal predictor geometry.  Complete it in
    # Pass 1 so no outcome can be opened before every source-side spatial
    # invariant and its associated geometry checks have succeeded.
    table = experiment63.add_invariant_features_by_unit(
        table,
        ["volume", "eye_id"],
        "Experiment 64 Arthur technical predictor geometry",
    )

    outcome = _deferred_binding(
        root,
        completion["sealed_outcome_manifest_binding"],
        "Arthur sealed outcome manifest binding",
    )
    if provenance["sealed_outcome_manifest_binding"] != completion["sealed_outcome_manifest_binding"]:
        _fail("Arthur completion/provenance outcome bindings differ")
    return TechnicalBundle(
        kind="arthur",
        unit_id="arthur_source",
        root=root.resolve(),
        table=table,
        completion_sha256=_sha256_bytes(completion_bytes),
        provenance_sha256=_sha256_bytes(provenance_bytes),
        technical_manifest=manifest,
        outcome_manifest=outcome,
        technical_config_sha256=str(completion["technical_config_sha256"]),
        technical_exclusions={
            **_arthur_technical_exclusion_report(table),
            "by_volume": {
                volume: _arthur_technical_exclusion_report(
                    table.loc[table["volume"].astype(str) == volume]
                )
                for volume in experiment63.ARTHUR_VOLUMES
            },
        },
    )


_CLEARANCE_AUTHORITY = object()
_OUTCOME_ATTEMPT_AUTHORITY = object()


@dataclass(frozen=True)
class Pass1Clearance:
    schema_version: str
    analysis_label: str
    expected_commit: str
    robust_core_config_sha256: str
    maike: tuple[TechnicalBundle, ...]
    arthur: TechnicalBundle
    pass1_reads: tuple[ReadEvent, ...]
    _authority: object = field(repr=False, compare=False)


@dataclass
class OutcomeAttemptAuthorization:
    expected_commit: str
    record_path: Path
    record_sha256: str
    record_size_bytes: int
    _authority: object = field(repr=False, compare=False)
    _consumed: bool = field(default=False, repr=False, compare=False)


def _assert_clearance(clearance: Pass1Clearance) -> None:
    if not isinstance(clearance, Pass1Clearance) or clearance._authority is not _CLEARANCE_AUTHORITY:
        _fail("Pass 2 requires an in-process clearance returned by successful Pass 1")
    if clearance.schema_version != PASS1_SCHEMA or clearance.analysis_label != ANALYSIS_LABEL:
        _fail("Pass 1 clearance metadata is invalid")
    if clearance.robust_core_config_sha256 != ROBUST_CORE_CONFIG_SHA256:
        _fail("Pass 1 clearance robust-core hash is invalid")
    if tuple(bundle.unit_id for bundle in clearance.maike) != tuple(EXPECTED_EYES):
        _fail("Pass 1 clearance does not cover the twelve named Maike eyes in frozen order")


def _assert_outcome_attempt_authorization(
    clearance: Pass1Clearance,
    authorization: OutcomeAttemptAuthorization | None,
) -> None:
    if (
        not isinstance(authorization, OutcomeAttemptAuthorization)
        or authorization._authority is not _OUTCOME_ATTEMPT_AUTHORITY
    ):
        _fail("Pass 2 requires the in-process authorization from a durable outcome attempt")
    if authorization._consumed:
        _fail("Experiment 64 outcome-attempt authorization has already been consumed")
    if authorization.expected_commit != clearance.expected_commit:
        _fail("Experiment 64 outcome-attempt commit differs from Pass 1")
    try:
        payload = authorization.record_path.read_bytes()
    except OSError as exc:
        raise ContractError(f"Cannot read durable Experiment 64 attempt record: {exc}") from exc
    if (
        len(payload) != authorization.record_size_bytes
        or _sha256_bytes(payload) != authorization.record_sha256
    ):
        _fail("Durable Experiment 64 attempt record changed after authorization")
    document = _json_object(payload, "Experiment 64 outcome attempt record")
    if (
        document.get("schema_version") != OUTCOME_ATTEMPT_SCHEMA
        or document.get("status") != "pass2_attempt_committed_before_outcome_access"
        or document.get("git") != {"commit": clearance.expected_commit, "dirty": False}
        or document.get("rerun_permitted") is not False
        or document.get("outcome_artifacts_opened_at_record_creation") is not False
        or document.get("pass1_clearance_bindings") != _clearance_bindings(clearance)
        or document.get("outcome_manifest_bindings") != _outcome_manifest_bindings(clearance)
    ):
        _fail("Durable Experiment 64 attempt record differs from Pass-1 authorization")
    authorization._consumed = True


def _maike_producer_module() -> Any:
    """Return the independent Maike producer once it is installed beside us."""

    candidates = (
        "experiment_64_extract_lens_surfaces",
        "experiment_64_prepare_maike_bundle",
        "extract_experiment_64_lens_surfaces",
    )
    for name in candidates:
        try:
            return __import__(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    _fail("Experiment 64 Maike producer module is missing")


def _maike_technical_fields(producer: Any) -> tuple[str, ...]:
    for name in ("TECHNICAL_FIELDS", "TECHNICAL_INVENTORY_FIELDS"):
        value = getattr(producer, name, None)
        if value is not None:
            return tuple(value)
    _fail("Experiment 64 Maike producer does not export its technical field tuple")


def _maike_target_fields(producer: Any) -> tuple[str, ...]:
    value = getattr(
        producer,
        "TARGET_FIELDS",
        getattr(producer, "TARGET_TABLE_FIELDS", None),
    )
    return tuple(value) if value is not None else MAIKE_TARGET_FIELDS


# Filled in below once the renderer/attester contract has been validated.  It
# remains a distinct function so leakage-barrier tests can inject a late-eye
# technical failure without constructing image artifacts.
def _validate_maike_biological_identity(
    document: Mapping[str, Any], eye_id: str
) -> tuple[str, str]:
    if eye_id not in EXPECTED_EYES:
        _fail(f"{eye_id} is not in the frozen Maike cohort")
    expected_species = (
        "Drosophila simulans"
        if eye_id.startswith("M3_")
        else "Drosophila mauritiana"
    )
    expected_sex = "F" if "_F_" in eye_id else "M"
    if document.get("species") != expected_species or document.get("sex") != expected_sex:
        _fail(f"{eye_id} species/sex metadata differs from its frozen eye identity")
    expected_independence = {
        "independent_unit": "animal",
        "animal_id": eye_id,
        "one_eye_per_animal_in_validation": True,
        "source_basis": "one supplied eye stack per uniquely named fly",
    }
    if document.get("biological_independence") != expected_independence:
        _fail(f"{eye_id} biological-independence metadata differs from the frozen cohort")
    return expected_species, expected_sex


def _load_maike_technical_bundle(
    reader: GuardedArtifactReader,
    root: Path,
    eye_id: str,
    expected_rows: int,
    expected_commit: str,
    producer: Any,
    repository_root: Path,
) -> TechnicalBundle:
    completion, completion_bytes = _read_document(
        reader, root, "technical_completion.json", f"{eye_id} technical completion"
    )
    provenance, provenance_bytes = _read_document(
        reader, root, "technical_provenance.json", f"{eye_id} technical provenance"
    )
    _exact_keys(completion, MAIKE_COMPLETION_KEYS, f"{eye_id} technical completion")
    _exact_keys(
        provenance,
        MAIKE_COMPLETION_KEYS | {"input_validation"},
        f"{eye_id} technical provenance",
    )
    _documents_agree(completion, provenance, f"{eye_id} technical documents")
    for name, document in (("completion", completion), ("provenance", provenance)):
        label = f"{eye_id} technical {name}"
        if document.get("schema_version") != MAIKE_TECHNICAL_SCHEMA:
            _fail(f"{label} has wrong schema")
        if document.get("status") != "complete" or document.get("experiment") != 64:
            _fail(f"{label} is not a clean completed Experiment 64 bundle")
        if document.get("eye_id") != eye_id:
            _fail(f"{label} has the wrong eye identity")
        if document.get("document_role") != f"technical_{name}":
            _fail(f"{label} has the wrong document role")
        _validate_git_record(document.get("git"), expected_commit, f"{label}.git")
        _validate_robust_core_binding(document, label)
        _validate_technical_coherence_binding(document, label)
        _validate_common_technical_config_hashes(document, label)
    expected_species, expected_sex = _validate_maike_biological_identity(
        completion, eye_id
    )
    if _strict_int(completion.get("n_expected"), f"{eye_id}.n_expected", 1) != expected_rows:
        _fail(f"{eye_id} expected denominator differs from the frozen ODA denominator")
    if _strict_int(completion.get("n_rows"), f"{eye_id}.n_rows", 1) != expected_rows:
        _fail(f"{eye_id} technical row count differs from its frozen ODA denominator")
    if completion.get("contiguous_indices") is not True:
        _fail(f"{eye_id} does not attest contiguous lens indices")
    if completion.get("instance_segmentation_validated") is not False:
        _fail(f"{eye_id} changed the frozen instance-segmentation status")
    expected_technical_pipeline, expected_sealed_config = producer._technical_configs(
        experiment63.EXPECTED_THRESHOLD_CONFIG,
        producer.DEFAULT_PIPELINE_CONFIG,
        robust64.ROBUST_CORE_CONFIG,
    )
    expected_technical_hash = _sha256_bytes(
        _canonical_json(expected_sealed_config).encode("utf-8")
    )
    if completion.get("technical_pipeline_config") != expected_technical_pipeline:
        _fail(f"{eye_id} technical pipeline configuration changed")
    if completion.get("technical_config_sha256") != expected_technical_hash:
        _fail(f"{eye_id} technical configuration differs from the frozen adapter contract")
    _validate_repository_code_bindings(
        reader,
        repository_root,
        completion.get("producer_code_sha256"),
        label=f"{eye_id} producer",
        maike_basenames=True,
    )

    manifest = _validated_manifest_bindings(
        completion.get("technical_output_manifest"),
        f"{eye_id} technical_output_manifest",
        phase="pass1",
    )
    table_relpath = "technical_inventory.csv"
    table = _csv_table(
        _read_verified_technical_artifact(
            reader, root, manifest, table_relpath, f"{eye_id} technical inventory"
        ),
        f"{eye_id} technical inventory",
    )
    _validate_table_header(table, _maike_technical_fields(producer), f"{eye_id} technical inventory")
    eligible_column = "distal_eligible" if "distal_eligible" in table else "distal_qc"
    table = _validate_technical_rows(
        table,
        label=f"{eye_id} technical inventory",
        eligible_column=eligible_column,
        expected_rows=expected_rows,
    )
    if set(table["eye_id"].astype(str)) != {eye_id}:
        _fail(f"{eye_id} technical inventory contains another eye")
    if set(table["species"].astype(str)) != {expected_species}:
        _fail(f"{eye_id} technical inventory has wrong species metadata")
    if set(table["sex"].astype(str)) != {expected_sex}:
        _fail(f"{eye_id} technical inventory has wrong sex metadata")
    indices = np.sort(table["lens_index"].to_numpy(np.int64))
    if not np.array_equal(indices, np.arange(expected_rows, dtype=np.int64)):
        _fail(f"{eye_id} technical inventory is not the complete 0..{expected_rows - 1} range")
    for row_number, row in table.iterrows():
        full = _strict_int(row["full_assigned_size"], f"{eye_id} row {row_number}.full_assigned_size", 0)
        main = _strict_int(row["main_component_size"], f"{eye_id} row {row_number}.main_component_size", 0)
        removed = _strict_int(row["component_removed_size"], f"{eye_id} row {row_number}.component_removed_size", 0)
        if main > full or removed != full - main:
            _fail(f"{eye_id} row {row_number} has inconsistent component supports")
        fraction = _strict_float(row["main_component_fraction"], f"{eye_id} row {row_number}.main_component_fraction")
        expected_fraction = main / full if full else 0.0
        if not math.isclose(fraction, expected_fraction, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail(f"{eye_id} row {row_number} has an inconsistent component fraction")
        component_gate = _strict_bool(
            row["component_fraction_gate_pass"],
            f"{eye_id} row {row_number}.component_fraction_gate_pass",
        )
        if component_gate != bool(full > 0 and main * 100 >= full * 99):
            _fail(f"{eye_id} row {row_number} changed the exact 99% component gate")
        raw_support = _strict_int(row["raw_distal_support"], f"{eye_id} row {row_number}.raw_distal_support", 0)
        core_support = _strict_int(row["robust_core_support"], f"{eye_id} row {row_number}.robust_core_support", 0)
        core_fraction = _strict_float(
            row["robust_core_retained_fraction"],
            f"{eye_id} row {row_number}.robust_core_retained_fraction",
        )
        expected_core_fraction = core_support / raw_support if raw_support else 0.0
        if not math.isclose(core_fraction, expected_core_fraction, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail(f"{eye_id} row {row_number} has an inconsistent robust-core fraction")
        if bool(row[eligible_column]) and row["robust_core_status"] != "pass":
            _fail(f"{eye_id} row {row_number} is eligible without robust-core eligibility")

    maike_label = f"{eye_id} technical inventory"
    _strict_count_mapping(
        completion["technical_counts"],
        _maike_producer_counts(table, maike_label),
        f"{eye_id} technical completion.technical_counts",
    )

    expected_manifest_paths = {
        table_relpath,
        "distal_qc_sampling.csv",
        "distal_frame_audit.json",
    }
    for row_number, row in table.iterrows():
        for field_name in ("instance_relpath", "sealed_distal_relpath"):
            relpath = str(row[field_name])
            expected_manifest_paths.add(relpath)
        core_relpath = str(row["sealed_distal_relpath"])
        arrays = _npz_arrays(
            _read_verified_technical_artifact(
                reader,
                root,
                manifest,
                core_relpath,
                f"{eye_id} sealed distal core row {row_number}",
            ),
            f"{eye_id} sealed distal core row {row_number}",
        )
        _validate_core_npz(
            arrays,
            label=f"{eye_id} sealed distal core row {row_number}",
            schema=MAIKE_CORE_SCHEMA,
            lens_index=int(row["lens_index"]),
            expected_core_support=_strict_int(row["robust_core_support"], f"{eye_id} row {row_number}.robust_core_support", 0),
            expected_raw_support=_strict_int(row["raw_distal_support"], f"{eye_id} row {row_number}.raw_distal_support", 0),
            arthur=False,
            expected_technical_config_sha256=str(completion["technical_config_sha256"]),
            expected_robust_status=str(row["robust_core_status"]),
            expected_diagnostics_json=str(row["robust_core_diagnostics_json"]),
        )
        instance_relpath = str(row["instance_relpath"])
        instance_arrays = _npz_arrays(
            _read_verified_technical_artifact(
                reader,
                root,
                manifest,
                instance_relpath,
                f"{eye_id} Stage-1 instance row {row_number}",
            ),
            f"{eye_id} Stage-1 instance row {row_number}",
        )
        expected_instance_members = set(getattr(producer, "INSTANCE_KEYS"))
        if set(instance_arrays) != expected_instance_members:
            _fail(f"{eye_id} Stage-1 instance row {row_number} has forbidden/missing members")
        if _scalar_text(
            instance_arrays["schema_version"],
            f"{eye_id} instance row {row_number}.schema_version",
        ) != getattr(producer, "INSTANCE_SCHEMA", MAIKE_INSTANCE_SCHEMA):
            _fail(f"{eye_id} Stage-1 instance row {row_number} has wrong schema")
        if _scalar_int64(
            instance_arrays["lens_index"],
            f"{eye_id} instance row {row_number}.lens_index",
        ) != int(row["lens_index"]):
            _fail(f"{eye_id} Stage-1 instance row {row_number} has wrong lens identity")
        for point_name in (
            "full_assigned_points_zyx",
            "main_component_points_zyx",
            "raw_distal_points_zyx",
        ):
            points = instance_arrays[point_name]
            if points.dtype != np.dtype("int32") or points.ndim != 2 or points.shape[1:] != (3,):
                _fail(f"{eye_id} instance row {row_number}.{point_name} has wrong dtype/shape")
            if np.any(points < 0) or len(np.unique(points, axis=0)) != len(points):
                _fail(f"{eye_id} instance row {row_number}.{point_name} is invalid")
        if len(instance_arrays["full_assigned_points_zyx"]) != int(row["full_assigned_size"]):
            _fail(f"{eye_id} instance row {row_number} full support differs from inventory")
        if len(instance_arrays["main_component_points_zyx"]) != int(row["main_component_size"]):
            _fail(f"{eye_id} instance row {row_number} component support differs from inventory")
        if len(instance_arrays["raw_distal_points_zyx"]) != int(row["raw_distal_support"]):
            _fail(f"{eye_id} instance row {row_number} raw distal support differs from inventory")
        instance_config_text = _scalar_text(
            instance_arrays["config_json"],
            f"{eye_id} instance row {row_number}.config_json",
        )
        instance_config_hash = _scalar_text(
            instance_arrays["config_sha256"],
            f"{eye_id} instance row {row_number}.config_sha256",
        )
        if _sha256_bytes(instance_config_text.encode("utf-8")) != instance_config_hash:
            _fail(f"{eye_id} instance row {row_number} has an invalid config hash")
        try:
            instance_config = json.loads(instance_config_text)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"{eye_id} instance row {row_number} config is invalid JSON"
            ) from exc
        if _canonical_json(instance_config) != instance_config_text:
            _fail(f"{eye_id} instance row {row_number} config is not canonical JSON")
        instance_spacing = instance_arrays["spacing_um"]
        core_spacing = arrays["spacing_um"]
        if (
            instance_spacing.dtype != np.dtype("float64")
            or instance_spacing.shape != (3,)
            or not np.array_equal(instance_spacing, core_spacing)
        ):
            _fail(f"{eye_id} instance/core spacing differs at row {row_number}")
        full_set = set(map(tuple, instance_arrays["full_assigned_points_zyx"].tolist()))
        main_set = set(map(tuple, instance_arrays["main_component_points_zyx"].tolist()))
        raw_set = set(map(tuple, instance_arrays["raw_distal_points_zyx"].tolist()))
        core_set = set(map(tuple, arrays["points_zyx"].tolist()))
        if not main_set.issubset(full_set):
            _fail(f"{eye_id} row {row_number} dominant component leaves its assignment")
        if not raw_set.issubset(main_set):
            _fail(f"{eye_id} row {row_number} raw distal points leave the dominant component")
        if not core_set.issubset(raw_set):
            _fail(f"{eye_id} row {row_number} robust core leaves the raw distal candidate")

    frame_relpath = "distal_frame_audit.json"
    compact_frame_binding = manifest.get(frame_relpath)
    if compact_frame_binding is None:
        _fail(f"{eye_id} technical manifest lacks {frame_relpath}")
    frame_audit = _read_bound_json(
        reader,
        root,
        frame_relpath,
        compact_frame_binding,
        phase="pass1",
        purpose=f"{eye_id} frame audit",
    )
    _validate_frame_audit(frame_audit, f"{eye_id} frame audit")

    sampling = _csv_table(
        _read_verified_technical_artifact(
            reader,
            root,
            manifest,
            "distal_qc_sampling.csv",
            f"{eye_id} QC sampling table",
        ),
        f"{eye_id} QC sampling table",
    )
    _validate_table_header(sampling, MAIKE_SAMPLING_FIELDS, f"{eye_id} QC sampling table")
    if len(sampling) != expected_rows:
        _fail(f"{eye_id} QC sampling table does not cover the fixed denominator")
    if set(sampling["eye_id"].astype(str)) != {eye_id}:
        _fail(f"{eye_id} QC sampling table contains another eye")
    if set(zip(sampling["lens_index"].map(int), sampling["seed_id"].astype(str), strict=True)) != set(
        zip(table["lens_index"].map(int), table["seed_id"].astype(str), strict=True)
    ):
        _fail(f"{eye_id} QC sampling identities differ from the technical inventory")
    sampling = sampling.sort_values("lens_index").reset_index(drop=True)
    inventory_for_sampling = table.sort_values("lens_index").reset_index(drop=True)
    sampling_eligible = np.asarray(
        [
            _strict_bool(value, f"{eye_id}.sampling.distal_eligible")
            for value in sampling["distal_eligible"]
        ],
        dtype=bool,
    )
    inventory_eligible = inventory_for_sampling[eligible_column].to_numpy(bool)
    if not np.array_equal(sampling_eligible, inventory_eligible):
        _fail(f"{eye_id} sampling distal_eligible differs from the technical inventory")
    for column in (
        "position_u_um",
        "position_v_um",
        "distal_scale_um",
        "coherence_margin",
    ):
        sampling_values = pd.to_numeric(sampling[column], errors="coerce").to_numpy(float)
        inventory_values = inventory_for_sampling[column].to_numpy(float)
        if not np.all(np.isfinite(sampling_values[sampling_eligible])):
            _fail(f"{eye_id} eligible sampling {column} is nonfinite")
        if not np.array_equal(
            sampling_values[sampling_eligible], inventory_values[sampling_eligible]
        ):
            _fail(f"{eye_id} sampling {column} differs from the technical inventory")
        if np.any(np.isfinite(sampling_values[~sampling_eligible])):
            _fail(f"{eye_id} ineligible sampling {column} must be an empty CSV cell")
    for column in ("instance_relpath", "sealed_distal_relpath"):
        if not np.array_equal(
            sampling[column].astype(str).to_numpy(),
            inventory_for_sampling[column].astype(str).to_numpy(),
        ):
            _fail(f"{eye_id} sampling {column} differs from the technical inventory")

    if set(manifest) != expected_manifest_paths:
        extra = sorted(set(manifest) - expected_manifest_paths)[:5]
        missing = sorted(expected_manifest_paths - set(manifest))[:5]
        _fail(
            f"{eye_id} technical manifest is not the exact sealed-outcome-free "
            "technical inventory; "
            f"missing={missing}, extra={extra}"
        )

    # Position invariants and central-flag consistency are target-independent.
    # They belong in Pass 1 so a failure cannot occur only after Pass 2 has
    # opened the sequestered proximal outcomes.
    table = experiment63.add_invariant_features_by_unit(
        table,
        ["eye_id"],
        f"Experiment 64 Maike technical predictor geometry {eye_id}",
    )
    outcome = _deferred_binding(
        root,
        completion.get("sealed_outcome_manifest_binding"),
        f"{eye_id} sealed outcome manifest binding",
    )

    qc_hash = _validate_maike_qc(
        reader=reader,
        bundle_root=root,
        eye_id=eye_id,
        repository_root=repository_root,
        technical_table=table,
        sampling_table=sampling,
        technical_manifest=manifest,
        robust_core_config=completion["robust_core_config"],
        robust_core_hash=completion["robust_core_config_sha256"],
        outcome_manifest_binding=completion["sealed_outcome_manifest_binding"],
        expected_commit=expected_commit,
    )
    return TechnicalBundle(
        kind="maike",
        unit_id=eye_id,
        root=root.resolve(),
        table=table,
        completion_sha256=_sha256_bytes(completion_bytes),
        provenance_sha256=_sha256_bytes(provenance_bytes),
        technical_manifest=manifest,
        outcome_manifest=outcome,
        qc_binding_sha256=qc_hash,
        technical_config_sha256=str(completion["technical_config_sha256"]),
        technical_exclusions=_maike_technical_exclusion_report(table, maike_label),
    )


def _validate_qc_code_binding(
    *,
    reader: GuardedArtifactReader,
    repository_root: Path,
    value: Any,
    declared_relative_path: str,
    repository_relative_path: str,
    imported_module: Any,
    label: str,
) -> None:
    binding = _exact_keys(
        value,
        {"relative_path", "sha256", "size_bytes"},
        f"{label}.binding",
    )
    if binding["relative_path"] != declared_relative_path:
        _fail(f"{label} declares the wrong source path")
    expected_path = (repository_root / repository_relative_path).resolve()
    module_path = Path(getattr(imported_module, "__file__", "")).resolve()
    if module_path != expected_path:
        _fail(f"{label} imported module is not the repository implementation")
    reader.read_bound(
        repository_root,
        repository_relative_path,
        {"sha256": binding["sha256"], "size_bytes": binding["size_bytes"]},
        phase="pass1",
        purpose=f"{label} frozen implementation",
    )


def _validate_maike_qc(
    *,
    reader: GuardedArtifactReader,
    bundle_root: Path,
    eye_id: str,
    repository_root: Path,
    technical_table: pd.DataFrame,
    sampling_table: pd.DataFrame,
    technical_manifest: Mapping[str, Mapping[str, Any]],
    robust_core_config: Mapping[str, Any],
    robust_core_hash: str,
    outcome_manifest_binding: Mapping[str, Any],
    expected_commit: str,
) -> str:
    """Validate the new disjoint sample, review, ledger and attestation."""

    try:
        import attest_experiment_64_instance_qc as attester
        import render_experiment_64_instance_qc_sample as renderer
    except ModuleNotFoundError as exc:
        raise ContractError("Experiment 64 QC renderer/attester modules are missing") from exc

    sample_dir_name = getattr(renderer, "SAMPLE_DIRECTORY_NAME")
    sample_relpath = f"{sample_dir_name}/sample_manifest.json"
    sample_bytes = reader.read(
        bundle_root,
        sample_relpath,
        phase="pass1",
        purpose=f"{eye_id} Experiment 64 sample manifest",
    )
    sample = _json_object(sample_bytes, f"{eye_id} Experiment 64 sample manifest")
    if sample.get("schema_version") != renderer.SCHEMA_VERSION:
        _fail(f"{eye_id} has wrong Experiment 64 sample schema")
    if sample.get("experiment") != 64 or sample.get("eye_id") != eye_id:
        _fail(f"{eye_id} sample experiment/eye identity mismatch")
    if sample.get("sampling_algorithm") != renderer.SAMPLING_ALGORITHM:
        _fail(f"{eye_id} sample uses the wrong sampling algorithm")
    if _strict_int(sample.get("n_selected"), f"{eye_id}.sample.n_selected", 0) != 32:
        _fail(f"{eye_id} sample must contain exactly 32 cases")
    if sample.get("selected_development_overlap_count") != 0:
        _fail(f"{eye_id} sample reports development overlap")
    if sample.get("robust_core_config") != dict(robust_core_config):
        _fail(f"{eye_id} sample robust-core configuration differs from its producer")
    if sample.get("robust_core_config_sha256") != robust_core_hash:
        _fail(f"{eye_id} sample robust-core hash differs from its producer")
    renderer_relpath = (
        "experiments/maike-modern-ground-truth/"
        "render_experiment_64_instance_qc_sample.py"
    )
    _validate_qc_code_binding(
        reader=reader,
        repository_root=repository_root,
        value=sample.get("renderer_code"),
        declared_relative_path=renderer_relpath,
        repository_relative_path=renderer_relpath,
        imported_module=renderer,
        label=f"{eye_id} QC renderer",
    )
    sequestration = _require_keys(
        sample.get("outcome_sequestration"),
        {
            "sampling_table_exact_field_allowlist",
            "target_proximal_prediction_error_model_or_sealed_outcome_opened",
        },
        f"{eye_id}.sample.outcome_sequestration",
    )
    if tuple(sequestration["sampling_table_exact_field_allowlist"]) != MAIKE_SAMPLING_FIELDS:
        _fail(f"{eye_id} sample changed the target-free sampling allowlist")
    if sequestration["target_proximal_prediction_error_model_or_sealed_outcome_opened"] is not False:
        _fail(f"{eye_id} sample does not attest zero forbidden outcome access")

    ledger_binding = _exact_keys(
        sample.get("development_exclusion_ledger"),
        {"relative_path", "sha256", "size_bytes", "tracked_at_head", "head_commit"},
        f"{eye_id}.sample.development_exclusion_ledger",
    )
    ledger_relpath = _safe_pure_relpath(
        ledger_binding["relative_path"], f"{eye_id}.ledger.relative_path"
    ).as_posix()
    if ledger_relpath != renderer.EXCLUSION_LEDGER_RELPATH.as_posix():
        _fail(f"{eye_id} sample does not bind the canonical Experiment 64 exclusion ledger")
    if ledger_binding["tracked_at_head"] is not True or ledger_binding["head_commit"] != expected_commit:
        _fail(f"{eye_id} exclusion ledger is not bound to the frozen HEAD commit")
    ledger = _read_bound_json(
        reader,
        repository_root,
        ledger_relpath,
        {"sha256": ledger_binding["sha256"], "size_bytes": ledger_binding["size_bytes"]},
        phase="pass1",
        purpose=f"{eye_id} development exclusion ledger",
    )
    if ledger.get("schema_version") != renderer.EXCLUSION_SCHEMA_VERSION:
        _fail(f"{eye_id} development exclusion ledger has wrong schema")
    if ledger.get("n_eyes") != 12 or ledger.get("n_excluded_per_eye") != 32:
        _fail(f"{eye_id} development exclusion ledger does not bind 12 x 32 cases")
    if ledger.get("n_eye_scoped_exclusions") != 384:
        _fail(f"{eye_id} development exclusion ledger does not bind 384 cases")
    ledger_eyes = ledger.get("eyes")
    if not isinstance(ledger_eyes, list) or len(ledger_eyes) != 12:
        _fail(f"{eye_id} development exclusion ledger must contain twelve eyes")
    old_by_eye: dict[str, set[tuple[int, str]]] = {}
    for entry in ledger_eyes:
        entry = _require_keys(entry, {"eye_id", "n_excluded", "identities"}, f"{eye_id}.ledger.eye")
        ledger_eye = str(entry["eye_id"])
        identities = entry["identities"]
        if not isinstance(identities, list) or len(identities) != 32 or entry["n_excluded"] != 32:
            _fail(f"{eye_id} development ledger eye {ledger_eye} does not contain 32 identities")
        ledger_identities: set[tuple[int, str]] = set()
        ledger_lenses: set[int] = set()
        for identity in identities:
            if not isinstance(identity, Mapping):
                _fail(f"{eye_id} development ledger eye {ledger_eye} has a non-object identity")
            lens_index = _strict_int(
                identity.get("lens_index"),
                f"{eye_id}.ledger.{ledger_eye}.lens_index",
                0,
            )
            seed_id = identity.get("seed_id")
            if not isinstance(seed_id, str) or not seed_id:
                _fail(f"{eye_id} development ledger eye {ledger_eye} has an invalid seed identity")
            ledger_identities.add((lens_index, seed_id))
            ledger_lenses.add(lens_index)
        if len(ledger_identities) != 32 or len(ledger_lenses) != 32:
            _fail(f"{eye_id} development ledger eye {ledger_eye} has duplicate lens identities")
        if ledger_eye in old_by_eye:
            _fail(f"{eye_id} development ledger repeats eye {ledger_eye}")
        old_by_eye[ledger_eye] = ledger_identities
    if set(old_by_eye) != set(EXPECTED_EYES):
        _fail(f"{eye_id} development ledger does not cover the twelve frozen eyes")
    stop_binding = _require_keys(
        ledger.get("experiment63_stop_record"),
        {"relative_path", "sha256", "size_bytes"},
        f"{eye_id}.ledger.experiment63_stop_record",
    )
    stop_relpath = _safe_pure_relpath(
        stop_binding["relative_path"], f"{eye_id}.ledger.stop_record.relative_path"
    ).as_posix()
    if stop_relpath != renderer.STOP_RECORD_RELPATH.as_posix():
        _fail(f"{eye_id} exclusion ledger binds the wrong Experiment 63 stop record")
    reader.read_bound(
        repository_root,
        stop_relpath,
        {"sha256": stop_binding["sha256"], "size_bytes": stop_binding["size_bytes"]},
        phase="pass1",
        purpose=f"{eye_id} committed Experiment 63 stop record",
    )
    prior_verification = sample.get("prior_sample_manifest_verification")
    if not isinstance(prior_verification, list) or len(prior_verification) != 12:
        _fail(f"{eye_id} sample lacks twelve prior-manifest verifications")
    expected_prior_hashes = {
        str(entry["eye_id"]): str(entry.get("old_sample_manifest_sha256"))
        for entry in ledger_eyes
    }
    actual_prior_hashes: dict[str, str] = {}
    for entry in prior_verification:
        entry = _require_keys(
            entry,
            {"eye_id", "sha256", "n_verified_identities"},
            f"{eye_id}.sample.prior_sample_manifest_verification",
        )
        prior_eye = str(entry["eye_id"])
        if entry["n_verified_identities"] != 32 or prior_eye in actual_prior_hashes:
            _fail(f"{eye_id} prior-manifest verification is incomplete or duplicated")
        actual_prior_hashes[prior_eye] = _strict_sha256(
            entry["sha256"], f"{eye_id}.prior_manifest.{prior_eye}.sha256"
        )
    if actual_prior_hashes != expected_prior_hashes:
        _fail(f"{eye_id} prior-manifest verifications differ from the exclusion ledger")

    _validate_table_header(
        sampling_table,
        MAIKE_SAMPLING_FIELDS,
        f"{eye_id} QC replay sampling table",
    )
    replay_rows: list[Any] = []
    replay_indices: set[int] = set()
    replay_seeds: set[str] = set()
    for row_number, row in sampling_table.iterrows():
        lens_index = _strict_int(
            row["lens_index"], f"{eye_id}.sampling[{row_number}].lens_index", 0
        )
        seed_id = str(row["seed_id"])
        if not seed_id or lens_index in replay_indices or seed_id in replay_seeds:
            _fail(f"{eye_id} QC replay sampling identities are empty or duplicated")
        if str(row["eye_id"]) != eye_id:
            _fail(f"{eye_id} QC replay sampling table contains another eye")
        eligible = _strict_bool(
            row["distal_eligible"],
            f"{eye_id}.sampling[{row_number}].distal_eligible",
        )
        expected_instance = f"instances/lens_{lens_index:06d}.npz"
        expected_core = f"sealed_distal/lens_{lens_index:06d}.npz"
        if row["instance_relpath"] != expected_instance or row["sealed_distal_relpath"] != expected_core:
            _fail(f"{eye_id} QC replay sampling artifact path is noncanonical")
        geometry: list[float | None] = []
        for column in (
            "position_u_um",
            "position_v_um",
            "distal_scale_um",
            "coherence_margin",
        ):
            value = row[column]
            if eligible:
                geometry.append(
                    _strict_float(value, f"{eye_id}.sampling[{row_number}].{column}")
                )
            else:
                if not pd.isna(value):
                    _fail(f"{eye_id} ineligible QC replay row exposes {column}")
                geometry.append(None)
        if eligible and float(geometry[2]) <= 0.0:
            _fail(f"{eye_id} eligible QC replay row has nonpositive scale")
        replay_rows.append(
            renderer.SamplingRow(
                eye_id=eye_id,
                lens_index=lens_index,
                seed_id=seed_id,
                distal_eligible=eligible,
                position_u_um=geometry[0],
                position_v_um=geometry[1],
                distal_scale_um=geometry[2],
                coherence_margin=geometry[3],
                instance_relpath=expected_instance,
                sealed_distal_relpath=expected_core,
            )
        )
        replay_indices.add(lens_index)
        replay_seeds.add(seed_id)
    if replay_indices != set(range(len(replay_rows))):
        _fail(f"{eye_id} QC replay sampling indices are not the complete 0..N-1 range")
    try:
        unseen_rows, eligible_removed = renderer.exclude_development_identities(
            replay_rows,
            old_by_eye[eye_id],
            eye_id=eye_id,
        )
        expected_selected = renderer.select_frozen_sample(unseen_rows)
    except renderer.QCError as exc:
        raise ContractError(f"{eye_id} deterministic QC replay failed: {exc}") from exc

    samples = sample.get("samples")
    if not isinstance(samples, list) or len(samples) != 32:
        _fail(f"{eye_id} sample must list exactly 32 cases")
    expected_summary = {
        "n_inventory_rows": len(replay_rows),
        "n_distal_qc_eligible_before_development_exclusion": sum(
            row.distal_eligible for row in replay_rows
        ),
        "n_development_identities_for_eye": 32,
        "n_eligible_development_identities_removed": eligible_removed,
        "n_unseen_distal_qc_eligible": len(unseen_rows),
        "n_selected": 32,
        "selected_development_overlap_count": 0,
    }
    for field_name, expected_value in expected_summary.items():
        if sample.get(field_name) != expected_value:
            _fail(f"{eye_id} sample summary {field_name} differs from deterministic replay")
    expected_cells = {
        f"r{radial}_s{scale}": 2
        for radial in range(renderer.N_RADIAL_STRATA)
        for scale in range(renderer.N_SCALE_STRATA)
    }
    if sample.get("cell_counts") != expected_cells:
        _fail(f"{eye_id} sample cell counts differ from deterministic replay")
    if sample.get("selection_role_counts") != {
        "near_worst_coherence": 16,
        "hash_minimal_remaining": 16,
    }:
        _fail(f"{eye_id} sample selection-role counts differ from deterministic replay")
    new_lenses: set[int] = set()
    technical_identities = set(
        zip(technical_table["lens_index"].map(int), technical_table["seed_id"].astype(str), strict=True)
    )
    technical_by_lens = technical_table.set_index("lens_index", drop=False)
    for ordinal, (entry, selected) in enumerate(
        zip(samples, expected_selected, strict=True)
    ):
        entry = _require_keys(
            entry,
            {"ordinal", "eye_id", "lens_index", "seed_id", "render"},
            f"{eye_id}.sample[{ordinal}]",
        )
        if entry["ordinal"] != ordinal or entry["eye_id"] != eye_id:
            _fail(f"{eye_id} sample ordinal/eye mismatch at {ordinal}")
        exact_selection = {
            "eye_id": eye_id,
            "lens_index": selected.row.lens_index,
            "seed_id": selected.row.seed_id,
            "radius_um": selected.row.radius_um,
            "distal_scale_um": selected.row.distal_scale_um,
            "coherence_margin": selected.row.coherence_margin,
            "radial_rank_after_exclusion": selected.radial_rank,
            "radial_stratum": selected.radial_stratum,
            "scale_rank_within_radial_after_exclusion": selected.scale_rank_within_radial,
            "scale_stratum": selected.scale_stratum,
            "selection_role": selected.selection_role,
            "selection_sha256": selected.selection_sha256,
        }
        if any(entry.get(key) != value for key, value in exact_selection.items()):
            _fail(f"{eye_id} sample {ordinal} differs from deterministic reselection")
        lens_index = _strict_int(entry["lens_index"], f"{eye_id}.sample[{ordinal}].lens_index", 0)
        seed_id = str(entry["seed_id"])
        if (lens_index, seed_id) not in technical_identities:
            _fail(f"{eye_id} sample {ordinal} is absent from the technical inventory")
        if lens_index in new_lenses:
            _fail(f"{eye_id} sample repeats lens {lens_index}")
        new_lenses.add(lens_index)
        technical_row = technical_by_lens.loc[lens_index]
        if isinstance(technical_row, pd.DataFrame):
            _fail(f"{eye_id} technical inventory repeats lens {lens_index}")
        for field_name, technical_relpath in (
            ("instance_artifact", selected.row.instance_relpath),
            ("sealed_robust_core_artifact", selected.row.sealed_distal_relpath),
        ):
            artifact_binding = _require_keys(
                entry.get(field_name),
                {"relative_path", "sha256", "size_bytes", "members_present", "members_accessed"},
                f"{eye_id}.sample[{ordinal}].{field_name}",
            )
            if artifact_binding["relative_path"] != technical_relpath:
                _fail(f"{eye_id} sample {ordinal} {field_name} path differs from reselection")
            if {
                "sha256": artifact_binding["sha256"],
                "size_bytes": artifact_binding["size_bytes"],
            } != dict(technical_manifest[technical_relpath]):
                _fail(f"{eye_id} sample {ordinal} {field_name} differs from technical manifest")
            expected_members = (
                (sorted(renderer.INSTANCE_MEMBERS), list(renderer.INSTANCE_ACCESSED_MEMBERS))
                if field_name == "instance_artifact"
                else (sorted(renderer.SEALED_CORE_MEMBERS), list(renderer.SEALED_CORE_ACCESSED_MEMBERS))
            )
            if (
                artifact_binding["members_present"] != expected_members[0]
                or artifact_binding["members_accessed"] != expected_members[1]
            ):
                _fail(f"{eye_id} sample {ordinal} {field_name} member disclosure changed")
        point_counts = _exact_keys(
            entry.get("point_counts"),
            {
                "full_assigned",
                "dominant_component",
                "raw_localized_distal",
                "final_robust_distal_core",
            },
            f"{eye_id}.sample[{ordinal}].point_counts",
        )
        expected_points = {
            "full_assigned": int(technical_row["full_assigned_size"]),
            "dominant_component": int(technical_row["main_component_size"]),
            "raw_localized_distal": int(technical_row["raw_distal_support"]),
            "final_robust_distal_core": int(technical_row["robust_core_support"]),
        }
        if dict(point_counts) != expected_points:
            _fail(f"{eye_id} sample {ordinal} point counts differ from technical inventory")
        render_binding = _exact_keys(
            entry["render"],
            {"relative_path", "sha256", "size_bytes"},
            f"{eye_id}.sample[{ordinal}].render",
        )
        render_relpath = (
            PurePosixPath(sample_dir_name)
            / _safe_pure_relpath(render_binding["relative_path"], f"{eye_id}.sample[{ordinal}].render.relative_path")
        ).as_posix()
        expected_render_relpath = (
            f"{sample_dir_name}/renders/sample_{ordinal:02d}_"
            f"r{selected.radial_stratum}_s{selected.scale_stratum}_"
            f"lens_{lens_index:06d}.png"
        )
        if render_relpath != expected_render_relpath:
            _fail(f"{eye_id} sample {ordinal} render path differs from deterministic reselection")
        reader.read_bound(
            bundle_root,
            render_relpath,
            {"sha256": render_binding["sha256"], "size_bytes": render_binding["size_bytes"]},
            phase="pass1",
            purpose=f"{eye_id} sample render {ordinal}",
        )
    overlap = sorted(
        (lens_index, seed_id)
        for lens_index, seed_id in old_by_eye[eye_id]
        if lens_index in new_lenses
    )
    if overlap:
        _fail(f"{eye_id} Experiment 64 sample overlaps Experiment 63 lenses: {overlap}")

    review_name = getattr(attester, "REVIEW_FILENAME")
    attestation_name = getattr(attester, "ATTESTATION_FILENAME")
    review_bytes = reader.read(
        bundle_root,
        review_name,
        phase="pass1",
        purpose=f"{eye_id} Experiment 64 review",
    )
    review = _json_object(review_bytes, f"{eye_id} Experiment 64 review")
    _exact_keys(review, attester.REVIEW_TOP_LEVEL_FIELDS, f"{eye_id} review")
    if review.get("schema_version") != attester.REVIEW_SCHEMA_VERSION:
        _fail(f"{eye_id} review has wrong schema")
    if (
        review.get("eye_id") != eye_id
        or review.get("review_scope") != "disjoint_stratified_sample_only"
        or review.get("review_mode") not in attester.REVIEW_MODES
        or review.get("sample_manifest_sha256") != _sha256_bytes(sample_bytes)
    ):
        _fail(f"{eye_id} review is not bound to the exact sample manifest")
    if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
        _fail(f"{eye_id} review has no reviewer identity")
    if not isinstance(review.get("reviewed_at_utc"), str) or not attester.UTC_RE.fullmatch(
        review["reviewed_at_utc"]
    ):
        _fail(f"{eye_id} review timestamp is not canonical UTC")
    decisions = review.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 32:
        _fail(f"{eye_id} review must contain exactly 32 decisions")
    expected_identities = [
        (int(entry["lens_index"]), str(entry["seed_id"])) for entry in samples
    ]
    actual_identities: list[tuple[int, str]] = []
    for ordinal, decision in enumerate(decisions):
        decision = _require_keys(
            decision,
            attester.REVIEW_DECISION_FIELDS,
            f"{eye_id}.review[{ordinal}]",
        )
        actual_identities.append((int(decision["lens_index"]), str(decision["seed_id"])))
        if decision["decision"] != "pass":
            _fail(f"{eye_id} review decision {ordinal} is not pass")
        if not isinstance(decision["notes"], str):
            _fail(f"{eye_id} review decision {ordinal} notes are not text")
    if actual_identities != expected_identities:
        _fail(f"{eye_id} review identities/order differ from the sample")

    attestation_bytes = reader.read(
        bundle_root,
        attestation_name,
        phase="pass1",
        purpose=f"{eye_id} Experiment 64 attestation",
    )
    attestation = _json_object(attestation_bytes, f"{eye_id} Experiment 64 attestation")
    attester_relpath = (
        "experiments/maike-modern-ground-truth/"
        "attest_experiment_64_instance_qc.py"
    )
    _validate_qc_code_binding(
        reader=reader,
        repository_root=repository_root,
        value=attestation.get("attester_code"),
        declared_relative_path="attest_experiment_64_instance_qc.py",
        repository_relative_path=attester_relpath,
        imported_module=attester,
        label=f"{eye_id} QC attester",
    )
    if attestation.get("schema_version") != attester.ATTESTATION_SCHEMA_VERSION:
        _fail(f"{eye_id} attestation has wrong schema")
    if (
        attestation.get("experiment") != 64
        or attestation.get("status") != "passed"
        or attestation.get("eye_id") != eye_id
        or attestation.get("review_scope") != "disjoint_stratified_sample_only"
    ):
        _fail(f"{eye_id} attestation has wrong eye identity")
    if attestation.get("stratified_sample_visual_qc_passed") is not True:
        _fail(f"{eye_id} attestation does not pass visual QC")
    if attestation.get("whole_run_stop_rule_satisfied_for_this_eye") is not True:
        _fail(f"{eye_id} attestation does not satisfy the whole-run stop rule")
    if attestation.get("technical_inventory_complete") is not True:
        _fail(f"{eye_id} attestation does not verify the technical inventory")
    if attestation.get("target_free_artifact_hash_verification_passed") is not True:
        _fail(f"{eye_id} attestation did not verify target-free artifact hashes")
    exclusion = _require_keys(
        attestation.get("development_exclusion_verification"),
        {
            "prior_sample_manifests_verified",
            "eye_scoped_identities_verified",
            "identities_for_this_eye",
            "selected_overlap_count",
            "exclusion_before_new_strata",
        },
        f"{eye_id}.attestation.development_exclusion_verification",
    )
    if (
        exclusion["prior_sample_manifests_verified"] != 12
        or exclusion["eye_scoped_identities_verified"] != 384
        or exclusion["identities_for_this_eye"] != 32
        or exclusion["selected_overlap_count"] != 0
        or exclusion["exclusion_before_new_strata"] is not True
    ):
        _fail(f"{eye_id} attestation development-exclusion evidence is invalid")
    sequestration = _require_keys(
        attestation.get("outcome_sequestration"),
        {
            "technical_pass_only",
            "sealed_outcome_manifest_binding_validated_but_file_not_opened",
            "target_proximal_prediction_error_model_or_sealed_outcome_opened",
            "model_and_error_blind",
        },
        f"{eye_id}.attestation.outcome_sequestration",
    )
    if (
        sequestration["technical_pass_only"] is not True
        or sequestration["sealed_outcome_manifest_binding_validated_but_file_not_opened"] is not True
        or sequestration["target_proximal_prediction_error_model_or_sealed_outcome_opened"] is not False
        or sequestration["model_and_error_blind"] is not True
    ):
        _fail(f"{eye_id} attestation does not verify outcome sequestration")
    reviewed = attestation.get("reviewed_samples")
    if not isinstance(reviewed, list) or len(reviewed) != 32:
        _fail(f"{eye_id} attestation does not bind 32 reviewed samples")
    expected_reviewed = [
        {
            "ordinal": ordinal,
            "lens_index": int(decision["lens_index"]),
            "seed_id": str(decision["seed_id"]),
            "decision": "pass",
        }
        for ordinal, decision in enumerate(decisions)
    ]
    if reviewed != expected_reviewed:
        _fail(f"{eye_id} attestation reviewed samples differ from the verified review")
    if attestation.get("review") != {
        "review_mode": review["review_mode"],
        "reviewer_id": review["reviewer_id"],
        "reviewed_at_utc": review["reviewed_at_utc"],
    }:
        _fail(f"{eye_id} attestation review metadata differs from the verified review")
    bindings = _require_keys(
        attestation.get("bindings"),
        {
            "sample_manifest",
            "review",
            "sealed_outcome_manifest",
            "development_exclusion_ledger",
        },
        f"{eye_id}.attestation.bindings",
    )
    expected_binding_hashes = {
        "sample_manifest": _sha256_bytes(sample_bytes),
        "review": _sha256_bytes(review_bytes),
    }
    for key, expected_hash in expected_binding_hashes.items():
        binding = _require_keys(bindings[key], {"sha256"}, f"{eye_id}.attestation.bindings.{key}")
        if binding["sha256"] != expected_hash:
            _fail(f"{eye_id} attestation {key} hash mismatch")
    sealed_binding = _exact_keys(
        bindings["sealed_outcome_manifest"],
        {"relative_path", "sha256", "size_bytes"},
        f"{eye_id}.attestation.sealed_outcome_manifest",
    )
    _strict_sha256(
        sealed_binding["sha256"],
        f"{eye_id}.attestation.sealed_outcome_manifest.sha256",
    )
    _strict_int(
        sealed_binding["size_bytes"],
        f"{eye_id}.attestation.sealed_outcome_manifest.size_bytes",
        0,
    )
    if sealed_binding != outcome_manifest_binding:
        _fail(f"{eye_id} attestation sealed-outcome binding differs from its producer")
    if bindings["development_exclusion_ledger"] != ledger_binding:
        _fail(f"{eye_id} attestation exclusion-ledger binding differs from its sample")
    return _sha256_bytes(attestation_bytes)


def run_pass1(
    *,
    repository_root: Path,
    expected_commit: str,
    arthur_root: Path,
    maike_root: Path,
    reader: GuardedArtifactReader | None = None,
    expected_eyes: Mapping[str, int] | None = None,
    git_validator: Callable[[Path, str], str] | None = None,
) -> Pass1Clearance:
    """Run the target-inaccessible technical preflight for all inputs."""

    reader = reader or GuardedArtifactReader()
    for root, label in (
        (repository_root, "repository"),
        (arthur_root, "Arthur bundle"),
        (maike_root, "Maike bundle"),
    ):
        if root.is_symlink():
            _fail(f"Experiment 64 forbids a symlinked {label} root: {root}")
    eye_map = dict(EXPECTED_EYES if expected_eyes is None else expected_eyes)
    if tuple(eye_map) != tuple(EXPECTED_EYES) or eye_map != EXPECTED_EYES:
        _fail("Experiment 64 requires exactly the twelve frozen eye denominators")
    git_validator = git_validator or experiment63.require_frozen_git
    expected_commit = git_validator(repository_root, expected_commit)
    if expected_commit is None:
        _fail("Git validator did not return the frozen commit")
    if ROBUST_CORE_CONFIG_SHA256 != "bb8053b79a1aa4a7f6398551e28064b10bd21471ddaf5ad8aec07962dbf8532d":
        _fail("Backend robust-core implementation/config hash changed")
    robust64.normalise_robust_core_config(robust64.ROBUST_CORE_CONFIG)
    producer = _maike_producer_module()

    # Ordering is a leakage control: all twelve Maike technical/QC gates are
    # complete before even Arthur's target-free source table is opened.
    maike_bundles: list[TechnicalBundle] = []
    for eye_id, expected_rows in eye_map.items():
        maike_bundles.append(
            _load_maike_technical_bundle(
                reader,
                maike_root / eye_id,
                eye_id,
                expected_rows,
                expected_commit,
                producer,
                repository_root,
            )
        )
    arthur_bundle = _load_arthur_technical_bundle(
        reader, arthur_root, expected_commit, repository_root
    )
    return Pass1Clearance(
        schema_version=PASS1_SCHEMA,
        analysis_label=ANALYSIS_LABEL,
        expected_commit=expected_commit,
        robust_core_config_sha256=ROBUST_CORE_CONFIG_SHA256,
        maike=tuple(maike_bundles),
        arthur=arthur_bundle,
        pass1_reads=tuple(reader.events),
        _authority=_CLEARANCE_AUTHORITY,
    )


def _validate_outcome_manifest_common(
    document: Mapping[str, Any],
    *,
    label: str,
    schema: str,
    expected_commit: str,
    technical_bundle: TechnicalBundle,
) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != schema:
        _fail(f"{label} has wrong schema")
    if document.get("status") != "complete" or document.get("experiment") != 64:
        _fail(f"{label} is not a clean completed Experiment 64 outcome bundle")
    _validate_git_record(document.get("git"), expected_commit, f"{label}.git")
    if document.get("robust_core_config_sha256") != ROBUST_CORE_CONFIG_SHA256:
        _fail(f"{label} robust-core hash differs from Pass 1")
    expected_threshold_hash = _sha256_bytes(
        experiment63.canonical_json(experiment63.EXPECTED_THRESHOLD_CONFIG).encode("utf-8")
    )
    if document.get("threshold_config_sha256") != expected_threshold_hash:
        _fail(f"{label} distal-geometry threshold hash differs from Pass 1")
    target_config = document.get("target_config")
    if not isinstance(target_config, Mapping):
        _fail(f"{label}.target_config must be an object")
    if _strict_sha256(
        document.get("target_config_sha256"), f"{label}.target_config_sha256"
    ) != _sha256_bytes(_canonical_json(target_config).encode("utf-8")):
        _fail(f"{label} target configuration hash is invalid")
    technical_binding_name = (
        "technical_table_binding"
        if technical_bundle.kind == "arthur"
        else "technical_inventory_binding"
    )
    technical_binding = _require_keys(
        document.get(technical_binding_name),
        {"relative_path", "sha256", "size_bytes"},
        f"{label}.{technical_binding_name}",
    )
    expected_table_name = "arthur_technical_table.csv" if technical_bundle.kind == "arthur" else "technical_inventory.csv"
    if technical_binding["relative_path"] != expected_table_name:
        _fail(f"{label} binds the wrong technical table")
    pass1_table_binding = technical_bundle.technical_manifest.get(expected_table_name)
    if pass1_table_binding is None or {
        "sha256": technical_binding["sha256"],
        "size_bytes": technical_binding["size_bytes"],
    } != dict(pass1_table_binding):
        _fail(f"{label} technical-table binding differs from Pass 1")
    if document.get("technical_config_sha256") != technical_bundle.technical_config_sha256:
        _fail(f"{label} technical configuration differs from Pass 1")
    if technical_bundle.kind == "maike":
        if document.get("eye_id") != technical_bundle.unit_id:
            _fail(f"{label} has the wrong eye identity")
    return _validated_manifest_bindings(document.get("output_manifest"), f"{label}.output_manifest", phase="pass2")


def _validate_target_reason_row(
    row: pd.Series,
    *,
    label: str,
    expected_resolvable: bool,
    expected_target_qc: bool,
    q05: float,
    rmse: float,
) -> None:
    """Validate the explanatory reason cells against recomputed target state."""

    resolvability = _reason_tokens(
        row["target_resolvability_reasons"],
        f"{label}.target_resolvability_reasons",
    )
    quality = _reason_tokens(
        row["target_qc_reasons"],
        f"{label}.target_qc_reasons",
    )
    if resolvability != tuple(sorted(resolvability)) or quality != tuple(sorted(quality)):
        _fail(f"{label} target reason tokens are not in canonical sorted order")
    allowed_structural = {
        "distal_qc_failed",
        "distal_refit_failed",
        "empty_component_or_distal",
        "invalid_distal_points",
        "invalid_distal_scale",
        "invalid_final_distal_frame",
        "invalid_proximal_points",
        "main_component_fraction_below_minimum",
        "no_distal_geometry",
        "no_spanning_lateral_bins",
        "nonfinite_target",
        "nonfinite_target_fit",
        "target_fit_failed",
        "target_fit_unavailable",
        "target_support_below_minimum",
    }
    allowed_quality = {
        "target_q05_raw_thickness_not_positive",
        "target_rmse_above_maximum",
    }
    if not set(resolvability).issubset(allowed_structural):
        _fail(f"{label} has an unrecognized target-resolvability reason")
    if not set(quality).issubset(allowed_structural | allowed_quality):
        _fail(f"{label} has an unrecognized target-QC reason")
    if not set(resolvability).issubset(quality):
        _fail(f"{label} target-QC reasons omit a resolvability reason")
    if expected_resolvable != (not resolvability):
        _fail(f"{label} target_resolvable and reason emptiness disagree")
    if expected_target_qc != (not quality):
        _fail(f"{label} target_qc and reason emptiness disagree")
    if expected_resolvable:
        expected_quality: set[str] = set()
        if not (math.isfinite(q05) and q05 > 0.0):
            expected_quality.add("target_q05_raw_thickness_not_positive")
        if not (math.isfinite(rmse) and rmse <= 2.5):
            expected_quality.add("target_rmse_above_maximum")
        if set(quality) != expected_quality:
            _fail(f"{label} target-QC reasons differ from recomputed q05/RMSE gates")


def _load_target_npz(
    arrays: Mapping[str, np.ndarray],
    *,
    label: str,
    schema: str,
    row: pd.Series,
    arthur: bool,
    technical_row: pd.Series,
    expected_members: set[str],
    technical_bundle: TechnicalBundle,
    outcome_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if set(arrays) != expected_members:
        _fail(f"{label} NPZ members differ from the Experiment 64 target contract")
    if _scalar_text(arrays["schema_version"], f"{label}.schema_version") != schema:
        _fail(f"{label} has wrong target schema")
    lens_index = int(row["lens_index"])
    if _scalar_int64(arrays["lens_index"], f"{label}.lens_index") != lens_index:
        _fail(f"{label} lens identity mismatch")
    if arthur:
        if _scalar_text(arrays["volume"], f"{label}.volume") != str(row["volume"]):
            _fail(f"{label} volume identity mismatch")
        if _scalar_int64(arrays["eye_id"], f"{label}.eye_id") != int(row["eye_id"]):
            _fail(f"{label} Arthur eye identity mismatch")
    elif _scalar_text(arrays["eye_id"], f"{label}.eye_id") != str(row["eye_id"]):
        _fail(f"{label} Maike eye identity mismatch")
    grid = arrays["canonical_grid_xy"]
    smooth = arrays["target_smoothed_thickness_um"]
    raw_xy = arrays["raw_target_xy_normalized"]
    raw_thickness = arrays["raw_target_thickness_um"]
    coefficients = arrays["target_coefficients_c0_c5"]
    if grid.dtype != np.dtype("float64") or grid.shape != (81, 2) or not np.array_equal(grid, experiment63.CANONICAL_GRID_XY):
        _fail(f"{label} canonical target grid mismatch")
    if smooth.dtype != np.dtype("float64") or smooth.shape != (81,):
        _fail(f"{label} smoothed target must be float64[81]")
    if raw_xy.dtype != np.dtype("float64") or raw_xy.ndim != 2 or raw_xy.shape[1:] != (2,):
        _fail(f"{label} raw target coordinates must be float64[N,2]")
    if raw_thickness.dtype != np.dtype("float64") or raw_thickness.shape != (len(raw_xy),):
        _fail(f"{label} raw target thickness must be float64[N]")
    if coefficients.dtype != np.dtype("float64") or coefficients.shape != (6,):
        _fail(f"{label} target coefficients must be float64[6]")
    if len(raw_xy) != int(row["target_support"]):
        _fail(f"{label} target support differs from target table")
    table_coefficients = row.loc[list(experiment63.TARGET_COLUMNS)].to_numpy(float)
    if not np.allclose(coefficients, table_coefficients, rtol=0.0, atol=0.0, equal_nan=True):
        _fail(f"{label} target coefficients differ from target table")
    finite_raw = np.all(np.isfinite(raw_xy)) and np.all(np.isfinite(raw_thickness))
    if len(raw_xy) and (not finite_raw or np.any(np.linalg.norm(raw_xy, axis=1) > 1.0 + 1.0e-10)):
        _fail(f"{label} raw target observations are invalid")
    design = np.column_stack(
        (
            np.ones(len(raw_xy)),
            raw_xy[:, 0],
            raw_xy[:, 1],
            raw_xy[:, 0] ** 2,
            raw_xy[:, 0] * raw_xy[:, 1],
            raw_xy[:, 1] ** 2,
        )
    )
    intrinsic = bool(
        len(raw_xy) >= 25
        and np.linalg.matrix_rank(design) == 6
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(smooth))
        and np.allclose(smooth, experiment63.CANONICAL_DESIGN @ coefficients, rtol=0.0, atol=1.0e-9)
    )
    eligible = _strict_bool(
        technical_row.get("distal_qc", technical_row.get("distal_eligible")),
        f"{label}.technical_distal_qc",
    )
    component_gate = (
        _strict_bool(
            technical_row["component_fraction_gate_pass"],
            f"{label}.component_fraction_gate_pass",
        )
        if "component_fraction_gate_pass" in technical_row
        else True
    )
    expected_resolvable = bool(eligible and component_gate and intrinsic)
    if bool(row["target_resolvable"]) != expected_resolvable:
        _fail(f"{label} target_resolvable differs from intrinsic target structure")
    q05 = math.nan
    rmse = math.nan
    if intrinsic:
        if not np.isclose(np.median(raw_thickness), float(row["target_depth_um"]), rtol=0.0, atol=1.0e-9):
            _fail(f"{label} target depth differs from raw median")
        q05 = float(np.quantile(raw_thickness, 0.05))
        if not np.isclose(q05, float(row["target_q05_raw_thickness_um"]), rtol=0.0, atol=1.0e-9):
            _fail(f"{label} target q05 differs from raw target")
        rmse = float(np.sqrt(np.mean((raw_thickness - design @ coefficients) ** 2)))
        if not np.isclose(rmse, float(row["target_rmse_um"]), rtol=0.0, atol=1.0e-9):
            _fail(f"{label} target RMSE differs from raw target")
    target_config = outcome_manifest.get("target_config")
    if not isinstance(target_config, Mapping):
        _fail(f"{label} outcome manifest target_config must be an object")
    q05_floor = float(target_config.get("target_q05_thickness_min_um_exclusive", 0.0))
    rmse_ceiling = float(target_config.get("target_fit_rmse_max_um", 2.5))
    if q05_floor != 0.0 or rmse_ceiling != 2.5:
        _fail(f"{label} outcome target-QC thresholds changed")
    expected_target_qc = bool(
        expected_resolvable
        and math.isfinite(q05)
        and q05 > q05_floor
        and math.isfinite(rmse)
        and rmse <= rmse_ceiling
    )
    if bool(row["target_qc"]) != expected_target_qc:
        _fail(f"{label} target_qc differs from raw q05/RMSE equivalence")
    _validate_target_reason_row(
        row,
        label=label,
        expected_resolvable=expected_resolvable,
        expected_target_qc=expected_target_qc,
        q05=q05,
        rmse=rmse,
    )
    if not arthur:
        proximal = arrays["proximal_points_xyz_um"]
        if proximal.dtype != np.dtype("float64") or proximal.ndim != 2 or proximal.shape[1:] != (3,):
            _fail(f"{label} proximal target points have wrong dtype/shape")
        if not np.all(np.isfinite(proximal)) or len(np.unique(proximal, axis=0)) != len(proximal):
            _fail(f"{label} proximal target points are invalid")
    sealed_relpath = _scalar_text(
        arrays["sealed_distal_relpath"], f"{label}.sealed_distal_relpath"
    )
    expected_sealed_relpath = str(technical_row["sealed_distal_relpath"])
    if sealed_relpath != expected_sealed_relpath:
        _fail(f"{label} binds the wrong sealed distal artifact")
    sealed_binding = technical_bundle.technical_manifest.get(sealed_relpath)
    if sealed_binding is None or _scalar_text(
        arrays["sealed_distal_sha256"], f"{label}.sealed_distal_sha256"
    ) != sealed_binding["sha256"]:
        _fail(f"{label} sealed distal SHA-256 differs from Pass 1")
    if _scalar_text(
        arrays["technical_config_sha256"], f"{label}.technical_config_sha256"
    ) != technical_bundle.technical_config_sha256:
        _fail(f"{label} technical configuration differs from Pass 1")
    outcome_config_text = _scalar_text(
        arrays["outcome_config_json"], f"{label}.outcome_config_json"
    )
    outcome_config_hash = _scalar_text(
        arrays["outcome_config_sha256"], f"{label}.outcome_config_sha256"
    )
    if _sha256_bytes(outcome_config_text.encode("utf-8")) != outcome_config_hash:
        _fail(f"{label} outcome configuration hash is invalid")
    try:
        outcome_config = json.loads(outcome_config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} outcome configuration is invalid JSON") from exc
    if _canonical_json(outcome_config) != outcome_config_text:
        _fail(f"{label} outcome configuration is not canonical JSON")
    if (
        outcome_config != outcome_manifest.get("target_config")
        or outcome_config_hash != outcome_manifest.get("target_config_sha256")
    ):
        _fail(f"{label} outcome configuration differs from its outcome manifest")
    return {
        "raw_target_xy_normalized": raw_xy.copy(),
        "raw_target_thickness_um": raw_thickness.copy(),
    }


def _load_outcome_bundle(
    reader: GuardedArtifactReader,
    technical: TechnicalBundle,
    expected_commit: str,
    producer: Any,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, np.ndarray]]]:
    deferred = technical.outcome_manifest
    payload = reader.read_bound(
        deferred.root,
        deferred.relative_path,
        deferred.binding(),
        phase="pass2",
        purpose=f"{deferred.owner} outcome manifest",
    )
    manifest = _json_object(payload, f"{deferred.owner} outcome manifest")
    arthur = technical.kind == "arthur"
    _exact_keys(
        manifest,
        ARTHUR_OUTCOME_MANIFEST_KEYS if arthur else MAIKE_OUTCOME_MANIFEST_KEYS,
        f"{technical.unit_id} outcome manifest",
    )
    schema = arthur64.OUTCOME_MANIFEST_SCHEMA if arthur else getattr(
        producer, "OUTCOME_MANIFEST_SCHEMA", MAIKE_OUTCOME_SCHEMA
    )
    artifacts = _validate_outcome_manifest_common(
        manifest,
        label=f"{technical.unit_id} outcome manifest",
        schema=schema,
        expected_commit=expected_commit,
        technical_bundle=technical,
    )
    target_relpath = "sealed_outcomes/target_table.csv"
    if target_relpath not in artifacts:
        _fail(f"{technical.unit_id} outcome manifest lacks target table")
    target_table = _csv_table(
        reader.read_bound(
            technical.root,
            target_relpath,
            artifacts[target_relpath],
            phase="pass2",
            purpose=f"{technical.unit_id} target table",
        ),
        f"{technical.unit_id} target table",
    )
    target_fields = ARTHUR_TARGET_FIELDS if arthur else _maike_target_fields(producer)
    _validate_table_header(target_table, target_fields, f"{technical.unit_id} target table")
    if len(target_table) != len(technical.table):
        _fail(f"{technical.unit_id} technical and target row counts differ")
    for column in ("target_resolvable", "target_qc"):
        target_table[column] = [
            _strict_bool(value, f"{technical.unit_id}.{column}") for value in target_table[column]
        ]
    target_table["lens_index"] = [
        _strict_int(value, f"{technical.unit_id}.lens_index", 0) for value in target_table["lens_index"]
    ]
    if not arthur:
        if (
            _strict_int(manifest.get("n_expected"), f"{technical.unit_id}.outcome.n_expected", 1)
            != len(technical.table)
            or _strict_int(manifest.get("n_rows"), f"{technical.unit_id}.outcome.n_rows", 1)
            != len(target_table)
            or manifest.get("contiguous_indices") is not True
            or manifest.get("index_range") != [0, len(target_table) - 1]
        ):
            _fail(f"{technical.unit_id} outcome denominator/contiguity contract failed")
    counts = _require_keys(
        manifest.get("counts"),
        {"target_resolvable", "target_qc"},
        f"{technical.unit_id}.outcome.counts",
    )
    if (
        _strict_int(counts["target_resolvable"], f"{technical.unit_id}.counts.target_resolvable", 0)
        != int(target_table["target_resolvable"].sum())
        or _strict_int(counts["target_qc"], f"{technical.unit_id}.counts.target_qc", 0)
        != int(target_table["target_qc"].sum())
    ):
        _fail(f"{technical.unit_id} outcome counts differ from the target table")
    if arthur and "rows" in counts and _strict_int(
        counts["rows"], f"{technical.unit_id}.counts.rows", 0
    ) != len(target_table):
        _fail(f"{technical.unit_id} outcome row count differs from target table")
    identity_columns = ["volume", "eye_id", "lens_index"] if arthur else ["eye_id", "lens_index"]
    if target_table.duplicated(identity_columns).any():
        _fail(f"{technical.unit_id} target table contains duplicate identities")
    expected_paths = {target_relpath}
    raw_targets: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    target_schema = arthur64.TARGET_SCHEMA if arthur else getattr(producer, "TARGET_SCHEMA", MAIKE_TARGET_SCHEMA)
    target_members = (
        set(arthur64.TARGET_KEYS)
        if arthur
        else set(getattr(producer, "TARGET_KEYS"))
    )
    technical_by_identity = technical.table.set_index(identity_columns, drop=False)
    for row_number, row in target_table.iterrows():
        relpath = str(row["lens_relpath"])
        expected_paths.add(relpath)
        if relpath not in artifacts:
            _fail(f"{technical.unit_id} outcome manifest lacks {relpath}")
        arrays = _npz_arrays(
            reader.read_bound(
                technical.root,
                relpath,
                artifacts[relpath],
                phase="pass2",
                purpose=f"{technical.unit_id} target NPZ row {row_number}",
            ),
            f"{technical.unit_id} target NPZ row {row_number}",
        )
        identity = tuple(row[column] for column in identity_columns)
        try:
            technical_row = technical_by_identity.loc[identity]
        except KeyError as exc:
            raise ContractError(
                f"{technical.unit_id} target identity is absent from Pass 1: {identity}"
            ) from exc
        if isinstance(technical_row, pd.DataFrame):
            _fail(f"{technical.unit_id} Pass 1 has duplicate target identity {identity}")
        raw = _load_target_npz(
            arrays,
            label=f"{technical.unit_id} target NPZ row {row_number}",
            schema=target_schema,
            row=row,
            arthur=arthur,
            technical_row=technical_row,
            expected_members=target_members,
            technical_bundle=technical,
            outcome_manifest=manifest,
        )
        raw_targets[(str(row["eye_id"]), int(row["lens_index"]))] = raw
    if set(artifacts) != expected_paths:
        _fail(f"{technical.unit_id} outcome manifest is not the exact target inventory")
    return target_table, raw_targets


def _join_technical_targets(
    technical: TechnicalBundle,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    arthur = technical.kind == "arthur"
    keys = ["volume", "eye_id", "lens_index"] if arthur else ["eye_id", "lens_index"]
    target_payload_columns = [column for column in targets if column not in keys and column != "lens_relpath"]
    joined = technical.table.merge(
        targets.loc[:, [*keys, *target_payload_columns]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(joined["_merge"]) != {"both"}:
        _fail(f"{technical.unit_id} technical/target identities do not match exactly")
    joined = joined.drop(columns="_merge")
    if "distal_qc" not in joined and "distal_eligible" in joined:
        joined["distal_qc"] = joined["distal_eligible"]
    return joined


def _outcome_availability_report(
    table: pd.DataFrame,
    *,
    label: str,
    denominator_name: str,
) -> dict[str, Any]:
    required = {
        "distal_qc",
        "target_resolvable",
        "target_qc",
        "target_resolvability_reasons",
        "target_qc_reasons",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        _fail(f"{label} lacks outcome-availability fields: {missing}")
    distal = np.asarray(
        [_strict_bool(value, f"{label}.distal_qc") for value in table["distal_qc"]],
        dtype=bool,
    )
    resolvable = np.asarray(
        [
            _strict_bool(value, f"{label}.target_resolvable")
            for value in table["target_resolvable"]
        ],
        dtype=bool,
    )
    target_qc = np.asarray(
        [_strict_bool(value, f"{label}.target_qc") for value in table["target_qc"]],
        dtype=bool,
    )
    if np.any(resolvable & ~distal) or np.any(target_qc & ~resolvable):
        _fail(f"{label} has impossible distal/target cohort nesting")
    primary = distal & resolvable
    unresolved = distal & ~resolvable
    target_qc_excluded = primary & ~target_qc
    return {
        denominator_name: len(table),
        "distal_qc_retained": int(distal.sum()),
        "distal_qc_excluded": int((~distal).sum()),
        "distal_qc_and_target_resolvable": int(primary.sum()),
        "target_unresolvable_within_distal_qc": int(unresolved.sum()),
        "target_unresolvable_reason_counts": _reason_counts(
            table.loc[unresolved],
            "target_resolvability_reasons",
            f"{label} target-unresolvable cohort",
        ),
        "distal_qc_and_target_qc": int((distal & target_qc).sum()),
        "target_qc_excluded_within_primary": int(target_qc_excluded.sum()),
        "target_qc_exclusion_reason_counts": _reason_counts(
            table.loc[target_qc_excluded],
            "target_qc_reasons",
            f"{label} target-QC exclusion cohort",
        ),
    }


def _score_with_loaded_raw_targets(
    table: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    raw_targets: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
) -> pd.DataFrame:
    # This is the unchanged Experiment 63 smoothed-grid scorer.  Passing None
    # is intentional: its old artifact loader knows only Experiment 63 NPZs.
    metrics = experiment63.score_predictions(table, predictions, artifact_roots=None)
    row_lookup = {
        (str(row.eye_id), int(row.lens_index)): position
        for position, row in enumerate(table.itertuples(index=False))
    }
    for method, coefficients in predictions.items():
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (len(table), 6):
            continue
        for identity, position in row_lookup.items():
            raw = raw_targets.get(identity)
            if raw is None:
                _fail(f"Missing already-loaded raw target for {identity}")
            xy = np.asarray(raw["raw_target_xy_normalized"], dtype=np.float64)
            observed = np.asarray(raw["raw_target_thickness_um"], dtype=np.float64)
            if not len(xy):
                continue
            design = np.column_stack(
                (
                    np.ones(len(xy)),
                    xy[:, 0],
                    xy[:, 1],
                    xy[:, 0] ** 2,
                    xy[:, 0] * xy[:, 1],
                    xy[:, 1] ** 2,
                )
            )
            absolute = np.abs(design @ coefficients[position] - observed)
            mask = (metrics["method"] == method) & (metrics["eye_id"].astype(str) == identity[0]) & (
                metrics["lens_index"].astype(int) == identity[1]
            )
            depth = float(table.iloc[position]["target_depth_um"])
            metrics.loc[mask, "raw_unsmoothed_available"] = True
            metrics.loc[mask, "raw_unsmoothed_mae_um"] = float(np.mean(absolute))
            metrics.loc[mask, "raw_unsmoothed_p90_error_um"] = float(np.quantile(absolute, 0.90))
            metrics.loc[mask, "raw_unsmoothed_normalized_mae"] = float(np.mean(absolute) / depth)
    return metrics


def descriptive_10_of_12_result(frozen_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Relabel the unchanged eye-win calculation at Experiment 64's boundary."""

    wins = _strict_int(frozen_summary.get("wins"), "primary summary wins", 0)
    losses = _strict_int(frozen_summary.get("losses"), "primary summary losses", 0)
    ties = _strict_int(frozen_summary.get("ties_nonwins"), "primary summary ties", 0)
    animals = _strict_int(
        frozen_summary.get("n_independent_animals"),
        "primary summary independent animals",
        1,
    )
    if animals != 12 or wins + losses + ties != 12:
        _fail("Descriptive primary summary must partition the twelve named animals")
    return {
        "analysis_label": ANALYSIS_LABEL,
        "inference_label": INFERENCE_LABEL,
        "decision_role": "descriptive_fixed_10_of_12_reference",
        "cohort": frozen_summary["cohort"],
        "independent_unit": frozen_summary["independent_unit"],
        "n_independent_animals": animals,
        "wins": wins,
        "losses": losses,
        "ties_nonwins": ties,
        "rule": "strict shape win in at least 10 of 12 named animals; ties are nonwins",
        "meets_descriptive_10_of_12_rule": wins >= 10,
        "fixed_denominator_two_sided_binomial_reference_at_10_wins": 0.03857421875,
        "observed_fixed_denominator_two_sided_binomial_p": (
            experiment63.two_sided_fixed_denominator_p(wins, 12)
        ),
        "observed_conventional_tie_dropping_two_sided_sign_p": (
            experiment63.conventional_tie_dropping_sign_p(wins, losses)
        ),
        "minimum_effect_threshold_prespecified": False,
        "material_effect_supported_by_direction_rule_alone": False,
        "pristine_external_confirmation": False,
    }


def run_pass2(
    clearance: Pass1Clearance,
    *,
    outcome_attempt: OutcomeAttemptAuthorization | None = None,
    reader: GuardedArtifactReader | None = None,
) -> dict[str, Any]:
    """Open outcomes only after Pass 1 and run the frozen Experiment 63 analyses."""

    _assert_clearance(clearance)
    _assert_outcome_attempt_authorization(clearance, outcome_attempt)
    reader = reader or GuardedArtifactReader()
    producer = _maike_producer_module()

    maike_technical_table = pd.concat(
        [bundle.table for bundle in clearance.maike], ignore_index=True
    )
    maike_technical_exclusions = {
        "aggregate": _maike_technical_exclusion_report(
            maike_technical_table, "consolidated Maike technical inventory"
        ),
        "by_eye": {
            bundle.unit_id: dict(bundle.technical_exclusions)
            for bundle in clearance.maike
        },
    }

    arthur_targets, _ = _load_outcome_bundle(
        reader, clearance.arthur, clearance.expected_commit, producer
    )
    source = _join_technical_targets(clearance.arthur, arthur_targets)
    source = experiment63.validate_lens_table(
        source,
        label="Experiment 64 Arthur source",
        expected_rows=EXPECTED_ARTHUR_ROWS,
        require_volume=True,
    )
    arthur_outcome_availability = {
        "aggregate": _outcome_availability_report(
            source,
            label="Experiment 64 Arthur source",
            denominator_name="fixed_source_denominator",
        ),
        "by_volume": {
            volume: _outcome_availability_report(
                source.loc[source["volume"].astype(str) == volume],
                label=f"Experiment 64 Arthur source volume {volume}",
                denominator_name="fixed_source_denominator",
            )
            for volume in experiment63.ARTHUR_VOLUMES
        },
    }
    maike_tables: list[pd.DataFrame] = []
    raw_targets: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    retention: dict[str, dict[str, Any]] = {}
    for technical in clearance.maike:
        targets, raw = _load_outcome_bundle(
            reader, technical, clearance.expected_commit, producer
        )
        joined = _join_technical_targets(technical, targets)
        joined = experiment63.validate_lens_table(
            joined,
            label=f"Experiment 64 Maike eye {technical.unit_id}",
            expected_eye=technical.unit_id,
            expected_rows=EXPECTED_EYES[technical.unit_id],
        )
        retained = experiment63.require_primary_resolution_gate(
            joined, EXPECTED_EYES[technical.unit_id], technical.unit_id
        )
        eye_retention = _outcome_availability_report(
            joined,
            label=f"Experiment 64 Maike eye {technical.unit_id}",
            denominator_name="fixed_oda_denominator",
        )
        if eye_retention["fixed_oda_denominator"] != EXPECTED_EYES[technical.unit_id]:
            _fail(f"{technical.unit_id} outcome-availability denominator changed")
        if eye_retention["distal_qc_and_target_resolvable"] != retained:
            _fail(f"{technical.unit_id} outcome-availability accounting disagrees with gate")
        retention[technical.unit_id] = eye_retention
        maike_tables.append(joined)
        overlap = set(raw_targets).intersection(raw)
        if overlap:
            _fail(f"Raw target identities overlap across Maike eyes: {sorted(overlap)[:3]}")
        raw_targets.update(raw)
    maike = pd.concat(maike_tables, ignore_index=True)
    if len(maike) != EXPECTED_TOTAL:
        _fail(f"Experiment 64 consolidated Maike table has {len(maike)} rows, not {EXPECTED_TOTAL}")
    target_availability_counts = _outcome_availability_report(
        maike,
        label="consolidated Experiment 64 Maike validation",
        denominator_name="fixed_oda_denominator",
    )
    if target_availability_counts["fixed_oda_denominator"] != EXPECTED_TOTAL:
        _fail("Consolidated Maike outcome-availability denominator changed")
    technical_exclusion_counts = {
        "counting_note": (
            "Retained/excluded lens counts are mutually exclusive within each gate; "
            "reason counts are non-exclusive when one lens has multiple reasons."
        ),
        "arthur_source_predictor_qc": dict(clearance.arthur.technical_exclusions),
        "arthur_source_outcome_availability": arthur_outcome_availability,
        "maike_validation_predictor_qc": maike_technical_exclusions,
        "maike_validation_outcome_availability": {
            "aggregate": target_availability_counts,
            "by_eye": retention,
        },
    }

    fitted = experiment63.run_primary_models(source, maike)
    source_primary = fitted["source_primary"]
    equal_volume_templates = np.vstack(
        [
            np.median(
                experiment63.symmetrized_targets(
                    source_primary[
                        source_primary["volume"].astype(str) == volume
                    ]
                ),
                axis=0,
            )
            for volume in experiment63.ARTHUR_VOLUMES
        ]
    )
    equal_volume_source_template = np.median(equal_volume_templates, axis=0)
    metrics = _score_with_loaded_raw_targets(
        fitted["test_primary"], fitted["predictions"], raw_targets
    )
    per_eye, frozen_summary = experiment63._aggregate_eye_comparison(
        metrics, "cohort_primary"
    )
    descriptive_result = descriptive_10_of_12_result(frozen_summary)
    species_sex_descriptive = experiment63.species_sex_descriptive_table(per_eye)
    per_eye_methods = experiment63.per_eye_method_descriptive_table(
        metrics, "cohort_primary"
    )
    internal_method_summary = experiment63.summarize_internal_methods(per_eye_methods)

    sensitivity = experiment63.run_target_qc_sensitivity_models(source, maike)
    sensitivity_metrics = _score_with_loaded_raw_targets(
        sensitivity["test"], sensitivity["predictions"], raw_targets
    )
    per_eye_sensitivity, sensitivity_nested_summary = experiment63._aggregate_eye_comparison(
        sensitivity_metrics, "cohort_target_qc"
    )
    sensitivity_result = experiment63.nonconfirmatory_nested_summary(
        sensitivity_nested_summary,
        "prespecified_target_qc_sensitivity_nonconfirmatory",
    )

    maike_secondary = experiment63.run_within_maike_nested_loao_secondary(maike)
    maike_secondary_metrics = _score_with_loaded_raw_targets(
        maike_secondary["cohort"], maike_secondary["predictions"], raw_targets
    )
    per_eye_maike_secondary, maike_secondary_result = (
        experiment63.aggregate_within_maike_secondary(maike_secondary_metrics)
    )
    return {
        "source": source,
        "maike": maike,
        "metrics": metrics,
        "per_eye": per_eye,
        "descriptive_result": descriptive_result,
        "species_sex_descriptive": species_sex_descriptive,
        "per_eye_methods": per_eye_methods,
        "internal_method_summary": internal_method_summary,
        "sensitivity_metrics": sensitivity_metrics,
        "per_eye_sensitivity": per_eye_sensitivity,
        "sensitivity_result": sensitivity_result,
        "sensitivity_selected_alphas": sensitivity["selected_alphas"],
        "sensitivity_alpha_audit": sensitivity["alpha_audit"],
        "sensitivity_models": sensitivity["models"],
        "maike_secondary_metrics": maike_secondary_metrics,
        "per_eye_maike_secondary": per_eye_maike_secondary,
        "maike_secondary_result": maike_secondary_result,
        "maike_secondary_selected_alphas": maike_secondary["selected_alphas"],
        "maike_secondary_alpha_audit": maike_secondary["alpha_audit"],
        "maike_secondary_models": maike_secondary["models"],
        "maike_secondary_metric_validity_gate": maike_secondary[
            "metric_validity_gate"
        ],
        "selected_alphas": fitted["selected_alphas"],
        "alpha_audit": fitted["alpha_audit"],
        "models": fitted["models"],
        "equal_volume_source_template": equal_volume_source_template,
        "metric_validity_gate": fitted["metric_validity_gate"],
        "retention": retention,
        "technical_exclusion_counts": technical_exclusion_counts,
        "pass2_reads": tuple(reader.events),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def _file_binding(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}


def _outcome_attempt_directory(maike_root: Path, expected_commit: str) -> Path:
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or expected_commit.lower() != expected_commit
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        _fail("Experiment 64 expected commit must be a lowercase 40-character Git hash")
    return maike_root.resolve().parent / f"experiment64_outcome_attempt_{expected_commit}"


def _clearance_bindings(clearance: Pass1Clearance) -> dict[str, Any]:
    return {
        "maike_attestation_sha256": {
            bundle.unit_id: bundle.qc_binding_sha256 for bundle in clearance.maike
        },
        "maike_technical_completion_sha256": {
            bundle.unit_id: bundle.completion_sha256 for bundle in clearance.maike
        },
        "maike_technical_provenance_sha256": {
            bundle.unit_id: bundle.provenance_sha256 for bundle in clearance.maike
        },
        "arthur_technical_completion_sha256": clearance.arthur.completion_sha256,
        "arthur_technical_provenance_sha256": clearance.arthur.provenance_sha256,
    }


def _outcome_manifest_bindings(clearance: Pass1Clearance) -> dict[str, Any]:
    return {
        "arthur": {
            "relative_path": clearance.arthur.outcome_manifest.relative_path,
            **clearance.arthur.outcome_manifest.binding(),
        },
        "maike": {
            bundle.unit_id: {
                "relative_path": bundle.outcome_manifest.relative_path,
                **bundle.outcome_manifest.binding(),
            }
            for bundle in clearance.maike
        },
    }


def execute_experiment64(
    *,
    repository_root: Path,
    expected_commit: str,
    arthur_root: Path,
    maike_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Execute one exclusive, atomically published Experiment 64 run."""

    output_directory = output_directory.resolve()
    frozen_execution_environment = experiment63.execution_environment()
    backend_relpath = (
        "experiments/maike-modern-ground-truth/experiment_64_two_pass_backend.py"
    )
    backend_binding = {
        "relative_path": backend_relpath,
        **_file_binding(Path(__file__).resolve()),
    }
    method_contract = {
        "ridge_alphas": experiment63.RIDGE_ALPHAS.tolist(),
        "source_weighting": "equal volume mass; all sample weights sum to one",
        "alpha_selection": (
            "Arthur-only leave-one-volume-out; equal mean of three held-out "
            "volume median 81-point normalized MAEs"
        ),
        "target_observation_contract": experiment63.TARGET_OBSERVATION_CONTRACT,
        "predictor_input_representation_contract": (
            experiment63.PREDICTOR_INPUT_REPRESENTATION_CONTRACT
        ),
        "trained_target_indices": experiment63.EVEN_TARGET_INDICES.tolist(),
        "zero_predicted_target_indices": experiment63.ODD_REFLECTION_INDICES.tolist(),
        "canonical_grid_xy": experiment63.CANONICAL_GRID_XY.tolist(),
    }
    attempt_directory = _outcome_attempt_directory(maike_root, expected_commit)
    if output_directory.exists():
        _fail(f"Exclusive Experiment 64 output already exists: {output_directory}")
    if attempt_directory.exists():
        _fail(
            "Experiment 64 outcome attempt already exists and cannot be retried: "
            f"{attempt_directory}"
        )
    reader = GuardedArtifactReader()
    clearance = run_pass1(
        repository_root=repository_root,
        expected_commit=expected_commit,
        arthur_root=arthur_root,
        maike_root=maike_root,
        reader=reader,
    )
    # Recheck the frozen tree after the complete technical pass.  The atomic
    # attempt directory is then created before Pass 2 and deliberately survives
    # every later exception, including a crash after the first outcome read.
    experiment63.require_frozen_git(repository_root, expected_commit)
    try:
        attempt_directory.mkdir()
    except FileExistsError as exc:
        raise ContractError(
            "Experiment 64 outcome attempt was claimed concurrently and cannot be retried: "
            f"{attempt_directory}"
        ) from exc
    attempt_record = attempt_directory / "attempt.json"
    attempt_payload = {
        "schema_version": OUTCOME_ATTEMPT_SCHEMA,
        "status": "pass2_attempt_committed_before_outcome_access",
        "analysis_label": ANALYSIS_LABEL,
        "inference_label": INFERENCE_LABEL,
        "git": {"commit": expected_commit, "dirty": False},
        "robust_core_config_sha256": ROBUST_CORE_CONFIG_SHA256,
        "arthur_root": str(arthur_root.resolve()),
        "maike_root": str(maike_root.resolve()),
        "output_directory": str(output_directory),
        "rerun_permitted": False,
        "outcome_artifacts_opened_at_record_creation": False,
        "pass1_clearance_bindings": _clearance_bindings(clearance),
        "outcome_manifest_bindings": _outcome_manifest_bindings(clearance),
        "attempted_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(attempt_record, attempt_payload)
    attempt_binding = {
        "directory": str(attempt_directory),
        "record": {
            "relative_path": "attempt.json",
            **_file_binding(attempt_record),
        },
    }
    outcome_authorization = OutcomeAttemptAuthorization(
        expected_commit=expected_commit,
        record_path=attempt_record,
        record_sha256=attempt_binding["record"]["sha256"],
        record_size_bytes=attempt_binding["record"]["size_bytes"],
        _authority=_OUTCOME_ATTEMPT_AUTHORITY,
    )
    analysis = run_pass2(
        clearance,
        outcome_attempt=outcome_authorization,
        reader=reader,
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent.resolve(),
        )
    )
    try:
        _write_csv(staging / "per_lens_primary_metrics.csv", analysis["metrics"])
        _write_csv(
            staging / "per_lens_target_qc_sensitivity_metrics.csv",
            analysis["sensitivity_metrics"],
        )
        _write_csv(
            staging / "per_lens_within_maike_nested_loao_secondary_metrics.csv",
            analysis["maike_secondary_metrics"],
        )
        _write_csv(staging / "per_eye_descriptive_10_of_12.csv", analysis["per_eye"])
        _write_csv(
            staging / "per_eye_internal_method_descriptive.csv",
            analysis["per_eye_methods"],
        )
        _write_csv(
            staging / "species_sex_descriptive.csv",
            analysis["species_sex_descriptive"],
        )
        _write_csv(
            staging / "per_eye_target_qc_sensitivity.csv",
            analysis["per_eye_sensitivity"],
        )
        _write_csv(
            staging / "per_eye_within_maike_nested_loao_secondary.csv",
            analysis["per_eye_maike_secondary"],
        )
        primary_alpha_audit = analysis["alpha_audit"].copy()
        primary_alpha_audit.insert(0, "analysis", "primary")
        sensitivity_alpha_audit = analysis["sensitivity_alpha_audit"].copy()
        sensitivity_alpha_audit.insert(0, "analysis", "target_qc_sensitivity")
        _write_csv(
            staging / "source_only_alpha_selection.csv",
            pd.concat(
                [primary_alpha_audit, sensitivity_alpha_audit], ignore_index=True
            ),
        )
        _write_csv(
            staging / "within_maike_nested_loao_alpha_selection.csv",
            analysis["maike_secondary_alpha_audit"],
        )
        _write_json(
            staging / "primary_result.json",
            {
                "schema_version": RUN_SCHEMA,
                "status": RUN_STATUS,
                "analysis_label": ANALYSIS_LABEL,
                "inference_label": INFERENCE_LABEL,
                "descriptive_10_of_12": analysis["descriptive_result"],
                "target_qc_sensitivity": analysis["sensitivity_result"],
                "within_maike_nested_loao_secondary": analysis[
                    "maike_secondary_result"
                ],
                "internal_method_descriptive": analysis[
                    "internal_method_summary"
                ],
                "metric_validity_gate": analysis["metric_validity_gate"],
                "within_maike_secondary_metric_validity_gate": analysis[
                    "maike_secondary_metric_validity_gate"
                ],
                "retention": analysis["retention"],
                "technical_exclusion_counts": analysis[
                    "technical_exclusion_counts"
                ],
                "selected_alphas": {
                    "primary": analysis["selected_alphas"],
                    "target_qc_sensitivity": analysis[
                        "sensitivity_selected_alphas"
                    ],
                    "within_maike_nested_loao_secondary": analysis[
                        "maike_secondary_selected_alphas"
                    ],
                },
                "primary_cohort": "distal_qc AND target_resolvable",
                "sensitivity_cohort": (
                    "distal_qc AND target_qc (models reselected and refit)"
                ),
                "minimum_effect_claim": None,
                "minimum_effect_threshold_prespecified": False,
                "material_effect_supported_by_direction_rule_alone": False,
                "outcome_attempt": attempt_binding,
                "backend": backend_binding,
                "execution_environment": frozen_execution_environment,
                "method_contract": method_contract,
                "representation_caveat": REPRESENTATION_CAVEAT,
                "created_utc": created_utc,
            },
        )
        model_payload: dict[str, np.ndarray] = {}
        for model_name, model in analysis["models"].items():
            model_payload[f"{model_name}_feature_mean"] = model.feature_mean
            model_payload[f"{model_name}_feature_scale"] = model.feature_scale
            model_payload[f"{model_name}_target_mean_even"] = model.target_mean
            model_payload[f"{model_name}_coefficients"] = model.coefficients
            model_payload[f"{model_name}_alpha"] = np.asarray(model.alpha, dtype=np.float64)
        model_payload["equal_volume_source_template_coefficients_c0_c5"] = np.asarray(
            analysis["equal_volume_source_template"], dtype=np.float64
        )
        for model_name, model in analysis["sensitivity_models"].items():
            prefix = f"target_qc_sensitivity_{model_name}"
            model_payload[f"{prefix}_feature_mean"] = model.feature_mean
            model_payload[f"{prefix}_feature_scale"] = model.feature_scale
            model_payload[f"{prefix}_target_mean_even"] = model.target_mean
            model_payload[f"{prefix}_coefficients"] = model.coefficients
            model_payload[f"{prefix}_alpha"] = np.asarray(
                model.alpha, dtype=np.float64
            )
        for outer_eye, outer_models in analysis["maike_secondary_models"].items():
            for model_name, model in outer_models.items():
                prefix = f"within_maike_loao_{outer_eye}_{model_name}"
                model_payload[f"{prefix}_feature_mean"] = model.feature_mean
                model_payload[f"{prefix}_feature_scale"] = model.feature_scale
                model_payload[f"{prefix}_target_mean_even"] = model.target_mean
                model_payload[f"{prefix}_coefficients"] = model.coefficients
                model_payload[f"{prefix}_alpha"] = np.asarray(
                    model.alpha, dtype=np.float64
                )
        np.savez(staging / "frozen_model_parameters.npz", **model_payload)
        output_manifest = {
            path.name: _file_binding(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        sealed = {
            "schema_version": RUN_SCHEMA,
            "status": RUN_STATUS,
            "analysis_label": ANALYSIS_LABEL,
            "inference_label": INFERENCE_LABEL,
            "pristine_external_confirmation": False,
            "git": {"commit": expected_commit, "dirty": False},
            "backend": backend_binding,
            "execution_environment": frozen_execution_environment,
            "robust_core_config": robust64.ROBUST_CORE_CONFIG,
            "robust_core_config_sha256": ROBUST_CORE_CONFIG_SHA256,
            "expected_eye_counts": EXPECTED_EYES,
            "expected_total_rows": EXPECTED_TOTAL,
            "pass1": {
                "schema_version": PASS1_SCHEMA,
                "all_twelve_maike_technical_qc_gates_passed_before_arthur_read": True,
                "outcome_artifacts_opened": False,
                **_clearance_bindings(clearance),
            },
            "outcome_attempt": attempt_binding,
            "outcome_manifest_bindings": _outcome_manifest_bindings(clearance),
            "control_features": list(experiment63.CONTROL_FEATURE_NAMES),
            "shape_additions": list(experiment63.SHAPE_ADDITION_NAMES),
            "method_contract": method_contract,
            "decision_rule": analysis["descriptive_result"],
            "technical_exclusion_counts": analysis["technical_exclusion_counts"],
            "retention_by_eye": analysis["retention"],
            "target_qc_sensitivity": analysis["sensitivity_result"],
            "internal_method_descriptive": analysis["internal_method_summary"],
            "within_maike_nested_loao_secondary_contract": dict(
                experiment63.SECONDARY_MAIKE_LOAO_CONTRACT
            ),
            "within_maike_nested_loao_secondary_result": analysis[
                "maike_secondary_result"
            ],
            "within_maike_secondary_metric_validity_gate": analysis[
                "maike_secondary_metric_validity_gate"
            ],
            "selected_alphas": {
                "primary": analysis["selected_alphas"],
                "target_qc_sensitivity": analysis[
                    "sensitivity_selected_alphas"
                ],
                "within_maike_nested_loao_secondary": analysis[
                    "maike_secondary_selected_alphas"
                ],
            },
            "evaluation_scope": "retained distal_qc AND target_resolvable cohort",
            "minimum_effect_claim": None,
            "minimum_effect_threshold_prespecified": False,
            "material_effect_supported_by_direction_rule_alone": False,
            "representation_caveat": REPRESENTATION_CAVEAT,
            "created_utc": created_utc,
            "output_manifest": output_manifest,
        }
        _write_json(staging / "sealed_run_manifest.json", sealed)
        seal_hash = _file_binding(staging / "sealed_run_manifest.json")["sha256"]
        (staging / "SEALED.sha256").write_text(
            f"{seal_hash}  sealed_run_manifest.json\n", encoding="ascii"
        )
        experiment63.require_frozen_git(repository_root, expected_commit)
        os.replace(staging, output_directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_directory": str(output_directory),
        **analysis["descriptive_result"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--arthur-root", type=Path, required=True)
    parser.add_argument("--maike-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--execute-sealed-first-experiment64",
        action="store_true",
        help="Required acknowledgement that this opens outcomes only after Pass 1.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not arguments.execute_sealed_first_experiment64:
        print(
            "Refusing to inspect Experiment 64 outcomes without "
            "--execute-sealed-first-experiment64.",
            file=sys.stderr,
        )
        return 2
    try:
        result = execute_experiment64(
            repository_root=arguments.repo,
            expected_commit=arguments.expected_commit,
            arthur_root=arguments.arthur_root,
            maike_root=arguments.maike_root,
            output_directory=arguments.output_directory,
        )
    except (ContractError, experiment63.ContractError) as exc:
        print(f"CONTRACT FAILURE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

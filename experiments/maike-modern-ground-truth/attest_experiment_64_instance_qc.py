#!/usr/bin/env python3
"""Attest one Experiment 64 disjoint visual-QC sample without opening outcomes.

The attester independently verifies the committed Experiment 63 stop record,
all twelve prior sample-manifest hashes, the exact 384-case exclusion ledger,
the target-free technical inventory, all technical artifact hashes, the
deterministic 32-case draw, its render hashes, and the human/AI-assisted review.
It does not open the separately bound sealed-outcome manifest or any artifact
below ``sealed_outcomes/``.  A failed or indeterminate decision produces no
attestation and requires the whole Experiment 64 run to stop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from render_experiment_64_instance_qc_sample import (
    EXPECTED_GLOBAL_EXCLUSIONS,
    EXPECTED_PRIOR_EYES,
    EXPECTED_PRIOR_PER_EYE,
    EXPECTED_SAMPLE_SIZE,
    EXPECTED_TECHNICAL_COHERENCE_CONFIG,
    EXCLUSION_LEDGER_RELPATH,
    EXCLUSION_SCHEMA_VERSION,
    INSTANCE_ACCESSED_MEMBERS,
    INSTANCE_MEMBERS,
    N_PER_CELL,
    N_RADIAL_STRATA,
    N_SCALE_STRATA,
    SAMPLE_DIRECTORY_NAME,
    SAMPLING_ALGORITHM,
    SEALED_CORE_ACCESSED_MEMBERS,
    SEALED_CORE_MEMBERS,
    SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    STOP_RECORD_RELPATH,
    TECHNICAL_BUNDLE_SCHEMA_VERSION,
    QCError,
    _artifact_binding,
    _load_json,
    _sha256_path,
    build_development_exclusion_ledger,
    exclude_development_identities,
    read_and_validate_producer_documents,
    read_sampling_table,
    select_frozen_sample,
    verify_committed_development_exclusion_ledger,
)


ATTESTATION_SCHEMA_VERSION = "experiment64.instance-qc-attestation.v1"
REVIEW_SCHEMA_VERSION = "experiment64.instance-qc-review.v1"
ATTESTATION_FILENAME = "experiment64_instance_qc_attestation.json"
REVIEW_FILENAME = "experiment64_instance_qc_review.json"
TECHNICAL_INVENTORY_FILENAME = "technical_inventory.csv"
REVIEW_SCOPES = frozenset({"disjoint_stratified_sample_only"})
REVIEW_MODES = frozenset({"human", "ai_assisted_visual_review_without_model_outputs"})
REVIEW_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "eye_id",
        "review_scope",
        "review_mode",
        "reviewer_id",
        "reviewed_at_utc",
        "sample_manifest_sha256",
        "decisions",
    }
)
REVIEW_DECISION_FIELDS = frozenset({"lens_index", "seed_id", "decision", "notes"})
HASH_ENTRY_FIELDS = frozenset({"sha256", "size_bytes"})
SEALED_OUTCOME_BINDING_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
FORBIDDEN_TECHNICAL_PATH_PARTS = frozenset(
    {
        "lenses",
        "sealed_outcomes",
        "targets",
        "target",
        "predictions",
        "prediction",
        "errors",
        "error",
        "models",
        "model",
    }
)
FORBIDDEN_INVENTORY_FIELD_TOKENS = (
    "proximal",
    "target",
    "prediction",
    "predicted",
    "model",
    "outcome",
)
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

FIT_SUPPORT_MINIMUM = int(EXPECTED_TECHNICAL_COHERENCE_CONFIG["fit_support_minimum"])
FIT_RMSE_MAX_UM = float(EXPECTED_TECHNICAL_COHERENCE_CONFIG["fit_rmse_max_um"])
FIT_26_LCC_FRACTION_MINIMUM = float(
    EXPECTED_TECHNICAL_COHERENCE_CONFIG["fit_largest_component_fraction_min"]
)
FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM = float(
    EXPECTED_TECHNICAL_COHERENCE_CONFIG["fit_abs_residual_p99_over_scale_max"]
)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, description: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise QCError(
            f"{description} fields are not exact: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonnegative_int(value: Any, *, description: str) -> int:
    if type(value) is not int or value < 0:
        raise QCError(f"{description} must be a nonnegative JSON integer")
    return value


def _validate_relpath(value: Any, *, description: str) -> str:
    if not isinstance(value, str):
        raise QCError(f"{description} must be a string")
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise QCError(f"Unsafe or noncanonical path in {description}: {value!r}")
    return value


def _validate_hash(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise QCError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _validate_hash_entry(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QCError(f"{description} must be an object")
    _exact_keys(value, HASH_ENTRY_FIELDS, description=description)
    return {
        "sha256": _validate_hash(value["sha256"], description=f"{description}.sha256"),
        "size_bytes": _require_nonnegative_int(
            value["size_bytes"], description=f"{description}.size_bytes"
        ),
    }


def _expected_technical_artifacts(n_expected: int) -> set[str]:
    expected = {
        TECHNICAL_INVENTORY_FILENAME,
        "distal_qc_sampling.csv",
        "distal_frame_audit.json",
    }
    for lens_index in range(n_expected):
        filename = f"lens_{lens_index:06d}.npz"
        expected.add(f"instances/{filename}")
        expected.add(f"sealed_distal/{filename}")
    return expected


def _validate_sealed_outcome_binding(value: Any, *, description: str) -> dict[str, Any]:
    """Validate metadata only; deliberately do not resolve or open its path."""

    if not isinstance(value, dict):
        raise QCError(f"{description} must be an object")
    _exact_keys(value, SEALED_OUTCOME_BINDING_FIELDS, description=description)
    relative_path = _validate_relpath(value["relative_path"], description=f"{description}.relative_path")
    if relative_path != "sealed_outcomes/manifest.json":
        raise QCError(f"{description} must bind exactly sealed_outcomes/manifest.json")
    return {
        "relative_path": relative_path,
        "sha256": _validate_hash(value["sha256"], description=f"{description}.sha256"),
        "size_bytes": _require_nonnegative_int(
            value["size_bytes"], description=f"{description}.size_bytes"
        ),
    }


def verify_target_free_technical_bundle(
    *, bundle_root: Path, eye_id: str
) -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """Verify all and only the target-free producer artifacts."""

    root = bundle_root.resolve()
    completion, provenance, _, _ = read_and_validate_producer_documents(
        bundle_root=root, eye_id=eye_id
    )
    n_expected = _require_nonnegative_int(
        completion.get("n_expected"), description="technical completion n_expected"
    )
    n_rows = _require_nonnegative_int(
        completion.get("n_rows"), description="technical completion n_rows"
    )
    if n_expected <= 0 or n_rows != n_expected:
        raise QCError("Technical completion has an incomplete row inventory")
    if provenance.get("n_expected") != n_expected or provenance.get("n_rows") != n_rows:
        raise QCError("Technical completion/provenance row counts disagree")

    manifests: list[dict[str, dict[str, Any]]] = []
    for name, document in (("completion", completion), ("provenance", provenance)):
        raw_manifest = document.get("technical_output_manifest")
        if not isinstance(raw_manifest, dict):
            raise QCError(f"Technical {name} has no technical_output_manifest object")
        validated: dict[str, dict[str, Any]] = {}
        for raw_path, raw_entry in raw_manifest.items():
            relpath = _validate_relpath(raw_path, description=f"technical {name} manifest path")
            pure = PurePosixPath(relpath)
            lowered_parts = {part.lower() for part in pure.parts}
            if lowered_parts & FORBIDDEN_TECHNICAL_PATH_PARTS:
                raise QCError(f"Forbidden outcome-bearing path in technical manifest: {relpath}")
            validated[relpath] = _validate_hash_entry(
                raw_entry, description=f"technical {name} manifest entry {relpath}"
            )
        manifests.append(validated)
    if manifests[0] != manifests[1]:
        raise QCError("Technical completion/provenance output manifests disagree")
    manifest = manifests[0]
    expected = _expected_technical_artifacts(n_expected)
    if set(manifest) != expected:
        raise QCError(
            "Technical output manifest is not the exact sealed-outcome-free "
            "technical inventory; "
            f"missing={sorted(expected - set(manifest))[:5]}, "
            f"extra={sorted(set(manifest) - expected)[:5]}"
        )

    completion_outcome = _validate_sealed_outcome_binding(
        completion.get("sealed_outcome_manifest_binding"),
        description="technical completion sealed-outcome binding",
    )
    provenance_outcome = _validate_sealed_outcome_binding(
        provenance.get("sealed_outcome_manifest_binding"),
        description="technical provenance sealed-outcome binding",
    )
    if completion_outcome != provenance_outcome:
        raise QCError("Technical completion/provenance sealed-outcome bindings disagree")

    for relpath, expected_binding in sorted(manifest.items()):
        path = (root / Path(*PurePosixPath(relpath).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise QCError(f"Technical artifact escapes bundle root: {relpath}") from exc
        if not path.is_file():
            raise QCError(f"Missing technical artifact: {relpath}")
        if path.stat().st_size != expected_binding["size_bytes"]:
            raise QCError(f"Technical artifact size mismatch: {relpath}")
        if _sha256_path(path) != expected_binding["sha256"]:
            raise QCError(f"Technical artifact SHA-256 mismatch: {relpath}")
    return manifest, n_expected, completion, provenance


def _parse_literal_bool(value: str, *, description: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise QCError(f"{description} must be literal true or false")


def _parse_csv_int(value: str, *, description: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise QCError(f"{description} must be an integer") from exc
    if parsed < 0:
        raise QCError(f"{description} must be nonnegative")
    return parsed


def _parse_csv_finite_float(value: str, *, description: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise QCError(f"{description} must be numeric") from exc
    if not math.isfinite(parsed):
        raise QCError(f"{description} must be finite")
    return parsed


def verify_technical_inventory(
    *, inventory_path: Path, eye_id: str, n_expected: int, sampling_rows: Sequence[Any]
) -> dict[str, Any]:
    """Check the sealed-outcome-free inventory and its agreement with sampling."""

    sampling_by_index = {row.lens_index: row for row in sampling_rows}
    inventory_rows: list[dict[str, Any]] = []
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields != TECHNICAL_INVENTORY_FIELDS:
            raise QCError(
                "Technical inventory header is not the exact target-free allowlist; "
                f"expected {TECHNICAL_INVENTORY_FIELDS}, got {fields}"
            )
        if any(
            token in field.lower()
            for field in fields
            for token in FORBIDDEN_INVENTORY_FIELD_TOKENS
        ):
            raise QCError("Technical inventory contains an outcome/proximal-bearing field name")
        for row_number, raw in enumerate(reader, start=2):
            if raw["eye_id"] != eye_id:
                raise QCError(f"Technical inventory row {row_number} has wrong eye_id")
            lens_index = _parse_csv_int(
                raw["lens_index"], description=f"technical inventory row {row_number} lens_index"
            )
            eligible = _parse_literal_bool(
                raw["distal_qc"], description=f"technical inventory row {row_number} distal_qc"
            )
            sampling = sampling_by_index.get(lens_index)
            if sampling is None:
                raise QCError(f"Technical inventory row {row_number} is absent from sampling")
            if (
                raw["seed_id"] != sampling.seed_id
                or eligible != sampling.distal_eligible
                or raw["instance_relpath"] != sampling.instance_relpath
                or raw["sealed_distal_relpath"] != sampling.sealed_distal_relpath
            ):
                raise QCError(f"Technical inventory/sampling disagreement at lens {lens_index}")
            gate_pass = _parse_literal_bool(
                raw["maike_final_fit_gate_pass"],
                description=f"technical inventory row {row_number} maike_final_fit_gate_pass",
            )
            if eligible:
                support = _parse_csv_int(
                    raw["distal_fit_support"],
                    description=f"technical inventory row {row_number} distal_fit_support",
                )
                rmse = _parse_csv_finite_float(
                    raw["distal_fit_rmse_um"],
                    description=f"technical inventory row {row_number} distal_fit_rmse_um",
                )
                p99_norm = _parse_csv_finite_float(
                    raw["distal_fit_p99_residual_over_scale"],
                    description=(
                        f"technical inventory row {row_number} "
                        "distal_fit_p99_residual_over_scale"
                    ),
                )
                component_count = _parse_csv_int(
                    raw["distal_fit_26_component_count"],
                    description=(
                        f"technical inventory row {row_number} distal_fit_26_component_count"
                    ),
                )
                largest_support = _parse_csv_int(
                    raw["distal_fit_26_largest_component_support"],
                    description=(
                        f"technical inventory row {row_number} "
                        "distal_fit_26_largest_component_support"
                    ),
                )
                largest_fraction = _parse_csv_finite_float(
                    raw["distal_fit_26_largest_component_fraction"],
                    description=(
                        f"technical inventory row {row_number} "
                        "distal_fit_26_largest_component_fraction"
                    ),
                )
                margin = _parse_csv_finite_float(
                    raw["coherence_margin"],
                    description=f"technical inventory row {row_number} coherence_margin",
                )
                position_u = _parse_csv_finite_float(
                    raw["position_u_um"],
                    description=f"technical inventory row {row_number} position_u_um",
                )
                position_v = _parse_csv_finite_float(
                    raw["position_v_um"],
                    description=f"technical inventory row {row_number} position_v_um",
                )
                scale = _parse_csv_finite_float(
                    raw["distal_scale_um"],
                    description=f"technical inventory row {row_number} distal_scale_um",
                )
                if scale <= 0:
                    raise QCError(f"Technical inventory row {row_number} has nonpositive scale")
                p99_abs = _parse_csv_finite_float(
                    raw["distal_abs_residual_p99_um"],
                    description=(
                        f"technical inventory row {row_number} distal_abs_residual_p99_um"
                    ),
                )
                if not math.isclose(
                    p99_norm, p99_abs / scale, rel_tol=1.0e-12, abs_tol=1.0e-12
                ):
                    raise QCError(f"Technical inventory row {row_number} has inconsistent p99/scale")
                if (
                    sampling.position_u_um != position_u
                    or sampling.position_v_um != position_v
                    or sampling.distal_scale_um != scale
                ):
                    raise QCError(f"Technical inventory/sampling geometry mismatch at lens {lens_index}")
                if support < 1 or component_count < 1 or largest_support > support:
                    raise QCError(f"Technical inventory row {row_number} has invalid 26-component counts")
                if not math.isclose(
                    largest_fraction,
                    largest_support / support,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise QCError(f"Technical inventory row {row_number} has inconsistent 26-LCC fraction")
                expected_margin = min(
                    (support - FIT_SUPPORT_MINIMUM) / FIT_SUPPORT_MINIMUM,
                    (FIT_RMSE_MAX_UM - rmse) / FIT_RMSE_MAX_UM,
                    (largest_fraction - FIT_26_LCC_FRACTION_MINIMUM)
                    / (1.0 - FIT_26_LCC_FRACTION_MINIMUM),
                    (FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM - p99_norm)
                    / FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM,
                )
                if not math.isclose(margin, expected_margin, rel_tol=1.0e-12, abs_tol=1.0e-12):
                    raise QCError(f"Technical inventory row {row_number} has wrong coherence_margin")
                expected_named_margins = {
                    "coherence_support_margin": (support - FIT_SUPPORT_MINIMUM)
                    / FIT_SUPPORT_MINIMUM,
                    "coherence_rmse_margin": (FIT_RMSE_MAX_UM - rmse) / FIT_RMSE_MAX_UM,
                    "coherence_lcc_margin": (
                        largest_fraction - FIT_26_LCC_FRACTION_MINIMUM
                    )
                    / (1.0 - FIT_26_LCC_FRACTION_MINIMUM),
                    "coherence_p99_over_scale_margin": (
                        FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM - p99_norm
                    )
                    / FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM,
                }
                for field, expected_value in expected_named_margins.items():
                    stored_value = _parse_csv_finite_float(
                        raw[field],
                        description=f"technical inventory row {row_number} {field}",
                    )
                    if not math.isclose(
                        stored_value, expected_value, rel_tol=1.0e-12, abs_tol=1.0e-12
                    ):
                        raise QCError(
                            f"Technical inventory row {row_number} has inconsistent {field}"
                        )
                gates_pass = (
                    support >= FIT_SUPPORT_MINIMUM
                    and rmse <= FIT_RMSE_MAX_UM
                    and largest_fraction >= FIT_26_LCC_FRACTION_MINIMUM
                    and p99_norm <= FIT_P99_RESIDUAL_OVER_SCALE_MAXIMUM
                )
                if not gates_pass or gate_pass is not True or raw["maike_final_fit_gate_reasons"] != "":
                    raise QCError(f"Eligible technical inventory row {row_number} fails a frozen final-fit gate")
                if sampling.coherence_margin is None or not math.isclose(
                    margin, sampling.coherence_margin, rel_tol=0.0, abs_tol=0.0
                ):
                    raise QCError(f"Technical inventory/sampling coherence mismatch at lens {lens_index}")
            else:
                margin = None
                if gate_pass is not False or not raw["maike_final_fit_gate_reasons"]:
                    raise QCError(
                        f"Ineligible technical inventory row {row_number} needs a failing gate reason"
                    )
                if any(
                    value is not None
                    for value in (
                        sampling.position_u_um,
                        sampling.position_v_um,
                        sampling.distal_scale_um,
                        sampling.coherence_margin,
                    )
                ):
                    raise QCError(f"Ineligible sampling row {row_number} exposes geometry values")
            inventory_rows.append(
                {
                    "lens_index": lens_index,
                    "seed_id": raw["seed_id"],
                    "distal_eligible": eligible,
                    "coherence_margin": margin,
                }
            )
    if len(inventory_rows) != n_expected:
        raise QCError("Technical inventory row count does not equal n_expected")
    if [row["lens_index"] for row in inventory_rows] != list(range(n_expected)):
        raise QCError("Technical inventory must be contiguous and ordered by lens_index")
    if len(sampling_rows) != n_expected:
        raise QCError("Sampling row count does not equal the complete technical inventory")
    return {
        "n_expected": n_expected,
        "n_inventory_rows": len(inventory_rows),
        "n_distal_qc_eligible": sum(row["distal_eligible"] for row in inventory_rows),
        "target_free_header": list(fields),
        "contiguous_indices": True,
        "sampling_inventory_agreement": True,
    }


def _verify_binding(
    value: Any,
    *,
    expected_path: Path,
    relative_to: Path,
    description: str,
    extra_fields: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(value, dict):
        raise QCError(f"{description} binding must be an object")
    expected_fields = frozenset({"relative_path", "sha256", "size_bytes"}) | extra_fields
    if frozenset(value) != expected_fields:
        raise QCError(f"{description} binding fields are not exact")
    expected = _artifact_binding(expected_path, relative_to=relative_to)
    for field in ("relative_path", "sha256", "size_bytes"):
        if value.get(field) != expected[field]:
            raise QCError(f"{description} binding mismatch for {field}")


def verify_sample_manifest(
    *,
    bundle_root: Path,
    sample_manifest_path: Path,
    repository_root: Path,
    stop_record_path: Path,
    prior_sample_manifests: Mapping[str, Path],
    technical_manifest: Mapping[str, Mapping[str, Any]],
    exclusion_ledger_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = bundle_root.resolve()
    expected_path = root / SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
    path = sample_manifest_path.resolve()
    if path != expected_path or not path.is_file():
        raise QCError(f"Sample manifest must be exactly {expected_path}")
    sample = _load_json(path, description="Experiment 64 sample manifest")
    eye_id = sample.get("eye_id")
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION or sample.get("experiment") != 64:
        raise QCError("Wrong Experiment 64 sample schema or experiment number")
    if not isinstance(eye_id, str) or not eye_id:
        raise QCError("Sample manifest has an invalid eye_id")
    if sample.get("sampling_algorithm") != SAMPLING_ALGORITHM:
        raise QCError("Wrong Experiment 64 sampling algorithm")
    if sample.get("review_scope") != "disjoint_stratified_sample_only":
        raise QCError("Wrong Experiment 64 review scope")
    if sample.get("all_instances_manually_reviewed") is not False:
        raise QCError("Sample must not claim that all instances were reviewed")
    if sample.get("n_selected") != EXPECTED_SAMPLE_SIZE:
        raise QCError("Experiment 64 sample must contain exactly 32 cases")
    if sample.get("selected_development_overlap_count") != 0:
        raise QCError("Experiment 64 sample declares development overlap")

    completion, provenance, robust_config, robust_hash = read_and_validate_producer_documents(
        bundle_root=root, eye_id=eye_id
    )
    _verify_binding(
        sample.get("technical_completion"),
        expected_path=root / "technical_completion.json",
        relative_to=root,
        description="technical completion",
    )
    _verify_binding(
        sample.get("technical_provenance"),
        expected_path=root / "technical_provenance.json",
        relative_to=root,
        description="technical provenance",
    )
    if sample.get("robust_core_config") != robust_config or sample.get("robust_core_config_sha256") != robust_hash:
        raise QCError("Sample manifest robust-core configuration binding is stale")

    rows, n_rows = read_sampling_table(root / "distal_qc_sampling.csv", eye_id=eye_id)
    _verify_binding(
        sample.get("sampling_table"),
        expected_path=root / "distal_qc_sampling.csv",
        relative_to=root,
        description="sampling table",
    )
    if n_rows != completion.get("n_expected") or provenance.get("n_expected") != n_rows:
        raise QCError("Sample/bundle inventory counts disagree")

    expected_ledger, exclusions = build_development_exclusion_ledger(
        repository_root=repository_root,
        stop_record_path=stop_record_path,
        prior_sample_manifests=prior_sample_manifests,
    )
    canonical_ledger_path = exclusion_ledger_path or (
        repository_root / Path(*EXCLUSION_LEDGER_RELPATH.parts)
    )
    committed_ledger_binding = verify_committed_development_exclusion_ledger(
        repository_root=repository_root,
        ledger_path=canonical_ledger_path,
        expected_ledger=expected_ledger,
    )
    if sample.get("development_exclusion_ledger") != committed_ledger_binding:
        raise QCError("Sample does not bind the current committed exclusion ledger")
    if expected_ledger.get("schema_version") != EXCLUSION_SCHEMA_VERSION:
        raise QCError("Development exclusion ledger has the wrong schema")
    if expected_ledger.get("n_eye_scoped_exclusions") != EXPECTED_GLOBAL_EXCLUSIONS:
        raise QCError("Development exclusion ledger does not contain 384 cases")
    if sample.get("experiment63_stop_record") != expected_ledger["experiment63_stop_record"]:
        raise QCError("Sample stop-record binding differs from the verified committed record")
    prior_verification = sample.get("prior_sample_manifest_verification")
    expected_prior = [
        {
            "eye_id": item["eye_id"],
            "sha256": item["old_sample_manifest_sha256"],
            "n_verified_identities": item["n_excluded"],
        }
        for item in expected_ledger["eyes"]
    ]
    if prior_verification != expected_prior:
        raise QCError("Sample does not bind all twelve verified prior sample manifests")

    unseen, eligible_removed = exclude_development_identities(
        rows, exclusions[eye_id], eye_id=eye_id
    )
    expected_selected = select_frozen_sample(unseen)
    samples = sample.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_SIZE:
        raise QCError("Sample entries must be a 32-element array")
    for ordinal, (entry, selected) in enumerate(zip(samples, expected_selected, strict=True)):
        if not isinstance(entry, dict) or entry.get("ordinal") != ordinal:
            raise QCError("Sample ordinals must be contiguous and ordered")
        exact_values = {
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
        if any(entry.get(key) != value for key, value in exact_values.items()):
            raise QCError(f"Sample entry {ordinal} does not match deterministic reselection")
        instance_path = root / selected.row.instance_relpath
        core_path = root / selected.row.sealed_distal_relpath
        instance_binding = entry.get("instance_artifact")
        _verify_binding(
            instance_binding,
            expected_path=instance_path,
            relative_to=root,
            description=f"sample {ordinal} instance",
            extra_fields=frozenset({"members_present", "members_accessed"}),
        )
        core_binding = entry.get("sealed_robust_core_artifact")
        _verify_binding(
            core_binding,
            expected_path=core_path,
            relative_to=root,
            description=f"sample {ordinal} robust core",
            extra_fields=frozenset({"members_present", "members_accessed"}),
        )
        if (
            instance_binding.get("members_present") != sorted(INSTANCE_MEMBERS)
            or instance_binding.get("members_accessed") != list(INSTANCE_ACCESSED_MEMBERS)
        ):
            raise QCError(f"Sample {ordinal} instance member-access disclosure is invalid")
        if (
            core_binding.get("members_present") != sorted(SEALED_CORE_MEMBERS)
            or core_binding.get("members_accessed") != list(SEALED_CORE_ACCESSED_MEMBERS)
        ):
            raise QCError(f"Sample {ordinal} robust-core member-access disclosure is invalid")
        if technical_manifest[selected.row.instance_relpath]["sha256"] != _sha256_path(instance_path):
            raise QCError(f"Sample {ordinal} instance differs from technical manifest")
        if technical_manifest[selected.row.sealed_distal_relpath]["sha256"] != _sha256_path(core_path):
            raise QCError(f"Sample {ordinal} robust core differs from technical manifest")
        render = entry.get("render")
        if not isinstance(render, dict):
            raise QCError(f"Sample {ordinal} render binding is missing")
        render_relpath = _validate_relpath(render.get("relative_path"), description=f"sample {ordinal} render")
        expected_render = root / SAMPLE_DIRECTORY_NAME / render_relpath
        _verify_binding(
            render,
            expected_path=expected_render,
            relative_to=root / SAMPLE_DIRECTORY_NAME,
            description=f"sample {ordinal} render",
        )
        expected_render_name = (
            f"sample_{ordinal:02d}_r{selected.radial_stratum}_s{selected.scale_stratum}_"
            f"lens_{selected.row.lens_index:06d}.png"
        )
        if (
            expected_render.suffix.lower() != ".png"
            or expected_render.parent.name != "renders"
            or expected_render.name != expected_render_name
        ):
            raise QCError(f"Sample {ordinal} render path is not an allowlisted PNG")

    if sample.get("n_development_identities_for_eye") != EXPECTED_PRIOR_PER_EYE:
        raise QCError("Sample does not record exactly 32 development identities for this eye")
    if sample.get("n_eligible_development_identities_removed") != eligible_removed:
        raise QCError("Sample eligible development-removal count is wrong")
    if sample.get("n_unseen_distal_qc_eligible") != len(unseen):
        raise QCError("Sample unseen eligible count is wrong")
    expected_cells = {
        f"r{radial}_s{scale}": N_PER_CELL
        for radial in range(N_RADIAL_STRATA)
        for scale in range(N_SCALE_STRATA)
    }
    if sample.get("cell_counts") != expected_cells:
        raise QCError("Sample does not have exactly two cases in every 4x4 cell")
    if sample.get("selection_role_counts") != {
        "near_worst_coherence": 16,
        "hash_minimal_remaining": 16,
    }:
        raise QCError("Sample does not have one near-worst and one hash-min case per cell")
    disclosure = sample.get("visual_disclosure")
    if not isinstance(disclosure, dict) or disclosure.get("proximal_anatomy_blind") is not False:
        raise QCError("Sample does not disclose that the full-body render is not proximal-anatomy blind")
    sequestration = sample.get("outcome_sequestration")
    if (
        not isinstance(sequestration, dict)
        or sequestration.get("target_proximal_prediction_error_model_or_sealed_outcome_opened") is not False
    ):
        raise QCError("Sample does not attest zero forbidden outcome access")
    renderer_path = Path(__file__).resolve().with_name(
        "render_experiment_64_instance_qc_sample.py"
    )
    _verify_binding(
        sample.get("renderer_code"),
        expected_path=renderer_path,
        relative_to=renderer_path.parents[2],
        description="renderer code",
    )
    return sample, samples, expected_ledger


def verify_review(
    *, review_path: Path, eye_id: str, sample_manifest_path: Path, samples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review = _load_json(review_path, description="Experiment 64 QC review")
    _exact_keys(review, REVIEW_TOP_LEVEL_FIELDS, description="review")
    if review["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise QCError("Wrong Experiment 64 review schema")
    if review["eye_id"] != eye_id or review["review_scope"] not in REVIEW_SCOPES:
        raise QCError("Review eye_id or scope does not match the sample")
    if review["review_mode"] not in REVIEW_MODES:
        raise QCError("Unsupported Experiment 64 review mode")
    if not isinstance(review["reviewer_id"], str) or not review["reviewer_id"].strip():
        raise QCError("Review reviewer_id must be nonempty")
    if not isinstance(review["reviewed_at_utc"], str) or not UTC_RE.fullmatch(review["reviewed_at_utc"]):
        raise QCError("Review reviewed_at_utc must be an ISO UTC timestamp ending in Z")
    if review["sample_manifest_sha256"] != _sha256_path(sample_manifest_path):
        raise QCError("Review is not bound to the current sample manifest")
    decisions = review["decisions"]
    if not isinstance(decisions, list) or len(decisions) != EXPECTED_SAMPLE_SIZE:
        raise QCError("Review must contain exactly 32 decisions; no replacement is permitted")
    validated: list[dict[str, Any]] = []
    for ordinal, (decision, sample) in enumerate(zip(decisions, samples, strict=True)):
        if not isinstance(decision, dict):
            raise QCError(f"Review decision {ordinal} must be an object")
        _exact_keys(decision, REVIEW_DECISION_FIELDS, description=f"review decision {ordinal}")
        if decision["lens_index"] != sample["lens_index"] or decision["seed_id"] != sample["seed_id"]:
            raise QCError(f"Review decision {ordinal} does not match the fixed sample order")
        verdict = decision["decision"]
        if verdict not in {"pass", "fail", "indeterminate"}:
            raise QCError(f"Review decision {ordinal} has an invalid verdict")
        if not isinstance(decision["notes"], str):
            raise QCError(f"Review decision {ordinal} notes must be a string")
        if verdict != "pass":
            raise QCError(
                f"Review decision {ordinal} for lens {decision['lens_index']} is {verdict}; "
                "the whole Experiment 64 run must stop and no attestation may be issued"
            )
        validated.append(dict(decision))
    return review, validated


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise QCError(f"Refusing to overwrite existing attestation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise QCError(f"Refusing to overwrite existing attestation: {path}")
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def attest_eye(
    *,
    bundle_root: Path,
    sample_manifest_path: Path,
    review_path: Path,
    output_path: Path,
    repository_root: Path,
    stop_record_path: Path,
    prior_sample_manifests: Mapping[str, Path],
    exclusion_ledger_path: Path | None = None,
) -> Path:
    """Issue one immutable passing eye attestation or fail closed."""

    root = bundle_root.resolve()
    canonical_ledger_path = (
        exclusion_ledger_path
        or (repository_root / Path(*EXCLUSION_LEDGER_RELPATH.parts))
    ).resolve()
    output = output_path.resolve()
    if output != root / ATTESTATION_FILENAME:
        raise QCError(f"Attestation output must be exactly {root / ATTESTATION_FILENAME}")
    if output.exists():
        raise QCError(f"Refusing to overwrite existing attestation: {output}")
    sample_preview = _load_json(sample_manifest_path, description="Experiment 64 sample manifest")
    eye_id = sample_preview.get("eye_id")
    if not isinstance(eye_id, str) or not eye_id:
        raise QCError("Sample manifest has no valid eye_id")

    technical_manifest, n_expected, completion, provenance = verify_target_free_technical_bundle(
        bundle_root=root, eye_id=eye_id
    )
    sampling_rows, _ = read_sampling_table(root / "distal_qc_sampling.csv", eye_id=eye_id)
    inventory = verify_technical_inventory(
        inventory_path=root / TECHNICAL_INVENTORY_FILENAME,
        eye_id=eye_id,
        n_expected=n_expected,
        sampling_rows=sampling_rows,
    )
    sample, samples, ledger = verify_sample_manifest(
        bundle_root=root,
        sample_manifest_path=sample_manifest_path,
        repository_root=repository_root,
        stop_record_path=stop_record_path,
        prior_sample_manifests=prior_sample_manifests,
        technical_manifest=technical_manifest,
        exclusion_ledger_path=canonical_ledger_path,
    )
    expected_review_path = root / REVIEW_FILENAME
    if review_path.resolve() != expected_review_path:
        raise QCError(f"Review input must be exactly {expected_review_path}")
    review, decisions = verify_review(
        review_path=review_path,
        eye_id=eye_id,
        sample_manifest_path=sample_manifest_path,
        samples=samples,
    )

    document = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "experiment": 64,
        "status": "passed",
        "eye_id": eye_id,
        "review_scope": "disjoint_stratified_sample_only",
        "all_instances_manually_reviewed": False,
        "stratified_sample_visual_qc_passed": True,
        "whole_run_stop_rule_satisfied_for_this_eye": True,
        "technical_inventory_complete": True,
        "target_free_artifact_hash_verification_passed": True,
        "development_exclusion_verification": {
            "experiment63_stop_record_committed": True,
            "prior_sample_manifests_verified": EXPECTED_PRIOR_EYES,
            "eye_scoped_identities_verified": EXPECTED_GLOBAL_EXCLUSIONS,
            "identities_for_this_eye": EXPECTED_PRIOR_PER_EYE,
            "selected_overlap_count": 0,
            "exclusion_before_new_strata": True,
        },
        "technical_inventory": inventory,
        "sample_coverage": sample["cell_counts"],
        "selection_role_counts": sample["selection_role_counts"],
        "reviewed_samples": [
            {
                "ordinal": ordinal,
                "lens_index": decision["lens_index"],
                "seed_id": decision["seed_id"],
                "decision": decision["decision"],
            }
            for ordinal, decision in enumerate(decisions)
        ],
        "outcome_sequestration": {
            "technical_pass_only": True,
            "sealed_outcome_manifest_binding_validated_but_file_not_opened": True,
            "target_proximal_prediction_error_model_or_sealed_outcome_opened": False,
            "model_and_error_blind": True,
            "proximal_anatomy_blind": False,
        },
        "bindings": {
            "technical_completion": _artifact_binding(root / "technical_completion.json", relative_to=root),
            "technical_provenance": _artifact_binding(root / "technical_provenance.json", relative_to=root),
            "technical_inventory": _artifact_binding(root / TECHNICAL_INVENTORY_FILENAME, relative_to=root),
            "sampling_table": _artifact_binding(root / "distal_qc_sampling.csv", relative_to=root),
            "sample_manifest": _artifact_binding(sample_manifest_path, relative_to=root),
            "development_exclusion_ledger": sample["development_exclusion_ledger"],
            "review": _artifact_binding(review_path, relative_to=root),
            "experiment63_stop_record": ledger["experiment63_stop_record"],
            "sealed_outcome_manifest": completion["sealed_outcome_manifest_binding"],
        },
        "verified_target_free_technical_artifact_count": len(technical_manifest),
        "review": {
            "review_mode": review["review_mode"],
            "reviewer_id": review["reviewer_id"],
            "reviewed_at_utc": review["reviewed_at_utc"],
        },
        "attester_code": _artifact_binding(Path(__file__).resolve()),
        "producer_document_schemas": {
            "completion": completion["schema_version"],
            "provenance": provenance["schema_version"],
            "expected": TECHNICAL_BUNDLE_SCHEMA_VERSION,
        },
    }
    _atomic_write_new_json(output, document)
    return output


def _parse_prior_argument(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise QCError("--prior-sample-manifest must use EYE_ID=/path/sample_manifest.json")
        eye_id, raw_path = value.split("=", 1)
        if not eye_id or eye_id in result or not raw_path:
            raise QCError("Duplicate or malformed --prior-sample-manifest argument")
        result[eye_id] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment63-stop-record", type=Path)
    parser.add_argument("--development-exclusion-ledger", type=Path)
    parser.add_argument("--prior-sample-manifest", action="append", default=[], metavar="EYE_ID=PATH")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stop = args.experiment63_stop_record or (
        args.repository_root / Path(*STOP_RECORD_RELPATH.parts)
    )
    try:
        prior = _parse_prior_argument(args.prior_sample_manifest)
        result = attest_eye(
            bundle_root=args.bundle_root,
            sample_manifest_path=args.sample_manifest
            or (args.bundle_root / SAMPLE_DIRECTORY_NAME / "sample_manifest.json"),
            review_path=args.review or (args.bundle_root / REVIEW_FILENAME),
            output_path=args.output or (args.bundle_root / ATTESTATION_FILENAME),
            repository_root=args.repository_root,
            stop_record_path=stop,
            prior_sample_manifests=prior,
            exclusion_ledger_path=args.development_exclusion_ledger,
        )
    except QCError as exc:
        print(f"Experiment 64 QC attestation failed: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

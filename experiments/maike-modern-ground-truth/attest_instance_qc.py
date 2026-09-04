#!/usr/bin/env python3
"""Issue an immutable Experiment 63 stratified instance-QC attestation.

The extraction bundle remains unchanged and continues to state
``instance_segmentation_validated=false``.  This tool verifies the complete
automatic inventory and every artifact hash, then records only that the frozen
32-lens stratified visual sample passed review.  It never claims that every
instance was manually inspected.
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

import numpy as np

from render_instance_qc_sample import (
    EXPECTED_SAMPLE_SIZE,
    FORBIDDEN_INPUT_ROLES,
    INSTANCE_ACCESSED_MEMBERS,
    INSTANCE_MEMBERS,
    N_PER_CELL,
    N_RADIAL_STRATA,
    N_SCALE_STRATA,
    SAMPLING_ALGORITHM,
    SAMPLING_FIELDS,
    SAMPLE_DIRECTORY_NAME,
    SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    SEALED_DISTAL_ACCESSED_MEMBERS,
    SEALED_DISTAL_MEMBERS,
    QCError,
    read_sampling_table,
    select_frozen_sample,
)


ATTESTATION_SCHEMA_VERSION = "experiment63.instance-qc-attestation.v1"
REVIEW_SCHEMA_VERSION = "experiment63.instance-qc-review.v1"
BUNDLE_SCHEMA_VERSION = "experiment63-eye-bundle-v2"
ATTESTATION_FILENAME = "instance_qc_attestation.json"
REVIEW_FILENAME = "instance_qc_review.json"
REVIEW_SCOPES = frozenset({"stratified_sample_only"})
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
SUMMARY_REQUIRED_FIELDS = frozenset(
    {
        "lens_index",
        "seed_id",
        "assignment_status",
        "full_assigned_size",
        "main_component_size",
        "component_removed_size",
        "main_component_fraction",
        "distal_eligible",
    }
)
PARTITION_FIELDS = frozenset(
    {
        "source_foreground_voxel_count",
        "assigned_voxel_count",
        "assigned_unique_voxel_count",
        "unassigned_foreground_voxel_count",
        "multiply_assigned_voxel_count",
        "exact_partition",
        "candidate_seeds_per_voxel",
    }
)
HASH_ENTRY_FIELDS = frozenset({"sha256", "size_bytes"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
OUTCOME_BLINDNESS_FIELDS = frozenset(
    {
        "sampling_table_exact_field_allowlist",
        "instance_members_accessed",
        "sealed_distal_members_accessed",
        "forbidden_input_roles",
        "fitted_lens_npz_opened",
        "proximal_target_prediction_error_or_model_data_opened",
    }
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    try:
        relpath = path.relative_to(relative_to).as_posix() if relative_to else path.name
    except ValueError as exc:
        raise QCError(f"Bound artifact escapes declared root: {path}") from exc
    return {
        "relative_path": relpath,
        "sha256": _sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QCError(f"Cannot read valid {description} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QCError(f"{description} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, description: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise QCError(
            f"{description} fields are not exact: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_bool(value: Any, *, description: str) -> bool:
    if type(value) is not bool:
        raise QCError(f"{description} must be a JSON boolean")
    return value


def _require_nonnegative_int(value: Any, *, description: str) -> int:
    if type(value) is not int or value < 0:
        raise QCError(f"{description} must be a nonnegative JSON integer")
    return value


def _validate_relpath(value: str) -> str:
    if not isinstance(value, str):
        raise QCError("Manifest relative path must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise QCError(f"Unsafe or noncanonical manifest path: {value!r}")
    return value


def _validate_hash_entry(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QCError(f"Manifest entry for {path} must be an object")
    _exact_keys(value, HASH_ENTRY_FIELDS, description=f"manifest entry {path}")
    sha = value["sha256"]
    size = value["size_bytes"]
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise QCError(f"Manifest entry for {path} has invalid sha256")
    _require_nonnegative_int(size, description=f"manifest size for {path}")
    return {"sha256": sha, "size_bytes": size}


def _validate_output_manifest(value: Any, *, description: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise QCError(f"{description} output_manifest must be an object")
    result: dict[str, dict[str, Any]] = {}
    for raw_path, entry in value.items():
        path = _validate_relpath(raw_path)
        if path in result:
            raise QCError(f"Duplicate manifest path: {path}")
        result[path] = _validate_hash_entry(entry, path=path)
    return result


def _expected_data_artifacts(n_expected: int) -> set[str]:
    expected = {"lens_summary.csv", "distal_qc_sampling.csv", "distal_frame_audit.json"}
    for index in range(n_expected):
        name = f"lens_{index:06d}.npz"
        expected.add(f"instances/{name}")
        expected.add(f"sealed_distal/{name}")
        expected.add(f"lenses/{name}")
    return expected


def _verify_bundle_manifests(
    bundle_root: Path, completion: Mapping[str, Any], provenance: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], int]:
    for name, document in (("completion", completion), ("provenance", provenance)):
        if document.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise QCError(f"Wrong {name} schema_version: {document.get('schema_version')!r}")
        if document.get("status") != "complete":
            raise QCError(f"{name} status must be complete")
        if document.get("contiguous_indices") is not True:
            raise QCError(f"{name} must attest contiguous_indices=true")
        if document.get("instance_segmentation_validated") is not False:
            raise QCError(
                f"{name} must preserve instance_segmentation_validated=false"
            )
        if "output_manifest" not in document:
            raise QCError(f"{name} is missing output_manifest")

    n_expected = _require_nonnegative_int(completion.get("n_expected"), description="completion n_expected")
    n_rows = _require_nonnegative_int(completion.get("n_rows"), description="completion n_rows")
    if n_expected <= 0 or n_rows != n_expected:
        raise QCError(f"Incomplete extraction inventory: n_expected={n_expected}, n_rows={n_rows}")
    if provenance.get("n_expected") != n_expected or provenance.get("n_rows") != n_rows:
        raise QCError("Completion/provenance row counts disagree")
    if completion.get("eye_id") != provenance.get("eye_id"):
        raise QCError("Completion/provenance eye_id disagree")
    provenance_manifest = _validate_output_manifest(
        provenance["output_manifest"], description="provenance"
    )
    completion_manifest = _validate_output_manifest(
        completion["output_manifest"], description="completion"
    )
    expected_data = _expected_data_artifacts(n_expected)
    if set(provenance_manifest) != expected_data:
        raise QCError(
            "Provenance manifest is not the exact complete data-artifact inventory: "
            f"missing={sorted(expected_data - set(provenance_manifest))[:5]}, "
            f"extra={sorted(set(provenance_manifest) - expected_data)[:5]}"
        )
    if set(completion_manifest) != expected_data:
        raise QCError("Completion manifest is not the exact complete data-artifact inventory")
    if completion_manifest != provenance_manifest:
        raise QCError("Completion/provenance output manifests must be identical")
    for path in expected_data:
        if completion_manifest[path] != provenance_manifest[path]:
            raise QCError(f"Completion/provenance manifest binding differs for {path}")

    verified = 0
    for relpath, expected in sorted(completion_manifest.items()):
        path = (bundle_root / relpath).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise QCError(f"Manifest artifact escapes bundle: {relpath}") from exc
        if not path.is_file():
            raise QCError(f"Manifest artifact is missing: {relpath}")
        if path.stat().st_size != expected["size_bytes"]:
            raise QCError(f"Artifact size mismatch: {relpath}")
        if _sha256_path(path) != expected["sha256"]:
            raise QCError(f"Artifact SHA-256 mismatch: {relpath}")
        verified += 1
    return completion_manifest, verified


def _partition_evidence(document: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    value = document.get("partition_evidence")
    if not isinstance(value, dict):
        raise QCError(f"{description} partition_evidence must be an object")
    _exact_keys(value, PARTITION_FIELDS, description=f"{description} partition_evidence")
    result: dict[str, Any] = {}
    for field in PARTITION_FIELDS - {"exact_partition"}:
        result[field] = _require_nonnegative_int(
            value[field], description=f"{description} partition_evidence.{field}"
        )
    result["exact_partition"] = _require_bool(
        value["exact_partition"], description=f"{description} partition_evidence.exact_partition"
    )
    return result


def _verify_partition(
    completion: Mapping[str, Any], provenance: Mapping[str, Any], *, summed_assigned_points: int
) -> dict[str, Any]:
    evidence = _partition_evidence(completion, description="completion")
    if evidence != _partition_evidence(provenance, description="provenance"):
        raise QCError("Completion/provenance partition evidence disagree")
    required = (
        evidence["exact_partition"] is True
        and evidence["candidate_seeds_per_voxel"] == 1
        and evidence["multiply_assigned_voxel_count"] == 0
        and evidence["unassigned_foreground_voxel_count"] == 0
        and evidence["assigned_voxel_count"] == evidence["assigned_unique_voxel_count"]
        and evidence["assigned_unique_voxel_count"] == evidence["source_foreground_voxel_count"]
        and summed_assigned_points == evidence["assigned_voxel_count"]
    )
    if not required:
        raise QCError(
            "Extraction does not provide a one-candidate, lossless, nonoverlapping partition "
            "consistent with instance point counts"
        )
    return evidence


def _parse_csv_int(value: str, *, field: str, row_number: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise QCError(f"Summary row {row_number}: {field} is not an integer") from exc
    if result < 0:
        raise QCError(f"Summary row {row_number}: {field} is negative")
    return result


def _parse_csv_bool(value: str, *, field: str, row_number: int) -> bool:
    if value not in {"true", "false"}:
        raise QCError(f"Summary row {row_number}: {field} must be literal true/false")
    return value == "true"


def _read_summary(path: Path, *, n_expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = frozenset(reader.fieldnames or ())
        if not SUMMARY_REQUIRED_FIELDS <= fields:
            raise QCError(f"lens_summary.csv is missing fields {sorted(SUMMARY_REQUIRED_FIELDS - fields)}")
        for row_number, raw in enumerate(reader, start=2):
            index = _parse_csv_int(raw["lens_index"], field="lens_index", row_number=row_number)
            full_size = _parse_csv_int(
                raw["full_assigned_size"], field="full_assigned_size", row_number=row_number
            )
            main_size = _parse_csv_int(
                raw["main_component_size"], field="main_component_size", row_number=row_number
            )
            removed = _parse_csv_int(
                raw["component_removed_size"], field="component_removed_size", row_number=row_number
            )
            status = raw["assignment_status"]
            if status not in {"ok", "empty_assignment"}:
                raise QCError(f"Summary row {row_number}: unknown assignment_status {status!r}")
            if main_size > full_size or removed != full_size - main_size:
                raise QCError(f"Summary row {row_number}: inconsistent component sizes")
            if (status == "empty_assignment") != (full_size == 0):
                raise QCError(f"Summary row {row_number}: empty_assignment/status mismatch")
            fraction_text = raw["main_component_fraction"]
            if full_size:
                try:
                    fraction = float(fraction_text)
                except ValueError as exc:
                    raise QCError(f"Summary row {row_number}: invalid main_component_fraction") from exc
                if not math.isfinite(fraction) or not math.isclose(
                    fraction, main_size / full_size, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise QCError(f"Summary row {row_number}: main_component_fraction mismatch")
            elif fraction_text not in {"", "nan", "NaN"}:
                try:
                    if float(fraction_text) != 0.0:
                        raise QCError(
                            f"Summary row {row_number}: empty assignment component fraction must be 0 or NaN"
                        )
                except ValueError as exc:
                    raise QCError(f"Summary row {row_number}: invalid empty component fraction") from exc
            rows.append(
                {
                    "lens_index": index,
                    "seed_id": raw["seed_id"],
                    "assignment_status": status,
                    "full_assigned_size": full_size,
                    "main_component_size": main_size,
                    "component_removed_size": removed,
                    "distal_eligible": _parse_csv_bool(
                        raw["distal_eligible"], field="distal_eligible", row_number=row_number
                    ),
                }
            )
    if len(rows) != n_expected or [row["lens_index"] for row in rows] != list(range(n_expected)):
        raise QCError("lens_summary.csv must contain every expected lens exactly once in index order")
    seed_ids = [row["seed_id"] for row in rows]
    if any(not seed_id for seed_id in seed_ids) or len(set(seed_ids)) != n_expected:
        raise QCError("lens_summary.csv seed_id values must be nonempty and unique")
    return rows


def _load_instance_counts(path: Path, *, lens_index: int) -> tuple[int, int]:
    """Open only the Stage-1 instance archive, never the fitted lens archive."""

    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != INSTANCE_MEMBERS:
            raise QCError(f"Instance {lens_index} has unexpected NPZ members")
        schema = str(np.asarray(archive["schema_version"]).item())
        index = int(np.asarray(archive["lens_index"]).item())
        full = np.asarray(archive["full_assigned_points_zyx"])
        main = np.asarray(archive["main_component_points_zyx"])
    if schema != "experiment63.instance.v2" or index != lens_index:
        raise QCError(f"Instance {lens_index} schema/index mismatch")
    for name, array in (("full", full), ("main", main)):
        if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iu":
            raise QCError(f"Instance {lens_index} {name} points must be integer N x 3")
    return int(len(full)), int(len(main))


def _verify_technical_inventory(
    bundle_root: Path,
    *,
    eye_id: str,
    n_expected: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    summary_rows = _read_summary(bundle_root / "lens_summary.csv", n_expected=n_expected)
    eligible_rows, sampling_n = read_sampling_table(
        bundle_root / "distal_qc_sampling.csv", eye_id=eye_id
    )
    if sampling_n != n_expected:
        raise QCError("distal_qc_sampling.csv is not the complete all-seed inventory")
    sampling_seed_map: dict[int, str] = {}
    with (bundle_root / "distal_qc_sampling.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLING_FIELDS:
            raise QCError("distal_qc_sampling.csv header changed during verification")
        for raw in reader:
            sampling_seed_map[int(raw["lens_index"])] = raw["seed_id"]
    eligible_indices = {row.lens_index for row in eligible_rows}
    summed_full = 0
    n_empty = 0
    for row in summary_rows:
        index = row["lens_index"]
        if sampling_seed_map.get(index) != row["seed_id"]:
            raise QCError(f"Seed identity mismatch between inventories at lens {index}")
        if (index in eligible_indices) != row["distal_eligible"]:
            raise QCError(f"Distal eligibility mismatch between inventories at lens {index}")
        instance_path = bundle_root / "instances" / f"lens_{index:06d}.npz"
        full_count, main_count = _load_instance_counts(instance_path, lens_index=index)
        if full_count != row["full_assigned_size"] or main_count != row["main_component_size"]:
            raise QCError(f"Instance point counts disagree with summary at lens {index}")
        summed_full += full_count
        n_empty += int(full_count == 0)
    return (
        {
            "complete": True,
            "n_expected": n_expected,
            "n_inventory_rows": len(summary_rows),
            "lens_indices_complete": True,
            "seed_ids_unique": True,
            "one_unique_instance_artifact_per_row_including_empty": True,
            "n_empty_assignment_rows": n_empty,
            "empty_assignment_rows_permitted": True,
            "artifact_point_counts_match_summary": True,
            "n_distal_qc_eligible": len(eligible_rows),
        },
        summary_rows,
        summed_full,
    )


def _verify_sample_manifest(
    *,
    sample_manifest_path: Path,
    bundle_root: Path,
    eye_id: str,
    completion_manifest: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(sample_manifest_path, description="sample manifest")
    if manifest.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise QCError("Wrong sample-manifest schema")
    if manifest.get("eye_id") != eye_id:
        raise QCError("Sample-manifest eye_id mismatch")
    if manifest.get("review_scope") != "stratified_sample_only":
        raise QCError("Sample manifest must state stratified_sample_only")
    if manifest.get("all_instances_manually_reviewed") is not False:
        raise QCError("Sample manifest must not claim all instances were reviewed")
    if manifest.get("sampling_algorithm") != SAMPLING_ALGORITHM:
        raise QCError("Wrong sample-selection algorithm")
    if manifest.get("n_selected") != EXPECTED_SAMPLE_SIZE:
        raise QCError("Sample manifest must contain exactly 32 selected lenses")
    expected_cells = {
        f"r{radial}_s{scale}": N_PER_CELL
        for radial in range(N_RADIAL_STRATA)
        for scale in range(N_SCALE_STRATA)
    }
    if manifest.get("cell_counts") != expected_cells:
        raise QCError("Sample manifest does not have exact 2-per-cell coverage")

    blindness = manifest.get("outcome_blindness")
    if not isinstance(blindness, dict):
        raise QCError("Sample manifest lacks outcome_blindness evidence")
    _exact_keys(blindness, OUTCOME_BLINDNESS_FIELDS, description="outcome_blindness")
    if (
        blindness["sampling_table_exact_field_allowlist"] != list(SAMPLING_FIELDS)
        or blindness["instance_members_accessed"] != list(INSTANCE_ACCESSED_MEMBERS)
        or blindness["sealed_distal_members_accessed"]
        != list(SEALED_DISTAL_ACCESSED_MEMBERS)
        or blindness["forbidden_input_roles"] != list(FORBIDDEN_INPUT_ROLES)
        or blindness["fitted_lens_npz_opened"] is not False
        or blindness["proximal_target_prediction_error_or_model_data_opened"] is not False
    ):
        raise QCError("Sample manifest outcome-blindness evidence is not the frozen allowlist")

    sampling_binding = manifest.get("sampling_table")
    if not isinstance(sampling_binding, dict):
        raise QCError("Sample manifest is missing its sampling-table binding")
    actual_sampling = _binding(bundle_root / "distal_qc_sampling.csv", relative_to=bundle_root)
    if sampling_binding != actual_sampling:
        raise QCError("Sample manifest sampling-table binding is stale")
    completion_sampling = completion_manifest["distal_qc_sampling.csv"]
    if {
        "sha256": sampling_binding["sha256"],
        "size_bytes": sampling_binding["size_bytes"],
    } != completion_sampling:
        raise QCError("Sample manifest does not bind the extracted sampling table")

    eligible, n_sampling_rows = read_sampling_table(
        bundle_root / "distal_qc_sampling.csv", eye_id=eye_id
    )
    if (
        manifest.get("n_inventory_rows") != n_sampling_rows
        or manifest.get("n_distal_qc_eligible") != len(eligible)
    ):
        raise QCError("Sample manifest population counts do not match the frozen sampling table")
    expected_selection = select_frozen_sample(eligible)
    expected_identity = [
        (
            selected.row.lens_index,
            selected.row.seed_id,
            selected.radial_stratum,
            selected.scale_stratum,
            selected.selection_sha256,
        )
        for selected in expected_selection
    ]
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_SIZE:
        raise QCError("Sample manifest samples must be a 32-item array")
    actual_identity: list[tuple[Any, ...]] = []
    sample_root = sample_manifest_path.parent.resolve()
    for ordinal, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("ordinal") != ordinal:
            raise QCError("Sample manifest ordinals must be complete and ordered")
        actual_identity.append(
            (
                sample.get("lens_index"),
                sample.get("seed_id"),
                sample.get("radial_stratum"),
                sample.get("scale_stratum"),
                sample.get("selection_sha256"),
            )
        )
        for key, prefix in (
            ("instance_artifact", "instances"),
            ("sealed_distal_artifact", "sealed_distal"),
        ):
            binding = sample.get(key)
            if not isinstance(binding, dict):
                raise QCError(f"Sample {ordinal} missing {key}")
            relpath = binding.get("relative_path")
            if relpath not in completion_manifest or not isinstance(relpath, str):
                raise QCError(f"Sample {ordinal} has unbound {key}")
            if not relpath.startswith(prefix + "/"):
                raise QCError(f"Sample {ordinal} {key} has forbidden role/path")
            if {
                "sha256": binding.get("sha256"),
                "size_bytes": binding.get("size_bytes"),
            } != completion_manifest[relpath]:
                raise QCError(f"Sample {ordinal} {key} binding differs from extraction")
            expected_present = sorted(
                INSTANCE_MEMBERS if key == "instance_artifact" else SEALED_DISTAL_MEMBERS
            )
            expected_accessed = list(
                INSTANCE_ACCESSED_MEMBERS
                if key == "instance_artifact"
                else SEALED_DISTAL_ACCESSED_MEMBERS
            )
            if (
                binding.get("members_present") != expected_present
                or binding.get("members_accessed") != expected_accessed
            ):
                raise QCError(f"Sample {ordinal} {key} violates the array-member allowlist")
        render = sample.get("render")
        if not isinstance(render, dict):
            raise QCError(f"Sample {ordinal} missing render binding")
        render_relpath = _validate_relpath(render.get("relative_path"))
        if not render_relpath.startswith("renders/"):
            raise QCError(f"Sample {ordinal} render path is outside renders/")
        render_path = (sample_root / render_relpath).resolve()
        try:
            render_path.relative_to(sample_root / "renders")
        except ValueError as exc:
            raise QCError(f"Sample {ordinal} render escapes render root") from exc
        if not render_path.is_file():
            raise QCError(f"Sample {ordinal} render is missing")
        if (
            render_path.stat().st_size != render.get("size_bytes")
            or _sha256_path(render_path) != render.get("sha256")
        ):
            raise QCError(f"Sample {ordinal} render hash/size mismatch")
    if actual_identity != expected_identity:
        raise QCError("Sample manifest is not the exact frozen deterministic selection")

    renderer_binding = manifest.get("renderer_code")
    if not isinstance(renderer_binding, dict):
        raise QCError("Sample manifest lacks renderer code binding")
    renderer_path = Path(__file__).resolve().with_name("render_instance_qc_sample.py")
    repository_root = renderer_path.parents[2]
    expected_renderer = {
        "relative_path": renderer_path.relative_to(repository_root).as_posix(),
        "sha256": _sha256_path(renderer_path),
        "size_bytes": renderer_path.stat().st_size,
    }
    if renderer_binding != expected_renderer:
        raise QCError("Renderer code changed after the sample was created")
    return manifest, expected_cells


def _validate_review(
    review: Mapping[str, Any],
    *,
    eye_id: str,
    sample_manifest_sha256: str,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    _exact_keys(review, REVIEW_TOP_LEVEL_FIELDS, description="review JSON")
    if review["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise QCError("Wrong review schema_version")
    if review["eye_id"] != eye_id:
        raise QCError("Review eye_id mismatch")
    if review["review_scope"] not in REVIEW_SCOPES:
        raise QCError("Review scope must be stratified_sample_only")
    if review["review_mode"] not in REVIEW_MODES:
        raise QCError(f"Unsupported review_mode {review['review_mode']!r}")
    if not isinstance(review["reviewer_id"], str) or not review["reviewer_id"].strip():
        raise QCError("reviewer_id must be a nonempty string")
    if not isinstance(review["reviewed_at_utc"], str) or not UTC_RE.fullmatch(
        review["reviewed_at_utc"]
    ):
        raise QCError("reviewed_at_utc must be an ISO-8601 UTC timestamp ending in Z")
    if review["sample_manifest_sha256"] != sample_manifest_sha256:
        raise QCError("Review does not bind the current sample manifest")
    decisions = review["decisions"]
    if not isinstance(decisions, list) or len(decisions) != EXPECTED_SAMPLE_SIZE:
        raise QCError("Review must contain exactly 32 decisions")
    for ordinal, (decision, sample) in enumerate(zip(decisions, samples)):
        if not isinstance(decision, dict):
            raise QCError(f"Review decision {ordinal} must be an object")
        _exact_keys(decision, REVIEW_DECISION_FIELDS, description=f"review decision {ordinal}")
        if decision["lens_index"] != sample["lens_index"] or decision["seed_id"] != sample["seed_id"]:
            raise QCError(f"Review decision {ordinal} does not match frozen sample order")
        if decision["decision"] != "pass":
            raise QCError(f"Review decision {ordinal} did not pass; no attestation issued")
        if not isinstance(decision["notes"], str):
            raise QCError(f"Review decision {ordinal} notes must be a string")


def _write_json_nonoverwriting(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise QCError(f"Refusing to overwrite existing attestation: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def attest_eye(
    *,
    bundle_root: Path,
    sample_manifest_path: Path,
    review_path: Path,
    output_path: Path,
) -> Path:
    bundle_root = bundle_root.resolve()
    sample_manifest_path = sample_manifest_path.resolve()
    review_path = review_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise QCError(f"Refusing to overwrite existing attestation: {output_path}")
    expected_sample_manifest = bundle_root / SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
    expected_review = bundle_root / REVIEW_FILENAME
    expected_output = bundle_root / ATTESTATION_FILENAME
    if sample_manifest_path != expected_sample_manifest:
        raise QCError(f"Sample manifest must be exactly {expected_sample_manifest}")
    if review_path != expected_review:
        raise QCError(f"Review JSON must be exactly {expected_review}")
    if output_path != expected_output:
        raise QCError(f"Attestation output must be exactly {expected_output}")

    completion_path = bundle_root / "completion.json"
    provenance_path = bundle_root / "provenance.json"
    completion = _load_json(completion_path, description="completion")
    provenance = _load_json(provenance_path, description="provenance")
    eye_id = completion.get("eye_id")
    if not isinstance(eye_id, str) or not eye_id:
        raise QCError("Completion eye_id must be a nonempty string")
    completion_manifest, n_hash_verified = _verify_bundle_manifests(
        bundle_root, completion, provenance
    )
    n_expected = int(completion["n_expected"])
    inventory, _, summed_full = _verify_technical_inventory(
        bundle_root, eye_id=eye_id, n_expected=n_expected
    )
    partition = _verify_partition(
        completion, provenance, summed_assigned_points=summed_full
    )
    sample_manifest, expected_cells = _verify_sample_manifest(
        sample_manifest_path=sample_manifest_path,
        bundle_root=bundle_root,
        eye_id=eye_id,
        completion_manifest=completion_manifest,
    )
    sample_manifest_sha256 = _sha256_path(sample_manifest_path)
    review = _load_json(review_path, description="review")
    _validate_review(
        review,
        eye_id=eye_id,
        sample_manifest_sha256=sample_manifest_sha256,
        samples=sample_manifest["samples"],
    )
    attester_path = Path(__file__).resolve()
    repository_root = attester_path.parents[2]
    reviewed_samples = [
        {
            "ordinal": sample["ordinal"],
            "lens_index": sample["lens_index"],
            "seed_id": sample["seed_id"],
            "radial_stratum": sample["radial_stratum"],
            "scale_stratum": sample["scale_stratum"],
            "decision": "pass",
            "render_sha256": sample["render"]["sha256"],
        }
        for sample in sample_manifest["samples"]
    ]
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "eye_id": eye_id,
        "review_scope": "stratified_sample_only",
        "review_mode": review["review_mode"],
        "reviewer_id": review["reviewer_id"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "all_instances_manually_reviewed": False,
        "stratified_sample_visual_qc_passed": True,
        "n_reviewed": EXPECTED_SAMPLE_SIZE,
        "n_expected": n_expected,
        "n_inventory_rows": inventory["n_inventory_rows"],
        "technical_inventory_complete": True,
        "artifact_hash_verification_passed": True,
        "partition_exact": True,
        "technical_inventory": inventory,
        "partition_evidence": partition,
        "artifact_verification": {
            "passed": True,
            "n_completion_manifest_artifacts_verified": n_hash_verified,
            "provenance_manifest_identical_to_completion": True,
            "fitted_lens_archives_hash_verified_but_not_opened": True,
        },
        "sample_coverage": expected_cells,
        "bindings": {
            "completion": _binding(completion_path, relative_to=bundle_root),
            "provenance": _binding(provenance_path, relative_to=bundle_root),
            "lens_summary": _binding(bundle_root / "lens_summary.csv", relative_to=bundle_root),
            "distal_qc_sampling": _binding(
                bundle_root / "distal_qc_sampling.csv", relative_to=bundle_root
            ),
            "sample_manifest": _binding(sample_manifest_path, relative_to=bundle_root),
            "review_json": _binding(review_path, relative_to=bundle_root),
            "renderer_code": sample_manifest["renderer_code"],
            "attester_code": {
                "relative_path": attester_path.relative_to(repository_root).as_posix(),
                "sha256": _sha256_path(attester_path),
                "size_bytes": attester_path.stat().st_size,
            },
        },
        "reviewed_samples": reviewed_samples,
        "claim_boundary": (
            "This sidecar records a passing visual review of the frozen 32-lens "
            "distal-QC-stratified sample only. It does not claim that every instance "
            "was manually reviewed, and it does not alter the extraction bundle's "
            "instance_segmentation_validated=false declaration."
        ),
    }
    _write_json_nonoverwriting(output_path, attestation)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help=(
            f"Defaults to BUNDLE_ROOT/{SAMPLE_DIRECTORY_NAME}/sample_manifest.json "
            "and cannot point elsewhere."
        ),
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        help=f"Defaults to BUNDLE_ROOT/{REVIEW_FILENAME} and cannot point elsewhere.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Defaults to BUNDLE_ROOT/{ATTESTATION_FILENAME}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_manifest = args.sample_manifest or (
        args.bundle_root / SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
    )
    review_json = args.review_json or (args.bundle_root / REVIEW_FILENAME)
    output_path = args.output or (args.bundle_root / ATTESTATION_FILENAME)
    try:
        result = attest_eye(
            bundle_root=args.bundle_root,
            sample_manifest_path=sample_manifest,
            review_path=review_json,
            output_path=output_path,
        )
    except QCError as exc:
        print(f"Instance-QC attestation failed: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

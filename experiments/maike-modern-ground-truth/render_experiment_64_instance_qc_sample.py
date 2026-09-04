#!/usr/bin/env python3
"""Render the disjoint, model/error-sequestered Experiment 64 instance-QC sample.

The input surface is deliberately narrow.  This program reads the two
target-free technical producer documents, the target-free sampling CSV, the
committed Experiment 63 stop record and its twelve already-frozen sample
manifests.  For selected lenses it opens only the Stage-1 instance and robust
sealed-distal-core NPZs.  It never opens a target, fitted lens, prediction,
error, model, or sealed-outcome artifact.

The image includes the complete assigned and dominant lens bodies.  It is
therefore model/error blind, but it is explicitly *not* proximal-anatomy
blind.  Raw localized distal points and the final robust core are rendered as
distinct overlays so the reviewer can see what the robust operator removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment_64_technical_metrics as technical_metrics


SCHEMA_VERSION = "experiment64.instance-qc-sample.v1"
EXCLUSION_SCHEMA_VERSION = "experiment64.development-exclusions.v1"
SAMPLING_ALGORITHM = "experiment64_instance_qc_v1"
SAMPLE_DIRECTORY_NAME = "experiment64_instance_qc_visual_sample"
EXPECTED_SAMPLE_SIZE = 32
EXPECTED_PRIOR_EYES = 12
EXPECTED_PRIOR_PER_EYE = 32
EXPECTED_GLOBAL_EXCLUSIONS = EXPECTED_PRIOR_EYES * EXPECTED_PRIOR_PER_EYE
N_RADIAL_STRATA = 4
N_SCALE_STRATA = 4
N_PER_CELL = 2
STOP_RECORD_RELPATH = PurePosixPath(
    "experiments/maike-modern-ground-truth/results/experiment_63_stop_record.json"
)
EXCLUSION_LEDGER_RELPATH = PurePosixPath(
    "experiments/maike-modern-ground-truth/results/experiment_64_development_exclusions.json"
)

TECHNICAL_BUNDLE_SCHEMA_VERSION = "experiment64.maike-technical-bundle.v1"
INSTANCE_SCHEMA_VERSION = "experiment64.maike-instance.v1"
SEALED_CORE_SCHEMA_VERSION = "experiment64.maike-sealed-distal-core.v1"
PRIOR_SAMPLE_SCHEMA_VERSION = "experiment63.instance-qc-sample.v1"
PRIOR_SAMPLING_ALGORITHM = "experiment63_instance_qc_v1"
STOP_RECORD_SCHEMA_VERSION = "experiment63.pre_outcome_stop.v1"

SAMPLING_FIELDS = (
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

INSTANCE_MEMBERS = frozenset(
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
INSTANCE_ACCESSED_MEMBERS = (
    "schema_version",
    "lens_index",
    "full_assigned_points_zyx",
    "main_component_points_zyx",
    "raw_distal_points_zyx",
    "spacing_um",
)
SEALED_CORE_MEMBERS = frozenset(
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
SEALED_CORE_ACCESSED_MEMBERS = (
    "schema_version",
    "lens_index",
    "points_zyx",
    "spacing_um",
    "raw_distal_support",
    "robust_core_config_sha256",
)
FORBIDDEN_INPUT_ROLES = (
    "target or proximal-target table",
    "fitted lens or sealed outcome",
    "prediction",
    "error or score",
    "trained model or model output",
)
FORBIDDEN_NPZ_MEMBER_TOKENS = (
    "proximal",
    "target",
    "prediction",
    "predicted",
    "error",
    "score",
    "model",
    "outcome",
)
EXPECTED_TECHNICAL_COHERENCE_CONFIG: dict[str, Any] = dict(
    technical_metrics.TECHNICAL_COHERENCE_CONFIG
)


class QCError(RuntimeError):
    """Raised when the frozen visual-QC input contract is not satisfied."""


@dataclass(frozen=True)
class SamplingRow:
    eye_id: str
    lens_index: int
    seed_id: str
    distal_eligible: bool
    position_u_um: float | None
    position_v_um: float | None
    distal_scale_um: float | None
    coherence_margin: float | None
    instance_relpath: str
    sealed_distal_relpath: str

    @property
    def radius_um(self) -> float:
        if self.position_u_um is None or self.position_v_um is None:
            raise QCError("An ineligible sampling row cannot be assigned a radial rank")
        return math.hypot(self.position_u_um, self.position_v_um)


@dataclass(frozen=True)
class SelectedRow:
    row: SamplingRow
    radial_rank: int
    radial_stratum: int
    scale_rank_within_radial: int
    scale_stratum: int
    selection_role: str
    selection_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    try:
        display = path.relative_to(relative_to).as_posix() if relative_to else path.name
    except ValueError as exc:
        raise QCError(f"Artifact escapes its declared root: {path}") from exc
    return {
        "relative_path": display,
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


def _run_git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise QCError("git is required to verify the committed Experiment 63 stop record") from exc


def verify_committed_stop_record(
    repository_root: Path, stop_record_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that the exact current stop-record bytes are committed at HEAD."""

    root = repository_root.resolve()
    stop = stop_record_path.resolve()
    expected = (root / Path(*STOP_RECORD_RELPATH.parts)).resolve()
    if stop != expected:
        raise QCError(f"Experiment 63 stop record must be exactly {expected}")
    if not stop.is_file():
        raise QCError(f"Experiment 63 stop record is missing: {stop}")

    top = _run_git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.decode().strip()).resolve() != root:
        raise QCError(f"Not the declared Git repository root: {root}")
    tracked = _run_git(root, "ls-files", "--error-unmatch", "--", STOP_RECORD_RELPATH.as_posix())
    if tracked.returncode != 0:
        raise QCError("Experiment 63 stop record is not tracked by Git")
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        STOP_RECORD_RELPATH.as_posix(),
    )
    if status.returncode != 0 or status.stdout.strip():
        raise QCError("Experiment 63 stop record differs from its committed HEAD version")
    head_blob = _run_git(root, "show", f"HEAD:{STOP_RECORD_RELPATH.as_posix()}")
    current_bytes = stop.read_bytes()
    if head_blob.returncode != 0 or head_blob.stdout != current_bytes:
        raise QCError("Experiment 63 stop-record bytes do not equal the committed HEAD blob")

    document = _load_json(stop, description="Experiment 63 stop record")
    if document.get("schema_version") != STOP_RECORD_SCHEMA_VERSION:
        raise QCError("Wrong Experiment 63 stop-record schema")
    if document.get("experiment") != 63:
        raise QCError("Experiment 63 stop record has the wrong experiment number")
    if document.get("status") != "stopped_before_model_outcome_evaluation":
        raise QCError("Experiment 63 stop record does not attest a pre-outcome stop")
    outcome_access = document.get("outcome_access")
    if not isinstance(outcome_access, dict) or any(value is not False for value in outcome_access.values()):
        raise QCError("Experiment 63 stop record does not attest zero outcome access")
    frozen_commit = document.get("frozen_git_commit")
    if not isinstance(frozen_commit, str) or len(frozen_commit) != 40:
        raise QCError("Experiment 63 frozen_git_commit is invalid")
    commit_check = _run_git(root, "cat-file", "-e", f"{frozen_commit}^{{commit}}")
    if commit_check.returncode != 0:
        raise QCError("Experiment 63 frozen_git_commit is not present in this repository")
    head = _run_git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise QCError("Cannot resolve repository HEAD")
    record_commit_result = _run_git(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        STOP_RECORD_RELPATH.as_posix(),
    )
    stop_record_commit = record_commit_result.stdout.decode().strip()
    if record_commit_result.returncode != 0 or len(stop_record_commit) != 40:
        raise QCError("Cannot resolve the commit that froze the Experiment 63 stop record")
    committed_at_source = _run_git(
        root, "show", f"{stop_record_commit}:{STOP_RECORD_RELPATH.as_posix()}"
    )
    if committed_at_source.returncode != 0 or committed_at_source.stdout != current_bytes:
        raise QCError("Experiment 63 stop record changed after its source commit")
    binding = {
        **_artifact_binding(stop, relative_to=root),
        "tracked_at_head": True,
        "head_commit": head.stdout.decode().strip(),
        "stop_record_commit": stop_record_commit,
        "experiment63_frozen_commit": frozen_commit,
    }
    return document, binding


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise QCError(f"{description} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise QCError(f"{description} must be a lowercase SHA-256 digest") from exc
    if value.lower() != value:
        raise QCError(f"{description} must be a lowercase SHA-256 digest")
    return value


def build_development_exclusion_ledger(
    *,
    repository_root: Path,
    stop_record_path: Path,
    prior_sample_manifests: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, set[tuple[int, str]]]]:
    """Verify all prior hashes and return the exact 384-case exclusion ledger."""

    stop, stop_binding = verify_committed_stop_record(repository_root, stop_record_path)
    eyes = stop.get("eyes")
    if not isinstance(eyes, list) or len(eyes) != EXPECTED_PRIOR_EYES:
        raise QCError("Experiment 63 stop record must contain exactly twelve eyes")
    expected_hashes: dict[str, str] = {}
    for position, entry in enumerate(eyes):
        if not isinstance(entry, dict):
            raise QCError(f"Experiment 63 stop-record eye {position} is not an object")
        eye_id = entry.get("eye_id")
        if not isinstance(eye_id, str) or not eye_id or eye_id in expected_hashes:
            raise QCError("Experiment 63 stop-record eye identifiers are invalid or duplicated")
        expected_hashes[eye_id] = _require_sha256(
            entry.get("sample_manifest_sha256"),
            description=f"Experiment 63 sample hash for {eye_id}",
        )
    if set(prior_sample_manifests) != set(expected_hashes):
        raise QCError(
            "Prior sample-manifest mapping must contain exactly the twelve stop-record eyes; "
            f"missing={sorted(set(expected_hashes) - set(prior_sample_manifests))}, "
            f"extra={sorted(set(prior_sample_manifests) - set(expected_hashes))}"
        )

    ledger_eyes: list[dict[str, Any]] = []
    exclusion_sets: dict[str, set[tuple[int, str]]] = {}
    global_pairs: set[tuple[str, int]] = set()
    for eye_id in expected_hashes:
        path = prior_sample_manifests[eye_id].resolve()
        if not path.is_file() or path.name != "sample_manifest.json":
            raise QCError(f"Missing or wrongly named Experiment 63 sample manifest for {eye_id}")
        actual_hash = _sha256_path(path)
        if actual_hash != expected_hashes[eye_id]:
            raise QCError(f"Experiment 63 sample-manifest SHA-256 mismatch for {eye_id}")
        manifest = _load_json(path, description=f"Experiment 63 sample manifest for {eye_id}")
        if manifest.get("schema_version") != PRIOR_SAMPLE_SCHEMA_VERSION:
            raise QCError(f"Wrong Experiment 63 sample schema for {eye_id}")
        if manifest.get("sampling_algorithm") != PRIOR_SAMPLING_ALGORITHM:
            raise QCError(f"Wrong Experiment 63 sampling algorithm for {eye_id}")
        if manifest.get("eye_id") != eye_id or manifest.get("n_selected") != EXPECTED_PRIOR_PER_EYE:
            raise QCError(f"Experiment 63 sample identity/count mismatch for {eye_id}")
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != EXPECTED_PRIOR_PER_EYE:
            raise QCError(f"Experiment 63 sample for {eye_id} must contain exactly 32 cases")
        identities: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for ordinal, sample in enumerate(samples):
            if not isinstance(sample, dict) or sample.get("ordinal") != ordinal:
                raise QCError(f"Experiment 63 sample ordinals are invalid for {eye_id}")
            if sample.get("eye_id") != eye_id:
                raise QCError(f"Experiment 63 sample entry has wrong eye_id for {eye_id}")
            lens_index = sample.get("lens_index")
            seed_id = sample.get("seed_id")
            selection_hash = _require_sha256(
                sample.get("selection_sha256"),
                description=f"Experiment 63 selection hash {eye_id}/{ordinal}",
            )
            if type(lens_index) is not int or lens_index < 0 or not isinstance(seed_id, str) or not seed_id:
                raise QCError(f"Invalid Experiment 63 identity for {eye_id}/{ordinal}")
            identity = (lens_index, seed_id)
            if identity in seen or (eye_id, lens_index) in global_pairs:
                raise QCError(f"Duplicate Experiment 63 development identity for {eye_id}")
            expected_selection = _sha256_bytes(
                f"{PRIOR_SAMPLING_ALGORITHM}|{eye_id}|{lens_index}|{seed_id}".encode("utf-8")
            )
            if selection_hash != expected_selection:
                raise QCError(f"Experiment 63 selection hash is invalid for {eye_id}/{ordinal}")
            seen.add(identity)
            global_pairs.add((eye_id, lens_index))
            identities.append(
                {
                    "old_ordinal": ordinal,
                    "lens_index": lens_index,
                    "seed_id": seed_id,
                    "old_selection_sha256": selection_hash,
                }
            )
        exclusion_sets[eye_id] = seen
        ledger_eyes.append(
            {
                "eye_id": eye_id,
                "old_sample_manifest_sha256": actual_hash,
                "n_excluded": len(identities),
                "identities": identities,
            }
        )

    if len(global_pairs) != EXPECTED_GLOBAL_EXCLUSIONS:
        raise QCError("Development exclusion ledger does not contain exactly 384 eye-scoped cases")
    stable_stop_binding = {
        field: stop_binding[field]
        for field in (
            "relative_path",
            "sha256",
            "size_bytes",
            "stop_record_commit",
            "experiment63_frozen_commit",
        )
    }
    ledger = {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "experiment": 64,
        "source_experiment": 63,
        "identity_scope": "eye_id+lens_index; seed_id must also match",
        "exclusion_timing": "before Experiment 64 radial and scale stable ranks",
        "experiment63_stop_record": stable_stop_binding,
        "n_eyes": EXPECTED_PRIOR_EYES,
        "n_excluded_per_eye": EXPECTED_PRIOR_PER_EYE,
        "n_eye_scoped_exclusions": EXPECTED_GLOBAL_EXCLUSIONS,
        "eyes": ledger_eyes,
    }
    return ledger, exclusion_sets


def verify_committed_development_exclusion_ledger(
    *, repository_root: Path, ledger_path: Path, expected_ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the canonical exclusion-ledger bytes to be tracked and unchanged."""

    root = repository_root.resolve()
    path = ledger_path.resolve()
    expected_path = (root / Path(*EXCLUSION_LEDGER_RELPATH.parts)).resolve()
    if path != expected_path or not path.is_file():
        raise QCError(f"Development exclusion ledger must be exactly {expected_path}")
    relpath = EXCLUSION_LEDGER_RELPATH.as_posix()
    tracked = _run_git(root, "ls-files", "--error-unmatch", "--", relpath)
    if tracked.returncode != 0:
        raise QCError("Experiment 64 development exclusion ledger is not tracked by Git")
    status = _run_git(
        root, "status", "--porcelain=v1", "--untracked-files=no", "--", relpath
    )
    if status.returncode != 0 or status.stdout.strip():
        raise QCError("Experiment 64 development exclusion ledger differs from HEAD")
    head_blob = _run_git(root, "show", f"HEAD:{relpath}")
    if head_blob.returncode != 0 or head_blob.stdout != path.read_bytes():
        raise QCError("Experiment 64 development exclusion ledger is not byte-identical to HEAD")
    actual = _load_json(path, description="canonical Experiment 64 development exclusion ledger")
    if actual != expected_ledger:
        raise QCError(
            "Committed Experiment 64 development exclusion ledger does not equal the "
            "independently reconstructed 384-case ledger"
        )
    head = _run_git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise QCError("Cannot resolve Git HEAD for the development exclusion ledger")
    return {
        **_artifact_binding(path, relative_to=root),
        "tracked_at_head": True,
        "head_commit": head.stdout.decode().strip(),
    }


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise QCError(f"Row {row_number}: {field} must be literal 'true' or 'false'")


def _parse_finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise QCError(f"Row {row_number}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise QCError(f"Row {row_number}: {field} must be finite")
    return result


def read_sampling_table(path: Path, *, eye_id: str) -> tuple[list[SamplingRow], int]:
    """Read only the exact target-free sampling allowlist."""

    rows: list[SamplingRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLING_FIELDS:
            raise QCError(
                "Sampling table header is not the exact target-free allowlist; "
                f"expected {SAMPLING_FIELDS}, got {tuple(reader.fieldnames or ())}"
            )
        seen_indices: set[int] = set()
        seen_seeds: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            if raw["eye_id"] != eye_id:
                raise QCError(f"Row {row_number}: wrong eye_id {raw['eye_id']!r}")
            try:
                lens_index = int(raw["lens_index"])
            except ValueError as exc:
                raise QCError(f"Row {row_number}: lens_index is not an integer") from exc
            seed_id = raw["seed_id"]
            if lens_index < 0 or lens_index in seen_indices:
                raise QCError(f"Row {row_number}: duplicate or negative lens_index")
            if not seed_id or seed_id in seen_seeds:
                raise QCError(f"Row {row_number}: empty or duplicate seed_id")
            expected_instance = f"instances/lens_{lens_index:06d}.npz"
            expected_core = f"sealed_distal/lens_{lens_index:06d}.npz"
            if raw["instance_relpath"] != expected_instance:
                raise QCError(
                    f"Row {row_number}: instance_relpath must be exactly {expected_instance!r}"
                )
            if raw["sealed_distal_relpath"] != expected_core:
                raise QCError(
                    f"Row {row_number}: sealed_distal_relpath must be exactly {expected_core!r}"
                )
            seen_indices.add(lens_index)
            seen_seeds.add(seed_id)
            eligible = _parse_bool(
                raw["distal_eligible"], field="distal_eligible", row_number=row_number
            )
            geometry_fields = (
                "position_u_um",
                "position_v_um",
                "distal_scale_um",
                "coherence_margin",
            )
            if eligible:
                position_u = _parse_finite_float(
                    raw["position_u_um"], field="position_u_um", row_number=row_number
                )
                position_v = _parse_finite_float(
                    raw["position_v_um"], field="position_v_um", row_number=row_number
                )
                scale = _parse_finite_float(
                    raw["distal_scale_um"], field="distal_scale_um", row_number=row_number
                )
                coherence_margin = _parse_finite_float(
                    raw["coherence_margin"], field="coherence_margin", row_number=row_number
                )
                if scale <= 0:
                    raise QCError(f"Row {row_number}: distal_scale_um must be positive")
            else:
                if any(raw[field] != "" for field in geometry_fields):
                    raise QCError(
                        f"Row {row_number}: ineligible geometry/coherence fields must be empty"
                    )
                position_u = position_v = scale = coherence_margin = None
            rows.append(
                SamplingRow(
                    eye_id=eye_id,
                    lens_index=lens_index,
                    seed_id=seed_id,
                    distal_eligible=eligible,
                    position_u_um=position_u,
                    position_v_um=position_v,
                    distal_scale_um=scale,
                    coherence_margin=coherence_margin,
                    instance_relpath=raw["instance_relpath"],
                    sealed_distal_relpath=raw["sealed_distal_relpath"],
                )
            )
    if not rows or sorted(seen_indices) != list(range(len(rows))):
        raise QCError("Sampling table must contain every lens_index exactly once from 0 to N-1")
    return rows, len(rows)


def exclude_development_identities(
    rows: Sequence[SamplingRow], exclusions: set[tuple[int, str]], *, eye_id: str
) -> tuple[list[SamplingRow], int]:
    if len(exclusions) != EXPECTED_PRIOR_PER_EYE:
        raise QCError(f"{eye_id}: exclusion ledger must contain exactly 32 identities")
    by_index = {row.lens_index: row for row in rows}
    for lens_index, seed_id in exclusions:
        row = by_index.get(lens_index)
        if row is None or row.seed_id != seed_id:
            raise QCError(
                f"{eye_id}: old development identity lens {lens_index}/{seed_id!r} "
                "is absent or changed in the Experiment 64 sampling inventory"
            )
    eligible_before = [row for row in rows if row.distal_eligible]
    unseen_rows = [
        row for row in rows if (row.lens_index, row.seed_id) not in exclusions
    ]
    # The ordering is scientifically material: remove development identities
    # from the complete inventory first, then form the eligible population and
    # only then compute new radial/scale ranks.
    filtered = [row for row in unseen_rows if row.distal_eligible]
    removed_eligible = len(eligible_before) - len(filtered)
    return filtered, removed_eligible


def _stable_stratum(rank: int, population_size: int, n_strata: int) -> int:
    if not 0 <= rank < population_size:
        raise QCError("Internal stable-rank error")
    return min(n_strata - 1, (rank * n_strata) // population_size)


def select_frozen_sample(rows: Sequence[SamplingRow]) -> list[SelectedRow]:
    """Take one cell-wise near-worst coherence case and one hash-min case."""

    if not rows:
        raise QCError("No unseen distal-eligible rows remain after development exclusion")
    if any(
        row.position_u_um is None
        or row.position_v_um is None
        or row.distal_scale_um is None
        or row.coherence_margin is None
        for row in rows
    ):
        raise QCError("Only eligible rows with finite geometry may enter Experiment 64 strata")
    radial_order = sorted(rows, key=lambda row: (row.radius_um, row.lens_index, row.seed_id))
    ranked: list[tuple[SamplingRow, int, int]] = []
    for radial_rank, row in enumerate(radial_order):
        radial = _stable_stratum(radial_rank, len(radial_order), N_RADIAL_STRATA)
        ranked.append((row, radial_rank, radial))

    cells: dict[tuple[int, int], list[SelectedRow]] = {
        (radial, scale): []
        for radial in range(N_RADIAL_STRATA)
        for scale in range(N_SCALE_STRATA)
    }
    for radial in range(N_RADIAL_STRATA):
        members = [entry for entry in ranked if entry[2] == radial]
        members.sort(key=lambda entry: (entry[0].distal_scale_um, entry[0].lens_index, entry[0].seed_id))
        for scale_rank, (row, radial_rank, _) in enumerate(members):
            scale = _stable_stratum(scale_rank, len(members), N_SCALE_STRATA)
            selection_hash = _sha256_bytes(
                f"{SAMPLING_ALGORITHM}|{row.eye_id}|{row.lens_index}|{row.seed_id}".encode("utf-8")
            )
            cells[(radial, scale)].append(
                SelectedRow(row, radial_rank, radial, scale_rank, scale, "candidate", selection_hash)
            )

    shortages = {cell: len(values) for cell, values in cells.items() if len(values) < N_PER_CELL}
    if shortages:
        raise QCError(f"Cannot construct the fixed 32-case sample; undersized cells: {shortages}")

    selected: list[SelectedRow] = []
    for cell in sorted(cells):
        values = cells[cell]
        near_worst = min(
            values,
            key=lambda item: (
                item.row.coherence_margin,
                item.row.lens_index,
                item.row.seed_id,
            ),
        )
        remaining = [item for item in values if item.row != near_worst.row]
        hash_minimal = min(
            remaining,
            key=lambda item: (item.selection_sha256, item.row.lens_index, item.row.seed_id),
        )
        selected.append(SelectedRow(
            near_worst.row,
            near_worst.radial_rank,
            near_worst.radial_stratum,
            near_worst.scale_rank_within_radial,
            near_worst.scale_stratum,
            "near_worst_coherence",
            near_worst.selection_sha256,
        ))
        selected.append(
            SelectedRow(
                hash_minimal.row,
                hash_minimal.radial_rank,
                hash_minimal.radial_stratum,
                hash_minimal.scale_rank_within_radial,
                hash_minimal.scale_stratum,
                "hash_minimal_remaining",
                hash_minimal.selection_sha256,
            )
        )
    if len(selected) != EXPECTED_SAMPLE_SIZE:
        raise QCError(f"Frozen sampling produced {len(selected)} rows instead of 32")
    return selected


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def read_and_validate_producer_documents(
    *, bundle_root: Path, eye_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    completion_path = bundle_root / "technical_completion.json"
    provenance_path = bundle_root / "technical_provenance.json"
    completion = _load_json(completion_path, description="Experiment 64 technical completion")
    provenance = _load_json(provenance_path, description="Experiment 64 technical provenance")
    for name, document in (("completion", completion), ("provenance", provenance)):
        if document.get("schema_version") != TECHNICAL_BUNDLE_SCHEMA_VERSION:
            raise QCError(f"Wrong Experiment 64 technical {name} schema")
        if document.get("status") != "complete" or document.get("experiment") != 64:
            raise QCError(f"Experiment 64 technical {name} is not complete")
        if document.get("eye_id") != eye_id:
            raise QCError(f"Experiment 64 technical {name} has the wrong eye_id")
        if document.get("contiguous_indices") is not True:
            raise QCError(f"Experiment 64 technical {name} does not attest contiguous indices")
        if document.get("instance_segmentation_validated") is not False:
            raise QCError(
                f"Experiment 64 technical {name} must preserve instance_segmentation_validated=false"
            )
    config = completion.get("robust_core_config")
    if not isinstance(config, dict) or provenance.get("robust_core_config") != config:
        raise QCError("Technical completion/provenance robust-core configs disagree")
    config_hash = _require_sha256(
        completion.get("robust_core_config_sha256"), description="robust-core config hash"
    )
    if provenance.get("robust_core_config_sha256") != config_hash:
        raise QCError("Technical completion/provenance robust-core config hashes disagree")
    if _canonical_json_sha256(config) != config_hash:
        raise QCError("Technical producer robust-core config hash is invalid")
    required_config = {
        "method": "geometric_median_euclidean_q90_intersect_trimmed_pca_lateral_q90_v1",
        "pca_trim_quantile": 0.90,
        "core_lateral_quantile": 0.90,
        "downstream_fit_quantile": 0.90,
        "minimum_input_points": 33,
        "minimum_retained_points": 27,
        "minimum_downstream_fit_points": 25,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise QCError(f"Robust-core config {key} is not frozen at {expected!r}")
    coherence_config = completion.get("technical_coherence_config")
    if (
        coherence_config != EXPECTED_TECHNICAL_COHERENCE_CONFIG
        or provenance.get("technical_coherence_config") != coherence_config
    ):
        raise QCError("Technical producer coherence/gate configuration is not frozen")
    coherence_hash = _require_sha256(
        completion.get("technical_coherence_config_sha256"),
        description="technical coherence config hash",
    )
    if (
        provenance.get("technical_coherence_config_sha256") != coherence_hash
        or _canonical_json_sha256(coherence_config) != coherence_hash
    ):
        raise QCError("Technical producer coherence/gate config hash is invalid")
    return completion, provenance, config, config_hash


def _resolve_exact_artifact(
    bundle_root: Path, relpath: str, *, prefix: str, lens_index: int
) -> Path:
    pure = PurePosixPath(relpath)
    expected = PurePosixPath(prefix) / f"lens_{lens_index:06d}.npz"
    if pure != expected or pure.is_absolute() or ".." in pure.parts:
        raise QCError(f"Lens {lens_index}: {prefix} path must be exactly {expected.as_posix()!r}")
    root = bundle_root.resolve()
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root / prefix)
    except ValueError as exc:
        raise QCError(f"Lens {lens_index}: artifact escapes {prefix}") from exc
    if not path.is_file():
        raise QCError(f"Lens {lens_index}: missing artifact {path}")
    return path


def _np_scalar_text(array: np.ndarray, *, name: str) -> str:
    value = np.asarray(array)
    if value.shape != ():
        raise QCError(f"{name} must be a scalar string")
    return str(value.item())


def _np_scalar_int(array: np.ndarray, *, name: str) -> int:
    value = np.asarray(array)
    if value.shape != () or value.dtype.kind not in "iu":
        raise QCError(f"{name} must be a scalar integer")
    return int(value.item())


def _validate_points(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[1:] != (3,) or value.dtype.kind not in "iu" or len(value) == 0:
        raise QCError(f"{name} must be a nonempty integer N x 3 array")
    value = value.astype(np.int64, copy=False)
    if len({tuple(row) for row in value.tolist()}) != len(value):
        raise QCError(f"{name} contains duplicate source coordinates")
    return value


def _load_instance_for_render(
    path: Path, *, lens_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        members = frozenset(archive.files)
        if members != INSTANCE_MEMBERS:
            raise QCError(f"Unexpected instance NPZ members in {path.name}: {sorted(members)}")
        if any(token in member.lower() for member in members for token in FORBIDDEN_NPZ_MEMBER_TOKENS):
            raise QCError(f"Forbidden outcome-bearing member name in {path.name}")
        schema = _np_scalar_text(archive["schema_version"], name="schema_version")
        stored_index = _np_scalar_int(archive["lens_index"], name="lens_index")
        full = _validate_points(archive["full_assigned_points_zyx"], name="full_assigned_points_zyx")
        main = _validate_points(archive["main_component_points_zyx"], name="main_component_points_zyx")
        raw_distal = _validate_points(
            archive["raw_distal_points_zyx"], name="raw_distal_points_zyx"
        )
        spacing = np.asarray(archive["spacing_um"], dtype=np.float64)
    if schema != INSTANCE_SCHEMA_VERSION or stored_index != lens_index:
        raise QCError(f"Lens {lens_index}: instance schema or index mismatch")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise QCError("spacing_um must contain three positive finite values")
    full_set = {tuple(row) for row in full.tolist()}
    if any(tuple(row) not in full_set for row in main.tolist()):
        raise QCError(f"Lens {lens_index}: dominant component is not a subset of assigned mask")
    main_set = {tuple(row) for row in main.tolist()}
    if any(tuple(row) not in main_set for row in raw_distal.tolist()):
        raise QCError(f"Lens {lens_index}: raw distal points are not a subset of dominant component")
    return full, main, raw_distal, spacing, sorted(members)


def _load_core_for_render(
    path: Path, *, lens_index: int, minimum_retained: int, expected_config_sha256: str
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        members = frozenset(archive.files)
        if members != SEALED_CORE_MEMBERS:
            raise QCError(f"Unexpected sealed-core NPZ members in {path.name}: {sorted(members)}")
        if any(token in member.lower() for member in members for token in FORBIDDEN_NPZ_MEMBER_TOKENS):
            raise QCError(f"Forbidden outcome-bearing member name in {path.name}")
        schema = _np_scalar_text(archive["schema_version"], name="schema_version")
        stored_index = _np_scalar_int(archive["lens_index"], name="lens_index")
        points = _validate_points(archive["points_zyx"], name="points_zyx")
        spacing = np.asarray(archive["spacing_um"], dtype=np.float64)
        raw_distal_support = _np_scalar_int(
            archive["raw_distal_support"], name="raw_distal_support"
        )
        stored_config_sha256 = _np_scalar_text(
            archive["robust_core_config_sha256"], name="robust_core_config_sha256"
        )
    if schema != SEALED_CORE_SCHEMA_VERSION or stored_index != lens_index:
        raise QCError(f"Lens {lens_index}: sealed-core schema or index mismatch")
    if len(points) < minimum_retained:
        raise QCError(f"Lens {lens_index}: robust core is below the producer-configured support floor")
    if raw_distal_support < len(points):
        raise QCError(f"Lens {lens_index}: raw_distal_support is smaller than the robust core")
    if stored_config_sha256 != expected_config_sha256:
        raise QCError(f"Lens {lens_index}: robust-core config hash differs from producer manifests")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise QCError("spacing_um must contain three positive finite values")
    return points, spacing, raw_distal_support, sorted(members)


def _plot_projection(
    axis: plt.Axes,
    full_xyz: np.ndarray,
    main_xyz: np.ndarray,
    raw_distal_xyz: np.ndarray,
    core_xyz: np.ndarray,
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
) -> None:
    a, b = dims
    axis.scatter(full_xyz[:, a], full_xyz[:, b], s=1.1, c="#777777", alpha=0.85, linewidths=0, rasterized=True)
    axis.scatter(main_xyz[:, a], main_xyz[:, b], s=0.8, c="#ffffff", alpha=0.92, linewidths=0, rasterized=True)
    axis.scatter(raw_distal_xyz[:, a], raw_distal_xyz[:, b], s=1.8, c="#ff8c00", alpha=0.82, linewidths=0, rasterized=True)
    axis.scatter(core_xyz[:, a], core_xyz[:, b], s=2.0, c="#00e1ef", alpha=1.0, linewidths=0, rasterized=True)
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#111111")
    axis.tick_params(labelsize=7)


def _render_lens(
    output_path: Path,
    *,
    selected: SelectedRow,
    full_zyx: np.ndarray,
    main_zyx: np.ndarray,
    raw_distal_zyx: np.ndarray,
    core_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> None:
    full_xyz = full_zyx[:, ::-1] * spacing_zyx[::-1]
    main_xyz = main_zyx[:, ::-1] * spacing_zyx[::-1]
    raw_distal_xyz = raw_distal_zyx[:, ::-1] * spacing_zyx[::-1]
    core_xyz = core_zyx[:, ::-1] * spacing_zyx[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _plot_projection(axes[0], full_xyz, main_xyz, raw_distal_xyz, core_xyz, (0, 1), ("x (µm)", "y (µm)"), "XY / project z")
    _plot_projection(axes[1], full_xyz, main_xyz, raw_distal_xyz, core_xyz, (0, 2), ("x (µm)", "z (µm)"), "XZ / project y")
    _plot_projection(axes[2], full_xyz, main_xyz, raw_distal_xyz, core_xyz, (1, 2), ("y (µm)", "z (µm)"), "YZ / project x")
    fig.suptitle(
        f"{selected.row.eye_id} · lens {selected.row.lens_index} · seed {selected.row.seed_id} · "
        f"r{selected.radial_stratum}/s{selected.scale_stratum} · {selected.selection_role}\n"
        "assigned: grey · dominant: white · raw distal: orange · final robust core: cyan\n"
        "model/error blind; not proximal-anatomy blind",
        fontsize=9,
    )
    fig.savefig(
        output_path,
        dpi=170,
        facecolor="white",
        metadata={"Software": "Experiment 64 model/error-sequestered QC renderer"},
    )
    plt.close(fig)


def render_eye_sample(
    *,
    eye_id: str,
    bundle_root: Path,
    sampling_table: Path,
    output_dir: Path,
    repository_root: Path,
    stop_record_path: Path,
    prior_sample_manifests: Mapping[str, Path],
    exclusion_ledger_path: Path | None = None,
) -> Path:
    """Render one immutable, disjoint 32-case eye sample."""

    if not eye_id or any(character in eye_id for character in "/\\"):
        raise QCError("eye_id must be a nonempty path-safe identifier")
    bundle_root = bundle_root.resolve()
    sampling_table = sampling_table.resolve()
    output_dir = output_dir.resolve()
    if sampling_table != bundle_root / "distal_qc_sampling.csv":
        raise QCError("Sampling input must be exactly BUNDLE_ROOT/distal_qc_sampling.csv")
    if output_dir != bundle_root / SAMPLE_DIRECTORY_NAME:
        raise QCError(f"QC output must be exactly BUNDLE_ROOT/{SAMPLE_DIRECTORY_NAME}")
    if output_dir.exists():
        raise QCError(f"Refusing to overwrite existing QC sample directory: {output_dir}")

    ledger, exclusion_sets = build_development_exclusion_ledger(
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
        expected_ledger=ledger,
    )
    if eye_id not in exclusion_sets:
        raise QCError(f"Eye {eye_id!r} is absent from the Experiment 63 exclusion ledger")
    completion, provenance, robust_config, robust_config_hash = read_and_validate_producer_documents(
        bundle_root=bundle_root, eye_id=eye_id
    )
    rows, n_inventory_rows = read_sampling_table(sampling_table, eye_id=eye_id)
    unseen_eligible, n_eligible_excluded = exclude_development_identities(
        rows, exclusion_sets[eye_id], eye_id=eye_id
    )
    selected_rows = select_frozen_sample(unseen_eligible)
    selected_identities = {
        (item.row.lens_index, item.row.seed_id) for item in selected_rows
    }
    overlap = selected_identities & exclusion_sets[eye_id]
    if overlap:
        raise QCError(f"Experiment 64 sample overlaps Experiment 63 development cases: {sorted(overlap)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        renders = staging / "renders"
        renders.mkdir()
        samples: list[dict[str, Any]] = []
        for ordinal, selected in enumerate(selected_rows):
            row = selected.row
            instance_path = _resolve_exact_artifact(
                bundle_root, row.instance_relpath, prefix="instances", lens_index=row.lens_index
            )
            core_path = _resolve_exact_artifact(
                bundle_root, row.sealed_distal_relpath, prefix="sealed_distal", lens_index=row.lens_index
            )
            full, main, raw_distal, instance_spacing, instance_members = _load_instance_for_render(
                instance_path, lens_index=row.lens_index
            )
            core, core_spacing, raw_distal_support, core_members = _load_core_for_render(
                core_path,
                lens_index=row.lens_index,
                minimum_retained=int(robust_config["minimum_retained_points"]),
                expected_config_sha256=robust_config_hash,
            )
            if not np.array_equal(instance_spacing, core_spacing):
                raise QCError(f"Lens {row.lens_index}: instance/core spacing mismatch")
            raw_set = {tuple(point) for point in raw_distal.tolist()}
            if len(raw_distal) < int(robust_config["minimum_input_points"]):
                raise QCError(f"Lens {row.lens_index}: raw distal set is below the configured input floor")
            if raw_distal_support != len(raw_distal):
                raise QCError(f"Lens {row.lens_index}: sealed raw_distal_support does not match instance points")
            if any(tuple(point) not in raw_set for point in core.tolist()):
                raise QCError(f"Lens {row.lens_index}: robust core is not a subset of raw distal points")
            render_name = (
                f"sample_{ordinal:02d}_r{selected.radial_stratum}_s{selected.scale_stratum}_"
                f"lens_{row.lens_index:06d}.png"
            )
            render_path = renders / render_name
            _render_lens(
                render_path,
                selected=selected,
                full_zyx=full,
                main_zyx=main,
                raw_distal_zyx=raw_distal,
                core_zyx=core,
                spacing_zyx=instance_spacing,
            )
            samples.append(
                {
                    "ordinal": ordinal,
                    "eye_id": eye_id,
                    "lens_index": row.lens_index,
                    "seed_id": row.seed_id,
                    "radius_um": row.radius_um,
                    "distal_scale_um": row.distal_scale_um,
                    "coherence_margin": row.coherence_margin,
                    "radial_rank_after_exclusion": selected.radial_rank,
                    "radial_stratum": selected.radial_stratum,
                    "scale_rank_within_radial_after_exclusion": selected.scale_rank_within_radial,
                    "scale_stratum": selected.scale_stratum,
                    "selection_role": selected.selection_role,
                    "selection_sha256": selected.selection_sha256,
                    "instance_artifact": {
                        **_artifact_binding(instance_path, relative_to=bundle_root),
                        "members_present": instance_members,
                        "members_accessed": list(INSTANCE_ACCESSED_MEMBERS),
                    },
                    "sealed_robust_core_artifact": {
                        **_artifact_binding(core_path, relative_to=bundle_root),
                        "members_present": core_members,
                        "members_accessed": list(SEALED_CORE_ACCESSED_MEMBERS),
                    },
                    "render": _artifact_binding(render_path, relative_to=staging),
                    "point_counts": {
                        "full_assigned": len(full),
                        "dominant_component": len(main),
                        "raw_localized_distal": len(raw_distal),
                        "final_robust_distal_core": len(core),
                    },
                }
            )

        cell_counts = {
            f"r{radial}_s{scale}": sum(
                sample["radial_stratum"] == radial and sample["scale_stratum"] == scale
                for sample in samples
            )
            for radial in range(N_RADIAL_STRATA)
            for scale in range(N_SCALE_STRATA)
        }
        role_counts = {
            role: sum(sample["selection_role"] == role for sample in samples)
            for role in ("near_worst_coherence", "hash_minimal_remaining")
        }
        if set(cell_counts.values()) != {2} or set(role_counts.values()) != {16}:
            raise QCError("Internal Experiment 64 sample coverage failure")

        renderer_path = Path(__file__).resolve()
        code_root = renderer_path.parents[2]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": 64,
            "eye_id": eye_id,
            "review_scope": "disjoint_stratified_sample_only",
            "all_instances_manually_reviewed": False,
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "sampling_rule": {
                "development_exclusion": "exact Experiment 63 eye-scoped identities before all new ranks",
                "radial_measure": "hypot(position_u_um,position_v_um)",
                "radial_strata": 4,
                "scale_measure": "distal_scale_um within radial stratum",
                "scale_strata_per_radial_stratum": 4,
                "rows_per_cell": 2,
                "near_worst_rule": "smallest coherence_margin; ties by lens_index,seed_id",
                "second_case_rule": "smallest sha256(experiment64_instance_qc_v1|eye_id|lens_index|seed_id) among remaining cell rows",
            },
            "n_inventory_rows": n_inventory_rows,
            "n_distal_qc_eligible_before_development_exclusion": sum(row.distal_eligible for row in rows),
            "n_development_identities_for_eye": EXPECTED_PRIOR_PER_EYE,
            "n_eligible_development_identities_removed": n_eligible_excluded,
            "n_unseen_distal_qc_eligible": len(unseen_eligible),
            "n_selected": len(samples),
            "selected_development_overlap_count": 0,
            "cell_counts": cell_counts,
            "selection_role_counts": role_counts,
            "robust_core_config": robust_config,
            "robust_core_config_sha256": robust_config_hash,
            "technical_coherence_config": completion["technical_coherence_config"],
            "technical_coherence_config_sha256": completion[
                "technical_coherence_config_sha256"
            ],
            "visual_disclosure": {
                "full_assigned_body_shown": True,
                "dominant_component_shown": True,
                "raw_localized_distal_points_available_in_allowlisted_instance": True,
                "raw_localized_distal_points_shown": True,
                "final_robust_core_shown": True,
                "model_and_error_blind": True,
                "proximal_anatomy_blind": False,
                "statement": "Complete lens bodies are shown; this review is not proximal-anatomy blind.",
            },
            "outcome_sequestration": {
                "sampling_table_exact_field_allowlist": list(SAMPLING_FIELDS),
                "instance_members_accessed": list(INSTANCE_ACCESSED_MEMBERS),
                "sealed_core_members_accessed": list(SEALED_CORE_ACCESSED_MEMBERS),
                "forbidden_input_roles": list(FORBIDDEN_INPUT_ROLES),
                "target_proximal_prediction_error_model_or_sealed_outcome_opened": False,
            },
            "experiment63_stop_record": ledger["experiment63_stop_record"],
            "prior_sample_manifest_verification": [
                {
                    "eye_id": item["eye_id"],
                    "sha256": item["old_sample_manifest_sha256"],
                    "n_verified_identities": item["n_excluded"],
                }
                for item in ledger["eyes"]
            ],
            "development_exclusion_ledger": committed_ledger_binding,
            "technical_completion": _artifact_binding(bundle_root / "technical_completion.json", relative_to=bundle_root),
            "technical_provenance": _artifact_binding(bundle_root / "technical_provenance.json", relative_to=bundle_root),
            "sampling_table": _artifact_binding(sampling_table, relative_to=bundle_root),
            "renderer_code": {
                "relative_path": renderer_path.relative_to(code_root).as_posix(),
                "sha256": _sha256_path(renderer_path),
                "size_bytes": renderer_path.stat().st_size,
            },
            "samples": samples,
        }
        manifest_path = staging / "sample_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        return output_dir / "sample_manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    parser.add_argument("--eye-id", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment63-stop-record", type=Path)
    parser.add_argument("--development-exclusion-ledger", type=Path)
    parser.add_argument(
        "--prior-sample-manifest",
        action="append",
        default=[],
        metavar="EYE_ID=PATH",
        help="Repeat exactly once for each of the twelve Experiment 63 eye samples.",
    )
    parser.add_argument("--sampling-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stop = args.experiment63_stop_record or (
        args.repository_root / Path(*STOP_RECORD_RELPATH.parts)
    )
    try:
        prior = _parse_prior_argument(args.prior_sample_manifest)
        manifest = render_eye_sample(
            eye_id=args.eye_id,
            bundle_root=args.bundle_root,
            sampling_table=args.sampling_table or (args.bundle_root / "distal_qc_sampling.csv"),
            output_dir=args.output_dir or (args.bundle_root / SAMPLE_DIRECTORY_NAME),
            repository_root=args.repository_root,
            stop_record_path=stop,
            prior_sample_manifests=prior,
            exclusion_ledger_path=args.development_exclusion_ledger,
        )
    except QCError as exc:
        print(f"Experiment 64 QC sample generation failed: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

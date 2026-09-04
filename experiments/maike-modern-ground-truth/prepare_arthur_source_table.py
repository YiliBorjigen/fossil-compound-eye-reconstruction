#!/usr/bin/env python3
"""Prepare the frozen Arthur Zhao source cohort for Experiment 63.

The three supplied whole-head scans are independent animals; the two eyes in
each scan are nested technical units.  Complete meshes and tip landmarks are
used only in Stage 1 to assign vertices to lenses and identify the distal and
proximal layers.  Stage 2 reopens hash-sealed distal-only point artifacts and
derives every predictor, eye frame, position, and distal QC decision without a
proximal point or tip landmark in its input.  Proximal targets are attached
only after the drop-only distal-QC fixed point has converged.

The command refuses to publish a primary-ready bundle from a dirty worktree.
``--allow-dirty-diagnostic`` exists only for source-side diagnostics and marks
the provenance status ``diagnostic`` so the frozen backend will reject it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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
from audit_distal_frame_stability import audit_frame_stability  # noqa: E402
from experiment_57_outer_only_validation import (  # noqa: E402
    parse_wrl_surfaces,
    prepare_oracle_split_records,
)
from experiment_58_cross_volume_confirmation import raw_positions_from_rdata  # noqa: E402


SCHEMA_VERSION = "experiment63.arthur-source.v2"
SEALED_DISTAL_SCHEMA = "experiment63.arthur-sealed-distal.v1"
LENS_TARGET_SCHEMA = "experiment63.arthur-target.v1"
ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
ISOLATION_BASIS = "stage2_reads_only_sha256_sealed_distal_artifacts"
ORACLE_STAGE1_SCOPE = "oracle_correspondence_and_distal_localization_only"
EYEMAP_COMMIT = "99d2a43123db636cedb55af9ff31a59657e7d17e"

VOLUMES = ("20231107", "20240530", "20240701")
EXPECTED_LANDMARK_COUNTS = {
    "20231107": 1632,
    "20240530": 1611,
    "20240701": 1709,
}
EXPECTED_STAGE1_COUNTS = {
    "20231107": 1622,
    "20240530": 1611,
    "20240701": 1709,
}
EXPECTED_SOURCE_COUNTS = {
    "stage1": 4942,
    "distal_qc": 4897,
}
EXPECTED_INPUT_IDENTITIES: dict[str, dict[str, dict[str, Any]]] = {
    "20231107": {
        "lens_mesh": {
            "name": "20231107-PTA-surface-lens.wrl",
            "size_bytes": 62_228_260,
            "sha256": "1a9e5b150208d7f71bc4848b798d988ee52d17bd1aa12d063e5e38dbce7c6b20",
        },
        "tip_mesh": {
            "name": "20231107-PTA-surface-tip.wrl",
            "size_bytes": 86_771_279,
            "sha256": "3498710995d31ebfbcab7437a8df92069ca27728e2347ac01380a46288fa68b5",
        },
        "rdata": {
            "name": "20231107.RData",
            "size_bytes": 296_662,
            "sha256": "e8cf395625b5fa2a415206614bd4e1a4d74d5372820038135bf77f312349a9ba",
        },
    },
    "20240530": {
        "lens_mesh": {
            "name": "20240530-PTA-surface-lens.wrl",
            "size_bytes": 328_800_390,
            "sha256": "671cebccc1561ed453e23f0cc58180dd29140f00fe80cc1fabed4054e2093c40",
        },
        "tip_mesh": {
            "name": "20240530-PTA-surface-tip.wrl",
            "size_bytes": 263_832_485,
            "sha256": "aa7f7da9fbb424c7e15d59e14505b39bf64095bc43555717128cce3cd6acc55c",
        },
        "rdata": {
            "name": "20240530.RData",
            "size_bytes": 293_647,
            "sha256": "6921aa284338b476cd7db14aea1e0fef19cc45bb141ea19d4baaf03f8d6ca684",
        },
    },
    "20240701": {
        "lens_mesh": {
            "name": "20240701-PTA-surface-lens.wrl",
            "size_bytes": 330_975_296,
            "sha256": "d4d7ab34517e744c5ce1a1d56f6b4fba0deaffcb32b4c9dbbfebf87529beecfa",
        },
        "tip_mesh": {
            "name": "20240701-PTA-surface-tip.wrl",
            "size_bytes": 256_352_807,
            "sha256": "1948916674147559ed23751016717b9b3cfc205eb37d4e691b70cd59b6253f08",
        },
        "rdata": {
            "name": "20240701.RData",
            "size_bytes": 310_361,
            "sha256": "8408447f8c5798fbc6d751de79ef4983d08c342a500c9a55b2a3b4b38d92deec",
        },
    },
}
RAW_PIXEL_PITCH_UM = {
    "20231107": 0.9882,
    "20240530": 0.9884,
    # The public deposition contains a 20240710 acquisition at 1.0065 um,
    # not an established mapping to Arthur's supplied 20240701 meshes.
    "20240701": None,
}

TARGET_CONFIG: dict[str, Any] = {
    "target_min_support": 25,
    "target_fit_rmse_max_um": 2.5,
    "target_q05_thickness_min_um_exclusive": 0.0,
    "target_fit_domain": (
        "supplied_proximal_surface_vertices_within_distal_q90_final_frame"
    ),
    "target_support_unit": "unique_proximal_mesh_vertices",
    "target_observation_type": "oracle_split_surface_mesh",
    "target_fit_method": "shared_huber_irls_normalized_quadratic",
    "target_coefficient_convention": "positive_distal_minus_proximal_thickness_c0_to_c5",
    "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
}

TABLE_FIELDS = (
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
    "stage1_eligible",
    "distal_qc",
    "distal_qc_reasons",
    "target_resolvable",
    "target_resolvability_reasons",
    "target_qc",
    "target_qc_reasons",
    "central",
    "position_u_um",
    "position_v_um",
    "distal_scale_um",
    "distal_gradient_magnitude",
    "distal_curvature_eigenvalue_1",
    "distal_curvature_eigenvalue_2",
    "distal_normalized_fit_residual",
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
    "sealed_distal_relpath",
    "lens_target_relpath",
)

CODE_PATHS = (
    "experiments/maike-modern-ground-truth/prepare_arthur_source_table.py",
    "experiments/maike-modern-ground-truth/distal_only_geometry.py",
    "experiments/maike-modern-ground-truth/audit_distal_frame_stability.py",
    "experiments/arthur-modern-ground-truth/experiment_57_outer_only_validation.py",
    "experiments/arthur-modern-ground-truth/experiment_58_cross_volume_confirmation.py",
)


class SourcePreparationError(RuntimeError):
    """Raised when an Arthur source-data invariant is not satisfied."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_binding(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def canonical_grid_xy() -> np.ndarray:
    axis = np.arange(-0.65, 0.65 + 0.13 / 2.0, 0.13, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    keep = xx * xx + yy * yy <= 0.65 * 0.65 + 1.0e-12
    grid = np.column_stack((xx[keep], yy[keep]))
    if grid.shape != (81, 2):
        raise SourcePreparationError("canonical Experiment 57 grid is not 81 by 2")
    return grid


def quadratic_design(xy: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(xy, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise SourcePreparationError("quadratic coordinates must have shape (n, 2)")
    x, y = coordinates.T
    return np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))


CANONICAL_GRID_XY = canonical_grid_xy()
CANONICAL_DESIGN = quadratic_design(CANONICAL_GRID_XY)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TABLE_FIELDS), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise SourcePreparationError(
            f"git {' '.join(arguments)} failed in {repository}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def git_identity(repository: Path) -> dict[str, Any]:
    commit = _run_git(repository, "rev-parse", "HEAD")
    dirty = bool(_run_git(repository, "status", "--porcelain=v1", "--untracked-files=all"))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SourcePreparationError(f"invalid git commit identity {commit!r}")
    return {"commit": commit, "dirty": dirty}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourcePreparationError(f"cannot read manifest {path}: {exc}") from exc
    rows = document.get("volumes") if isinstance(document, Mapping) else None
    if not isinstance(rows, list):
        raise SourcePreparationError("manifest must contain a volumes array")
    required = {"volume", "lens_mesh", "tip_mesh", "rdata"}
    if any(not isinstance(row, Mapping) or set(row) != required for row in rows):
        raise SourcePreparationError(
            f"every manifest row must contain exactly {sorted(required)}"
        )
    by_volume = {str(row["volume"]): dict(row) for row in rows}
    if len(by_volume) != len(rows) or set(by_volume) != set(VOLUMES):
        raise SourcePreparationError(f"manifest must contain exactly {list(VOLUMES)}")
    result = []
    for volume in VOLUMES:
        row = by_volume[volume]
        resolved = {"volume": volume}
        for key in ("lens_mesh", "tip_mesh", "rdata"):
            candidate = Path(row[key]).expanduser().resolve()
            if not candidate.is_file():
                raise SourcePreparationError(f"missing {key} for {volume}: {candidate}")
            expected = EXPECTED_INPUT_IDENTITIES[volume][key]
            observed = {"name": candidate.name, **artifact_binding(candidate)}
            if observed != expected:
                raise SourcePreparationError(
                    f"{volume} {key} identity differs from the frozen input: "
                    f"expected {expected}, got {observed}"
                )
            resolved[key] = candidate
        if len({resolved[key] for key in ("lens_mesh", "tip_mesh", "rdata")}) != 3:
            raise SourcePreparationError(f"{volume} manifest paths must be distinct")
        result.append(resolved)
    return result


def write_sealed_distal(
    path: Path,
    *,
    volume: str,
    eye_id: int,
    lens_index: int,
    points_xyz_um: np.ndarray,
    threshold_config: Mapping[str, Any],
) -> dict[str, Any]:
    points = np.asarray(points_xyz_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise SourcePreparationError("Arthur Stage-1 distal cap must be float64[N,3]")
    if not np.all(np.isfinite(points)):
        raise SourcePreparationError("Arthur Stage-1 distal cap is nonfinite")
    # WRL coordinate arrays can repeat vertices along mesh seams.  Canonical
    # sealed caps represent geometric samples, not face-incidence weights, so
    # exact duplicates are removed before support is checked and before any
    # downstream fit. np.unique also establishes lexicographic XYZ order.
    points = np.unique(points, axis=0)
    if len(points) < 25:
        raise SourcePreparationError(
            "Arthur Stage-1 distal cap has fewer than 25 unique vertices"
        )
    config_json = distal_geometry.canonical_config_json(threshold_config)
    config_hash = sha256_bytes(config_json.encode("utf-8"))
    _atomic_npz(
        path,
        schema_version=np.asarray(SEALED_DISTAL_SCHEMA),
        volume=np.asarray(volume),
        eye_id=np.asarray(eye_id, dtype=np.int64),
        lens_index=np.asarray(lens_index, dtype=np.int64),
        points_xyz_um=points,
        config_json=np.asarray(config_json),
        config_sha256=np.asarray(config_hash),
    )
    return load_sealed_distal(path, threshold_config)


def _scalar(array: np.ndarray, label: str) -> Any:
    if array.shape != ():
        raise SourcePreparationError(f"{label} must be a scalar array")
    return array.item()


def load_sealed_distal(
    path: Path, threshold_config: Mapping[str, Any]
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "volume",
        "eye_id",
        "lens_index",
        "points_xyz_um",
        "config_json",
        "config_sha256",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise SourcePreparationError("Arthur sealed-distal keys differ from contract")
            schema = str(_scalar(archive["schema_version"], "schema_version"))
            volume = str(_scalar(archive["volume"], "volume"))
            eye_id = int(_scalar(archive["eye_id"], "eye_id"))
            lens_index = int(_scalar(archive["lens_index"], "lens_index"))
            points = np.asarray(archive["points_xyz_um"])
            config_json = str(_scalar(archive["config_json"], "config_json"))
            config_hash = str(_scalar(archive["config_sha256"], "config_sha256"))
    except (OSError, ValueError) as exc:
        if isinstance(exc, SourcePreparationError):
            raise
        raise SourcePreparationError(f"cannot read sealed distal {path}: {exc}") from exc
    if schema != SEALED_DISTAL_SCHEMA or volume not in VOLUMES or eye_id not in (0, 1):
        raise SourcePreparationError(f"invalid Arthur sealed-distal identity in {path}")
    if lens_index < 0 or points.dtype != np.dtype("float64") or points.ndim != 2:
        raise SourcePreparationError(f"invalid Arthur sealed-distal arrays in {path}")
    if points.shape[1:] != (3,) or len(points) < 25 or not np.all(np.isfinite(points)):
        raise SourcePreparationError(f"invalid Arthur distal points in {path}")
    if not np.array_equal(points, np.unique(points, axis=0)):
        raise SourcePreparationError(
            f"Arthur distal points are duplicated or not in canonical XYZ order in {path}"
        )
    if config_json != distal_geometry.canonical_config_json(threshold_config):
        raise SourcePreparationError(f"threshold config differs in {path}")
    if config_hash != sha256_bytes(config_json.encode("utf-8")):
        raise SourcePreparationError(f"config hash mismatch in {path}")
    return {
        "schema_version": schema,
        "volume": volume,
        "eye_id": eye_id,
        "lens_index": lens_index,
        "points_xyz_um": points.copy(),
        "config": dict(threshold_config),
        "config_json": config_json,
        "config_sha256": config_hash,
        "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "sealed_distal_artifact": True,
        "stage1_eligible": True,
    }


def fit_source_target(
    distal_points_xyz_um: np.ndarray,
    proximal_points_xyz_um: np.ndarray,
    geometry: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct one target in the final distal-only frame.

    Target availability and target quality are deliberately separate.  The
    former requires support and finite fit quantities.  Positive q05 thickness
    and the 2.5-um proximal-fit RMSE are quality filters used only by the
    prespecified sensitivity analysis; neither can feed back into distal QC.
    """

    empty = {
        "target_resolvable": False,
        "target_qc": False,
        "target_resolvability_reasons": [],
        "target_qc_reasons": [],
        "target_coefficients": np.full(6, np.nan, dtype=np.float64),
        "target_depth_um": math.nan,
        "target_q05_raw_thickness_um": math.nan,
        "target_support": 0,
        "target_rmse_um": math.nan,
        "canonical_grid_xy": CANONICAL_GRID_XY.copy(),
        "target_smoothed_thickness_um": np.full(81, np.nan, dtype=np.float64),
        "raw_target_xy_normalized": np.empty((0, 2), dtype=np.float64),
        "raw_target_thickness_um": np.empty(0, dtype=np.float64),
    }
    distal = np.asarray(distal_points_xyz_um, dtype=np.float64)
    proximal = np.asarray(proximal_points_xyz_um, dtype=np.float64)
    if distal.ndim != 2 or distal.shape[1:] != (3,) or len(distal) < 25:
        empty["target_resolvability_reasons"] = ["invalid_distal_points"]
        empty["target_qc_reasons"] = ["invalid_distal_points"]
        return empty
    if proximal.ndim != 2 or proximal.shape[1:] != (3,) or not np.all(
        np.isfinite(proximal)
    ):
        empty["target_resolvability_reasons"] = ["invalid_proximal_points"]
        empty["target_qc_reasons"] = ["invalid_proximal_points"]
        return empty
    proximal = np.unique(proximal, axis=0)
    origin = np.asarray(geometry["origin"], dtype=np.float64)
    u = np.asarray(geometry["u"], dtype=np.float64)
    v = np.asarray(geometry["v"], dtype=np.float64)
    w = np.asarray(geometry["w"], dtype=np.float64)
    scale = float(geometry["scale_um"])
    distal_beta = np.asarray(geometry["quadratic_beta_normalized"], dtype=np.float64)
    if (
        origin.shape != (3,)
        or u.shape != (3,)
        or v.shape != (3,)
        or w.shape != (3,)
        or distal_beta.shape != (6,)
        or not np.all(np.isfinite(distal_beta))
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        empty["target_resolvability_reasons"] = ["invalid_final_distal_frame"]
        empty["target_qc_reasons"] = ["invalid_final_distal_frame"]
        return empty

    # Arthur supplied an already extracted proximal surface mesh, rather than
    # Maike's filled binary lens volume.  Its unique proximal vertices are the
    # available surface observations; imposing Maike's axial-span voxel-column
    # sampler on this sparse boundary representation would make target support
    # depend on unrelated distal/proximal mesh tessellation alignment.  Both
    # adapters nevertheless use the same final distal-q90 frame, robust
    # normalized quadratic fitter, coefficient convention, and 81-point grid.
    relative = proximal - origin
    local = np.column_stack((relative @ u, relative @ v, relative @ w))
    xy_all = local[:, :2] / scale
    keep = np.linalg.norm(xy_all, axis=1) <= 1.0 + 1.0e-10
    xy = xy_all[keep]
    proximal_z = local[keep, 2]
    support = int(len(xy))
    empty["target_support"] = support
    empty["raw_target_xy_normalized"] = xy
    raw_thickness = quadratic_design(xy) @ distal_beta - proximal_z
    empty["raw_target_thickness_um"] = raw_thickness
    reasons: list[str] = []
    if support < int(TARGET_CONFIG["target_min_support"]):
        reasons.append("target_support_below_minimum")
    if not (
        np.all(np.isfinite(xy))
        and np.all(np.isfinite(proximal_z))
        and np.all(np.isfinite(raw_thickness))
    ):
        reasons.append("nonfinite_target")
    if reasons:
        empty["target_resolvability_reasons"] = sorted(set(reasons))
        empty["target_qc_reasons"] = sorted(set(reasons))
        return empty

    proximal_fit = distal_geometry.fit_robust_quadratic(
        xy[:, 0], xy[:, 1], proximal_z, config=threshold_config
    )
    proximal_beta = np.asarray(proximal_fit["beta"], dtype=np.float64)
    proximal_rmse = float(proximal_fit["rmse_um"])
    coefficients = distal_beta - proximal_beta
    smoothed = CANONICAL_DESIGN @ coefficients
    if not (
        np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(raw_thickness))
        and np.all(np.isfinite(smoothed))
        and math.isfinite(proximal_rmse)
    ):
        empty["target_resolvability_reasons"] = ["nonfinite_target_fit"]
        empty["target_qc_reasons"] = ["nonfinite_target_fit"]
        return empty

    target_depth = float(np.median(raw_thickness))
    q05 = float(np.quantile(raw_thickness, 0.05))
    common = {
        "target_coefficients": coefficients,
        "target_depth_um": target_depth,
        "target_q05_raw_thickness_um": q05,
        "target_support": support,
        "target_rmse_um": proximal_rmse,
        "canonical_grid_xy": CANONICAL_GRID_XY.copy(),
        "target_smoothed_thickness_um": smoothed,
        "raw_target_xy_normalized": xy,
        "raw_target_thickness_um": raw_thickness,
    }
    if not q05 > float(TARGET_CONFIG["target_q05_thickness_min_um_exclusive"]):
        reasons.append("target_q05_raw_thickness_not_positive")
    if proximal_rmse > float(TARGET_CONFIG["target_fit_rmse_max_um"]):
        reasons.append("target_rmse_above_maximum")
    return {
        "target_resolvable": True,
        "target_qc": not reasons,
        "target_resolvability_reasons": [],
        "target_qc_reasons": sorted(set(reasons)),
        **common,
    }


def fixed_point_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "converged",
        "max_iterations",
        "iterations",
        "eligible_counts",
        "readded_count",
    )
    if any(key not in result for key in required):
        raise SourcePreparationError("shared geometry omitted fixed-point evidence")
    evidence = {key: result[key] for key in required}
    if not evidence["converged"] or evidence["readded_count"] != 0:
        raise SourcePreparationError("Arthur distal-QC fixed point failed")
    reason_counts: Counter[str] = Counter()
    for row in result.get("per_lens", []):
        reason_counts.update(map(str, row.get("drop_reasons", [])))
    evidence.update(
        {
            "initial_count": len(result.get("initial_record_indices", [])),
            "eligible_count": len(result.get("eligible_indices", [])),
            "excluded_count": len(result.get("initial_record_indices", []))
            - len(result.get("eligible_indices", [])),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "iteration_diagnostics": result.get("iteration_diagnostics", []),
        }
    )
    return evidence


def _empty_row(volume: str, eye_id: int, lens_index: int, sealed_relpath: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "volume": volume,
        "animal_id": f"Arthur_{volume}",
        "eye_id": eye_id,
        "source_eye_unit": f"{volume}:eye_{eye_id}",
        "lens_index": lens_index,
        "species": "Drosophila melanogaster",
        "sex": "female",
        "age_days_min": 6,
        "age_days_max": 7,
        "raw_pixel_pitch_um": (
            RAW_PIXEL_PITCH_UM[volume]
            if RAW_PIXEL_PITCH_UM[volume] is not None
            else math.nan
        ),
        "stage1_eligible": True,
        "distal_qc": False,
        "distal_qc_reasons": "",
        "target_resolvable": False,
        "target_resolvability_reasons": "distal_qc_failed",
        "target_qc": False,
        "target_qc_reasons": "distal_qc_failed",
        "central": False,
        "position_u_um": math.nan,
        "position_v_um": math.nan,
        "distal_scale_um": math.nan,
        "distal_gradient_magnitude": math.nan,
        "distal_curvature_eigenvalue_1": math.nan,
        "distal_curvature_eigenvalue_2": math.nan,
        "distal_normalized_fit_residual": math.nan,
        "target_depth_um": math.nan,
        "target_q05_raw_thickness_um": math.nan,
        "target_support": 0,
        "target_rmse_um": math.nan,
        "sealed_distal_relpath": sealed_relpath,
        "lens_target_relpath": "",
    }
    for index in range(6):
        row[f"target_c{index}"] = math.nan
    return row


def _process_eye(
    *,
    volume: str,
    eye_id: int,
    stage1_records: Sequence[Mapping[str, Any]],
    staging: Path,
    threshold_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    unit = f"{volume}:eye_{eye_id}"
    distal_records: list[dict[str, Any]] = []
    proximal_surface_points: dict[int, np.ndarray] = {}
    distal_bindings: dict[str, Any] = {}
    for record in stage1_records:
        lens_index = int(record["landmark_id"])
        relpath = f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        path = staging / relpath
        loaded = write_sealed_distal(
            path,
            volume=volume,
            eye_id=eye_id,
            lens_index=lens_index,
            points_xyz_um=np.asarray(record["outer"], dtype=np.float64),
            threshold_config=threshold_config,
        )
        distal_records.append(loaded)
        proximal_surface_points[lens_index] = np.unique(
            np.asarray(record["inner"], dtype=np.float64), axis=0
        )
        distal_bindings[relpath] = artifact_binding(path)

    fixed = distal_geometry.run_monotone_fixed_point(
        distal_records, config=threshold_config
    )
    evidence = fixed_point_evidence(fixed)
    eligible_indices = [int(value) for value in fixed["eligible_indices"]]
    geometry_rows = fixed["geometry"]["per_lens"]
    geometry_by_lens = {
        int(record["lens_index"]): geometry
        for record, geometry in zip(
            fixed["eligible_records"], geometry_rows, strict=True
        )
    }
    fixed_by_lens = {int(row["lens_index"]): row for row in fixed["per_lens"]}
    eligible_set = set(eligible_indices)
    distal_by_lens = {int(row["lens_index"]): row for row in distal_records}

    # This audit reruns the same target-blind fixed point and perturbs only
    # distal origins.  It receives the verified sealed records, never targets.
    audit = audit_frame_stability(distal_records, eye_id=unit, config=threshold_config)
    if not audit["gate_passed"]:
        raise SourcePreparationError(f"distal frame stability gate failed for {unit}")
    audit_fixed = audit.get("fixed_point", {})
    for key in ("converged", "max_iterations", "iterations", "eligible_counts", "readded_count"):
        if audit_fixed.get(key) != evidence[key]:
            raise SourcePreparationError(
                f"frame audit fixed point differs from source preparation for {unit}"
            )
    # Use the richer source evidence in both places so the independently rerun
    # audit and provenance are literally identical, not merely count-equal.
    audit["fixed_point"] = evidence
    for entry in audit["input_artifacts"]:
        lens_index = int(entry["lens_index"])
        entry["relative_path"] = (
            f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        )
    audit_relpath = f"distal_frame_audits/{volume}_eye_{eye_id}.json"
    audit_path = staging / audit_relpath
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(audit_path, audit)

    rows = []
    target_bindings: dict[str, Any] = {}
    for record in stage1_records:
        lens_index = int(record["landmark_id"])
        sealed_relpath = f"sealed_distal/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        row = _empty_row(volume, eye_id, lens_index, sealed_relpath)
        fixed_row = fixed_by_lens[lens_index]
        row["distal_qc"] = lens_index in eligible_set
        row["distal_qc_reasons"] = "|".join(fixed_row.get("drop_reasons", []))
        geometry = geometry_by_lens.get(lens_index)
        if geometry is None:
            rows.append(row)
            continue

        position = np.asarray(geometry["position_2d_um"], dtype=np.float64)
        curvature = np.asarray(geometry["curvature_eigenvalues"], dtype=np.float64)
        row.update(
            {
                "central": bool(geometry["central"]),
                "position_u_um": float(position[0]),
                "position_v_um": float(position[1]),
                "distal_scale_um": float(geometry["scale_um"]),
                "distal_gradient_magnitude": float(geometry["gradient_magnitude"]),
                "distal_curvature_eigenvalue_1": float(curvature[0]),
                "distal_curvature_eigenvalue_2": float(curvature[1]),
                "distal_normalized_fit_residual": float(geometry["normalised_rmse"]),
            }
        )
        distal_record = distal_by_lens[lens_index]
        target = fit_source_target(
            np.asarray(distal_record["points_xyz_um"], dtype=np.float64),
            proximal_surface_points[lens_index],
            geometry,
            threshold_config,
        )
        coefficients = np.asarray(target["target_coefficients"], dtype=np.float64)
        row.update(
            {
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
            }
        )
        for coefficient_index in range(6):
            row[f"target_c{coefficient_index}"] = float(coefficients[coefficient_index])
        target_relpath = f"lenses/{volume}/eye_{eye_id}/lens_{lens_index:06d}.npz"
        target_path = staging / target_relpath
        _atomic_npz(
            target_path,
            schema_version=np.asarray(LENS_TARGET_SCHEMA),
            volume=np.asarray(volume),
            eye_id=np.asarray(eye_id, dtype=np.int64),
            lens_index=np.asarray(lens_index, dtype=np.int64),
            canonical_grid_xy=np.asarray(target["canonical_grid_xy"], dtype=np.float64),
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
        )
        row["lens_target_relpath"] = target_relpath
        target_bindings[target_relpath] = artifact_binding(target_path)
        rows.append(row)
    frame_binding = {"relative_path": audit_relpath, **artifact_binding(audit_path)}
    artifact_bindings = {
        **distal_bindings,
        **target_bindings,
        audit_relpath: artifact_binding(audit_path),
    }
    return rows, evidence, frame_binding, artifact_bindings


def _validate_expected_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "stage1": len(rows),
        "distal_qc": sum(bool(row["distal_qc"]) for row in rows),
        "target_resolvable": sum(bool(row["target_resolvable"]) for row in rows),
        "target_qc": sum(bool(row["target_qc"]) for row in rows),
    }
    protected_counts = {key: counts[key] for key in EXPECTED_SOURCE_COUNTS}
    if protected_counts != EXPECTED_SOURCE_COUNTS:
        raise SourcePreparationError(
            "source Stage-1/distal-QC counts changed: "
            f"expected {EXPECTED_SOURCE_COUNTS}, got {protected_counts}"
        )
    return counts


def _cohort_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "stage1": len(rows),
        "distal_qc": sum(bool(row["distal_qc"]) for row in rows),
        "target_resolvable": sum(bool(row["target_resolvable"]) for row in rows),
        "target_qc": sum(bool(row["target_qc"]) for row in rows),
    }


def _reason_counts(
    rows: Sequence[Mapping[str, Any]], field: str, *, include_empty: bool = False
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row.get(field, ""))
        reasons = [reason for reason in raw.split("|") if reason]
        if not reasons and include_empty:
            reasons = ["pass"]
        counts.update(reasons)
    return dict(sorted(counts.items()))


def build_source_bundle(
    *,
    manifest_path: Path,
    eyemap_root: Path,
    output_directory: Path,
    repository_root: Path = REPOSITORY_ROOT,
    allow_dirty_diagnostic: bool = False,
) -> Path:
    """Build one immutable source bundle and return its final directory."""

    repository_root = repository_root.resolve()
    eyemap_root = eyemap_root.resolve()
    git = git_identity(repository_root)
    if git["dirty"] and not allow_dirty_diagnostic:
        raise SourcePreparationError(
            "repository is dirty; commit the frozen implementation before source preparation"
        )
    eyemap_git = git_identity(eyemap_root)
    if eyemap_git["commit"] != EYEMAP_COMMIT or eyemap_git["dirty"]:
        raise SourcePreparationError(
            f"eyemap_T4 must be clean at frozen commit {EYEMAP_COMMIT}"
        )
    manifest = load_manifest(manifest_path)
    if output_directory.exists():
        raise SourcePreparationError(f"refusing to overwrite {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging.", dir=output_directory.parent
        )
    )
    threshold_config = dict(distal_geometry.DEFAULT_CONFIG)
    try:
        all_rows: list[dict[str, Any]] = []
        stage1_diagnostics: dict[str, Any] = {}
        fixed_points: dict[str, Any] = {}
        frame_audits: dict[str, Any] = {}
        input_files: dict[str, Any] = {}
        output_artifacts: dict[str, Any] = {}
        for manifest_row in manifest:
            volume = str(manifest_row["volume"])
            lens_mesh = Path(manifest_row["lens_mesh"])
            tip_mesh = Path(manifest_row["tip_mesh"])
            rdata = Path(manifest_row["rdata"])
            _, lens_surfaces = parse_wrl_surfaces(lens_mesh)
            _, tip_surfaces = parse_wrl_surfaces(tip_mesh)
            if len(tip_surfaces) != 2:
                raise SourcePreparationError(f"{volume} tip mesh does not contain two eyes")
            lens_positions, tip_positions = raw_positions_from_rdata(rdata)
            stage1, diagnostics = prepare_oracle_split_records(
                lens_surfaces, lens_positions, tip_positions
            )
            if diagnostics["n_landmarks"] != EXPECTED_LANDMARK_COUNTS[volume]:
                raise SourcePreparationError(f"{volume} landmark count changed")
            if len(stage1) != EXPECTED_STAGE1_COUNTS[volume]:
                raise SourcePreparationError(f"{volume} Stage-1 cohort count changed")
            stage1_diagnostics[volume] = diagnostics
            input_files[volume] = {
                "lens_mesh": {"name": lens_mesh.name, **artifact_binding(lens_mesh)},
                "tip_mesh": {"name": tip_mesh.name, **artifact_binding(tip_mesh)},
                "rdata": {"name": rdata.name, **artifact_binding(rdata)},
            }
            for eye_id in (0, 1):
                eye_records = [record for record in stage1 if int(record["eye"]) == eye_id]
                rows, fixed, audit_binding, artifacts = _process_eye(
                    volume=volume,
                    eye_id=eye_id,
                    stage1_records=eye_records,
                    staging=staging,
                    threshold_config=threshold_config,
                )
                unit = f"{volume}:eye_{eye_id}"
                all_rows.extend(rows)
                fixed_points[unit] = fixed
                frame_audits[unit] = audit_binding
                output_artifacts.update(artifacts)

        all_rows.sort(key=lambda row: (VOLUMES.index(str(row["volume"])), int(row["eye_id"]), int(row["lens_index"])))
        counts = _validate_expected_counts(all_rows)
        counts_by_volume = {
            volume: _cohort_counts(
                [row for row in all_rows if str(row["volume"]) == volume]
            )
            for volume in VOLUMES
        }
        table_path = staging / "arthur_source_table.csv"
        _atomic_csv(table_path, all_rows)
        table_binding = artifact_binding(table_path)
        output_artifacts["arthur_source_table.csv"] = table_binding

        pipeline_config = {
            "fixed_point_policy": "monotone_drop_only_no_reentry",
            **TARGET_CONFIG,
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "status": "diagnostic" if git["dirty"] else "complete",
            "analysis_scope": ANALYSIS_SCOPE,
            "isolation_basis": ISOLATION_BASIS,
            "oracle_stage1_scope": ORACLE_STAGE1_SCOPE,
            "table_sha256": table_binding["sha256"],
            "table_size_bytes": table_binding["size_bytes"],
            "n_rows": len(all_rows),
            "counts": counts,
            "counts_by_volume": counts_by_volume,
            "cohorts": {
                "both_models_and_primary_targets": counts["target_resolvable"],
                "both_models_target_qc_sensitivity": counts["target_qc"],
                "cross_validation_independence_unit": "volume_animal",
                "n_cross_validation_units": 3,
                "eyes_are_nested_within_volume_animal": True,
            },
            "exclusion_reason_counts": {
                "distal_qc": _reason_counts(all_rows, "distal_qc_reasons"),
                "target_resolvability": _reason_counts(
                    all_rows, "target_resolvability_reasons"
                ),
                "target_qc": _reason_counts(all_rows, "target_qc_reasons"),
            },
            "volumes": list(VOLUMES),
            "threshold_config": threshold_config,
            "threshold_config_sha256": sha256_bytes(
                canonical_json(threshold_config).encode("utf-8")
            ),
            "pipeline_config": pipeline_config,
            "pipeline_config_sha256": sha256_bytes(
                canonical_json(pipeline_config).encode("utf-8")
            ),
            "target_observation_contract": {
                "common_latent_response": (
                    "positive distal-minus-proximal thickness represented by an "
                    "unsymmetrized six-coefficient robust quadratic in the final "
                    "distal-only q90 frame and evaluated on the common 81-point grid"
                ),
                "arthur_source_operator": (
                    "unique vertices of the supplied oracle-split proximal surface "
                    "mesh inside the final distal q90 domain"
                ),
                "maike_test_operator": (
                    "one minimum-axial proximal voxel per 0.325-um lateral bin with "
                    "at least 0.650-um axial span inside the final distal q90 domain"
                ),
                "representation_shift": True,
                "why_operators_differ": (
                    "Arthur provides sparse extracted boundary meshes whereas Maike "
                    "provides filled binary lens volumes; axial-column sampling is "
                    "not defined by corresponding filled columns in a surface mesh"
                ),
                "interpretation_limit": (
                    "an external-transfer failure cannot by itself separate biological "
                    "domain shift from source/test observation-representation shift"
                ),
            },
            "fixed_points": fixed_points,
            "frame_audits": frame_audits,
            "stage1_diagnostics": stage1_diagnostics,
            "input_files": input_files,
            "output_artifacts": output_artifacts,
            "git": git,
            "code_sha256": {
                relpath: sha256_file(repository_root / relpath) for relpath in CODE_PATHS
            },
            "eyemap": {"commit": eyemap_git["commit"], "dirty": eyemap_git["dirty"]},
            "biological_independence": {
                "independent_unit": "whole_head_scan_animal",
                "n_independent_animals": 3,
                "eyes_per_animal": 2,
                "nesting": "bilateral_eyes_nested_within_animal",
                "animals": [f"Arthur_{volume}" for volume in VOLUMES],
                "sex": "female",
                "age_days": [6, 7],
            },
            "coordinate_calibration": {
                "20231107_raw_pixel_pitch_um": 0.9882,
                "20240530_raw_pixel_pitch_um": 0.9884,
                "20240701_raw_pixel_pitch_um": None,
                "20240701_note": (
                    "unresolved: public deposit identifies 20240710 at 1.0065 um; "
                    "no specimen mapping was assumed"
                ),
                "mesh_coordinates": "physical_micrometres_from_supplied_WRL",
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(staging / "arthur_source_provenance.json", provenance)
        os.replace(staging, output_directory)
        return output_directory
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eyemap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--allow-dirty-diagnostic",
        action="store_true",
        help="write a backend-rejected source-side diagnostic bundle",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    output = build_source_bundle(
        manifest_path=arguments.manifest,
        eyemap_root=arguments.eyemap_root,
        output_directory=arguments.output,
        repository_root=arguments.repository_root,
        allow_dirty_diagnostic=arguments.allow_dirty_diagnostic,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

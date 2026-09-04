#!/usr/bin/env python3
"""Frozen primary backend for Experiment 63.

This module answers one deliberately narrow question: conditional on an oracle
having identified each distal corneal-lens cap, do distal-cap shape descriptors
improve prediction of the hidden proximal thickness surface beyond a nested,
rotation/reflection-invariant position-and-scale control?

The backend is intentionally fail closed.  It accepts only hash-bound producer
bundles made at the same clean Git commit, validates every sealed distal artifact
semantically, trains exclusively on Arthur Zhao's three source volumes, and
compares the nested models once in twelve independent Maike Kittelmann eyes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, distance


ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
ISOLATION_BASIS = "stage2_reads_only_sha256_sealed_distal_artifacts"
EYE_BUNDLE_SCHEMA = "experiment63-eye-bundle-v2"
SEALED_DISTAL_SCHEMA = "experiment63.sealed-distal.v2"
ARTHUR_SOURCE_SCHEMA = "experiment63.arthur-source.v2"
ARTHUR_SEALED_DISTAL_SCHEMA = "experiment63.arthur-sealed-distal.v1"
ARTHUR_TARGET_SCHEMA = "experiment63.arthur-target.v1"
FRAME_AUDIT_SCHEMA = "experiment63.distal-frame-audit.v1"
ATTESTATION_SCHEMA = "experiment63.instance-qc-attestation.v1"
RUN_SCHEMA = "experiment63.primary-run.v1"
MAIKE_MASK_PROVENANCE_SCHEMA = "maike-mask-provenance-v2"
MAIKE_SEED_PROVENANCE_SCHEMA = "experiment63-oda-source-map-v3"
MAIKE_SEED_ROLE = "oracle_correspondence_and_distal_localization_stage1_only"

EXPECTED_EYES: dict[str, int] = {
    "M3_F_24_01": 1001,
    "M3_F_28_03": 1011,
    "M3_F_35_03": 1023,
    "M3_M_26_01": 855,
    "M3_M_32_01": 944,
    "M3_M_36_01": 970,
    "RED3_25_F_36": 1008,
    "RED3_25_F_37": 984,
    "RED3_25_F_38": 1003,
    "RED3_25_M_26": 882,
    "RED3_25_M_27": 822,
    "RED3_25_M_28": 866,
}
EXPECTED_TOTAL = 11_369
ARTHUR_VOLUMES = ("20231107", "20240530", "20240701")
ARTHUR_STAGE1_COUNTS = {"20231107": 1622, "20240530": 1611, "20240701": 1709}
ARTHUR_LANDMARK_COUNTS = {"20231107": 1632, "20240530": 1611, "20240701": 1709}
MAIKE_SOURCE_ARCHIVES: dict[str, dict[str, Any]] = {
    "M3_F_24_01": {
        "name": "tiffs_M3_F_24_01_eye_lenses-20260903T135137Z-1-001.zip",
        "sha256": "a1af8a4b3f04f9b22f4f416854189072b14dd59b13b6708048d4766580e4a638",
        "size_bytes": 8_656_592,
    },
    "M3_F_28_03": {
        "name": "tiffs_M3_F_28_03_eye_lenses-20260903T135112Z-1-001.zip",
        "sha256": "d18763929208c1fff12ada55124e1f489cd420ec66f101ebb9309656fce38ffa",
        "size_bytes": 7_929_219,
    },
    "M3_F_35_03": {
        "name": "tiffs_M3_F_35_03_eye_lenses-20260903T135109Z-1-001.zip",
        "sha256": "6a205f1d691e88829861fd56d16269b4c6867d1ae7bd3f14ca558279b18ccef0",
        "size_bytes": 8_363_277,
    },
    "M3_M_26_01": {
        "name": "tiffs_M3_M_26_01_eye_lenses-20260903T135105Z-1-001.zip",
        "sha256": "69b0206668e185d333f5fcdf8302a85cec048e4bb39ad93cd54001d5ec415f37",
        "size_bytes": 6_172_316,
    },
    "M3_M_32_01": {
        "name": "tiffs_M3_M_32_01_eye_lenses-20260903T135052Z-1-001.zip",
        "sha256": "adc4aa40945bcba20956679262b211444ec307391361f775e15ab4555e797d51",
        "size_bytes": 6_382_619,
    },
    "M3_M_36_01": {
        "name": "tiffs_M3_M_36_01_eye_lenses-20260903T135121Z-1-001.zip",
        "sha256": "024cb4ba7de8c56a98fa96bf881b67d94cdb49a2d0c43160cfdffe8b9a30f2bc",
        "size_bytes": 6_859_529,
    },
    "RED3_25_F_36": {
        "name": "tiffs_RED3_25_F_36_eye_lenses-20260903T135103Z-1-001.zip",
        "sha256": "9208899f9c2481ebbe43b760b3efe6dfb60cfeb8b3b7fe156c8ee7884feed495",
        "size_bytes": 11_412_517,
    },
    "RED3_25_F_37": {
        "name": "tiffs_RED3_25_F_37_eye_lenses-20260903T135100Z-1-001.zip",
        "sha256": "1d230daea5bacacc92f5ab5aec4da13c1d4b10e7a2aedc554ed3baf7cc0e3012",
        "size_bytes": 9_907_754,
    },
    "RED3_25_F_38": {
        "name": "tiffs_RED3_25_F_38_eye_lenses-20260903T135116Z-1-001.zip",
        "sha256": "b735f6767029f512ba094a7ffeb4c1d46a3cc04039022890cce1f54b4e8a6b57",
        "size_bytes": 10_740_551,
    },
    "RED3_25_M_26": {
        "name": "tiffs_RED3_25_M_26_eye_lenses-20260903T135055Z-1-001.zip",
        "sha256": "9c1ab44205610e940617e19249df8c5364a0b88115ff736b3c6a3952bd63b3de",
        "size_bytes": 8_599_547,
    },
    "RED3_25_M_27": {
        "name": "tiffs_RED3_25_M_27_eye_lenses-20260903T135132Z-1-001.zip",
        "sha256": "5997775e6366fc4c268eb1cc65e467b101cc07ecaad2c0d8d4363bd16729d4ef",
        "size_bytes": 9_110_763,
    },
    "RED3_25_M_28": {
        "name": "tiffs_RED3_25_M_28_eye_lenses-20260903T135127Z-1-001.zip",
        "sha256": "d3492e58cde9a6182729f0c9803fcc368f3bb7debdeb798d6e633f03febc40ce",
        "size_bytes": 8_047_098,
    },
}
MAIKE_SOURCE_STACK_GEOMETRY: dict[str, dict[str, Any]] = {
    "M3_F_24_01": {"slice_count": 864, "uncropped_shape_zyx": [864, 1635, 1740]},
    "M3_F_28_03": {"slice_count": 875, "uncropped_shape_zyx": [875, 1470, 1914]},
    "M3_F_35_03": {"slice_count": 1031, "uncropped_shape_zyx": [1031, 1425, 1602]},
    "M3_M_26_01": {"slice_count": 694, "uncropped_shape_zyx": [694, 1575, 1689]},
    "M3_M_32_01": {"slice_count": 714, "uncropped_shape_zyx": [714, 1587, 1485]},
    "M3_M_36_01": {"slice_count": 684, "uncropped_shape_zyx": [684, 1659, 1569]},
    "RED3_25_F_36": {"slice_count": 1158, "uncropped_shape_zyx": [1158, 1794, 2004]},
    "RED3_25_F_37": {"slice_count": 1172, "uncropped_shape_zyx": [1172, 1449, 2079]},
    "RED3_25_F_38": {"slice_count": 1135, "uncropped_shape_zyx": [1135, 1845, 1542]},
    "RED3_25_M_26": {"slice_count": 1040, "uncropped_shape_zyx": [1040, 1620, 1560]},
    "RED3_25_M_27": {"slice_count": 1252, "uncropped_shape_zyx": [1252, 1458, 1365]},
    "RED3_25_M_28": {"slice_count": 1283, "uncropped_shape_zyx": [1283, 1062, 1698]},
}
ARTHUR_INPUT_FILES: dict[str, dict[str, dict[str, Any]]] = {
    "20231107": {
        "lens_mesh": {
            "name": "20231107-PTA-surface-lens.wrl",
            "sha256": "1a9e5b150208d7f71bc4848b798d988ee52d17bd1aa12d063e5e38dbce7c6b20",
            "size_bytes": 62_228_260,
        },
        "tip_mesh": {
            "name": "20231107-PTA-surface-tip.wrl",
            "sha256": "3498710995d31ebfbcab7437a8df92069ca27728e2347ac01380a46288fa68b5",
            "size_bytes": 86_771_279,
        },
        "rdata": {
            "name": "20231107.RData",
            "sha256": "e8cf395625b5fa2a415206614bd4e1a4d74d5372820038135bf77f312349a9ba",
            "size_bytes": 296_662,
        },
    },
    "20240530": {
        "lens_mesh": {
            "name": "20240530-PTA-surface-lens.wrl",
            "sha256": "671cebccc1561ed453e23f0cc58180dd29140f00fe80cc1fabed4054e2093c40",
            "size_bytes": 328_800_390,
        },
        "tip_mesh": {
            "name": "20240530-PTA-surface-tip.wrl",
            "sha256": "aa7f7da9fbb424c7e15d59e14505b39bf64095bc43555717128cce3cd6acc55c",
            "size_bytes": 263_832_485,
        },
        "rdata": {
            "name": "20240530.RData",
            "sha256": "6921aa284338b476cd7db14aea1e0fef19cc45bb141ea19d4baaf03f8d6ca684",
            "size_bytes": 293_647,
        },
    },
    "20240701": {
        "lens_mesh": {
            "name": "20240701-PTA-surface-lens.wrl",
            "sha256": "d4d7ab34517e744c5ce1a1d56f6b4fba0deaffcb32b4c9dbbfebf87529beecfa",
            "size_bytes": 330_975_296,
        },
        "tip_mesh": {
            "name": "20240701-PTA-surface-tip.wrl",
            "sha256": "1948916674147559ed23751016717b9b3cfc205eb37d4e691b70cd59b6253f08",
            "size_bytes": 256_352_807,
        },
        "rdata": {
            "name": "20240701.RData",
            "sha256": "8408447f8c5798fbc6d751de79ef4983d08c342a500c9a55b2a3b4b38d92deec",
            "size_bytes": 310_361,
        },
    },
}

EXPECTED_THRESHOLD_CONFIG: dict[str, Any] = {
    "distal_scale_statistic": "q90_radius_um",
    "distal_scale_min_um": 3.0,
    "distal_scale_max_um": 13.0,
    "min_sealed_distal_cap_points": 25,
    "distal_fit_rmse_max_um": 2.5,
    "quadratic_design_condition_max": 1_000_000.0,
    "fixed_point_max_iterations": 20,
    "lateral_bin_um": 0.325,
    "min_axial_span_um": 0.650,
    "connectivity": 26,
    "candidate_seeds_per_voxel": 1,
}
EXPECTED_PIPELINE_CONFIG: dict[str, Any] = {
    "fixed_point_policy": "monotone_drop_only_no_reentry",
    "target_main_component_fraction_min": 0.99,
    "target_min_support": 25,
    "target_fit_rmse_max_um": 2.5,
    "target_q05_thickness_min_um_exclusive": 0.0,
    "original_spacing_um": [0.325, 0.325, 0.325],
    "distal_split": "largest_26_component_boundary_deterministic_1d_two_means",
    "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
}
EXPECTED_PREDICTOR_PIPELINE_CONFIG: dict[str, Any] = {
    key: EXPECTED_PIPELINE_CONFIG[key]
    for key in ("fixed_point_policy", "original_spacing_um", "distal_split", "canonical_grid")
}
EXPECTED_ARTHUR_PIPELINE_CONFIG: dict[str, Any] = {
    "fixed_point_policy": "monotone_drop_only_no_reentry",
    "target_min_support": 25,
    "target_fit_rmse_max_um": 2.5,
    "target_q05_thickness_min_um_exclusive": 0.0,
    "target_fit_domain": "supplied_proximal_surface_vertices_within_distal_q90_final_frame",
    "target_support_unit": "unique_proximal_mesh_vertices",
    "target_observation_type": "oracle_split_surface_mesh",
    "target_fit_method": "shared_huber_irls_normalized_quadratic",
    "target_coefficient_convention": "positive_distal_minus_proximal_thickness_c0_to_c5",
    "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
}
TARGET_OBSERVATION_CONTRACT: dict[str, Any] = {
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
}
PREDICTOR_INPUT_REPRESENTATION_CONTRACT: dict[str, Any] = {
    "arthur_source_distal_input": (
        "irregular float64 vertices from the supplied surface mesh after oracle distal localization"
    ),
    "maike_test_distal_input": (
        "int32 indices from a 0.325-um isotropic voxelized filled-volume boundary cap after oracle distal localization"
    ),
    "common_descriptor_map": (
        "the same distal-only fixed-point frame and scale, gradient magnitude, ordered curvature eigenvalues, and normalized fit residual"
    ),
    "representation_shift": True,
    "interpretation_limit": (
        "distal descriptors can be sensitive to mesh tessellation versus voxelized-boundary sampling, so external-transfer failure is not biology-specific"
    ),
}
SECONDARY_MAIKE_LOAO_CONTRACT: dict[str, Any] = {
    "analysis_status": "prespecified_secondary_diagnostic_not_primary",
    "cohort": "distal_qc AND target_resolvable",
    "outer_validation": "leave_one_Maike_animal_out",
    "inner_alpha_selection": (
        "within_each_outer_fold_leave_one_of_the_remaining_11_animals_out"
    ),
    "alpha_objective": (
        "equal_animal_mean_of_inner_heldout_animal_median_81pt_normalized_MAE"
    ),
    "training_weighting": "equal_animal_mass",
    "target_handling": (
        "same_frozen_c0_c1_c3_c5_fit_c2_c4_zero_full_grid_penalty_and_central_symmetrization"
    ),
    "decision_role": (
        "descriptive_only_cannot_modify_rescue_or_replace_the_Arthur_to_Maike_primary_result"
    ),
    "interpretation_limit": (
        "tests_within_Maike_representation_predictability_not_cross_taxon_or_cross_representation_generalization"
    ),
}
EXPECTED_TARGET_COHORT_DEFINITIONS: dict[str, str] = {
    "predictor_eligibility": (
        "distal_qc is fixed before and independently of all target fields"
    ),
    "target_resolvable": (
        "distal frame available; main component fraction >=0.99; at least 25 "
        "spanning lateral bins; finite full-rank target fit"
    ),
    "target_qc_sensitivity": (
        "target_resolvable and raw thickness q05 >0 and target RMSE <=2.5 um"
    ),
}
EXPECTED_FRAME_THRESHOLDS: dict[str, Any] = {
    "noncentral_poleward_u_angle_p95_max_deg": 5.0,
    "outward_axis_angle_p95_max_deg": 2.0,
    "central_classification_change_rate_max": 0.02,
    "numerical_fallback_outside_central_max": 0,
}
EXPECTED_PERTURBATIONS = {
    "exhaustive_leave_one_origin_out",
    "ninety_percent_subsample",
    "one_percent_nearest_neighbor_gaussian",
}

RIDGE_ALPHAS = np.logspace(-2, 3, 12)
EVEN_TARGET_INDICES = np.array([0, 1, 3, 5], dtype=np.int64)
ODD_REFLECTION_INDICES = np.array([2, 4], dtype=np.int64)
POSITION_FEATURE_NAMES = (
    "position_radius_um",
    "position_boundary_distance_um",
    "position_nn1_um",
    "position_nn3_um",
    "position_nn6_um",
    "position_pairwise_q10_um",
    "position_pairwise_q25_um",
    "position_pairwise_q50_um",
    "position_pairwise_q75_um",
    "position_pairwise_q90_um",
)
CONTROL_FEATURE_NAMES = POSITION_FEATURE_NAMES + ("distal_scale_um",)
SHAPE_ADDITION_NAMES = (
    "distal_gradient_magnitude",
    "distal_curvature_eigenvalue_1",
    "distal_curvature_eigenvalue_2",
    "distal_normalized_fit_residual",
)
SHAPE_FEATURE_NAMES = CONTROL_FEATURE_NAMES + SHAPE_ADDITION_NAMES
TARGET_COLUMNS = tuple(f"target_c{i}" for i in range(6))

REQUIRED_TABLE_COLUMNS = {
    "eye_id",
    "lens_index",
    "distal_qc",
    "target_resolvable",
    "target_qc",
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
    *TARGET_COLUMNS,
}
REQUIRED_MAIKE_COMPONENT_COLUMNS = {
    "seed_id",
    "assignment_status",
    "full_assigned_size",
    "main_component_size",
    "component_removed_size",
    "main_component_fraction",
    "partition_target_resolvable",
    "distal_eligible",
}

SOURCE_CODE_FILES = (
    "experiments/maike-modern-ground-truth/prepare_arthur_source_table.py",
    "experiments/maike-modern-ground-truth/distal_only_geometry.py",
    "experiments/maike-modern-ground-truth/audit_distal_frame_stability.py",
    "experiments/arthur-modern-ground-truth/experiment_57_outer_only_validation.py",
    "experiments/arthur-modern-ground-truth/experiment_58_cross_volume_confirmation.py",
)
MAIKE_PRODUCER_CODE_FILES = {
    "prepare_maike_masks.py": (
        "experiments/maike-modern-ground-truth/prepare_maike_masks.py"
    ),
    "map_oda_to_source.py": (
        "experiments/maike-modern-ground-truth/map_oda_to_source.py"
    ),
    "extract_lens_surfaces.py": (
        "experiments/maike-modern-ground-truth/extract_lens_surfaces.py"
    ),
    "distal_only_geometry.py": (
        "experiments/maike-modern-ground-truth/distal_only_geometry.py"
    ),
    "audit_distal_frame_stability.py": (
        "experiments/maike-modern-ground-truth/audit_distal_frame_stability.py"
    ),
}


class ContractError(RuntimeError):
    """Raised when a frozen-input or leakage-control contract is violated."""


def _fail(message: str) -> None:
    raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def execution_environment() -> dict[str, Any]:
    packages = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ("numpy", "pandas", "scipy", "Pillow")
    }
    numpy_config = getattr(np.__config__, "CONFIG", {})
    build_dependencies = (
        numpy_config.get("Build Dependencies", {})
        if isinstance(numpy_config, Mapping)
        else {}
    )
    numerical_thread_environment = {
        key: os.environ.get(key)
        for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "numpy_build_dependencies": build_dependencies,
        "numerical_thread_environment": numerical_thread_environment,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"Expected a JSON object in {path}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    _fail(f"{label} is not a strict boolean: {value!r}")


def _strict_int(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        _fail(f"{label} is a boolean, not an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not an integer: {value!r}") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(integer)
    if not math.isfinite(numeric) or numeric != integer:
        _fail(f"{label} is not an exact integer: {value!r}")
    if minimum is not None and integer < minimum:
        _fail(f"{label} must be >= {minimum}; got {integer}")
    return integer


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        _fail(f"{label} is missing required keys: {missing}")


def _require_exact_mapping(actual: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(actual, Mapping):
        _fail(f"{label} must be an object")
    if set(actual) != set(expected):
        _fail(
            f"{label} keys differ from the frozen contract; "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            try:
                finite = math.isfinite(float(actual_value))
            except (TypeError, ValueError):
                finite = False
            if not finite or float(actual_value) != expected_value:
                _fail(f"{label}.{key} must equal {expected_value!r}")
        elif actual_value != expected_value:
            _fail(f"{label}.{key} must equal {expected_value!r}")


def _safe_relative(root: Path, relpath: str, label: str) -> Path:
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"Unsafe relative path in {label}: {relpath!r}")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"Missing path bound by {label}: {candidate}") from exc
    if resolved_root != resolved and resolved_root not in resolved.parents:
        _fail(f"Path escapes {root}: {relpath!r}")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"Symlink is forbidden in a frozen bundle: {cursor}")
    return candidate


def validate_file_binding(root: Path, relpath: str, binding: Any, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"sha256", "size_bytes"}:
        _fail(f"{label}[{relpath!r}] must contain exactly sha256 and size_bytes")
    expected_hash = str(binding["sha256"])
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        _fail(f"Invalid lowercase SHA-256 in {label}[{relpath!r}]")
    expected_size = _strict_int(binding["size_bytes"], f"{label}[{relpath!r}].size_bytes", 0)
    path = _safe_relative(root, relpath, label)
    if not path.is_file():
        _fail(f"Bound artifact is not a regular file: {path}")
    if path.stat().st_size != expected_size:
        _fail(f"Size mismatch for {path}")
    if sha256_file(path) != expected_hash:
        _fail(f"SHA-256 mismatch for {path}")
    return path


def validate_output_manifest(root: Path, manifest: Any, label: str) -> dict[str, Path]:
    if not isinstance(manifest, Mapping) or not manifest:
        _fail(f"{label} must be a nonempty object")
    validated: dict[str, Path] = {}
    for relpath in sorted(manifest):
        if not isinstance(relpath, str):
            _fail(f"{label} path keys must be strings")
        validated[relpath] = validate_file_binding(root, relpath, manifest[relpath], label)
    return validated


def get_git_identity(repo: Path) -> tuple[str, bool]:
    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode:
            _fail(f"Git command failed in {repo}: {process.stderr.strip()}")
        return process.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        _fail(f"Invalid Git commit identity: {commit!r}")
    dirty = bool(run("status", "--porcelain=v1", "--untracked-files=all"))
    return commit, dirty


def require_frozen_git(repo: Path, expected_commit: str) -> str:
    commit, dirty = get_git_identity(repo)
    if dirty:
        _fail("Repository is dirty; the frozen primary run requires a clean worktree")
    if commit != expected_commit:
        _fail(f"Current commit {commit} does not match frozen commit {expected_commit}")
    return commit


def validate_git_record(record: Any, expected_commit: str, label: str) -> None:
    if not isinstance(record, Mapping):
        _fail(f"{label}.git must be an object")
    _require_keys(record, {"commit", "dirty"}, f"{label}.git")
    if record["commit"] != expected_commit:
        _fail(f"{label} was produced at a different Git commit")
    if _strict_bool(record["dirty"], f"{label}.git.dirty"):
        _fail(f"{label} was produced from a dirty worktree")


def canonical_grid_xy() -> np.ndarray:
    axis = np.arange(-0.65, 0.65 + 0.13 / 2.0, 0.13, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    keep = xx * xx + yy * yy <= 0.65 * 0.65 + 1e-12
    grid = np.column_stack([xx[keep], yy[keep]])
    if grid.shape != (81, 2):  # defensive guard against numerical/platform drift
        _fail(f"Canonical grid unexpectedly has shape {grid.shape}")
    return grid


CANONICAL_GRID_XY = canonical_grid_xy()
CANONICAL_DESIGN = np.column_stack(
    [
        np.ones(81),
        CANONICAL_GRID_XY[:, 0],
        CANONICAL_GRID_XY[:, 1],
        CANONICAL_GRID_XY[:, 0] ** 2,
        CANONICAL_GRID_XY[:, 0] * CANONICAL_GRID_XY[:, 1],
        CANONICAL_GRID_XY[:, 1] ** 2,
    ]
)


def validate_sealed_distal_npz(
    path: Path, lens_index: int, *, require_distal_qc_support: bool = True
) -> dict[str, Any]:
    """Validate the exact, predictor-only v2 artifact and return decoded config."""
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_keys = {
                "schema_version",
                "lens_index",
                "points_zyx",
                "spacing_um",
                "config_json",
                "config_sha256",
            }
            if set(data.files) != expected_keys:
                _fail(
                    f"{path} has forbidden/missing arrays; expected exactly "
                    f"{sorted(expected_keys)}, got {sorted(data.files)}"
                )
            schema = data["schema_version"]
            index = data["lens_index"]
            points = data["points_zyx"]
            spacing = data["spacing_um"]
            config_text_array = data["config_json"]
            config_hash_array = data["config_sha256"]
            for name, array in {
                "schema_version": schema,
                "lens_index": index,
                "config_json": config_text_array,
                "config_sha256": config_hash_array,
            }.items():
                if array.shape != ():
                    _fail(f"{path}:{name} must be a scalar array")
            if schema.dtype.kind != "U" or str(schema.item()) != SEALED_DISTAL_SCHEMA:
                _fail(f"{path}: invalid sealed schema")
            if index.dtype != np.dtype("int64") or int(index.item()) != lens_index:
                _fail(f"{path}: lens_index must be int64 scalar {lens_index}")
            if points.dtype != np.dtype("int32") or points.ndim != 2 or points.shape[1] != 3:
                _fail(f"{path}: points_zyx must be int32[N,3]")
            if require_distal_qc_support and len(points) < EXPECTED_THRESHOLD_CONFIG[
                "min_sealed_distal_cap_points"
            ]:
                _fail(f"{path}: sealed distal cap has fewer than 25 points")
            if np.any(points < 0):
                _fail(f"{path}: points_zyx contains negative indices")
            if len(np.unique(points, axis=0)) != len(points):
                _fail(f"{path}: points_zyx contains duplicates")
            order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
            if not np.array_equal(order, np.arange(len(points))):
                _fail(f"{path}: points_zyx is not canonically lexicographically sorted")
            if spacing.dtype != np.dtype("float64") or spacing.shape != (3,):
                _fail(f"{path}: spacing_um must be float64[3]")
            if not np.array_equal(spacing, np.array([0.325, 0.325, 0.325], dtype=np.float64)):
                _fail(f"{path}: spacing_um differs from the original 0.325-um voxels")
            if config_text_array.dtype.kind != "U" or config_hash_array.dtype.kind != "U":
                _fail(f"{path}: config fields must be Unicode scalars")
            config_text = str(config_text_array.item())
            config_hash = str(config_hash_array.item())
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot safely read sealed NPZ {path}: {exc}") from exc

    if sha256_text(config_text) != config_hash:
        _fail(f"{path}: config_sha256 does not bind config_json")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: config_json is invalid") from exc
    if not isinstance(config, dict) or canonical_json(config) != config_text:
        _fail(f"{path}: config_json is not a canonical JSON object")
    forbidden_fragments = ("target", "proximal", "prediction", "error", "tip", "oda_axis")
    lowered = config_text.lower()
    if any(fragment in lowered for fragment in forbidden_fragments):
        _fail(f"{path}: sealed predictor config contains target/oracle outcome language")
    _require_exact_mapping(config.get("threshold_config"), EXPECTED_THRESHOLD_CONFIG, f"{path}:threshold_config")
    if config.get("analysis_scope") != ANALYSIS_SCOPE:
        _fail(f"{path}: wrong analysis scope")
    if config.get("isolation_basis") != ISOLATION_BASIS:
        _fail(f"{path}: wrong isolation basis")
    if set(config) != {
        "analysis_scope",
        "isolation_basis",
        "threshold_config",
        "predictor_pipeline_config",
    }:
        _fail(f"{path}: sealed config has unexpected top-level keys")
    _require_exact_mapping(
        config.get("predictor_pipeline_config"),
        EXPECTED_PREDICTOR_PIPELINE_CONFIG,
        f"{path}:predictor_pipeline_config",
    )
    return config


def validate_arthur_sealed_distal_npz(
    path: Path,
    row: pd.Series,
    threshold_config_sha256: str,
) -> None:
    """Validate a source mesh cap without pretending its vertices are voxel indices."""

    expected_arrays = {
        "schema_version", "volume", "eye_id", "lens_index", "points_xyz_um",
        "config_json", "config_sha256",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                _fail(f"{path}: Arthur sealed-distal arrays differ from the exact contract")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot safely read Arthur sealed distal {path}: {exc}") from exc
    for name in ("schema_version", "volume", "eye_id", "lens_index", "config_json", "config_sha256"):
        if arrays[name].shape != ():
            _fail(f"{path}:{name} must be scalar")
    if arrays["schema_version"].dtype.kind != "U" or str(arrays["schema_version"].item()) != ARTHUR_SEALED_DISTAL_SCHEMA:
        _fail(f"{path}: wrong Arthur sealed-distal schema")
    if arrays["volume"].dtype.kind != "U" or str(arrays["volume"].item()) != str(row["volume"]):
        _fail(f"{path}: wrong Arthur volume identity")
    if arrays["eye_id"].dtype != np.dtype("int64") or int(arrays["eye_id"].item()) != int(row["eye_id"]):
        _fail(f"{path}: wrong Arthur eye identity")
    if arrays["lens_index"].dtype != np.dtype("int64") or int(arrays["lens_index"].item()) != int(row["lens_index"]):
        _fail(f"{path}: wrong Arthur lens identity")
    points = arrays["points_xyz_um"]
    if points.dtype != np.dtype("float64") or points.ndim != 2 or points.shape[1] != 3:
        _fail(f"{path}: Arthur points_xyz_um must be float64[N,3]")
    if len(points) < 25 or not np.all(np.isfinite(points)):
        _fail(f"{path}: Arthur distal cap has invalid support or coordinates")
    if len(np.unique(points, axis=0)) != len(points):
        _fail(f"{path}: Arthur distal cap contains duplicate vertices")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    if not np.array_equal(order, np.arange(len(points))):
        _fail(f"{path}: Arthur distal vertices are not canonically sorted")
    if arrays["config_json"].dtype.kind != "U" or arrays["config_sha256"].dtype.kind != "U":
        _fail(f"{path}: Arthur sealed config fields must be Unicode")
    config_text = str(arrays["config_json"].item())
    config_hash = str(arrays["config_sha256"].item())
    if sha256_text(config_text) != config_hash or config_hash != threshold_config_sha256:
        _fail(f"{path}: Arthur sealed threshold config hash mismatch")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: invalid Arthur sealed config JSON") from exc
    if canonical_json(config) != config_text:
        _fail(f"{path}: Arthur sealed config JSON is noncanonical")
    _require_exact_mapping(config, EXPECTED_THRESHOLD_CONFIG, f"{path}:threshold_config")


def validate_arthur_target_npz(path: Path, row: pd.Series) -> bool:
    expected_arrays = {
        "schema_version", "volume", "eye_id", "lens_index", "canonical_grid_xy",
        "target_smoothed_thickness_um", "raw_target_xy_normalized",
        "raw_target_thickness_um", "target_coefficients_c0_c5",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                _fail(f"{path}: Arthur target arrays differ from the exact contract")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot safely read Arthur target {path}: {exc}") from exc
    for name in ("schema_version", "volume", "eye_id", "lens_index"):
        if arrays[name].shape != ():
            _fail(f"{path}:{name} must be scalar")
    if arrays["schema_version"].dtype.kind != "U" or str(arrays["schema_version"].item()) != ARTHUR_TARGET_SCHEMA:
        _fail(f"{path}: wrong Arthur target schema")
    if arrays["volume"].dtype.kind != "U" or str(arrays["volume"].item()) != str(row["volume"]):
        _fail(f"{path}: wrong Arthur target volume")
    if arrays["eye_id"].dtype != np.dtype("int64") or int(arrays["eye_id"].item()) != int(row["eye_id"]):
        _fail(f"{path}: wrong Arthur target eye")
    if arrays["lens_index"].dtype != np.dtype("int64") or int(arrays["lens_index"].item()) != int(row["lens_index"]):
        _fail(f"{path}: wrong Arthur target lens")
    grid = arrays["canonical_grid_xy"]
    raw_xy = arrays["raw_target_xy_normalized"]
    raw_thickness = arrays["raw_target_thickness_um"]
    smooth = arrays["target_smoothed_thickness_um"]
    coefficients = arrays["target_coefficients_c0_c5"]
    if grid.dtype != np.dtype("float64") or grid.shape != (81, 2) or not np.array_equal(grid, CANONICAL_GRID_XY):
        _fail(f"{path}: Arthur canonical target grid mismatch")
    if raw_xy.dtype != np.dtype("float64") or raw_xy.ndim != 2 or raw_xy.shape[1] != 2:
        _fail(f"{path}: Arthur raw target coordinates must be float64[N,2]")
    if raw_thickness.dtype != np.dtype("float64") or raw_thickness.shape != (len(raw_xy),):
        _fail(f"{path}: Arthur raw target thickness must be float64[N]")
    if smooth.dtype != np.dtype("float64") or smooth.shape != (81,):
        _fail(f"{path}: Arthur smoothed target must be float64[81]")
    if coefficients.dtype != np.dtype("float64") or coefficients.shape != (6,):
        _fail(f"{path}: Arthur target coefficients must be float64[6]")
    support = _strict_int(row["target_support"], f"{path}:target_support", 0)
    if len(raw_xy) != support:
        _fail(f"{path}: Arthur target support differs from the source table")
    if not np.all(np.isfinite(raw_xy)) or not np.all(np.isfinite(raw_thickness)):
        _fail(f"{path}: Arthur raw target geometry is nonfinite")
    raw_observations = np.column_stack([raw_xy, raw_thickness])
    if len(np.unique(raw_observations, axis=0)) != len(raw_observations):
        _fail(f"{path}: Arthur target repeats a proximal mesh observation")
    if np.any(np.linalg.norm(raw_xy, axis=1) > 1.0 + 1e-10):
        _fail(f"{path}: Arthur raw target coordinates leave the normalized distal cap")
    table_coefficients = row.loc[list(TARGET_COLUMNS)].to_numpy(float)
    if not np.allclose(coefficients, table_coefficients, rtol=0.0, atol=0.0, equal_nan=True):
        _fail(f"{path}: Arthur target coefficients differ from the source table")
    intrinsic_structure = bool(
        support >= 25
        and np.linalg.matrix_rank(_evaluate_quadratic_design(raw_xy)) == 6
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(smooth))
        and np.allclose(smooth, CANONICAL_DESIGN @ coefficients, rtol=0.0, atol=1e-9)
    )
    if bool(row["target_resolvable"]) != intrinsic_structure:
        _fail(f"{path}: Arthur target_resolvable is not the exact structural flag")
    if intrinsic_structure:
        if not np.isclose(np.median(raw_thickness), float(row["target_depth_um"]), rtol=0.0, atol=1e-9):
            _fail(f"{path}: Arthur target depth differs from raw median")
        if not np.isclose(np.quantile(raw_thickness, 0.05), float(row["target_q05_raw_thickness_um"]), rtol=0.0, atol=1e-9):
            _fail(f"{path}: Arthur target raw q05 differs from the source table")
        raw_rmse = float(
            np.sqrt(
                np.mean(
                    (raw_thickness - _evaluate_quadratic_design(raw_xy) @ coefficients) ** 2
                )
            )
        )
        if not np.isclose(raw_rmse, float(row["target_rmse_um"]), rtol=0.0, atol=1e-9):
            _fail(f"{path}: Arthur target RMSE differs from raw target geometry")
    return intrinsic_structure


def _parse_bool_columns(table: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        result[column] = [_strict_bool(v, f"{label}.{column}") for v in result[column]]
    return result


def validate_lens_table(
    table: pd.DataFrame,
    *,
    label: str,
    expected_eye: str | None = None,
    expected_rows: int | None = None,
    require_volume: bool = False,
) -> pd.DataFrame:
    required = set(REQUIRED_TABLE_COLUMNS)
    if require_volume:
        required.add("volume")
    missing = sorted(required - set(table.columns))
    if missing:
        _fail(f"{label} is missing columns: {missing}")
    if expected_rows is not None and len(table) != expected_rows:
        _fail(f"{label} has {len(table)} rows; expected exactly {expected_rows}")
    result = _parse_bool_columns(
        table,
        ("distal_qc", "target_resolvable", "target_qc", "central"),
        label,
    )
    try:
        result["lens_index"] = result["lens_index"].map(
            lambda value: _strict_int(value, f"{label}.lens_index", 0)
        )
    except ContractError:
        raise
    if result["lens_index"].duplicated().any() and not require_volume:
        _fail(f"{label} contains duplicate lens_index values")
    if expected_rows is not None:
        actual_indices = np.sort(result["lens_index"].to_numpy(np.int64))
        if not np.array_equal(actual_indices, np.arange(expected_rows, dtype=np.int64)):
            _fail(f"{label} lens indices are not the complete 0..{expected_rows - 1} range")
    if expected_eye is not None:
        if set(result["eye_id"].astype(str)) != {expected_eye}:
            _fail(f"{label} contains a wrong eye_id")

    numeric_columns = [
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
        *TARGET_COLUMNS,
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    distal = result["distal_qc"].to_numpy(bool)
    predictor_columns = [
        "position_u_um",
        "position_v_um",
        "distal_scale_um",
        *SHAPE_ADDITION_NAMES,
    ]
    predictor_values = result.loc[distal, predictor_columns].to_numpy(float)
    if not np.all(np.isfinite(predictor_values)):
        _fail(f"{label} has nonfinite predictor values in distal-QC rows")
    scales = result.loc[distal, "distal_scale_um"].to_numpy(float)
    if np.any((scales < 3.0) | (scales > 13.0)):
        _fail(f"{label} has distal-QC scales outside frozen bounds")
    gradients = result.loc[distal, "distal_gradient_magnitude"].to_numpy(float)
    residuals = result.loc[distal, "distal_normalized_fit_residual"].to_numpy(float)
    if np.any(gradients < 0) or np.any(residuals < 0):
        _fail(f"{label} has negative shape magnitude/residual")
    eig1 = result.loc[distal, "distal_curvature_eigenvalue_1"].to_numpy(float)
    eig2 = result.loc[distal, "distal_curvature_eigenvalue_2"].to_numpy(float)
    if np.any(eig1 > eig2 + 1e-12):
        _fail(f"{label} curvature eigenvalues are not ordered")

    resolvable = result["target_resolvable"].to_numpy(bool)
    target_qc = result["target_qc"].to_numpy(bool)
    if np.any(resolvable & ~distal):
        _fail(f"{label} marks target_resolvable true outside distal-QC rows")
    if np.any(target_qc & ~resolvable):
        _fail(f"{label} marks target_qc true where target_resolvable is false")
    target_rows = distal & resolvable
    target_values = result.loc[
        target_rows,
        [*TARGET_COLUMNS, "target_depth_um", "target_q05_raw_thickness_um", "target_rmse_um"],
    ].to_numpy(float)
    if not np.all(np.isfinite(target_values)):
        _fail(f"{label} has nonfinite targets in the primary cohort")
    support = result.loc[target_rows, "target_support"].to_numpy(float)
    if np.any(~np.isfinite(support)) or np.any(support < 25) or np.any(support != np.floor(support)):
        _fail(f"{label} target support must be integer-like and >=25 when resolvable")
    if np.any(target_qc & ~distal):
        _fail(f"{label} marks target_qc true outside distal-QC rows")
    expected_target_qc = (
        distal
        & resolvable
        & (result["target_q05_raw_thickness_um"].to_numpy(float) > 0.0)
        & (result["target_rmse_um"].to_numpy(float) <= 2.5)
    )
    if not np.array_equal(target_qc, expected_target_qc):
        _fail(
            f"{label} target_qc is not exactly structural target_resolvable plus "
            "raw-thickness q05>0 and target RMSE<=2.5 um"
        )
    return result


def validate_metric_denominator_gate(table: pd.DataFrame, label: str) -> dict[str, Any]:
    """Abort, rather than filter, if the frozen normalized-MAE metric is undefined."""

    primary = table["distal_qc"].to_numpy(bool) & table["target_resolvable"].to_numpy(bool)
    depths = table["target_depth_um"].to_numpy(float)
    invalid = primary & (~np.isfinite(depths) | (depths <= 0.0))
    n_primary = int(np.sum(primary))
    n_valid = int(np.sum(primary & np.isfinite(depths) & (depths > 0.0)))
    if np.any(invalid):
        indices = table.loc[invalid, "lens_index"].astype(int).head(10).tolist()
        _fail(
            f"{label} whole-run metric-validity gate failed: frozen target_depth_um "
            f"denominator is not finite and positive for {int(np.sum(invalid))}/{n_primary} "
            f"primary rows (first lens indices {indices}); rows may not be dropped"
        )
    return {
        "metric": "81-point MAE divided by median raw target thickness_um",
        "policy": "abort_entire_experiment_never_drop_invalid_rows",
        "n_primary_rows": n_primary,
        "n_finite_positive_target_depth": n_valid,
        "n_invalid": 0,
        "passed": True,
    }


def validate_maike_component_table(table: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_MAIKE_COMPONENT_COLUMNS - set(table.columns))
    if missing:
        _fail(f"{label} is missing component/partition fields: {missing}")
    result = table.copy()
    result["partition_target_resolvable"] = [
        _strict_bool(value, f"{label}.partition_target_resolvable")
        for value in result["partition_target_resolvable"]
    ]
    result["distal_eligible"] = [
        _strict_bool(value, f"{label}.distal_eligible") for value in result["distal_eligible"]
    ]
    if not np.array_equal(
        result["distal_eligible"].to_numpy(bool), result["distal_qc"].to_numpy(bool)
    ):
        _fail(f"{label} distal_eligible and distal_qc flags differ")
    seed_ids = result["seed_id"].astype(str)
    if (seed_ids.str.len() == 0).any() or seed_ids.duplicated().any():
        _fail(f"{label} seed identities are empty or duplicated")
    for row_number, row in result.iterrows():
        full = _strict_int(row["full_assigned_size"], f"{label}[{row_number}].full_assigned_size", 0)
        main = _strict_int(row["main_component_size"], f"{label}[{row_number}].main_component_size", 0)
        removed = _strict_int(row["component_removed_size"], f"{label}[{row_number}].component_removed_size", 0)
        if main > full or removed != full - main:
            _fail(f"{label} row {row_number} has inconsistent component sizes")
        status = str(row["assignment_status"])
        if status not in {"ok", "empty_assignment"} or (status == "empty_assignment") != (full == 0):
            _fail(f"{label} row {row_number} has inconsistent assignment status")
        try:
            fraction = float(row["main_component_fraction"])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label} row {row_number} has invalid component fraction") from exc
        expected_fraction = main / full if full else 0.0
        if not math.isfinite(fraction) or not math.isclose(
            fraction, expected_fraction, rel_tol=1e-9, abs_tol=1e-12
        ):
            _fail(f"{label} row {row_number} component fraction is inconsistent")
        gate = full > 0 and main * 100 >= full * 99
        if bool(row["partition_target_resolvable"]) != gate:
            _fail(f"{label} row {row_number} changed the exact 99% component gate")
        if bool(row["target_resolvable"]) and not gate:
            _fail(f"{label} row {row_number} resolves a target that failed component QC")
        if bool(row["target_qc"]) and float(row["target_rmse_um"]) > 2.5:
            _fail(f"{label} row {row_number} target_qc exceeds frozen RMSE limit")
    return result


def _point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    denom = np.einsum("ij,ij->i", segment, segment)
    if np.any(denom <= 0):
        _fail("Convex-hull boundary contains a zero-length edge")
    relative = points[:, None, :] - a[None, :, :]
    t = np.einsum("nki,ki->nk", relative, segment) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    nearest = a[None, :, :] + t[:, :, None] * segment[None, :, :]
    return np.sqrt(np.min(np.sum((points[:, None, :] - nearest) ** 2, axis=2), axis=1))


def invariant_position_features(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute all ten O(2)-invariant position descriptors from distal-QC rows."""
    result = table.copy()
    for name in POSITION_FEATURE_NAMES:
        result[name] = np.nan

    reference = result["distal_qc"].to_numpy(bool)
    reference_indices = np.flatnonzero(reference)
    points = result.loc[reference, ["position_u_um", "position_v_um"]].to_numpy(float)
    if len(points) < 7:
        _fail(f"{label} has fewer than seven distal-QC reference lenses")
    if len(np.unique(points, axis=0)) != len(points):
        _fail(f"{label} has duplicate distal lens positions")
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ContractError(f"{label} distal positions do not define a 2D hull") from exc
    vertices = hull.vertices
    a = points[vertices]
    b = points[np.roll(vertices, -1)]
    boundary = _point_segment_distances(points, a, b)
    pairwise = distance.squareform(distance.pdist(points))
    if not np.all(np.isfinite(pairwise)):
        _fail(f"{label} pairwise distances are nonfinite")
    # Exclude the diagonal explicitly; multiplying an identity matrix by
    # infinity would create NaNs in its off-diagonal zero entries.
    pairwise_no_self = pairwise.copy()
    np.fill_diagonal(pairwise_no_self, np.inf)
    without_self = np.sort(pairwise_no_self, axis=1)[:, :-1]
    radius = np.linalg.norm(points, axis=1)
    values = np.column_stack(
        [
            radius,
            boundary,
            without_self[:, 0],
            without_self[:, 2],
            without_self[:, 5],
            *[np.quantile(without_self, q, axis=1) for q in (0.10, 0.25, 0.50, 0.75, 0.90)],
        ]
    )
    if not np.all(np.isfinite(values)):
        _fail(f"{label} invariant position features are nonfinite")
    result.loc[reference, list(POSITION_FEATURE_NAMES)] = values

    central_expected = radius <= 0.5 * float(np.median(without_self[:, 0]))
    central_reported = result.loc[reference, "central"].to_numpy(bool)
    if not np.array_equal(central_expected, central_reported):
        differing = int(np.sum(central_expected != central_reported))
        _fail(f"{label} has {differing} central flags inconsistent with frozen geometry")
    return result


def add_invariant_features_by_unit(
    table: pd.DataFrame, unit_columns: Sequence[str], label: str
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    group_key: str | list[str] = list(unit_columns)
    if len(unit_columns) == 1:
        group_key = unit_columns[0]
    for unit, group in table.groupby(group_key, sort=True, dropna=False):
        pieces.append(invariant_position_features(group, f"{label} unit {unit!r}"))
    if not pieces:
        _fail(f"{label} contains no units")
    return pd.concat(pieces).sort_index()


def require_primary_resolution_gate(table: pd.DataFrame, expected_rows: int, eye_id: str) -> int:
    expected_rows = _strict_int(expected_rows, "expected_rows", 1)
    numerator = int((table["distal_qc"] & table["target_resolvable"]).sum())
    required = math.ceil(0.80 * expected_rows)
    if numerator < required:
        _fail(
            f"{eye_id} has {numerator}/{expected_rows} distal-QC and target-resolvable lenses; "
            f"the frozen 80% gate requires {required}"
        )
    return numerator


def symmetrized_targets(table: pd.DataFrame) -> np.ndarray:
    targets = table.loc[:, TARGET_COLUMNS].to_numpy(float, copy=True)
    central = table["central"].to_numpy(bool)
    targets[central, 1] = 0.0
    targets[central, 2] = 0.0
    targets[central, 4] = 0.0
    curvature = 0.5 * (targets[central, 3] + targets[central, 5])
    targets[central, 3] = curvature
    targets[central, 5] = curvature
    return targets


def target_grids(target_coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(target_coefficients, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] != 6:
        _fail("Target coefficient matrix must have shape [N,6]")
    return coefficients @ CANONICAL_DESIGN.T


@dataclass(frozen=True)
class WeightedRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict_even(self, features: np.ndarray, central: np.ndarray | None = None) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        standardized = (features - self.feature_mean) / self.feature_scale
        even = self.target_mean + standardized @ self.coefficients
        full = np.zeros((len(features), 6), dtype=float)
        full[:, EVEN_TARGET_INDICES] = even
        if central is not None:
            central = np.asarray(central, dtype=bool)
            full[central, 1] = 0.0
            average = 0.5 * (full[central, 3] + full[central, 5])
            full[central, 3] = average
            full[central, 5] = average
        return full


def equal_group_weights(groups: Sequence[Any]) -> np.ndarray:
    values = np.asarray(groups)
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) < 1 or np.any(counts == 0):
        _fail("Cannot form equal group weights")
    count_by_group = dict(zip(unique.tolist(), counts.tolist()))
    # Sum to one, with exactly 1/G total mass per source volume.  The constant
    # total keeps the frozen ridge-alpha scale comparable across LOOV folds
    # with different row counts and against the final three-volume fit.
    return np.array(
        [1.0 / (len(unique) * count_by_group[value]) for value in values],
        dtype=float,
    )


def fit_weighted_ridge(
    features: np.ndarray,
    targets_full: np.ndarray,
    groups: Sequence[Any],
    alpha: float,
) -> WeightedRidge:
    x = np.asarray(features, dtype=float)
    y_full = np.asarray(targets_full, dtype=float)
    if x.ndim != 2 or y_full.shape != (len(x), 6) or len(x) == 0:
        _fail("Invalid ridge training arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y_full)):
        _fail("Ridge training arrays contain NaN or infinity")
    if not math.isfinite(float(alpha)) or alpha <= 0:
        _fail("Ridge alpha must be finite and positive")
    weights = equal_group_weights(groups)
    weight_sum = float(np.sum(weights))
    feature_mean = np.sum(x * weights[:, None], axis=0) / weight_sum
    variance = np.sum((x - feature_mean) ** 2 * weights[:, None], axis=0) / weight_sum
    if np.any(variance <= 1e-24):
        _fail("A frozen model feature has zero weighted variance")
    feature_scale = np.sqrt(variance)
    xs = (x - feature_mean) / feature_scale
    y = y_full[:, EVEN_TARGET_INDICES]
    target_mean = np.sum(y * weights[:, None], axis=0) / weight_sum
    yc = y - target_mean
    weighted_x = xs * np.sqrt(weights)[:, None]
    weighted_y = yc * np.sqrt(weights)[:, None]
    gram = weighted_x.T @ weighted_x
    rhs = weighted_x.T @ weighted_y
    coefficients = np.linalg.solve(gram + float(alpha) * np.eye(x.shape[1]), rhs)
    return WeightedRidge(feature_mean, feature_scale, target_mean, coefficients, float(alpha))


def normalized_grid_mae(predicted_coefficients: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    target_coefficients = symmetrized_targets(table)
    predicted = target_grids(predicted_coefficients)
    target = target_grids(target_coefficients)
    depths = table["target_depth_um"].to_numpy(float)
    if np.any(depths <= 0) or not np.all(np.isfinite(depths)):
        _fail("Normalized-MAE denominators must be finite and positive")
    return np.mean(np.abs(predicted - target), axis=1) / depths


def select_alpha_leave_one_volume_out(
    source: pd.DataFrame, feature_names: Sequence[str]
) -> tuple[float, pd.DataFrame]:
    volumes = tuple(sorted(source["volume"].astype(str).unique()))
    if volumes != tuple(sorted(ARTHUR_VOLUMES)):
        _fail(f"Alpha selection requires exactly Arthur volumes {ARTHUR_VOLUMES}")
    rows: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        heldout_medians: list[float] = []
        for heldout in ARTHUR_VOLUMES:
            train = source["volume"].astype(str) != heldout
            test = ~train
            model = fit_weighted_ridge(
                source.loc[train, feature_names].to_numpy(float),
                symmetrized_targets(source.loc[train]),
                source.loc[train, "volume"].astype(str).to_numpy(),
                float(alpha),
            )
            prediction = model.predict_even(
                source.loc[test, feature_names].to_numpy(float),
                source.loc[test, "central"].to_numpy(bool),
            )
            median = float(np.median(normalized_grid_mae(prediction, source.loc[test])))
            heldout_medians.append(median)
            rows.append(
                {
                    "model": "shape" if tuple(feature_names) == SHAPE_FEATURE_NAMES else "control",
                    "alpha": float(alpha),
                    "heldout_volume": heldout,
                    "heldout_median_81pt_normalized_mae": median,
                }
            )
        rows.append(
            {
                "model": "shape" if tuple(feature_names) == SHAPE_FEATURE_NAMES else "control",
                "alpha": float(alpha),
                "heldout_volume": "equal_volume_mean",
                "heldout_median_81pt_normalized_mae": float(np.mean(heldout_medians)),
            }
        )
    audit = pd.DataFrame(rows)
    means = audit[audit["heldout_volume"] == "equal_volume_mean"]
    minimum = float(means["heldout_median_81pt_normalized_mae"].min())
    selected = float(means.loc[means["heldout_median_81pt_normalized_mae"] == minimum, "alpha"].min())
    return selected, audit


def select_alpha_nested_maike_loao(
    training: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    outer_heldout_animal: str,
) -> tuple[float, pd.DataFrame]:
    """Select alpha without reading the outer held-out animal's targets."""

    outer_heldout_animal = str(outer_heldout_animal)
    if outer_heldout_animal not in EXPECTED_EYES:
        _fail(f"Unknown outer held-out Maike animal {outer_heldout_animal!r}")
    expected_training_animals = set(EXPECTED_EYES) - {outer_heldout_animal}
    animals = training["eye_id"].astype(str)
    if set(animals) != expected_training_animals:
        _fail(
            "Nested Maike alpha selection requires exactly the other eleven animals"
        )
    model_name = "shape" if tuple(feature_names) == SHAPE_FEATURE_NAMES else "control"
    rows: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        heldout_medians: list[float] = []
        for inner_heldout in sorted(expected_training_animals):
            inner_train = animals != inner_heldout
            inner_test = ~inner_train
            model = fit_weighted_ridge(
                training.loc[inner_train, feature_names].to_numpy(float),
                symmetrized_targets(training.loc[inner_train]),
                animals.loc[inner_train].to_numpy(),
                float(alpha),
            )
            prediction = model.predict_even(
                training.loc[inner_test, feature_names].to_numpy(float),
                training.loc[inner_test, "central"].to_numpy(bool),
            )
            median = float(
                np.median(normalized_grid_mae(prediction, training.loc[inner_test]))
            )
            heldout_medians.append(median)
            rows.append(
                {
                    "model": model_name,
                    "outer_heldout_animal": outer_heldout_animal,
                    "inner_heldout_animal": inner_heldout,
                    "alpha": float(alpha),
                    "inner_heldout_median_81pt_normalized_mae": median,
                }
            )
        rows.append(
            {
                "model": model_name,
                "outer_heldout_animal": outer_heldout_animal,
                "inner_heldout_animal": "equal_animal_mean",
                "alpha": float(alpha),
                "inner_heldout_median_81pt_normalized_mae": float(
                    np.mean(heldout_medians)
                ),
            }
        )
    audit = pd.DataFrame(rows)
    means = audit[audit["inner_heldout_animal"] == "equal_animal_mean"]
    minimum = float(means["inner_heldout_median_81pt_normalized_mae"].min())
    selected = float(
        means.loc[
            means["inner_heldout_median_81pt_normalized_mae"] == minimum,
            "alpha",
        ].min()
    )
    return selected, audit


def two_sided_fixed_denominator_p(wins: int, n: int = 12) -> float:
    wins = _strict_int(wins, "wins", 0)
    n = _strict_int(n, "n", 1)
    if wins > n:
        _fail("wins cannot exceed n")
    tail_count = min(wins, n - wins)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail_count + 1)) / (2**n)
    return min(1.0, float(probability))


def conventional_tie_dropping_sign_p(wins: int, losses: int) -> float:
    wins = _strict_int(wins, "wins", 0)
    losses = _strict_int(losses, "losses", 0)
    n = wins + losses
    if n == 0:
        return 1.0
    tail_count = min(wins, losses)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail_count + 1)) / (2**n)
    return min(1.0, float(probability))


def validate_fixed_point(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    _require_keys(
        value,
        {"converged", "max_iterations", "iterations", "eligible_counts", "readded_count"},
        label,
    )
    if not _strict_bool(value["converged"], f"{label}.converged"):
        _fail(f"{label} did not converge")
    if _strict_int(value["max_iterations"], f"{label}.max_iterations") != 20:
        _fail(f"{label} changed the frozen iteration limit")
    iterations = _strict_int(value["iterations"], f"{label}.iterations", 0)
    if iterations > 20:
        _fail(f"{label} exceeded 20 iterations")
    if _strict_int(value["readded_count"], f"{label}.readded_count", 0) != 0:
        _fail(f"{label} re-added a failed lens")
    counts_raw = value["eligible_counts"]
    if not isinstance(counts_raw, list) or not counts_raw:
        _fail(f"{label}.eligible_counts must be a nonempty list")
    counts = [_strict_int(v, f"{label}.eligible_counts", 0) for v in counts_raw]
    if any(b > a for a, b in zip(counts, counts[1:])):
        _fail(f"{label} eligibility is not monotone decreasing")
    if len(counts) >= 2 and counts[-1] != counts[-2]:
        _fail(f"{label} did not record a stable terminal iteration")
    if iterations != len(counts) - 1:
        _fail(f"{label}.iterations does not match eligible_counts")


def validate_partition_evidence(value: Any, label: str) -> None:
    fields = {
        "source_foreground_voxel_count",
        "assigned_voxel_count",
        "assigned_unique_voxel_count",
        "unassigned_foreground_voxel_count",
        "multiply_assigned_voxel_count",
        "exact_partition",
        "candidate_seeds_per_voxel",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(f"{label} differs from the exact partition-evidence schema")
    counts = {
        key: _strict_int(value[key], f"{label}.{key}", 0)
        for key in fields
        if key != "exact_partition"
    }
    if not _strict_bool(value["exact_partition"], f"{label}.exact_partition"):
        _fail(f"{label} is not an exact partition")
    if counts["candidate_seeds_per_voxel"] != 1:
        _fail(f"{label} does not use one-candidate centre-Voronoi assignment")
    if counts["unassigned_foreground_voxel_count"] != 0:
        _fail(f"{label} leaves foreground voxels unassigned")
    if counts["multiply_assigned_voxel_count"] != 0:
        _fail(f"{label} assigns foreground voxels more than once")
    if counts["assigned_voxel_count"] != counts["assigned_unique_voxel_count"]:
        _fail(f"{label} assigned and unique-assigned counts differ")
    if counts["assigned_unique_voxel_count"] != counts["source_foreground_voxel_count"]:
        _fail(f"{label} does not cover the source foreground losslessly")


def validate_frame_audit(audit: Mapping[str, Any], eye_id: str, label: str) -> None:
    _require_keys(audit, {"schema_version", "eye_id", "gate_passed", "thresholds", "perturbations"}, label)
    if audit["schema_version"] != FRAME_AUDIT_SCHEMA or str(audit["eye_id"]) != eye_id:
        _fail(f"{label} has wrong schema or eye")
    if not _strict_bool(audit["gate_passed"], f"{label}.gate_passed"):
        _fail(f"{label} failed its hard stability gate")
    _require_exact_mapping(audit["thresholds"], EXPECTED_FRAME_THRESHOLDS, f"{label}.thresholds")
    perturbations = audit["perturbations"]
    if not isinstance(perturbations, Mapping) or set(perturbations) != EXPECTED_PERTURBATIONS:
        _fail(f"{label}.perturbations differ from the frozen audit")
    for name, metrics in perturbations.items():
        metric_label = f"{label}.perturbations.{name}"
        required = {
            "noncentral_poleward_u_angle_p95_deg",
            "outward_axis_angle_p95_deg",
            "central_classification_change_rate",
            "n_numerical_fallback_outside_central",
        }
        if not isinstance(metrics, Mapping):
            _fail(f"{metric_label} must be an object")
        _require_keys(metrics, required, metric_label)
        try:
            u_angle = float(metrics["noncentral_poleward_u_angle_p95_deg"])
            outward_angle = float(metrics["outward_axis_angle_p95_deg"])
            change = float(metrics["central_classification_change_rate"])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{metric_label} contains a nonnumeric metric") from exc
        if not all(math.isfinite(v) for v in (u_angle, outward_angle, change)):
            _fail(f"{metric_label} contains nonfinite metrics")
        fallbacks = _strict_int(
            metrics["n_numerical_fallback_outside_central"],
            f"{metric_label}.n_numerical_fallback_outside_central",
            0,
        )
        if u_angle > 5.0 or outward_angle > 2.0 or change > 0.02 or fallbacks != 0:
            _fail(f"{metric_label} exceeds a hard frozen threshold")


def validate_frame_audit_bindings(
    root: Path,
    audit: Mapping[str, Any],
    table: pd.DataFrame,
    expected_rows: int,
    sealed_config_sha256: str,
    fixed_point: Mapping[str, Any],
    label: str,
) -> None:
    required = {
        "analysis_scope",
        "target_blind",
        "input_type",
        "config_sha256",
        "n_input_artifacts",
        "n_distal_qc_eligible",
        "eligible_lens_indices",
        "input_artifacts",
        "fixed_point",
    }
    _require_keys(audit, required, label)
    if audit["analysis_scope"] != ANALYSIS_SCOPE:
        _fail(f"{label} has wrong analysis scope")
    if not _strict_bool(audit["target_blind"], f"{label}.target_blind"):
        _fail(f"{label} is not target blind")
    if audit["input_type"] != "sealed_distal_artifacts_only":
        _fail(f"{label} did not consume sealed distal artifacts only")
    if audit["config_sha256"] != sealed_config_sha256:
        _fail(f"{label} config hash differs from the sealed distal artifacts")
    if _strict_int(audit["n_input_artifacts"], f"{label}.n_input_artifacts") != expected_rows:
        _fail(f"{label} did not audit all expected sealed artifacts")
    eligible = sorted(table.loc[table["distal_qc"], "lens_index"].astype(int).tolist())
    if _strict_int(audit["n_distal_qc_eligible"], f"{label}.n_distal_qc_eligible") != len(eligible):
        _fail(f"{label} distal-QC count differs from lens_summary")
    try:
        audit_eligible = [_strict_int(v, f"{label}.eligible_lens_indices", 0) for v in audit["eligible_lens_indices"]]
    except TypeError as exc:
        raise ContractError(f"{label}.eligible_lens_indices must be an array") from exc
    if audit_eligible != eligible:
        _fail(f"{label} eligible indices differ from lens_summary")
    if audit["fixed_point"] != fixed_point:
        _fail(f"{label} fixed-point evidence differs from the bundle provenance")
    artifacts = audit["input_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != expected_rows:
        _fail(f"{label} input artifact audit is incomplete")
    seen: set[int] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            _fail(f"{label} input artifact entry must be an object")
        _require_keys(entry, {"lens_index", "relative_path", "sha256"}, f"{label}.input_artifact")
        index = _strict_int(entry["lens_index"], f"{label}.input_artifact.lens_index", 0)
        if index in seen or index >= expected_rows:
            _fail(f"{label} contains a duplicate/out-of-range input artifact index")
        seen.add(index)
        relpath = f"sealed_distal/lens_{index:06d}.npz"
        if entry["relative_path"] != relpath:
            _fail(f"{label} has wrong relative path for lens {index}")
        if entry["sha256"] != sha256_file(root / relpath):
            _fail(f"{label} input artifact hash mismatch for lens {index}")
    if seen != set(range(expected_rows)):
        _fail(f"{label} input artifact indices are incomplete")


def validate_arthur_frame_audit_bindings(
    root: Path,
    audit: Mapping[str, Any],
    unit: str,
    unit_table: pd.DataFrame,
    threshold_config_sha256: str,
    fixed_point: Mapping[str, Any],
    output_manifest: Mapping[str, Path],
) -> None:
    label = f"Arthur frame audit {unit}"
    required = {
        "analysis_scope", "target_blind", "input_type", "config_sha256",
        "sealed_config_json", "n_input_artifacts", "n_distal_qc_eligible",
        "eligible_lens_indices", "input_artifacts", "fixed_point",
    }
    _require_keys(audit, required, label)
    if audit["analysis_scope"] != ANALYSIS_SCOPE:
        _fail(f"{label} has wrong analysis scope")
    if not _strict_bool(audit["target_blind"], f"{label}.target_blind"):
        _fail(f"{label} is not target blind")
    if audit["input_type"] != "sealed_distal_artifacts_only":
        _fail(f"{label} did not consume sealed distal artifacts only")
    if audit["config_sha256"] != threshold_config_sha256:
        _fail(f"{label} threshold-config hash differs from provenance")
    sealed_config_text = audit["sealed_config_json"]
    if not isinstance(sealed_config_text, str) or sha256_text(sealed_config_text) != threshold_config_sha256:
        _fail(f"{label} sealed config JSON does not bind its hash")
    try:
        sealed_config = json.loads(sealed_config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} sealed config JSON is invalid") from exc
    if canonical_json(sealed_config) != sealed_config_text:
        _fail(f"{label} sealed config JSON is noncanonical")
    _require_exact_mapping(sealed_config, EXPECTED_THRESHOLD_CONFIG, f"{label}.sealed_config")
    if audit["fixed_point"] != fixed_point:
        _fail(f"{label} fixed-point evidence differs from source provenance")

    expected_indices = sorted(unit_table["lens_index"].astype(int).tolist())
    eligible_indices = sorted(
        unit_table.loc[unit_table["distal_qc"], "lens_index"].astype(int).tolist()
    )
    if _strict_int(audit["n_input_artifacts"], f"{label}.n_input_artifacts") != len(expected_indices):
        _fail(f"{label} did not audit every source cap")
    if _strict_int(audit["n_distal_qc_eligible"], f"{label}.n_distal_qc_eligible") != len(eligible_indices):
        _fail(f"{label} distal-QC count differs from the source table")
    try:
        reported_eligible = sorted(
            _strict_int(value, f"{label}.eligible_lens_indices", 0)
            for value in audit["eligible_lens_indices"]
        )
    except TypeError as exc:
        raise ContractError(f"{label}.eligible_lens_indices must be an array") from exc
    if reported_eligible != eligible_indices:
        _fail(f"{label} eligible lens identities differ from the source table")

    volume, eye_fragment = unit.split(":eye_", 1)
    eye = _strict_int(eye_fragment, f"{label}.eye", 0)
    artifacts = audit["input_artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_indices):
        _fail(f"{label} input artifact inventory is incomplete")
    seen: set[int] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping) or set(entry) != {
            "lens_index", "relative_path", "sha256", "size_bytes"
        }:
            _fail(f"{label} input artifact entry has wrong schema")
        index = _strict_int(entry["lens_index"], f"{label}.input_artifact.lens_index", 0)
        if index in seen:
            _fail(f"{label} repeats sealed source lens {index}")
        seen.add(index)
        relpath = f"sealed_distal/{volume}/eye_{eye}/lens_{index:06d}.npz"
        if entry["relative_path"] != relpath or relpath not in output_manifest:
            _fail(f"{label} has wrong sealed source path for lens {index}")
        manifest_path = output_manifest[relpath]
        if str(entry["sha256"]) != sha256_file(manifest_path):
            _fail(f"{label} sealed source hash mismatch for lens {index}")
        if _strict_int(entry["size_bytes"], f"{label}.input_artifact.size_bytes", 0) != manifest_path.stat().st_size:
            _fail(f"{label} sealed source size mismatch for lens {index}")
    if seen != set(expected_indices):
        _fail(f"{label} sealed source lens identities are incomplete")


def _validate_binding_object(root: Path, binding: Any, expected_relpath: str, label: str) -> None:
    if not isinstance(binding, Mapping):
        _fail(f"{label} must be an object")
    _require_keys(binding, {"relative_path", "sha256", "size_bytes"}, label)
    if binding["relative_path"] != expected_relpath:
        _fail(f"{label} must bind {expected_relpath}")
    validate_file_binding(
        root,
        expected_relpath,
        {"sha256": binding["sha256"], "size_bytes": binding["size_bytes"]},
        label,
    )


def validate_visual_sample_and_review(
    root: Path,
    eye_id: str,
    attestation: Mapping[str, Any],
) -> None:
    sample_path = root / "instance_qc_visual_sample/sample_manifest.json"
    review_path = root / "instance_qc_review.json"
    sample = load_json(sample_path)
    review = load_json(review_path)
    if sample.get("schema_version") != "experiment63.instance-qc-sample.v1":
        _fail(f"{eye_id} sample manifest has wrong schema")
    if sample.get("eye_id") != eye_id or sample.get("review_scope") != "stratified_sample_only":
        _fail(f"{eye_id} sample manifest has wrong eye/scope")
    if sample.get("all_instances_manually_reviewed") is not False:
        _fail(f"{eye_id} sample manifest overclaims complete manual review")
    if _strict_int(sample.get("n_selected"), "sample_manifest.n_selected") != 32:
        _fail(f"{eye_id} sample manifest does not select exactly 32 lenses")
    if sample.get("cell_counts") != attestation["sample_coverage"]:
        _fail(f"{eye_id} sample-manifest cells differ from attestation")
    blindness = sample.get("outcome_blindness")
    if not isinstance(blindness, Mapping):
        _fail(f"{eye_id} sample manifest lacks outcome-blindness evidence")
    sampling_allowlist = [
        "eye_id",
        "lens_index",
        "seed_id",
        "distal_eligible",
        "position_u_um",
        "position_v_um",
        "distal_scale_um",
        "instance_relpath",
        "sealed_distal_relpath",
    ]
    if blindness.get("sampling_table_exact_field_allowlist") != sampling_allowlist:
        _fail(f"{eye_id} visual sample used a nonfrozen sampling table")
    if blindness.get("fitted_lens_npz_opened") is not False:
        _fail(f"{eye_id} visual sampler opened fitted target-bearing lens archives")
    if blindness.get("proximal_target_prediction_error_or_model_data_opened") is not False:
        _fail(f"{eye_id} visual sampler was not outcome blind")

    samples = sample.get("samples")
    reviewed = attestation["reviewed_samples"]
    if not isinstance(samples, list) or len(samples) != 32:
        _fail(f"{eye_id} sample manifest does not enumerate 32 samples")
    if not isinstance(reviewed, list) or len(reviewed) != 32:
        _fail(f"{eye_id} attestation does not enumerate 32 reviewed samples")
    seen_lenses: set[int] = set()
    calculated_cells = {f"r{r}_s{s}": 0 for r in range(4) for s in range(4)}
    sample_identity: list[tuple[int, str]] = []
    for ordinal, (sample_row, reviewed_row) in enumerate(zip(samples, reviewed, strict=True)):
        if not isinstance(sample_row, Mapping) or not isinstance(reviewed_row, Mapping):
            _fail(f"{eye_id} visual sample/reviewed row {ordinal} is not an object")
        if _strict_int(sample_row.get("ordinal"), f"sample[{ordinal}].ordinal") != ordinal:
            _fail(f"{eye_id} sample ordinals are not exact and ordered")
        if _strict_int(reviewed_row.get("ordinal"), f"reviewed[{ordinal}].ordinal") != ordinal:
            _fail(f"{eye_id} reviewed ordinals are not exact and ordered")
        lens_index = _strict_int(sample_row.get("lens_index"), f"sample[{ordinal}].lens_index", 0)
        if lens_index in seen_lenses:
            _fail(f"{eye_id} visual sample repeats lens {lens_index}")
        seen_lenses.add(lens_index)
        seed_id = sample_row.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id:
            _fail(f"{eye_id} sample {ordinal} has invalid seed identity")
        radial = _strict_int(sample_row.get("radial_stratum"), f"sample[{ordinal}].radial_stratum", 0)
        scale = _strict_int(sample_row.get("scale_stratum"), f"sample[{ordinal}].scale_stratum", 0)
        cell = f"r{radial}_s{scale}"
        if cell not in calculated_cells:
            _fail(f"{eye_id} sample {ordinal} is outside the frozen 4x4 strata")
        calculated_cells[cell] += 1
        for key, expected in {
            "lens_index": lens_index,
            "seed_id": seed_id,
            "radial_stratum": radial,
            "scale_stratum": scale,
            "decision": "pass",
        }.items():
            if reviewed_row.get(key) != expected:
                _fail(f"{eye_id} attested reviewed sample {ordinal} differs on {key}")
        render = sample_row.get("render")
        if not isinstance(render, Mapping):
            _fail(f"{eye_id} sample {ordinal} lacks a render binding")
        render_relpath = render.get("relative_path")
        if not isinstance(render_relpath, str) or not render_relpath.startswith("renders/"):
            _fail(f"{eye_id} sample {ordinal} render has a forbidden path")
        validate_file_binding(
            sample_path.parent,
            render_relpath,
            {"sha256": render.get("sha256"), "size_bytes": render.get("size_bytes")},
            f"{eye_id}.sample[{ordinal}].render",
        )
        if reviewed_row.get("render_sha256") != render.get("sha256"):
            _fail(f"{eye_id} reviewed sample {ordinal} does not bind its render")
        sample_identity.append((lens_index, seed_id))
    if calculated_cells != attestation["sample_coverage"]:
        _fail(f"{eye_id} sampled rows do not actually provide two per stratum")

    required_review_keys = {
        "schema_version",
        "eye_id",
        "review_scope",
        "review_mode",
        "reviewer_id",
        "reviewed_at_utc",
        "sample_manifest_sha256",
        "decisions",
    }
    if set(review) != required_review_keys:
        _fail(f"{eye_id} review JSON keys differ from the frozen schema")
    if review["schema_version"] != "experiment63.instance-qc-review.v1":
        _fail(f"{eye_id} review JSON has wrong schema")
    if review["eye_id"] != eye_id or review["review_scope"] != "stratified_sample_only":
        _fail(f"{eye_id} review JSON has wrong eye/scope")
    if review["review_mode"] != attestation["review_mode"]:
        _fail(f"{eye_id} review mode differs from attestation")
    if review["sample_manifest_sha256"] != sha256_file(sample_path):
        _fail(f"{eye_id} review does not hash-bind the sample manifest")
    decisions = review["decisions"]
    if not isinstance(decisions, list) or len(decisions) != 32:
        _fail(f"{eye_id} review JSON does not contain exactly 32 decisions")
    for ordinal, (decision, identity) in enumerate(zip(decisions, sample_identity, strict=True)):
        if not isinstance(decision, Mapping) or set(decision) != {
            "lens_index", "seed_id", "decision", "notes"
        }:
            _fail(f"{eye_id} review decision {ordinal} has wrong schema")
        if (decision["lens_index"], decision["seed_id"]) != identity:
            _fail(f"{eye_id} review decision {ordinal} differs from frozen sample order")
        if decision["decision"] != "pass" or not isinstance(decision["notes"], str):
            _fail(f"{eye_id} review decision {ordinal} did not pass cleanly")


def validate_attestation(
    root: Path,
    repo: Path,
    eye_id: str,
    expected_rows: int,
    expected_partition: Mapping[str, Any],
    expected_distal_qc: int,
) -> dict[str, Any]:
    path = root / "instance_qc_attestation.json"
    attestation = load_json(path)
    required = {
        "schema_version",
        "eye_id",
        "review_scope",
        "review_mode",
        "all_instances_manually_reviewed",
        "stratified_sample_visual_qc_passed",
        "n_reviewed",
        "technical_inventory_complete",
        "artifact_hash_verification_passed",
        "partition_exact",
        "partition_evidence",
        "bindings",
        "technical_inventory",
        "artifact_verification",
        "sample_coverage",
        "reviewed_samples",
    }
    _require_keys(attestation, required, f"attestation {eye_id}")
    if attestation["schema_version"] != ATTESTATION_SCHEMA or attestation["eye_id"] != eye_id:
        _fail(f"Wrong attestation schema or eye for {eye_id}")
    if attestation["review_scope"] != "stratified_sample_only":
        _fail(f"{eye_id} attestation has wrong review scope")
    if attestation["review_mode"] not in {
        "human",
        "ai_assisted_visual_review_without_model_outputs",
    }:
        _fail(f"{eye_id} attestation has an unapproved review mode")
    if _strict_bool(attestation["all_instances_manually_reviewed"], "all_instances_manually_reviewed"):
        _fail(f"{eye_id} attestation falsely claims all-instance manual review")
    for key in (
        "stratified_sample_visual_qc_passed",
        "technical_inventory_complete",
        "artifact_hash_verification_passed",
    ):
        if not _strict_bool(attestation[key], f"attestation.{key}"):
            _fail(f"{eye_id} attestation failed {key}")
    if _strict_int(attestation["n_reviewed"], "attestation.n_reviewed") != 32:
        _fail(f"{eye_id} attestation must review exactly 32 instances")
    if not _strict_bool(attestation["partition_exact"], "attestation.partition_exact"):
        _fail(f"{eye_id} instance assignment is not an exact foreground partition")
    strata = attestation["sample_coverage"]
    if not isinstance(strata, Mapping) or len(strata) != 16:
        _fail(f"{eye_id} attestation needs all sixteen 4x4 strata")
    expected_strata = {f"r{radius}_s{scale}" for radius in range(4) for scale in range(4)}
    if set(strata) != expected_strata:
        _fail(f"{eye_id} attestation has wrong 4x4 stratum identifiers")
    if any(_strict_int(v, f"attestation.stratum.{k}") != 2 for k, v in strata.items()):
        _fail(f"{eye_id} attestation needs exactly two reviewed instances per stratum")
    reviewed = attestation["reviewed_samples"]
    if not isinstance(reviewed, list) or len(reviewed) != 32:
        _fail(f"{eye_id} attestation must enumerate exactly 32 reviewed samples")
    partition = attestation["partition_evidence"]
    if partition != expected_partition:
        _fail(f"{eye_id} attestation partition evidence does not exactly bind the extractor evidence")

    inventory = attestation["technical_inventory"]
    if not isinstance(inventory, Mapping):
        _fail(f"{eye_id} technical_inventory must be an object")
    inventory_keys = {
        "complete",
        "n_expected",
        "n_inventory_rows",
        "lens_indices_complete",
        "seed_ids_unique",
        "one_unique_instance_artifact_per_row_including_empty",
        "n_empty_assignment_rows",
        "empty_assignment_rows_permitted",
        "artifact_point_counts_match_summary",
        "n_distal_qc_eligible",
    }
    _require_keys(inventory, inventory_keys, "technical_inventory")
    for key in (
        "complete",
        "lens_indices_complete",
        "seed_ids_unique",
        "one_unique_instance_artifact_per_row_including_empty",
        "empty_assignment_rows_permitted",
        "artifact_point_counts_match_summary",
    ):
        if not _strict_bool(inventory[key], f"technical_inventory.{key}"):
            _fail(f"{eye_id} technical inventory failed {key}")
    if _strict_int(inventory["n_expected"], "technical_inventory.n_expected") != expected_rows:
        _fail(f"{eye_id} technical inventory expected count mismatch")
    if _strict_int(inventory["n_inventory_rows"], "technical_inventory.n_inventory_rows") != expected_rows:
        _fail(f"{eye_id} technical inventory is truncated")
    empty_rows = _strict_int(
        inventory["n_empty_assignment_rows"], "technical_inventory.n_empty_assignment_rows", 0
    )
    if empty_rows > expected_rows:
        _fail(f"{eye_id} technical inventory has impossible empty-row count")
    if _strict_int(inventory["n_distal_qc_eligible"], "technical_inventory.n_distal_qc_eligible", 0) != expected_distal_qc:
        _fail(f"{eye_id} attested distal-QC count differs from lens_summary")

    verification = attestation["artifact_verification"]
    if not isinstance(verification, Mapping):
        _fail(f"{eye_id} artifact_verification must be an object")
    verification_keys = {
        "passed",
        "n_completion_manifest_artifacts_verified",
        "provenance_manifest_identical_to_completion",
        "fitted_lens_archives_hash_verified_but_not_opened",
    }
    _require_keys(verification, verification_keys, "artifact_verification")
    for key in (
        "passed",
        "provenance_manifest_identical_to_completion",
        "fitted_lens_archives_hash_verified_but_not_opened",
    ):
        if not _strict_bool(verification[key], f"artifact_verification.{key}"):
            _fail(f"{eye_id} artifact verification failed {key}")
    if _strict_int(
        verification["n_completion_manifest_artifacts_verified"],
        "artifact_verification.n_completion_manifest_artifacts_verified",
        1,
    ) < expected_rows * 3:
        _fail(f"{eye_id} artifact verifier did not cover all three per-lens archives")

    bindings = attestation["bindings"]
    if not isinstance(bindings, Mapping):
        _fail(f"{eye_id} attestation bindings must be an object")
    critical_bindings = {
        "completion": "completion.json",
        "provenance": "provenance.json",
        "lens_summary": "lens_summary.csv",
        "distal_qc_sampling": "distal_qc_sampling.csv",
        "sample_manifest": "instance_qc_visual_sample/sample_manifest.json",
        "review_json": "instance_qc_review.json",
    }
    required_binding_keys = {*critical_bindings, "renderer_code", "attester_code"}
    if set(bindings) != required_binding_keys:
        _fail(f"{eye_id} attestation binding keys differ from the frozen contract")
    for key, relpath in critical_bindings.items():
        if key not in bindings:
            _fail(f"{eye_id} attestation is missing the {key} binding")
        _validate_binding_object(root, bindings[key], relpath, f"attestation.bindings.{key}")
    for key, relpath in {
        "renderer_code": "experiments/maike-modern-ground-truth/render_instance_qc_sample.py",
        "attester_code": "experiments/maike-modern-ground-truth/attest_instance_qc.py",
    }.items():
        _validate_binding_object(repo, bindings[key], relpath, f"attestation.bindings.{key}")
    validate_visual_sample_and_review(root, eye_id, attestation)
    # The attestation producer validates the complete artifact inventory.  The
    # backend independently rehashes the four analysis-critical files above and
    # the sidecar is itself included in the sealed run input manifest.
    return attestation


def _validated_hash_size_core(binding: Any, label: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        _fail(f"{label} must be an object")
    _require_keys(binding, {"sha256", "size_bytes"}, label)
    digest = str(binding["sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _fail(f"{label}.sha256 is not a lowercase SHA-256")
    size = _strict_int(binding["size_bytes"], f"{label}.size_bytes", 0)
    return {"sha256": digest, "size_bytes": size}


def _embedded_pretty_json_binding(value: Any, label: str) -> dict[str, Any]:
    """Recreate the exact sorted/indented JSON bytes used by both Stage-1 producers."""

    try:
        payload = (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} cannot be serialized as producer JSON") from exc
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def validate_maike_producer_hashes(repo: Path, mapping: Any, label: str) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(MAIKE_PRODUCER_CODE_FILES):
        _fail(f"{label} does not contain the exact Maike producer implementation set")
    for basename, relpath in MAIKE_PRODUCER_CODE_FILES.items():
        binding = mapping[basename]
        if not isinstance(binding, Mapping) or set(binding) != {"sha256", "size_bytes"}:
            _fail(f"{label}.{basename} has wrong binding schema")
        validate_file_binding(repo, relpath, binding, f"{label}.{basename}")


def validate_maike_input_provenance(
    provenance: Mapping[str, Any],
    input_hashes: Any,
    eye_id: str,
    expected_rows: int,
    label: str,
) -> None:
    """Bind the exact TIFF ZIP -> mask -> oracle-seed chain embedded in a bundle."""

    expected_input_keys = {
        "mask_npy", "mask_provenance", "seed_csv", "seed_provenance"
    }
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != expected_input_keys:
        _fail(f"{label}.input_hashes must contain exactly {sorted(expected_input_keys)}")
    input_core: dict[str, dict[str, Any]] = {}
    for input_name in sorted(expected_input_keys):
        entry = input_hashes[input_name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "path", "sha256", "size_bytes"
        }:
            _fail(f"{label}.input_hashes.{input_name} has wrong schema")
        if not isinstance(entry["path"], str) or not entry["path"]:
            _fail(f"{label}.input_hashes.{input_name}.path is empty")
        input_core[input_name] = _validated_hash_size_core(
            entry, f"{label}.input_hashes.{input_name}"
        )

    _require_keys(
        provenance,
        {"mask_source_provenance", "seed_source_provenance"},
        f"{label} provenance",
    )
    mask = provenance["mask_source_provenance"]
    seed = provenance["seed_source_provenance"]
    if not isinstance(mask, Mapping) or not isinstance(seed, Mapping):
        _fail(f"{label} embedded mask/seed provenance must be objects")
    if _embedded_pretty_json_binding(mask, f"{label}.mask_source_provenance") != input_core[
        "mask_provenance"
    ]:
        _fail(f"{label} embedded mask provenance does not match its bound JSON bytes")
    if _embedded_pretty_json_binding(seed, f"{label}.seed_source_provenance") != input_core[
        "seed_provenance"
    ]:
        _fail(f"{label} embedded seed provenance does not match its bound JSON bytes")

    _require_keys(
        mask,
        {
            "schema_version", "eye_id", "axis_order", "spacing_um",
            "original_spacing_um", "array_sha256", "npy_sha256",
            "source_archive", "archive_sha256", "source_slices", "uncropped",
            "output",
        },
        f"{label}.mask_source_provenance",
    )
    if mask["schema_version"] != MAIKE_MASK_PROVENANCE_SCHEMA or mask["eye_id"] != eye_id:
        _fail(f"{label} embedded mask provenance has wrong schema/eye")
    if mask["axis_order"] != "zyx":
        _fail(f"{label} embedded mask provenance has wrong axis order")
    if mask["spacing_um"] != [0.325, 0.325, 0.325] or mask[
        "original_spacing_um"
    ] != [0.325, 0.325, 0.325]:
        _fail(f"{label} embedded mask provenance has wrong physical spacing")
    mask_output = mask["output"]
    mask_output_core = _validated_hash_size_core(
        mask_output, f"{label}.mask_source_provenance.output"
    )
    if mask_output_core != input_core["mask_npy"]:
        _fail(f"{label} mask NPY binding disagrees with embedded provenance")
    if (
        mask.get("array_sha256") != mask_output_core["sha256"]
        or mask.get("npy_sha256") != mask_output_core["sha256"]
    ):
        _fail(f"{label} embedded mask NPY digests disagree")
    if not isinstance(mask_output, Mapping) or any(
        mask_output.get(key) != expected
        for key, expected in {
            "format": "npy",
            "dtype": "uint8",
            "order": "C",
            "axis_order": "zyx",
        }.items()
    ):
        _fail(f"{label} embedded mask output contract changed")

    source_archive = mask["source_archive"]
    if not isinstance(source_archive, Mapping):
        _fail(f"{label} source archive binding is missing")
    _require_keys(source_archive, {"path", "name", "sha256", "size_bytes"}, f"{label}.source_archive")
    if not isinstance(source_archive["path"], str) or not source_archive["path"]:
        _fail(f"{label} source archive path is empty")
    observed_archive = {
        "name": source_archive["name"],
        **_validated_hash_size_core(source_archive, f"{label}.source_archive"),
    }
    if observed_archive != MAIKE_SOURCE_ARCHIVES[eye_id]:
        _fail(f"{label} is not derived from the frozen TIFF ZIP for {eye_id}")
    if mask.get("archive_sha256") != observed_archive["sha256"]:
        _fail(f"{label} embedded source archive digests disagree")
    expected_stack = MAIKE_SOURCE_STACK_GEOMETRY[eye_id]
    source_slices = mask["source_slices"]
    uncropped = mask["uncropped"]
    if not isinstance(source_slices, Mapping) or _strict_int(
        source_slices.get("count"), f"{label}.source_slices.count", 1
    ) != expected_stack["slice_count"]:
        _fail(f"{label} source TIFF slice count changed")
    if not isinstance(uncropped, Mapping) or uncropped.get(
        "shape_zyx"
    ) != expected_stack["uncropped_shape_zyx"]:
        _fail(f"{label} uncropped TIFF stack shape changed")
    if uncropped.get("image_shape_yx") != expected_stack["uncropped_shape_zyx"][1:]:
        _fail(f"{label} uncropped TIFF image shape is inconsistent")

    _require_keys(
        seed,
        {
            "schema_version", "eye_id", "seed_role", "seed_csv_sha256",
            "csv_sha256", "n_expected", "n_rows", "n_foreground_hits",
            "lens_index_range", "candidate_seeds_per_voxel",
            "input_axis_direction", "output_axis_direction", "input_hashes",
            "output",
        },
        f"{label}.seed_source_provenance",
    )
    exact_seed_fields = {
        "schema_version": MAIKE_SEED_PROVENANCE_SCHEMA,
        "eye_id": eye_id,
        "seed_role": MAIKE_SEED_ROLE,
        "n_expected": expected_rows,
        "n_rows": expected_rows,
        "n_foreground_hits": expected_rows,
        "lens_index_range": [0, expected_rows - 1],
        "candidate_seeds_per_voxel": 1,
        "input_axis_direction": "toward_eye_center",
        "output_axis_direction": "away_from_eye_center",
    }
    for key, expected in exact_seed_fields.items():
        if seed.get(key) != expected:
            _fail(f"{label}.seed_source_provenance.{key} changed")
    seed_output_core = _validated_hash_size_core(
        seed["output"], f"{label}.seed_source_provenance.output"
    )
    if seed_output_core != input_core["seed_csv"]:
        _fail(f"{label} seed CSV binding disagrees with embedded provenance")
    if (
        seed.get("seed_csv_sha256") != seed_output_core["sha256"]
        or seed.get("csv_sha256") != seed_output_core["sha256"]
    ):
        _fail(f"{label} embedded seed CSV digests disagree")

    seed_inputs = seed["input_hashes"]
    expected_seed_inputs = {
        "oda_csv", "transform_json", "mask_npy", "mask_provenance", "source_archive"
    }
    if not isinstance(seed_inputs, Mapping) or set(seed_inputs) != expected_seed_inputs:
        _fail(f"{label} embedded seed input inventory has wrong schema")
    for key in ("oda_csv", "transform_json", "mask_npy", "mask_provenance"):
        binding = seed_inputs[key]
        if not isinstance(binding, Mapping) or set(binding) != {
            "path", "sha256", "size_bytes"
        }:
            _fail(f"{label}.seed_source_provenance.input_hashes.{key} is malformed")
        if not isinstance(binding["path"], str) or not binding["path"]:
            _fail(f"{label}.seed_source_provenance.input_hashes.{key}.path is empty")
        _validated_hash_size_core(
            binding, f"{label}.seed_source_provenance.input_hashes.{key}"
        )
    if _validated_hash_size_core(
        seed_inputs["mask_npy"], f"{label}.seed_source_provenance.input_hashes.mask_npy"
    ) != input_core["mask_npy"]:
        _fail(f"{label} seed map does not bind the same mask NPY")
    if _validated_hash_size_core(
        seed_inputs["mask_provenance"],
        f"{label}.seed_source_provenance.input_hashes.mask_provenance",
    ) != input_core["mask_provenance"]:
        _fail(f"{label} seed map does not bind the same mask provenance")
    if seed_inputs["source_archive"] != source_archive:
        _fail(f"{label} seed and mask provenance bind different TIFF ZIPs")


def load_maike_eye_bundle(
    root: Path, eye_id: str, expected_commit: str, repo: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_rows = EXPECTED_EYES[eye_id]
    completion = load_json(root / "completion.json")
    provenance = load_json(root / "provenance.json")
    label = f"Maike bundle {eye_id}"
    common_exact_keys = {
        "schema_version",
        "status",
        "eye_id",
        "species",
        "sex",
        "biological_independence",
        "analysis_scope",
        "isolation_basis",
        "threshold_config",
        "threshold_config_sha256",
        "pipeline_config",
        "pipeline_config_sha256",
        "predictor_pipeline_config",
        "sealed_config_sha256",
        "n_expected",
        "n_rows",
        "counts",
        "contiguous_indices",
        "index_range",
        "instance_segmentation_validated",
        "target_cohort_definitions",
        "partition_evidence",
        "fixed_point",
        "sealed_distal_stage1_manifest",
        "input_hashes",
        "git",
        "producer_implementation_hashes",
        "created_utc",
        "output_manifest",
    }
    expected_document_keys = {
        "completion": common_exact_keys,
        "provenance": common_exact_keys
        | {"mask_source_provenance", "seed_source_provenance"},
    }
    for document_name, document in (("completion", completion), ("provenance", provenance)):
        doc_label = f"{label} {document_name}"
        if set(document) != expected_document_keys[document_name]:
            missing = sorted(expected_document_keys[document_name] - set(document))
            extra = sorted(set(document) - expected_document_keys[document_name])
            _fail(
                f"{doc_label} keys differ from the exact producer schema; "
                f"missing={missing}, extra={extra}"
            )
        if document["schema_version"] != EYE_BUNDLE_SCHEMA or document["status"] != "complete":
            _fail(f"{doc_label} has wrong schema/status")
        if document["eye_id"] != eye_id:
            _fail(f"{doc_label} has wrong eye_id")
        expected_species_fragment = "simulans" if eye_id.startswith("M3_") else "mauritiana"
        if expected_species_fragment not in str(document["species"]).strip().lower():
            _fail(f"{doc_label} species disagrees with the frozen eye identity")
        expected_sex = "female" if "_F_" in eye_id else "male"
        reported_sex = str(document["sex"]).strip().lower()
        accepted_sex = {expected_sex, expected_sex[0]}
        if reported_sex not in accepted_sex:
            _fail(f"{doc_label} sex disagrees with the frozen eye identity")
        _require_exact_mapping(
            document["biological_independence"],
            {
                "independent_unit": "animal",
                "animal_id": eye_id,
                "one_eye_per_animal_in_validation": True,
                "source_basis": "one supplied eye stack per uniquely named fly",
            },
            f"{doc_label}.biological_independence",
        )
        _require_exact_mapping(
            document["target_cohort_definitions"],
            EXPECTED_TARGET_COHORT_DEFINITIONS,
            f"{doc_label}.target_cohort_definitions",
        )
        if document["analysis_scope"] != ANALYSIS_SCOPE or document["isolation_basis"] != ISOLATION_BASIS:
            _fail(f"{doc_label} violates the conditional Stage-2 scope")
        if _strict_int(document["n_expected"], f"{doc_label}.n_expected") != expected_rows:
            _fail(f"{doc_label} expected count mismatch")
        if _strict_int(document["n_rows"], f"{doc_label}.n_rows") != expected_rows:
            _fail(f"{doc_label} is truncated")
        _require_exact_mapping(document["threshold_config"], EXPECTED_THRESHOLD_CONFIG, f"{doc_label}.threshold_config")
        if document["threshold_config_sha256"] != sha256_text(canonical_json(EXPECTED_THRESHOLD_CONFIG)):
            _fail(f"{doc_label} threshold hash mismatch")
        _require_exact_mapping(document["pipeline_config"], EXPECTED_PIPELINE_CONFIG, f"{doc_label}.pipeline_config")
        if document["pipeline_config_sha256"] != sha256_text(canonical_json(EXPECTED_PIPELINE_CONFIG)):
            _fail(f"{doc_label} pipeline hash mismatch")
        _require_exact_mapping(
            document["predictor_pipeline_config"],
            EXPECTED_PREDICTOR_PIPELINE_CONFIG,
            f"{doc_label}.predictor_pipeline_config",
        )
        validate_git_record(document["git"], expected_commit, doc_label)
        if not _strict_bool(document["contiguous_indices"], f"{doc_label}.contiguous_indices"):
            _fail(f"{doc_label} does not contain every expected row")
        if _strict_bool(
            document["instance_segmentation_validated"],
            f"{doc_label}.instance_segmentation_validated",
        ):
            _fail(f"{doc_label} must not overclaim full manual segmentation validation")
        validate_maike_producer_hashes(
            repo,
            document["producer_implementation_hashes"],
            f"{doc_label}.producer_implementation_hashes",
        )
    if completion["output_manifest"] != provenance["output_manifest"]:
        _fail(f"{label} completion/provenance manifests disagree")
    for key in common_exact_keys:
        if completion[key] != provenance[key]:
            _fail(f"{label} completion/provenance disagree on {key}")
    validated_manifest = validate_output_manifest(root, completion["output_manifest"], f"{label}.output_manifest")
    for relpath in ("lens_summary.csv", "distal_qc_sampling.csv", "distal_frame_audit.json"):
        if relpath not in validated_manifest:
            _fail(f"{label} output manifest does not bind {relpath}")
    stage1_manifest = completion["sealed_distal_stage1_manifest"]
    if not isinstance(stage1_manifest, Mapping) or len(stage1_manifest) != expected_rows:
        _fail(f"{label} Stage-1 sealed manifest is incomplete")
    for index in range(expected_rows):
        relpath = f"sealed_distal/lens_{index:06d}.npz"
        if relpath not in stage1_manifest or relpath not in completion["output_manifest"]:
            _fail(f"{label} Stage-1 sealed manifest is missing {relpath}")
        if stage1_manifest[relpath] != completion["output_manifest"][relpath]:
            _fail(f"{label} sealed artifact {relpath} changed after Stage 1")
    if completion["index_range"] != [0, expected_rows - 1]:
        _fail(f"{label} index_range is not the complete frozen range")
    validate_maike_input_provenance(
        provenance,
        completion["input_hashes"],
        eye_id,
        expected_rows,
        label,
    )

    _require_keys(provenance, {"fixed_point", "partition_evidence"}, f"{label} provenance")
    _require_keys(completion, {"partition_evidence"}, f"{label} completion")
    if completion["partition_evidence"] != provenance["partition_evidence"]:
        _fail(f"{label} completion/provenance partition evidence disagrees")
    validate_partition_evidence(completion["partition_evidence"], f"{label}.partition_evidence")
    validate_fixed_point(provenance["fixed_point"], f"{label}.fixed_point")
    frame_audit = load_json(root / "distal_frame_audit.json")
    validate_frame_audit(frame_audit, eye_id, f"{label}.frame_audit")

    try:
        table = pd.read_csv(root / "lens_summary.csv")
    except (OSError, pd.errors.ParserError) as exc:
        raise ContractError(f"Cannot read {label} lens_summary.csv") from exc
    table = validate_lens_table(
        table,
        label=f"{label}.lens_summary",
        expected_eye=eye_id,
        expected_rows=expected_rows,
    )
    table = validate_maike_component_table(table, f"{label}.lens_summary")
    table = add_invariant_features_by_unit(table, ["eye_id"], label)
    counts = completion["counts"]
    expected_counts = {
        "stage1": expected_rows,
        "distal_qc": int(table["distal_qc"].sum()),
        "target_resolvable": int(table["target_resolvable"].sum()),
        "target_qc": int(table["target_qc"].sum()),
    }
    if not isinstance(counts, Mapping) or set(counts) != set(expected_counts):
        _fail(f"{label}.counts has wrong schema")
    observed_counts = {
        key: _strict_int(value, f"{label}.counts.{key}", 0) for key, value in counts.items()
    }
    if observed_counts != expected_counts:
        _fail(f"{label}.counts differs from lens_summary")
    validate_attestation(
        root,
        repo,
        eye_id,
        expected_rows,
        completion["partition_evidence"],
        int(table["distal_qc"].sum()),
    )
    n_raw_resolvable = int(table["target_resolvable"].sum())
    n_resolvable = require_primary_resolution_gate(table, expected_rows, eye_id)

    sealed_config_hashes: set[str] = set()
    for index in range(expected_rows):
        relpath = f"sealed_distal/lens_{index:06d}.npz"
        if relpath not in validated_manifest:
            _fail(f"{label} manifest is missing {relpath}")
        row = table.loc[table["lens_index"] == index]
        if len(row) != 1:
            _fail(f"{label} has no unique summary row for sealed lens {index}")
        config = validate_sealed_distal_npz(
            root / relpath,
            index,
            require_distal_qc_support=bool(row.iloc[0]["distal_qc"]),
        )
        sealed_config_hashes.add(sha256_text(canonical_json(config)))
        lens_relpath = f"lenses/lens_{index:06d}.npz"
        if lens_relpath not in validated_manifest:
            _fail(f"{label} manifest is missing {lens_relpath}")
        validate_lens_artifact(root / lens_relpath, row.iloc[0])
    if len(sealed_config_hashes) != 1:
        _fail(f"{label} sealed artifacts do not share one frozen Stage-2 config")
    sealed_config_hash = next(iter(sealed_config_hashes))
    if completion["sealed_config_sha256"] != sealed_config_hash:
        _fail(f"{label} completion sealed-config hash disagrees with artifacts")
    if provenance["sealed_config_sha256"] != sealed_config_hash:
        _fail(f"{label} provenance sealed-config hash disagrees with artifacts")
    validate_frame_audit_bindings(
        root,
        frame_audit,
        table,
        expected_rows,
        sealed_config_hash,
        provenance["fixed_point"],
        f"{label}.frame_audit",
    )

    metadata = {
        "eye_id": eye_id,
        "animal_id": completion["biological_independence"]["animal_id"],
        "biological_independence": completion["biological_independence"],
        "source_archive": MAIKE_SOURCE_ARCHIVES[eye_id],
        "source_stack_geometry": MAIKE_SOURCE_STACK_GEOMETRY[eye_id],
        "n_expected": expected_rows,
        "completion_sha256": sha256_file(root / "completion.json"),
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "summary_sha256": sha256_file(root / "lens_summary.csv"),
        "frame_audit_sha256": sha256_file(root / "distal_frame_audit.json"),
        "attestation_sha256": sha256_file(root / "instance_qc_attestation.json"),
        "sealed_config_sha256": sealed_config_hash,
        "n_distal_qc": int(table["distal_qc"].sum()),
        "n_target_resolvable": n_raw_resolvable,
        "n_primary_cohort": n_resolvable,
        "n_target_qc": int(table["target_qc"].sum()),
    }
    return table, metadata


def _verify_repo_code_hashes(repo: Path, mapping: Any, label: str) -> None:
    if not isinstance(mapping, Mapping) or set(mapping) != set(SOURCE_CODE_FILES):
        _fail(f"{label} must bind exactly the frozen Arthur source implementation")
    for relpath in SOURCE_CODE_FILES:
        value = mapping[relpath]
        if not isinstance(value, str):
            _fail(f"{label}[{relpath!r}] must be a SHA-256 string")
        expected = value
        path = _safe_relative(repo, relpath, label)
        if str(expected) != sha256_file(path):
            _fail(f"{label} hash mismatch for {relpath}")


def validate_arthur_stage1_diagnostics(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(ARTHUR_VOLUMES):
        _fail("Arthur stage1_diagnostics has wrong volume keys")
    required = {
        "n_landmarks", "n_eye_0", "n_eye_1",
        "median_lens_tip_distance_um", "median_landmark_mesh_distance_um",
        "n_raw_patches", "stage1_failure_counts",
    }
    for volume in ARTHUR_VOLUMES:
        diagnostic = value[volume]
        label = f"Arthur stage1_diagnostics.{volume}"
        if not isinstance(diagnostic, Mapping) or set(diagnostic) != required:
            _fail(f"{label} has wrong schema")
        n_landmarks = _strict_int(diagnostic["n_landmarks"], f"{label}.n_landmarks", 1)
        n_eye_0 = _strict_int(diagnostic["n_eye_0"], f"{label}.n_eye_0", 1)
        n_eye_1 = _strict_int(diagnostic["n_eye_1"], f"{label}.n_eye_1", 1)
        n_raw = _strict_int(diagnostic["n_raw_patches"], f"{label}.n_raw_patches", 1)
        if n_landmarks != ARTHUR_LANDMARK_COUNTS[volume] or n_raw != ARTHUR_STAGE1_COUNTS[volume]:
            _fail(f"{label} changed the frozen landmark/Stage-1 counts")
        if n_eye_0 + n_eye_1 != n_landmarks:
            _fail(f"{label} eye membership does not partition all landmarks")
        failures = diagnostic["stage1_failure_counts"]
        if not isinstance(failures, Mapping) or set(failures) != {
            "patch_support", "distal_layer_support"
        }:
            _fail(f"{label}.stage1_failure_counts has wrong schema")
        n_failed = sum(
            _strict_int(count, f"{label}.stage1_failure_counts.{reason}", 0)
            for reason, count in failures.items()
        )
        if n_raw + n_failed != n_landmarks:
            _fail(f"{label} Stage-1 accounting is not exact")
        for key in (
            "median_lens_tip_distance_um", "median_landmark_mesh_distance_um"
        ):
            try:
                metric = float(diagnostic[key])
            except (TypeError, ValueError) as exc:
                raise ContractError(f"{label}.{key} is not numeric") from exc
            if not math.isfinite(metric) or metric <= 0.0:
                _fail(f"{label}.{key} must be finite and positive")


def load_arthur_source_bundle(
    table_path: Path,
    provenance_path: Path,
    repo: Path,
    expected_commit: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = load_json(provenance_path)
    label = "Arthur source provenance"
    required = {
        "schema_version",
        "status",
        "analysis_scope",
        "isolation_basis",
        "oracle_stage1_scope",
        "table_sha256",
        "table_size_bytes",
        "n_rows",
        "volumes",
        "threshold_config",
        "threshold_config_sha256",
        "pipeline_config",
        "pipeline_config_sha256",
        "counts",
        "counts_by_volume",
        "cohorts",
        "exclusion_reason_counts",
        "target_observation_contract",
        "fixed_points",
        "frame_audits",
        "stage1_diagnostics",
        "input_files",
        "output_artifacts",
        "git",
        "code_sha256",
        "eyemap",
        "biological_independence",
        "coordinate_calibration",
        "created_utc",
    }
    if set(provenance) != required:
        missing = sorted(required - set(provenance))
        extra = sorted(set(provenance) - required)
        _fail(
            f"{label} keys differ from the exact producer schema; "
            f"missing={missing}, extra={extra}"
        )
    if provenance["schema_version"] != ARTHUR_SOURCE_SCHEMA or provenance["status"] != "complete":
        _fail(f"{label} has wrong schema/status")
    if provenance["analysis_scope"] != ANALYSIS_SCOPE or provenance["isolation_basis"] != ISOLATION_BASIS:
        _fail(f"{label} violates the conditional Stage-2 scope")
    if provenance["oracle_stage1_scope"] != "oracle_correspondence_and_distal_localization_only":
        _fail(f"{label} has wrong oracle Stage-1 scope")
    validate_git_record(provenance["git"], expected_commit, label)
    _require_exact_mapping(
        provenance["eyemap"],
        {"commit": "99d2a43123db636cedb55af9ff31a59657e7d17e", "dirty": False},
        f"{label}.eyemap",
    )
    _require_exact_mapping(provenance["threshold_config"], EXPECTED_THRESHOLD_CONFIG, f"{label}.threshold_config")
    if provenance["threshold_config_sha256"] != sha256_text(canonical_json(EXPECTED_THRESHOLD_CONFIG)):
        _fail(f"{label} threshold hash mismatch")
    _require_exact_mapping(
        provenance["pipeline_config"], EXPECTED_ARTHUR_PIPELINE_CONFIG, f"{label}.pipeline_config"
    )
    if provenance["pipeline_config_sha256"] != sha256_text(
        canonical_json(EXPECTED_ARTHUR_PIPELINE_CONFIG)
    ):
        _fail(f"{label} pipeline hash mismatch")
    _require_exact_mapping(
        provenance["target_observation_contract"],
        TARGET_OBSERVATION_CONTRACT,
        f"{label}.target_observation_contract",
    )
    validate_arthur_stage1_diagnostics(provenance["stage1_diagnostics"])
    if table_path.is_symlink() or not table_path.is_file():
        _fail("Arthur source table must be an existing regular non-symlink file")
    if sha256_file(table_path) != provenance["table_sha256"]:
        _fail("Arthur source table SHA-256 mismatch")
    if table_path.stat().st_size != _strict_int(provenance["table_size_bytes"], "Arthur table size", 0):
        _fail("Arthur source table size mismatch")
    if tuple(sorted(map(str, provenance["volumes"]))) != tuple(sorted(ARTHUR_VOLUMES)):
        _fail("Arthur provenance must contain exactly the three frozen source volumes")
    _verify_repo_code_hashes(repo, provenance["code_sha256"], f"{label}.code_sha256")
    input_files = provenance["input_files"]
    _require_exact_mapping(input_files, ARTHUR_INPUT_FILES, f"{label}.input_files")
    validated_source_manifest = validate_output_manifest(
        provenance_path.parent,
        provenance["output_artifacts"],
        f"{label}.output_artifacts",
    )
    if "arthur_source_table.csv" not in validated_source_manifest:
        _fail("Arthur output manifest does not bind arthur_source_table.csv")
    try:
        same_table = table_path.resolve(strict=True) == validated_source_manifest[
            "arthur_source_table.csv"
        ].resolve(strict=True)
    except OSError as exc:
        raise ContractError("Arthur source table path cannot be resolved") from exc
    if not same_table:
        _fail("Arthur table argument is not the table inside the verified producer bundle")
    independence = provenance["biological_independence"]
    if not isinstance(independence, Mapping):
        _fail("Arthur provenance lacks biological independence metadata")
    expected_independence = {
        "independent_unit": "whole_head_scan_animal",
        "n_independent_animals": 3,
        "eyes_per_animal": 2,
        "nesting": "bilateral_eyes_nested_within_animal",
        "animals": [f"Arthur_{volume}" for volume in ARTHUR_VOLUMES],
        "sex": "female",
        "age_days": [6, 7],
    }
    _require_exact_mapping(independence, expected_independence, "Arthur biological_independence")

    fixed_points = provenance["fixed_points"]
    frame_audits = provenance["frame_audits"]
    if not isinstance(fixed_points, Mapping) or not isinstance(frame_audits, Mapping):
        _fail("Arthur fixed_points/frame_audits must be objects")
    if set(fixed_points) != set(frame_audits):
        _fail("Arthur fixed-point and frame-audit unit sets differ")
    expected_source_units = {
        f"{volume}:eye_{eye}" for volume in ARTHUR_VOLUMES for eye in (0, 1)
    }
    if set(fixed_points) != expected_source_units:
        _fail("Arthur provenance must contain both nested eyes from each of three volumes")
    loaded_frame_audits: dict[str, Mapping[str, Any]] = {}
    for unit, fixed_point in fixed_points.items():
        validate_fixed_point(fixed_point, f"Arthur fixed point {unit}")
        audit_record = frame_audits[unit]
        if not isinstance(audit_record, Mapping) or set(audit_record) != {
            "relative_path", "sha256", "size_bytes"
        }:
            _fail(f"Arthur frame audit {unit} must be an exact hash/size binding")
        relpath = str(audit_record["relative_path"])
        if relpath not in validated_source_manifest:
            _fail(f"Arthur output manifest does not contain frame audit {unit}")
        audit_path = validate_file_binding(
            provenance_path.parent,
            relpath,
            {"sha256": audit_record["sha256"], "size_bytes": audit_record["size_bytes"]},
            f"Arthur frame audit {unit}",
        )
        if validated_source_manifest[relpath] != audit_path:
            _fail(f"Arthur frame audit {unit} binding differs from output manifest")
        audit = load_json(audit_path)
        validate_frame_audit(audit, str(unit), f"Arthur frame audit {unit}")
        loaded_frame_audits[unit] = audit

    try:
        table = pd.read_csv(table_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ContractError("Cannot read Arthur source table") from exc
    if len(table) != _strict_int(provenance["n_rows"], "Arthur provenance.n_rows", 1):
        _fail("Arthur source table row count mismatch")
    table = validate_lens_table(table, label="Arthur source table", require_volume=True)
    if tuple(sorted(table["volume"].astype(str).unique())) != tuple(sorted(ARTHUR_VOLUMES)):
        _fail("Arthur source table has wrong volumes")
    identity_columns = {
        "animal_id", "source_eye_unit", "stage1_eligible", "species", "sex",
        "age_days_min", "age_days_max", "sealed_distal_relpath",
        "lens_target_relpath", "distal_qc_reasons", "target_resolvability_reasons",
        "target_qc_reasons",
    }
    missing_identity = sorted(identity_columns - set(table.columns))
    if missing_identity:
        _fail(f"Arthur source table is missing identity columns: {missing_identity}")
    table["eye_id"] = table["eye_id"].map(
        lambda value: _strict_int(value, "Arthur source table.eye_id", 0)
    )
    if not set(table["eye_id"]).issubset({0, 1}):
        _fail("Arthur source table eye_id must be 0 or 1")
    stage1_eligible = np.array(
        [_strict_bool(value, "Arthur source table.stage1_eligible") for value in table["stage1_eligible"]]
    )
    if not np.all(stage1_eligible):
        _fail("Arthur source table contains a row outside the frozen Stage-1 cohort")
    for row_index, row in table.iterrows():
        volume = str(row["volume"])
        eye = int(row["eye_id"])
        if str(row["animal_id"]) != f"Arthur_{volume}":
            _fail(f"Arthur source row {row_index} has wrong animal nesting")
        if str(row["source_eye_unit"]) != f"{volume}:eye_{eye}":
            _fail(f"Arthur source row {row_index} has wrong source-eye identity")
        if str(row["species"]).strip() != "Drosophila melanogaster":
            _fail(f"Arthur source row {row_index} has wrong species")
        if str(row["sex"]).strip().lower() != "female":
            _fail(f"Arthur source row {row_index} has wrong sex")
        if _strict_int(row["age_days_min"], "Arthur age_days_min") != 6:
            _fail(f"Arthur source row {row_index} has wrong minimum age")
        if _strict_int(row["age_days_max"], "Arthur age_days_max") != 7:
            _fail(f"Arthur source row {row_index} has wrong maximum age")
    volume_counts = table["volume"].astype(str).value_counts().to_dict()
    if volume_counts != ARTHUR_STAGE1_COUNTS:
        _fail("Arthur source Stage-1 counts differ from the three frozen animal volumes")
    counts = provenance["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {
        "stage1", "distal_qc", "target_resolvable", "target_qc"
    }:
        _fail("Arthur provenance counts have wrong schema")
    observed_counts = {
        "stage1": len(table),
        "distal_qc": int(table["distal_qc"].sum()),
        "target_resolvable": int(table["target_resolvable"].sum()),
        "target_qc": int(table["target_qc"].sum()),
    }
    if {key: _strict_int(value, f"Arthur counts.{key}", 0) for key, value in counts.items()} != observed_counts:
        _fail("Arthur provenance counts differ from the source table")
    if observed_counts["stage1"] != 4942 or observed_counts["distal_qc"] != 4897:
        _fail("Arthur frozen Stage-1/distal-QC counts changed")
    if table.groupby([table["volume"].astype(str), "eye_id"])["lens_index"].apply(lambda x: x.duplicated().any()).any():
        _fail("Arthur source table contains duplicate lens indices within an eye")

    counts_by_volume = provenance["counts_by_volume"]
    if not isinstance(counts_by_volume, Mapping) or set(counts_by_volume) != set(ARTHUR_VOLUMES):
        _fail("Arthur counts_by_volume has wrong volume keys")
    observed_by_volume: dict[str, dict[str, int]] = {}
    for volume in ARTHUR_VOLUMES:
        volume_table = table[table["volume"].astype(str) == volume]
        observed_by_volume[volume] = {
            "stage1": len(volume_table),
            "distal_qc": int(volume_table["distal_qc"].sum()),
            "target_resolvable": int(volume_table["target_resolvable"].sum()),
            "target_qc": int(volume_table["target_qc"].sum()),
        }
        recorded = counts_by_volume[volume]
        if not isinstance(recorded, Mapping) or set(recorded) != set(observed_by_volume[volume]):
            _fail(f"Arthur counts_by_volume.{volume} has wrong schema")
        normalized = {
            key: _strict_int(value, f"Arthur counts_by_volume.{volume}.{key}", 0)
            for key, value in recorded.items()
        }
        if normalized != observed_by_volume[volume]:
            _fail(f"Arthur counts_by_volume.{volume} differs from the source table")
    _require_exact_mapping(
        provenance["cohorts"],
        {
            "both_models_and_primary_targets": observed_counts["target_resolvable"],
            "both_models_target_qc_sensitivity": observed_counts["target_qc"],
            "cross_validation_independence_unit": "volume_animal",
            "n_cross_validation_units": 3,
            "eyes_are_nested_within_volume_animal": True,
        },
        "Arthur provenance.cohorts",
    )

    def reason_counts(column: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for raw in table[column].fillna("").astype(str):
            for reason in (part for part in raw.split("|") if part):
                output[reason] = output.get(reason, 0) + 1
        return dict(sorted(output.items()))

    _require_exact_mapping(
        provenance["exclusion_reason_counts"],
        {
            "distal_qc": reason_counts("distal_qc_reasons"),
            "target_resolvability": reason_counts("target_resolvability_reasons"),
            "target_qc": reason_counts("target_qc_reasons"),
        },
        "Arthur provenance.exclusion_reason_counts",
    )
    _require_exact_mapping(
        provenance["coordinate_calibration"],
        {
            "20231107_raw_pixel_pitch_um": 0.9882,
            "20240530_raw_pixel_pitch_um": 0.9884,
            "20240701_raw_pixel_pitch_um": None,
            "20240701_note": (
                "unresolved: public deposit identifies 20240710 at 1.0065 um; "
                "no specimen mapping was assumed"
            ),
            "mesh_coordinates": "physical_micrometres_from_supplied_WRL",
        },
        "Arthur provenance.coordinate_calibration",
    )

    expected_output_paths = {"arthur_source_table.csv"}
    expected_output_paths.update(str(binding["relative_path"]) for binding in frame_audits.values())
    threshold_hash = str(provenance["threshold_config_sha256"])
    for row_index, row in table.iterrows():
        volume = str(row["volume"])
        eye = int(row["eye_id"])
        lens_index = int(row["lens_index"])
        sealed_relpath = f"sealed_distal/{volume}/eye_{eye}/lens_{lens_index:06d}.npz"
        if str(row["sealed_distal_relpath"]) != sealed_relpath:
            _fail(f"Arthur source row {row_index} has wrong sealed-distal path")
        if sealed_relpath not in validated_source_manifest:
            _fail(f"Arthur output manifest omits sealed source row {row_index}")
        expected_output_paths.add(sealed_relpath)
        validate_arthur_sealed_distal_npz(
            validated_source_manifest[sealed_relpath], row, threshold_hash
        )

        target_value = row["lens_target_relpath"]
        if bool(row["distal_qc"]):
            target_relpath = f"lenses/{volume}/eye_{eye}/lens_{lens_index:06d}.npz"
            if str(target_value) != target_relpath:
                _fail(f"Arthur source row {row_index} has wrong target-artifact path")
            if target_relpath not in validated_source_manifest:
                _fail(f"Arthur output manifest omits target source row {row_index}")
            expected_output_paths.add(target_relpath)
            validate_arthur_target_npz(validated_source_manifest[target_relpath], row)
        elif not pd.isna(target_value) and str(target_value).strip():
            _fail(f"Arthur source row {row_index} binds a target outside distal QC")
    if set(validated_source_manifest) != expected_output_paths:
        extra = sorted(set(validated_source_manifest) - expected_output_paths)[:10]
        missing = sorted(expected_output_paths - set(validated_source_manifest))[:10]
        _fail(
            "Arthur output manifest is not the exact producer inventory; "
            f"first extra={extra}, first missing={missing}"
        )
    for unit, audit in loaded_frame_audits.items():
        volume, eye_fragment = unit.split(":eye_", 1)
        eye = int(eye_fragment)
        unit_table = table[
            (table["volume"].astype(str) == volume) & (table["eye_id"] == eye)
        ]
        validate_arthur_frame_audit_bindings(
            provenance_path.parent,
            audit,
            unit,
            unit_table,
            threshold_hash,
            fixed_points[unit],
            validated_source_manifest,
        )
    table = add_invariant_features_by_unit(table, ["volume", "eye_id"], "Arthur source")
    primary = table["distal_qc"] & table["target_resolvable"]
    if not primary.any():
        _fail("Arthur source primary cohort is empty")
    if set(table.loc[primary, "volume"].astype(str)) != set(ARTHUR_VOLUMES):
        _fail("Every Arthur volume must contribute primary training targets")
    return table, provenance


def validate_lens_artifact(path: Path, row: pd.Series) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_arrays = {
                "schema_version",
                "lens_index",
                "distal_points_xyz_um",
                "proximal_points_xyz_um",
                "canonical_grid_xy",
                "target_smoothed_thickness_um",
                "raw_target_xy_normalized",
                "raw_target_thickness_um",
                "target_coefficients_c0_c5",
                "distal_frame_origin_xyz_um",
                "distal_frame_u_xyz",
                "distal_frame_v_xyz",
                "distal_frame_outward_xyz",
                "distal_coefficients_normalized",
                "config_json",
                "config_sha256",
            }
            if set(data.files) != expected_arrays:
                _fail(f"{path} fitted-lens arrays differ from the exact v2 contract")
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot safely read lens artifact {path}: {exc}") from exc
    grid = arrays["canonical_grid_xy"]
    smooth = arrays["target_smoothed_thickness_um"]
    raw_xy = arrays["raw_target_xy_normalized"]
    raw_thickness = arrays["raw_target_thickness_um"]
    schema = arrays["schema_version"]
    artifact_index = arrays["lens_index"]
    artifact_coefficients = arrays["target_coefficients_c0_c5"]
    config_text_array = arrays["config_json"]
    config_hash_array = arrays["config_sha256"]
    if schema.shape != () or schema.dtype.kind != "U" or schema.item() != "experiment63.lens.v2":
        _fail(f"{path}: wrong fitted-lens schema")
    expected_index = int(row["lens_index"])
    if artifact_index.shape != () or artifact_index.dtype != np.dtype("int64") or int(artifact_index.item()) != expected_index:
        _fail(f"{path}: fitted-lens index mismatch")
    if grid.dtype != np.dtype("float64") or grid.shape != (81, 2) or not np.array_equal(grid, CANONICAL_GRID_XY):
        _fail(f"{path}: canonical grid mismatch")
    if smooth.dtype != np.dtype("float64") or smooth.shape != (81,):
        _fail(f"{path}: smoothed target must be float64[81]")
    if raw_xy.dtype != np.dtype("float64") or raw_xy.ndim != 2 or raw_xy.shape[1] != 2:
        _fail(f"{path}: raw target coordinates must be float64[N,2]")
    if raw_thickness.dtype != np.dtype("float64") or raw_thickness.shape != (len(raw_xy),):
        _fail(f"{path}: raw target thickness must be float64[N]")
    if len(raw_xy) != int(row["target_support"]):
        _fail(f"{path}: raw target support differs from lens_summary")
    if not np.all(np.isfinite(raw_xy)) or not np.all(np.isfinite(raw_thickness)):
        _fail(f"{path}: raw target geometry is nonfinite")
    if np.any(np.linalg.norm(raw_xy, axis=1) > 1.0 + 1e-10):
        _fail(f"{path}: raw target coordinates leave the normalized distal cap")
    coefficients = row.loc[list(TARGET_COLUMNS)].to_numpy(float)
    if (
        artifact_coefficients.dtype != np.dtype("float64")
        or artifact_coefficients.shape != (6,)
        or not np.allclose(
            artifact_coefficients, coefficients, rtol=0.0, atol=0.0, equal_nan=True
        )
    ):
        _fail(f"{path}: target coefficients differ from lens_summary")
    if (
        config_text_array.shape != ()
        or config_text_array.dtype.kind != "U"
        or config_hash_array.shape != ()
        or config_hash_array.dtype.kind != "U"
    ):
        _fail(f"{path}: fitted-lens config binding has wrong dtype/shape")
    config_text = str(config_text_array.item())
    if sha256_text(config_text) != str(config_hash_array.item()):
        _fail(f"{path}: fitted-lens config hash mismatch")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: invalid fitted-lens config JSON") from exc
    if canonical_json(config) != config_text:
        _fail(f"{path}: fitted-lens config JSON is noncanonical")
    if set(config) != {
        "analysis_scope", "isolation_basis", "threshold_config", "predictor_pipeline_config"
    }:
        _fail(f"{path}: fitted-lens config has unexpected top-level keys")
    if config.get("analysis_scope") != ANALYSIS_SCOPE or config.get("isolation_basis") != ISOLATION_BASIS:
        _fail(f"{path}: fitted-lens config violates analysis scope")
    _require_exact_mapping(
        config.get("threshold_config"), EXPECTED_THRESHOLD_CONFIG, f"{path}:threshold_config"
    )
    _require_exact_mapping(
        config.get("predictor_pipeline_config"),
        EXPECTED_PREDICTOR_PIPELINE_CONFIG,
        f"{path}:predictor_pipeline_config",
    )
    design = _evaluate_quadratic_design(raw_xy)
    intrinsic_structure = bool(
        len(raw_xy) >= 25
        and np.linalg.matrix_rank(design) == 6
        and np.all(np.isfinite(artifact_coefficients))
        and np.all(np.isfinite(smooth))
        and np.allclose(smooth, CANONICAL_DESIGN @ artifact_coefficients, rtol=0, atol=1e-9)
    )
    expected_resolvable = bool(
        row["distal_qc"]
        and row.get("partition_target_resolvable", True)
        and intrinsic_structure
    )
    if bool(row["target_resolvable"]) != expected_resolvable:
        _fail(f"{path}: target_resolvable is not the exact structural flag")
    if not expected_resolvable:
        return arrays
    if not np.isclose(np.median(raw_thickness), float(row["target_depth_um"]), rtol=0, atol=1e-9):
        _fail(f"{path}: target depth differs from raw target median")
    if np.quantile(raw_thickness, 0.05) <= 0:
        # Negative local thickness is allowed in the primary structural cohort;
        # it is excluded only by the target-QC sensitivity flag.
        if bool(row["target_qc"]):
            _fail(f"{path}: target_qc is true despite nonpositive raw q05 thickness")
    q05 = float(np.quantile(raw_thickness, 0.05))
    if not np.isclose(
        q05, float(row["target_q05_raw_thickness_um"]), rtol=0, atol=1e-9
    ):
        _fail(f"{path}: raw target q05 differs from lens_summary")
    raw_rmse = float(
        np.sqrt(
            np.mean(
                (raw_thickness - design @ coefficients) ** 2
            )
        )
    )
    if not np.isclose(raw_rmse, float(row["target_rmse_um"]), rtol=0, atol=1e-9):
        _fail(f"{path}: target RMSE differs from raw target geometry")
    return arrays


def _evaluate_quadratic_design(xy: np.ndarray) -> np.ndarray:
    x, y = xy.T
    return np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])


def _evaluate_coefficients_on_xy(coefficients: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return _evaluate_quadratic_design(xy) @ coefficients


def score_predictions(
    table: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    artifact_roots: Mapping[str, Path] | None = None,
) -> pd.DataFrame:
    targets = symmetrized_targets(table)
    target_grid = target_grids(targets)
    depths = table["target_depth_um"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    raw_artifact_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for method, prediction_values in predictions.items():
        prediction_values = np.asarray(prediction_values, dtype=float)
        if prediction_values.shape == targets.shape:
            coefficients: np.ndarray | None = prediction_values
            predicted_grid = target_grids(coefficients)
        elif prediction_values.shape == target_grid.shape:
            coefficients = None
            predicted_grid = prediction_values
        else:
            _fail(
                f"Prediction {method} must have shape {targets.shape} coefficients "
                f"or {target_grid.shape} grid values"
            )
        if not np.all(np.isfinite(predicted_grid)):
            _fail(f"Prediction {method} contains NaN or infinity")
        absolute = np.abs(predicted_grid - target_grid)
        for position, (index, record) in enumerate(table.iterrows()):
            row: dict[str, Any] = {
                "eye_id": str(record["eye_id"]),
                "lens_index": int(record["lens_index"]),
                "method": method,
                "cohort_primary": bool(record["distal_qc"] and record["target_resolvable"]),
                "cohort_target_qc": bool(record["distal_qc"] and record["target_qc"]),
                "central": bool(record["central"]),
                "target_depth_um": float(depths[position]),
                "target_support": int(record["target_support"]),
                "smoothed_81pt_mae_um": float(np.mean(absolute[position])),
                "smoothed_81pt_p90_error_um": float(np.quantile(absolute[position], 0.90)),
                "smoothed_81pt_normalized_mae": float(np.mean(absolute[position]) / depths[position]),
                "raw_unsmoothed_available": False,
                "raw_unsmoothed_mae_um": np.nan,
                "raw_unsmoothed_p90_error_um": np.nan,
                "raw_unsmoothed_normalized_mae": np.nan,
            }
            if artifact_roots is not None and coefficients is not None:
                eye_id = str(record["eye_id"])
                if eye_id in artifact_roots:
                    lens_index = int(record["lens_index"])
                    artifact_key = (eye_id, lens_index)
                    if artifact_key not in raw_artifact_cache:
                        artifact_path = (
                            artifact_roots[eye_id]
                            / f"lenses/lens_{lens_index:06d}.npz"
                        )
                        raw_artifact_cache[artifact_key] = validate_lens_artifact(
                            artifact_path, record
                        )
                    arrays = raw_artifact_cache[artifact_key]
                    raw_prediction = _evaluate_coefficients_on_xy(coefficients[position], arrays["raw_target_xy_normalized"])
                    raw_error = np.abs(raw_prediction - arrays["raw_target_thickness_um"])
                    row.update(
                        {
                            "raw_unsmoothed_available": True,
                            "raw_unsmoothed_mae_um": float(np.mean(raw_error)),
                            "raw_unsmoothed_p90_error_um": float(np.quantile(raw_error, 0.90)),
                            "raw_unsmoothed_normalized_mae": float(np.mean(raw_error) / depths[position]),
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_eye_comparison(metrics: pd.DataFrame, cohort_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = metrics[metrics[cohort_column]].copy()
    pivot = subset.pivot(index=["eye_id", "lens_index"], columns="method", values="smoothed_81pt_normalized_mae")
    absolute_pivot = subset.pivot(
        index=["eye_id", "lens_index"],
        columns="method",
        values="smoothed_81pt_mae_um",
    )
    if not {"position_scale_control", "position_scale_plus_distal_shape"}.issubset(pivot.columns):
        _fail("Primary metrics lack both nested models")
    raw_pivot: pd.DataFrame | None = None
    nested_methods = {"position_scale_control", "position_scale_plus_distal_shape"}
    raw_subset = subset[subset["method"].isin(nested_methods)]
    if (
        "raw_unsmoothed_available" in raw_subset
        and len(raw_subset)
        and raw_subset["raw_unsmoothed_available"].all()
    ):
        raw_pivot = raw_subset.pivot(
            index=["eye_id", "lens_index"],
            columns="method",
            values="raw_unsmoothed_normalized_mae",
        )
    rows: list[dict[str, Any]] = []
    for eye_id in EXPECTED_EYES:
        try:
            eye = pivot.xs(eye_id, level="eye_id")
        except KeyError as exc:
            raise ContractError(f"No scored lenses for eye {eye_id} in {cohort_column}") from exc
        control = float(np.median(eye["position_scale_control"]))
        shape = float(np.median(eye["position_scale_plus_distal_shape"]))
        difference = control - shape
        absolute_eye = absolute_pivot.xs(eye_id, level="eye_id")
        control_absolute = float(np.median(absolute_eye["position_scale_control"]))
        shape_absolute = float(
            np.median(absolute_eye["position_scale_plus_distal_shape"])
        )
        outcome = "win" if difference > 0.0 else ("loss" if difference < 0.0 else "tie")
        species = (
            "Drosophila simulans"
            if eye_id.startswith("M3_")
            else "Drosophila mauritiana"
        )
        sex = "female" if "_F_" in eye_id else "male"
        row = {
                "cohort": cohort_column,
                "eye_id": eye_id,
                "species": species,
                "sex": sex,
                "independent_unit": "animal (one eye per fly)",
                "n_lenses": len(eye),
                "control_median_normalized_mae": control,
                "shape_median_normalized_mae": shape,
                "control_minus_shape": difference,
                "control_median_smoothed_81pt_mae_um": control_absolute,
                "shape_median_smoothed_81pt_mae_um": shape_absolute,
                "control_minus_shape_median_smoothed_81pt_mae_um": (
                    control_absolute - shape_absolute
                ),
                "shape_outcome": outcome,
            }
        if raw_pivot is not None:
            raw_eye = raw_pivot.xs(eye_id, level="eye_id")
            row.update(
                {
                    "control_median_raw_unsmoothed_normalized_mae": float(
                        np.median(raw_eye["position_scale_control"])
                    ),
                    "shape_median_raw_unsmoothed_normalized_mae": float(
                        np.median(raw_eye["position_scale_plus_distal_shape"])
                    ),
                }
            )
        rows.append(row)
    comparison = pd.DataFrame(rows)
    wins = int((comparison["shape_outcome"] == "win").sum())
    losses = int((comparison["shape_outcome"] == "loss").sum())
    ties = int((comparison["shape_outcome"] == "tie").sum())
    result = {
        "cohort": cohort_column,
        "independent_unit": "animal (one eye per fly)",
        "n_independent_animals": 12,
        "wins": wins,
        "losses": losses,
        "ties_nonwins": ties,
        "pass_rule": "strict shape win in at least 10 of 12 animals; ties are nonwins",
        "passes_frozen_primary_rule": wins >= 10,
        "fixed_denominator_two_sided_binomial_reference_p": two_sided_fixed_denominator_p(wins, 12),
        "conventional_tie_dropping_two_sided_sign_p": conventional_tie_dropping_sign_p(wins, losses),
    }
    return comparison, result


def species_sex_descriptive_table(per_eye_primary: pd.DataFrame) -> pd.DataFrame:
    """Frozen descriptive strata; never a confirmatory rescue analysis."""

    required = {"eye_id", "species", "sex", "control_minus_shape", "shape_outcome"}
    if not required.issubset(per_eye_primary.columns):
        _fail("Per-eye primary table lacks fields for species/sex description")
    if len(per_eye_primary) != 12 or set(per_eye_primary["eye_id"].astype(str)) != set(
        EXPECTED_EYES
    ):
        _fail("Species/sex description requires exactly the twelve named Maike animals")
    strata: list[tuple[str, str, str]] = [
        ("species", species, "all")
        for species in ("Drosophila simulans", "Drosophila mauritiana")
    ]
    strata.extend(("sex", "all", sex) for sex in ("female", "male"))
    strata.extend(
        ("species_x_sex", species, sex)
        for species in ("Drosophila simulans", "Drosophila mauritiana")
        for sex in ("female", "male")
    )
    rows: list[dict[str, Any]] = []
    for stratum_type, species, sex in strata:
        keep = np.ones(len(per_eye_primary), dtype=bool)
        if species != "all":
            keep &= per_eye_primary["species"].to_numpy(str) == species
        if sex != "all":
            keep &= per_eye_primary["sex"].to_numpy(str) == sex
        cell = per_eye_primary.loc[keep].sort_values("eye_id")
        expected_n = 6 if stratum_type in {"species", "sex"} else 3
        if len(cell) != expected_n:
            _fail(
                f"Descriptive stratum {stratum_type}, {species}, {sex} "
                f"does not contain {expected_n} animals"
            )
        effects = {
            str(record.eye_id): float(record.control_minus_shape)
            for record in cell.itertuples(index=False)
        }
        outcomes = cell["shape_outcome"].astype(str)
        rows.append(
            {
                "analysis_status": "prespecified_descriptive_nonconfirmatory",
                "stratum_type": stratum_type,
                "species": species,
                "sex": sex,
                "n_animals": expected_n,
                "named_eye_control_minus_shape_json": canonical_json(effects),
                "median_control_minus_shape": float(np.median(list(effects.values()))),
                "strict_shape_wins": int((outcomes == "win").sum()),
                "strict_shape_losses": int((outcomes == "loss").sum()),
                "ties": int((outcomes == "tie").sum()),
            }
        )
    return pd.DataFrame(rows)


def per_eye_method_descriptive_table(
    metrics: pd.DataFrame, cohort_column: str
) -> pd.DataFrame:
    subset = metrics[metrics[cohort_column]].copy()
    rows: list[dict[str, Any]] = []
    for (eye_id, method), group in subset.groupby(["eye_id", "method"], sort=True):
        eye_id = str(eye_id)
        row = {
            "eye_id": eye_id,
            "species": (
                "Drosophila simulans"
                if eye_id.startswith("M3_")
                else "Drosophila mauritiana"
            ),
            "sex": "female" if "_F_" in eye_id else "male",
            "method": str(method),
            "n_lenses": len(group),
            "median_smoothed_81pt_mae_um": float(
                np.median(group["smoothed_81pt_mae_um"])
            ),
            "median_smoothed_81pt_normalized_mae": float(
                np.median(group["smoothed_81pt_normalized_mae"])
            ),
        }
        if group["raw_unsmoothed_available"].all():
            row["median_raw_unsmoothed_mae_um"] = float(
                np.median(group["raw_unsmoothed_mae_um"])
            )
            row["median_raw_unsmoothed_normalized_mae"] = float(
                np.median(group["raw_unsmoothed_normalized_mae"])
            )
        else:
            row["median_raw_unsmoothed_mae_um"] = math.nan
            row["median_raw_unsmoothed_normalized_mae"] = math.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_methods = {
        "position_scale_control",
        "position_scale_plus_distal_shape",
        "equal_volume_source_template",
    }
    if set(result["method"]) != expected_methods:
        _fail("Internal primary method summary has an unexpected method set")
    if any(
        set(result.loc[result["method"] == method, "eye_id"].astype(str))
        != set(EXPECTED_EYES)
        for method in expected_methods
    ):
        _fail("Internal primary method summary does not cover all twelve animals")
    return result


def summarize_internal_methods(per_eye_methods: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {
        "analysis_status": "descriptive_internal_frozen_comparator",
        "inference_role": "does_not_modify_the_nested_primary_pass_rule",
        "methods": {},
    }
    for method, group in per_eye_methods.groupby("method", sort=True):
        output["methods"][str(method)] = {
            "n_animals": len(group),
            "median_of_eye_median_smoothed_81pt_mae_um": float(
                np.median(group["median_smoothed_81pt_mae_um"])
            ),
            "median_of_eye_median_smoothed_81pt_normalized_mae": float(
                np.median(group["median_smoothed_81pt_normalized_mae"])
            ),
        }
    return output


def nonconfirmatory_nested_summary(
    nested_summary: Mapping[str, Any], analysis_status: str
) -> dict[str, Any]:
    return {
        "analysis_status": analysis_status,
        "cohort": nested_summary["cohort"],
        "independent_unit": "animal (one eye per fly)",
        "n_independent_animals": 12,
        "strict_shape_wins": nested_summary["wins"],
        "strict_shape_losses": nested_summary["losses"],
        "ties_counted_as_nonwins": nested_summary["ties_nonwins"],
        "confirmatory_pass_fail_rule": None,
        "decision_role": "cannot_modify_rescue_or_replace_the_primary_result",
    }


def run_primary_models(source: pd.DataFrame, maike: pd.DataFrame) -> dict[str, Any]:
    metric_validity_gate = {
        "source": validate_metric_denominator_gate(source, "Arthur source"),
        "maike_all": validate_metric_denominator_gate(maike, "Maike validation cohort"),
        "maike_by_eye": {
            eye_id: validate_metric_denominator_gate(
                maike[maike["eye_id"].astype(str) == eye_id], f"Maike eye {eye_id}"
            )
            for eye_id in EXPECTED_EYES
        },
    }
    source_primary = source[source["distal_qc"] & source["target_resolvable"]].copy()
    test_primary = maike[maike["distal_qc"] & maike["target_resolvable"]].copy()
    if set(test_primary["eye_id"].astype(str)) != set(EXPECTED_EYES):
        _fail("Every independent Maike animal must contribute to the primary cohort")

    control_alpha, control_audit = select_alpha_leave_one_volume_out(source_primary, CONTROL_FEATURE_NAMES)
    shape_alpha, shape_audit = select_alpha_leave_one_volume_out(source_primary, SHAPE_FEATURE_NAMES)
    source_targets = symmetrized_targets(source_primary)
    groups = source_primary["volume"].astype(str).to_numpy()
    control_model = fit_weighted_ridge(
        source_primary.loc[:, CONTROL_FEATURE_NAMES].to_numpy(float),
        source_targets,
        groups,
        control_alpha,
    )
    shape_model = fit_weighted_ridge(
        source_primary.loc[:, SHAPE_FEATURE_NAMES].to_numpy(float),
        source_targets,
        groups,
        shape_alpha,
    )
    control_prediction = control_model.predict_even(
        test_primary.loc[:, CONTROL_FEATURE_NAMES].to_numpy(float),
        test_primary["central"].to_numpy(bool),
    )
    shape_prediction = shape_model.predict_even(
        test_primary.loc[:, SHAPE_FEATURE_NAMES].to_numpy(float),
        test_primary["central"].to_numpy(bool),
    )

    # Supporting source-only template.  Taking a median within each source
    # volume and then across volumes gives every source volume equal weight.
    volume_templates = np.vstack(
        [
            np.median(symmetrized_targets(source_primary[source_primary["volume"].astype(str) == volume]), axis=0)
            for volume in ARTHUR_VOLUMES
        ]
    )
    template = np.median(volume_templates, axis=0)
    template_prediction = np.repeat(template[None, :], len(test_primary), axis=0)
    central = test_primary["central"].to_numpy(bool)
    template_prediction[central, 1] = 0.0
    average = 0.5 * (template_prediction[central, 3] + template_prediction[central, 5])
    template_prediction[central, 3] = average
    template_prediction[central, 5] = average

    return {
        "source_primary": source_primary,
        "test_primary": test_primary,
        "predictions": {
            "position_scale_control": control_prediction,
            "position_scale_plus_distal_shape": shape_prediction,
            "equal_volume_source_template": template_prediction,
        },
        "alpha_audit": pd.concat([control_audit, shape_audit], ignore_index=True),
        "selected_alphas": {"control": control_alpha, "shape": shape_alpha},
        "models": {"control": control_model, "shape": shape_model},
        "metric_validity_gate": metric_validity_gate,
    }


def run_target_qc_sensitivity_models(source: pd.DataFrame, maike: pd.DataFrame) -> dict[str, Any]:
    """Repeat model selection, fitting, and evaluation on strict target-QC rows."""
    source_cohort = source[source["distal_qc"] & source["target_qc"]].copy()
    test_cohort = maike[maike["distal_qc"] & maike["target_qc"]].copy()
    if set(source_cohort["volume"].astype(str)) != set(ARTHUR_VOLUMES):
        _fail("Every Arthur volume must contribute to the target-QC sensitivity training cohort")
    if set(test_cohort["eye_id"].astype(str)) != set(EXPECTED_EYES):
        _fail("Every Maike animal must contribute to the target-QC sensitivity cohort")
    control_alpha, control_audit = select_alpha_leave_one_volume_out(
        source_cohort, CONTROL_FEATURE_NAMES
    )
    shape_alpha, shape_audit = select_alpha_leave_one_volume_out(
        source_cohort, SHAPE_FEATURE_NAMES
    )
    targets = symmetrized_targets(source_cohort)
    groups = source_cohort["volume"].astype(str).to_numpy()
    control_model = fit_weighted_ridge(
        source_cohort.loc[:, CONTROL_FEATURE_NAMES].to_numpy(float),
        targets,
        groups,
        control_alpha,
    )
    shape_model = fit_weighted_ridge(
        source_cohort.loc[:, SHAPE_FEATURE_NAMES].to_numpy(float),
        targets,
        groups,
        shape_alpha,
    )
    predictions = {
        "position_scale_control": control_model.predict_even(
            test_cohort.loc[:, CONTROL_FEATURE_NAMES].to_numpy(float),
            test_cohort["central"].to_numpy(bool),
        ),
        "position_scale_plus_distal_shape": shape_model.predict_even(
            test_cohort.loc[:, SHAPE_FEATURE_NAMES].to_numpy(float),
            test_cohort["central"].to_numpy(bool),
        ),
    }
    return {
        "source": source_cohort,
        "test": test_cohort,
        "predictions": predictions,
        "alpha_audit": pd.concat([control_audit, shape_audit], ignore_index=True),
        "selected_alphas": {"control": control_alpha, "shape": shape_alpha},
        "models": {"control": control_model, "shape": shape_model},
    }


def run_within_maike_nested_loao_secondary(maike: pd.DataFrame) -> dict[str, Any]:
    """Prespecified descriptive diagnostic, isolated from the external primary fit."""

    cohort = maike[maike["distal_qc"] & maike["target_resolvable"]].copy()
    animals = cohort["eye_id"].astype(str)
    if set(animals) != set(EXPECTED_EYES):
        _fail("Every Maike animal must contribute to the within-Maike LOAO diagnostic")
    metric_gate = validate_metric_denominator_gate(
        cohort, "within-Maike nested-LOAO secondary cohort"
    )
    predictions = {
        "position_scale_control": np.full((len(cohort), 6), np.nan, dtype=float),
        "position_scale_plus_distal_shape": np.full((len(cohort), 6), np.nan, dtype=float),
    }
    selected_alphas: dict[str, dict[str, float]] = {}
    models: dict[str, dict[str, WeightedRidge]] = {}
    audits: list[pd.DataFrame] = []
    for outer_heldout in EXPECTED_EYES:
        train = animals != outer_heldout
        test = ~train
        training = cohort.loc[train]
        control_alpha, control_audit = select_alpha_nested_maike_loao(
            training,
            CONTROL_FEATURE_NAMES,
            outer_heldout_animal=outer_heldout,
        )
        shape_alpha, shape_audit = select_alpha_nested_maike_loao(
            training,
            SHAPE_FEATURE_NAMES,
            outer_heldout_animal=outer_heldout,
        )
        audits.extend([control_audit, shape_audit])
        training_groups = training["eye_id"].astype(str).to_numpy()
        training_targets = symmetrized_targets(training)
        control_model = fit_weighted_ridge(
            training.loc[:, CONTROL_FEATURE_NAMES].to_numpy(float),
            training_targets,
            training_groups,
            control_alpha,
        )
        shape_model = fit_weighted_ridge(
            training.loc[:, SHAPE_FEATURE_NAMES].to_numpy(float),
            training_targets,
            training_groups,
            shape_alpha,
        )
        predictions["position_scale_control"][test.to_numpy()] = control_model.predict_even(
            cohort.loc[test, CONTROL_FEATURE_NAMES].to_numpy(float),
            cohort.loc[test, "central"].to_numpy(bool),
        )
        predictions["position_scale_plus_distal_shape"][test.to_numpy()] = shape_model.predict_even(
            cohort.loc[test, SHAPE_FEATURE_NAMES].to_numpy(float),
            cohort.loc[test, "central"].to_numpy(bool),
        )
        selected_alphas[outer_heldout] = {
            "control": control_alpha,
            "shape": shape_alpha,
        }
        models[outer_heldout] = {
            "control": control_model,
            "shape": shape_model,
        }
    for method, prediction in predictions.items():
        if not np.all(np.isfinite(prediction)):
            _fail(f"Within-Maike LOAO prediction {method} is incomplete")
    return {
        "cohort": cohort,
        "predictions": predictions,
        "alpha_audit": pd.concat(audits, ignore_index=True),
        "selected_alphas": selected_alphas,
        "models": models,
        "metric_validity_gate": metric_gate,
        "contract": dict(SECONDARY_MAIKE_LOAO_CONTRACT),
    }


def aggregate_within_maike_secondary(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    comparison, nested_summary = _aggregate_eye_comparison(metrics, "cohort_primary")
    comparison.insert(0, "analysis", "within_maike_nested_loao_secondary")
    result = {
        **SECONDARY_MAIKE_LOAO_CONTRACT,
        "independent_unit": "animal (one eye per fly)",
        "n_independent_animals": 12,
        "strict_shape_wins": nested_summary["wins"],
        "strict_shape_losses": nested_summary["losses"],
        "ties_counted_as_nonwins": nested_summary["ties_nonwins"],
        "confirmatory_pass_fail_rule": None,
    }
    return comparison, result


def _write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def _write_json_atomic(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_binding(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def execute_frozen_primary(
    *,
    repo: Path,
    expected_commit: str,
    arthur_table: Path,
    arthur_provenance: Path,
    maike_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate, score exactly once, and atomically seal all outputs."""
    if output_dir.exists():
        _fail(f"Exclusive primary output already exists: {output_dir}")
    expected_commit = require_frozen_git(repo, expected_commit)
    frozen_execution_environment = execution_environment()
    run_created_utc = datetime.now(timezone.utc).isoformat()
    if EXPECTED_TOTAL != sum(EXPECTED_EYES.values()):
        _fail("Internal expected-count map no longer sums to 11,369")

    source, source_provenance = load_arthur_source_bundle(
        arthur_table, arthur_provenance, repo, expected_commit
    )
    eye_tables: list[pd.DataFrame] = []
    eye_metadata: list[dict[str, Any]] = []
    eye_roots: dict[str, Path] = {}
    for eye_id in EXPECTED_EYES:
        eye_root = maike_root / eye_id
        table, metadata = load_maike_eye_bundle(eye_root, eye_id, expected_commit, repo)
        eye_tables.append(table)
        eye_metadata.append(metadata)
        eye_roots[eye_id] = eye_root
    if any(
        not isinstance(metadata, Mapping)
        or "animal_id" not in metadata
        or "eye_id" not in metadata
        or "biological_independence" not in metadata
        for metadata in eye_metadata
    ):
        _fail("Maike validation metadata lacks explicit animal/eye independence evidence")
    animal_ids = [str(metadata["animal_id"]) for metadata in eye_metadata]
    eye_ids = [str(metadata["eye_id"]) for metadata in eye_metadata]
    if (
        len(animal_ids) != 12
        or len(set(animal_ids)) != 12
        or set(animal_ids) != set(EXPECTED_EYES)
        or len(set(eye_ids)) != 12
        or set(eye_ids) != set(EXPECTED_EYES)
    ):
        _fail(
            "Maike validation requires exactly 12 distinct animals with one uniquely "
            "identified eye per fly"
        )
    maike = pd.concat(eye_tables, ignore_index=True)
    if len(maike) != EXPECTED_TOTAL:
        _fail(f"Maike consolidated table has {len(maike)} rows, not 11,369")

    fitted = run_primary_models(source, maike)
    for metadata in eye_metadata:
        metadata["metric_validity_gate"] = fitted["metric_validity_gate"]["maike_by_eye"][
            str(metadata["eye_id"])
        ]
    primary_table = fitted["test_primary"]
    metrics = score_predictions(primary_table, fitted["predictions"], artifact_roots=eye_roots)
    per_eye_primary, primary_result = _aggregate_eye_comparison(metrics, "cohort_primary")
    species_sex_descriptive = species_sex_descriptive_table(per_eye_primary)
    per_eye_methods = per_eye_method_descriptive_table(metrics, "cohort_primary")
    internal_method_summary = summarize_internal_methods(per_eye_methods)
    sensitivity = run_target_qc_sensitivity_models(source, maike)
    sensitivity_metrics = score_predictions(
        sensitivity["test"], sensitivity["predictions"], artifact_roots=eye_roots
    )
    per_eye_sensitivity, sensitivity_nested_summary = _aggregate_eye_comparison(
        sensitivity_metrics, "cohort_target_qc"
    )
    sensitivity_result = nonconfirmatory_nested_summary(
        sensitivity_nested_summary, "prespecified_target_qc_sensitivity_nonconfirmatory"
    )
    maike_secondary = run_within_maike_nested_loao_secondary(maike)
    maike_secondary_metrics = score_predictions(
        maike_secondary["cohort"],
        maike_secondary["predictions"],
        artifact_roots=eye_roots,
    )
    per_eye_maike_secondary, maike_secondary_result = aggregate_within_maike_secondary(
        maike_secondary_metrics
    )

    parent = output_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        _write_csv_atomic(metrics, staging / "per_lens_metrics.csv")
        _write_csv_atomic(
            sensitivity_metrics, staging / "per_lens_target_qc_sensitivity_metrics.csv"
        )
        _write_csv_atomic(
            maike_secondary_metrics,
            staging / "per_lens_within_maike_nested_loao_secondary_metrics.csv",
        )
        _write_csv_atomic(per_eye_primary, staging / "per_eye_primary.csv")
        _write_csv_atomic(
            per_eye_methods, staging / "per_eye_internal_method_descriptive.csv"
        )
        _write_csv_atomic(
            species_sex_descriptive, staging / "species_sex_descriptive.csv"
        )
        _write_csv_atomic(per_eye_sensitivity, staging / "per_eye_target_qc_sensitivity.csv")
        _write_csv_atomic(
            per_eye_maike_secondary,
            staging / "per_eye_within_maike_nested_loao_secondary.csv",
        )
        primary_alpha_audit = fitted["alpha_audit"].copy()
        primary_alpha_audit.insert(0, "analysis", "primary")
        sensitivity_alpha_audit = sensitivity["alpha_audit"].copy()
        sensitivity_alpha_audit.insert(0, "analysis", "target_qc_sensitivity")
        _write_csv_atomic(
            pd.concat([primary_alpha_audit, sensitivity_alpha_audit], ignore_index=True),
            staging / "source_only_alpha_selection.csv",
        )
        _write_csv_atomic(
            maike_secondary["alpha_audit"],
            staging / "within_maike_nested_loao_alpha_selection.csv",
        )
        _write_json_atomic(
            {
                "schema_version": RUN_SCHEMA,
                "analysis_scope": ANALYSIS_SCOPE,
                "independent_unit": "animal (one eye per fly)",
                "primary": primary_result,
                "target_qc_sensitivity": sensitivity_result,
                "within_maike_nested_loao_secondary": maike_secondary_result,
                "internal_method_descriptive": internal_method_summary,
                "metric_validity_gate": fitted["metric_validity_gate"],
                "within_maike_secondary_metric_validity_gate": maike_secondary[
                    "metric_validity_gate"
                ],
                "selected_alphas": {
                    "primary": fitted["selected_alphas"],
                    "target_qc_sensitivity": sensitivity["selected_alphas"],
                    "within_maike_nested_loao_secondary": maike_secondary[
                        "selected_alphas"
                    ],
                },
                "fixed_reference_at_ten_wins": 0.03857421875,
                "primary_cohort": "distal_qc AND target_resolvable",
                "sensitivity_cohort": "distal_qc AND target_qc (models reselected and refit)",
                "target_handling": (
                    "fit c0,c1,c3,c5; predict c2,c4=0; score all six; "
                    "symmetrize central targets and predictions"
                ),
                "target_observation_contract": TARGET_OBSERVATION_CONTRACT,
                "predictor_input_representation_contract": (
                    PREDICTOR_INPUT_REPRESENTATION_CONTRACT
                ),
                "execution_environment": frozen_execution_environment,
                "created_utc": run_created_utc,
            },
            staging / "primary_result.json",
        )
        models_payload: dict[str, np.ndarray] = {}
        for model_name, model in fitted["models"].items():
            models_payload[f"{model_name}_feature_mean"] = model.feature_mean
            models_payload[f"{model_name}_feature_scale"] = model.feature_scale
            models_payload[f"{model_name}_target_mean_even"] = model.target_mean
            models_payload[f"{model_name}_coefficients"] = model.coefficients
            models_payload[f"{model_name}_alpha"] = np.array(model.alpha, dtype=np.float64)
        for model_name, model in sensitivity["models"].items():
            prefix = f"target_qc_sensitivity_{model_name}"
            models_payload[f"{prefix}_feature_mean"] = model.feature_mean
            models_payload[f"{prefix}_feature_scale"] = model.feature_scale
            models_payload[f"{prefix}_target_mean_even"] = model.target_mean
            models_payload[f"{prefix}_coefficients"] = model.coefficients
            models_payload[f"{prefix}_alpha"] = np.array(model.alpha, dtype=np.float64)
        for outer_eye, outer_models in maike_secondary["models"].items():
            for model_name, model in outer_models.items():
                prefix = f"within_maike_loao_{outer_eye}_{model_name}"
                models_payload[f"{prefix}_feature_mean"] = model.feature_mean
                models_payload[f"{prefix}_feature_scale"] = model.feature_scale
                models_payload[f"{prefix}_target_mean_even"] = model.target_mean
                models_payload[f"{prefix}_coefficients"] = model.coefficients
                models_payload[f"{prefix}_alpha"] = np.array(model.alpha, dtype=np.float64)
        np.savez(staging / "frozen_model_parameters.npz", **models_payload)

        output_files = {
            path.name: _file_binding(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        input_manifest = {
            "arthur_table": {
                "path": str(arthur_table),
                **_file_binding(arthur_table),
                "metric_validity_gate": fitted["metric_validity_gate"]["source"],
            },
            "arthur_provenance": {"path": str(arthur_provenance), **_file_binding(arthur_provenance)},
            "maike_eyes": eye_metadata,
        }
        sealed_manifest = {
            "schema_version": RUN_SCHEMA,
            "status": "sealed_first_complete_run",
            "analysis_scope": ANALYSIS_SCOPE,
            "isolation_basis": ISOLATION_BASIS,
            "independent_unit": "animal (one eye per fly)",
            "n_independent_animals": 12,
            "expected_eye_counts": EXPECTED_EYES,
            "expected_total_rows": EXPECTED_TOTAL,
            "git": {"commit": expected_commit, "dirty": False},
            "created_utc": run_created_utc,
            "execution_environment": frozen_execution_environment,
            "backend": {
                "relative_path": "experiments/maike-modern-ground-truth/experiment_63_primary_backend.py",
                "sha256": sha256_file(Path(__file__)),
            },
            "threshold_config": EXPECTED_THRESHOLD_CONFIG,
            "threshold_config_sha256": sha256_text(canonical_json(EXPECTED_THRESHOLD_CONFIG)),
            "ridge_alphas": RIDGE_ALPHAS.tolist(),
            "source_weighting": "equal volume mass; all sample weights sum to one",
            "source_grouping": (
                "three independent Arthur scan/animal volumes; bilateral eyes are nested "
                "within volume and are not treated as six independent animals"
            ),
            "alpha_selection": (
                "Arthur-only leave-one-volume-out; equal mean of three held-out "
                "volume median 81-point normalized MAEs"
            ),
            "primary_cohort": "distal_qc AND target_resolvable",
            "target_qc_sensitivity": "reselect and refit both models on distal_qc AND target_qc",
            "target_observation_contract": TARGET_OBSERVATION_CONTRACT,
            "predictor_input_representation_contract": (
                PREDICTOR_INPUT_REPRESENTATION_CONTRACT
            ),
            "internal_method_descriptive": internal_method_summary,
            "within_maike_nested_loao_secondary_contract": SECONDARY_MAIKE_LOAO_CONTRACT,
            "within_maike_nested_loao_secondary_result": maike_secondary_result,
            "within_maike_secondary_metric_validity_gate": maike_secondary[
                "metric_validity_gate"
            ],
            "metric_validity_gate": fitted["metric_validity_gate"],
            "control_features": list(CONTROL_FEATURE_NAMES),
            "shape_additions": list(SHAPE_ADDITION_NAMES),
            "trained_target_indices": EVEN_TARGET_INDICES.tolist(),
            "zero_predicted_target_indices": ODD_REFLECTION_INDICES.tolist(),
            "input_manifest": input_manifest,
            "output_manifest": output_files,
            "source_provenance_sha256": sha256_file(arthur_provenance),
            "selected_alphas": {
                "primary": fitted["selected_alphas"],
                "target_qc_sensitivity": sensitivity["selected_alphas"],
                "within_maike_nested_loao_secondary": maike_secondary["selected_alphas"],
            },
            "primary_result_sha256": output_files["primary_result.json"]["sha256"],
        }
        _write_json_atomic(sealed_manifest, staging / "sealed_run_manifest.json")
        seal_hash = sha256_file(staging / "sealed_run_manifest.json")
        (staging / "SEALED.sha256").write_text(
            f"{seal_hash}  sealed_run_manifest.json\n", encoding="ascii"
        )
        # Catch a race or an accidental analysis-time edit before publishing.
        require_frozen_git(repo, expected_commit)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        **primary_result,
        "metric_validity_gate": fitted["metric_validity_gate"],
        "within_maike_nested_loao_secondary": maike_secondary_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--arthur-table", type=Path, required=True)
    parser.add_argument("--arthur-provenance", type=Path, required=True)
    parser.add_argument("--maike-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--execute-frozen-primary",
        action="store_true",
        help="Required fail-safe acknowledgement that this is the immutable first outcome run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_frozen_primary:
        print(
            "Refusing to inspect Maike outcomes without --execute-frozen-primary. "
            "Freeze and commit all code first.",
            file=sys.stderr,
        )
        return 2
    try:
        result = execute_frozen_primary(
            repo=args.repo,
            expected_commit=args.expected_commit,
            arthur_table=args.arthur_table,
            arthur_provenance=args.arthur_provenance,
            maike_root=args.maike_root,
            output_dir=args.output_dir,
        )
    except ContractError as exc:
        print(f"CONTRACT FAILURE: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

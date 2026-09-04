from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "experiment_63_primary_backend.py"
SPEC = importlib.util.spec_from_file_location("experiment_63_primary_backend", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backend
SPEC.loader.exec_module(backend)


def load_sibling_module(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e63_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_stage2_config() -> dict:
    return {
        "analysis_scope": backend.ANALYSIS_SCOPE,
        "isolation_basis": backend.ISOLATION_BASIS,
        "threshold_config": backend.EXPECTED_THRESHOLD_CONFIG,
        "predictor_pipeline_config": backend.EXPECTED_PREDICTOR_PIPELINE_CONFIG,
    }


def write_sealed(path: Path, index: int = 0, **overrides) -> None:
    points = np.array(
        [[z, y, x] for z in range(3) for y in range(3) for x in range(3)],
        dtype=np.int32,
    )
    config_text = backend.canonical_json(valid_stage2_config())
    arrays = {
        "schema_version": np.array(backend.SEALED_DISTAL_SCHEMA),
        "lens_index": np.array(index, dtype=np.int64),
        "points_zyx": points,
        "spacing_um": np.array([0.325, 0.325, 0.325], dtype=np.float64),
        "config_json": np.array(config_text),
        "config_sha256": np.array(backend.sha256_text(config_text)),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)


def synthetic_table(n: int = 25, *, eye_id: str = "eye") -> pd.DataFrame:
    side = int(math.ceil(math.sqrt(n)))
    positions = []
    for iy in range(side):
        for ix in range(side):
            positions.append(((ix - (side - 1) / 2) * 10.0, (iy - (side - 1) / 2) * 10.0))
    positions = np.asarray(positions[:n], dtype=float)
    nearest = np.sqrt(np.min(
        np.where(
            np.eye(n, dtype=bool),
            np.inf,
            np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2),
        ),
        axis=1,
    ))
    central = np.linalg.norm(positions, axis=1) <= 0.5 * np.median(nearest)
    rows = {
        "eye_id": [eye_id] * n,
        "lens_index": np.arange(n),
        "distal_qc": [True] * n,
        "target_resolvable": [True] * n,
        "target_qc": [True] * n,
        "central": central,
        "position_u_um": positions[:, 0],
        "position_v_um": positions[:, 1],
        "distal_scale_um": np.linspace(5.0, 7.0, n),
        "distal_gradient_magnitude": np.linspace(0.1, 0.9, n),
        "distal_curvature_eigenvalue_1": np.linspace(-0.2, 0.0, n),
        "distal_curvature_eigenvalue_2": np.linspace(0.1, 0.3, n),
        "distal_normalized_fit_residual": np.linspace(0.01, 0.09, n),
        "target_depth_um": np.linspace(10.0, 12.0, n),
        "target_q05_raw_thickness_um": np.linspace(9.0, 11.0, n),
        "target_support": [30] * n,
        "target_rmse_um": [0.5] * n,
    }
    for coefficient in range(6):
        rows[f"target_c{coefficient}"] = np.linspace(10.0 if coefficient == 0 else 0.0, 11.0 if coefficient == 0 else 0.2, n)
    return pd.DataFrame(rows)


def with_component_fields(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["seed_id"] = [f"seed-{index}" for index in range(len(result))]
    result["assignment_status"] = "ok"
    result["full_assigned_size"] = 100
    result["main_component_size"] = 99
    result["component_removed_size"] = 1
    result["main_component_fraction"] = 0.99
    result["partition_target_resolvable"] = True
    result["distal_eligible"] = result["distal_qc"]
    return result


def valid_fixed_point() -> dict:
    return {
        "converged": True,
        "max_iterations": 20,
        "iterations": 3,
        "eligible_counts": [100, 95, 94, 94],
        "readded_count": 0,
    }


def valid_frame_audit(eye_id: str = "eye") -> dict:
    metric = {
        "noncentral_poleward_u_angle_p95_deg": 4.9,
        "outward_axis_angle_p95_deg": 1.9,
        "central_classification_change_rate": 0.019,
        "n_numerical_fallback_outside_central": 0,
    }
    return {
        "schema_version": backend.FRAME_AUDIT_SCHEMA,
        "eye_id": eye_id,
        "gate_passed": True,
        "thresholds": backend.EXPECTED_FRAME_THRESHOLDS.copy(),
        "perturbations": {key: metric.copy() for key in backend.EXPECTED_PERTURBATIONS},
    }


def synthetic_maike_cohort(n_per_eye: int = 9) -> pd.DataFrame:
    pieces = []
    for ordinal, eye_id in enumerate(backend.EXPECTED_EYES):
        eye = synthetic_table(n_per_eye, eye_id=eye_id)
        lens_order = np.arange(n_per_eye, dtype=float)
        eye["position_u_um"] += 0.013 * (ordinal + 1) * lens_order
        eye["position_v_um"] += 0.007 * (ordinal + 1) * (lens_order % 4) ** 2
        positions = eye[["position_u_um", "position_v_um"]].to_numpy(float)
        pairwise = np.sqrt(
            np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(pairwise, np.inf)
        eye["central"] = np.linalg.norm(positions, axis=1) <= 0.5 * float(
            np.median(np.min(pairwise, axis=1))
        )
        eye["distal_scale_um"] += ordinal * 0.01
        eye["distal_gradient_magnitude"] += ordinal * 0.001
        pieces.append(eye)
    combined = pd.concat(pieces, ignore_index=True)
    return backend.add_invariant_features_by_unit(
        combined, ["eye_id"], "synthetic Maike cohort"
    )


def valid_maike_embedded_input_chain(
    eye_id: str | None = None,
) -> tuple[dict, dict]:
    eye_id = eye_id or next(iter(backend.EXPECTED_EYES))
    expected_rows = backend.EXPECTED_EYES[eye_id]
    archive = {
        "path": f"/frozen/{backend.MAIKE_SOURCE_ARCHIVES[eye_id]['name']}",
        **backend.MAIKE_SOURCE_ARCHIVES[eye_id],
    }
    mask_npy = {"sha256": "1" * 64, "size_bytes": 101}
    mask = {
        "schema_version": backend.MAIKE_MASK_PROVENANCE_SCHEMA,
        "eye_id": eye_id,
        "axis_order": "zyx",
        "spacing_um": [0.325, 0.325, 0.325],
        "original_spacing_um": [0.325, 0.325, 0.325],
        "array_sha256": mask_npy["sha256"],
        "npy_sha256": mask_npy["sha256"],
        "source_archive": archive,
        "archive_sha256": archive["sha256"],
        "source_slices": {
            "count": backend.MAIKE_SOURCE_STACK_GEOMETRY[eye_id]["slice_count"]
        },
        "uncropped": {
            "shape_zyx": backend.MAIKE_SOURCE_STACK_GEOMETRY[eye_id][
                "uncropped_shape_zyx"
            ],
            "image_shape_yx": backend.MAIKE_SOURCE_STACK_GEOMETRY[eye_id][
                "uncropped_shape_zyx"
            ][1:],
        },
        "output": {
            "format": "npy",
            "dtype": "uint8",
            "order": "C",
            "axis_order": "zyx",
            **mask_npy,
        },
    }
    mask_provenance = backend._embedded_pretty_json_binding(mask, "mask")
    seed_csv = {"sha256": "2" * 64, "size_bytes": 202}
    seed = {
        "schema_version": backend.MAIKE_SEED_PROVENANCE_SCHEMA,
        "eye_id": eye_id,
        "seed_role": backend.MAIKE_SEED_ROLE,
        "seed_csv_sha256": seed_csv["sha256"],
        "csv_sha256": seed_csv["sha256"],
        "n_expected": expected_rows,
        "n_rows": expected_rows,
        "n_foreground_hits": expected_rows,
        "lens_index_range": [0, expected_rows - 1],
        "candidate_seeds_per_voxel": 1,
        "input_axis_direction": "toward_eye_center",
        "output_axis_direction": "away_from_eye_center",
        "input_hashes": {
            "oda_csv": {"path": "/frozen/oda.csv", "sha256": "3" * 64, "size_bytes": 303},
            "transform_json": {
                "path": "/frozen/transform.json",
                "sha256": "4" * 64,
                "size_bytes": 404,
            },
            "mask_npy": {"path": "/frozen/mask.npy", **mask_npy},
            "mask_provenance": {
                "path": "/frozen/mask.json",
                **mask_provenance,
            },
            "source_archive": archive,
        },
        "output": {"path": "/frozen/seeds.csv", **seed_csv},
    }
    seed_provenance = backend._embedded_pretty_json_binding(seed, "seed")
    provenance = {
        "mask_source_provenance": mask,
        "seed_source_provenance": seed,
    }
    input_hashes = {
        "mask_npy": {"path": "/frozen/mask.npy", **mask_npy},
        "mask_provenance": {"path": "/frozen/mask.json", **mask_provenance},
        "seed_csv": {"path": "/frozen/seeds.csv", **seed_csv},
        "seed_provenance": {
            "path": "/frozen/seeds.json",
            **seed_provenance,
        },
    }
    return provenance, input_hashes


class FrozenConstantsTests(unittest.TestCase):
    def test_exact_eye_counts(self) -> None:
        self.assertEqual(len(backend.EXPECTED_EYES), 12)
        self.assertEqual(sum(backend.EXPECTED_EYES.values()), 11_369)
        self.assertEqual(backend.EXPECTED_TOTAL, 11_369)

    def test_scope_is_explicitly_conditional(self) -> None:
        self.assertEqual(
            backend.ANALYSIS_SCOPE,
            "conditional_on_oracle_distal_surface_localization",
        )

    def test_frozen_alphas(self) -> None:
        np.testing.assert_allclose(backend.RIDGE_ALPHAS, np.logspace(-2, 3, 12), rtol=0, atol=0)

    def test_distal_support_key_names_the_sealed_cap_semantics(self) -> None:
        self.assertEqual(
            backend.EXPECTED_THRESHOLD_CONFIG["min_sealed_distal_cap_points"], 25
        )
        self.assertNotIn("min_distal_fit_points", backend.EXPECTED_THRESHOLD_CONFIG)

    def test_canonical_grid_matches_experiment_57(self) -> None:
        self.assertEqual(backend.CANONICAL_GRID_XY.shape, (81, 2))
        self.assertAlmostEqual(float(np.max(np.linalg.norm(backend.CANONICAL_GRID_XY, axis=1))), 0.65)

    def test_fixed_reference_probability(self) -> None:
        self.assertEqual(backend.two_sided_fixed_denominator_p(10, 12), 0.03857421875)
        self.assertEqual(backend.two_sided_fixed_denominator_p(6, 12), 1.0)

    def test_tie_dropping_probability(self) -> None:
        self.assertEqual(backend.conventional_tie_dropping_sign_p(0, 0), 1.0)
        self.assertEqual(backend.conventional_tie_dropping_sign_p(10, 0), 2 / 1024)

    def test_actual_arthur_adapter_schema_matches_backend(self) -> None:
        adapter = load_sibling_module("prepare_arthur_source_table")
        self.assertEqual(adapter.SCHEMA_VERSION, backend.ARTHUR_SOURCE_SCHEMA)
        self.assertEqual(
            {"fixed_point_policy": "monotone_drop_only_no_reentry", **adapter.TARGET_CONFIG},
            backend.EXPECTED_ARTHUR_PIPELINE_CONFIG,
        )
        self.assertTrue(
            (backend.REQUIRED_TABLE_COLUMNS | {"volume"}).issubset(adapter.TABLE_FIELDS)
        )
        self.assertEqual(tuple(adapter.CODE_PATHS), backend.SOURCE_CODE_FILES)
        self.assertEqual(adapter.EXPECTED_INPUT_IDENTITIES, backend.ARTHUR_INPUT_FILES)

    def test_actual_maike_archive_contract_matches_backend(self) -> None:
        mask_producer = load_sibling_module("prepare_maike_masks")
        backend_contract = {
            eye_id: {
                **backend.MAIKE_SOURCE_ARCHIVES[eye_id],
                **backend.MAIKE_SOURCE_STACK_GEOMETRY[eye_id],
            }
            for eye_id in backend.EXPECTED_EYES
        }
        self.assertEqual(mask_producer.EXPECTED_SOURCE_ARCHIVES, backend_contract)

    def test_target_observation_operators_are_explicitly_not_symmetric(self) -> None:
        contract = backend.TARGET_OBSERVATION_CONTRACT
        self.assertTrue(contract["representation_shift"])
        self.assertNotEqual(
            contract["arthur_source_operator"], contract["maike_test_operator"]
        )
        self.assertIn("cannot", contract["interpretation_limit"])
        predictor = backend.PREDICTOR_INPUT_REPRESENTATION_CONTRACT
        self.assertTrue(predictor["representation_shift"])
        self.assertIn("irregular float64", predictor["arthur_source_distal_input"])
        self.assertIn("0.325-um", predictor["maike_test_distal_input"])
        self.assertIn("cannot_modify_rescue", backend.SECONDARY_MAIKE_LOAO_CONTRACT["decision_role"])

    def test_execution_environment_records_numerical_runtime(self) -> None:
        environment = backend.execution_environment()
        self.assertEqual(
            set(environment["packages"]), {"numpy", "pandas", "scipy", "Pillow"}
        )
        self.assertTrue(environment["python"])
        self.assertIn("numpy_build_dependencies", environment)


class SealedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "lens_000000.npz"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_exact_v2_artifact(self) -> None:
        write_sealed(self.path)
        config = backend.validate_sealed_distal_npz(self.path, 0)
        self.assertEqual(config["analysis_scope"], backend.ANALYSIS_SCOPE)

    def test_actual_extractor_payload_matches_backend_contract(self) -> None:
        extractor = load_sibling_module("extract_lens_surfaces")
        self.assertEqual(extractor.DEFAULT_THRESHOLD_CONFIG, backend.EXPECTED_THRESHOLD_CONFIG)
        self.assertEqual(extractor.DEFAULT_PIPELINE_CONFIG, backend.EXPECTED_PIPELINE_CONFIG)
        sealed_config = {
            "analysis_scope": extractor.ANALYSIS_SCOPE,
            "isolation_basis": extractor.ISOLATION_BASIS,
            "threshold_config": extractor.DEFAULT_THRESHOLD_CONFIG,
            "predictor_pipeline_config": {
                key: extractor.DEFAULT_PIPELINE_CONFIG[key]
                for key in backend.EXPECTED_PREDICTOR_PIPELINE_CONFIG
            },
        }
        points = np.array(
            [[z, y, x] for z in range(3) for y in range(3) for x in range(3)],
            dtype=np.int32,
        )
        payload = extractor._sealed_payload(
            0, points, [0.325, 0.325, 0.325], sealed_config
        )
        np.savez(self.path, **payload)
        backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_extra_outcome_array(self) -> None:
        write_sealed(self.path, target=np.array([1.0]))
        with self.assertRaisesRegex(backend.ContractError, "forbidden/missing"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_wrong_point_dtype(self) -> None:
        points = np.arange(81, dtype=np.int64).reshape(27, 3)
        write_sealed(self.path, points_zyx=points)
        with self.assertRaisesRegex(backend.ContractError, "int32"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_unsorted_points(self) -> None:
        points = np.array(
            [[z, y, x] for z in range(3) for y in range(3) for x in range(3)], dtype=np.int32
        )[::-1]
        write_sealed(self.path, points_zyx=points)
        with self.assertRaisesRegex(backend.ContractError, "not canonically"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_duplicate_points(self) -> None:
        points = np.zeros((25, 3), dtype=np.int32)
        write_sealed(self.path, points_zyx=points)
        with self.assertRaisesRegex(backend.ContractError, "duplicates"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_empty_sealed_row_allowed_only_outside_distal_qc(self) -> None:
        write_sealed(self.path, points_zyx=np.empty((0, 3), dtype=np.int32))
        backend.validate_sealed_distal_npz(
            self.path, 0, require_distal_qc_support=False
        )
        with self.assertRaisesRegex(backend.ContractError, "fewer than 25"):
            backend.validate_sealed_distal_npz(
                self.path, 0, require_distal_qc_support=True
            )

    def test_sealed_cap_support_gate_uses_all_artifact_points(self) -> None:
        points = np.array(
            [[z, y, x] for z in range(2) for y in range(3) for x in range(4)],
            dtype=np.int32,
        )
        write_sealed(self.path, points_zyx=points)
        with self.assertRaisesRegex(backend.ContractError, "fewer than 25"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_wrong_spacing(self) -> None:
        write_sealed(self.path, spacing_um=np.array([1.3, 1.3, 1.3], dtype=np.float64))
        with self.assertRaisesRegex(backend.ContractError, "0.325"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_config_hash_tamper(self) -> None:
        write_sealed(self.path, config_sha256=np.array("0" * 64))
        with self.assertRaisesRegex(backend.ContractError, "does not bind"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_noncanonical_config(self) -> None:
        text = json.dumps(valid_stage2_config(), indent=2)
        write_sealed(
            self.path,
            config_json=np.array(text),
            config_sha256=np.array(backend.sha256_text(text)),
        )
        with self.assertRaisesRegex(backend.ContractError, "not a canonical"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_target_language_in_config(self) -> None:
        config = valid_stage2_config()
        config["target_depth"] = 10
        text = backend.canonical_json(config)
        write_sealed(
            self.path,
            config_json=np.array(text),
            config_sha256=np.array(backend.sha256_text(text)),
        )
        with self.assertRaisesRegex(backend.ContractError, "target/oracle"):
            backend.validate_sealed_distal_npz(self.path, 0)

    def test_rejects_threshold_drift(self) -> None:
        config = valid_stage2_config()
        config["threshold_config"] = config["threshold_config"].copy()
        config["threshold_config"]["distal_fit_rmse_max_um"] = 2.6
        text = backend.canonical_json(config)
        write_sealed(
            self.path,
            config_json=np.array(text),
            config_sha256=np.array(backend.sha256_text(text)),
        )
        with self.assertRaisesRegex(backend.ContractError, "must equal"):
            backend.validate_sealed_distal_npz(self.path, 0)


class ArthurArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.row = synthetic_table().iloc[0].copy()
        self.row["volume"] = "20231107"
        self.row["eye_id"] = 0
        self.row["lens_index"] = 7
        self.sealed_path = self.root / "sealed.npz"
        self.target_path = self.root / "target.npz"
        points = np.array(
            [[x, y, z] for x in range(3) for y in range(3) for z in range(3)],
            dtype=np.float64,
        )
        config_text = backend.canonical_json(backend.EXPECTED_THRESHOLD_CONFIG)
        np.savez(
            self.sealed_path,
            schema_version=np.array(backend.ARTHUR_SEALED_DISTAL_SCHEMA),
            volume=np.array("20231107"),
            eye_id=np.array(0, dtype=np.int64),
            lens_index=np.array(7, dtype=np.int64),
            points_xyz_um=points,
            config_json=np.array(config_text),
            config_sha256=np.array(backend.sha256_text(config_text)),
        )
        raw_xy = backend.CANONICAL_GRID_XY[:30].copy()
        coefficients = self.row.loc[list(backend.TARGET_COLUMNS)].to_numpy(float)
        raw_thickness = backend._evaluate_coefficients_on_xy(coefficients, raw_xy)
        self.row["target_support"] = len(raw_xy)
        self.row["target_depth_um"] = float(np.median(raw_thickness))
        self.row["target_q05_raw_thickness_um"] = float(np.quantile(raw_thickness, 0.05))
        self.row["target_rmse_um"] = 0.0
        np.savez(
            self.target_path,
            schema_version=np.array(backend.ARTHUR_TARGET_SCHEMA),
            volume=np.array("20231107"),
            eye_id=np.array(0, dtype=np.int64),
            lens_index=np.array(7, dtype=np.int64),
            canonical_grid_xy=backend.CANONICAL_GRID_XY,
            target_smoothed_thickness_um=backend.CANONICAL_DESIGN @ coefficients,
            raw_target_xy_normalized=raw_xy,
            raw_target_thickness_um=raw_thickness,
            target_coefficients_c0_c5=coefficients,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_exact_source_artifacts(self) -> None:
        backend.validate_arthur_sealed_distal_npz(
            self.sealed_path,
            self.row,
            backend.sha256_text(backend.canonical_json(backend.EXPECTED_THRESHOLD_CONFIG)),
        )
        backend.validate_arthur_target_npz(self.target_path, self.row)

    def test_rejects_duplicate_source_vertices(self) -> None:
        with np.load(self.sealed_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["points_xyz_um"] = np.vstack(
            [arrays["points_xyz_um"], arrays["points_xyz_um"][-1]]
        )
        np.savez(self.sealed_path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "duplicate vertices"):
            backend.validate_arthur_sealed_distal_npz(
                self.sealed_path,
                self.row,
                backend.sha256_text(backend.canonical_json(backend.EXPECTED_THRESHOLD_CONFIG)),
            )

    def test_rejects_source_target_table_tamper(self) -> None:
        altered = self.row.copy()
        altered["target_c0"] += 1.0
        with self.assertRaisesRegex(backend.ContractError, "differ from the source table"):
            backend.validate_arthur_target_npz(self.target_path, altered)

    def test_rejects_repeated_source_target_observation(self) -> None:
        with np.load(self.target_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["raw_target_xy_normalized"][-1] = arrays[
            "raw_target_xy_normalized"
        ][-2]
        arrays["raw_target_thickness_um"][-1] = arrays[
            "raw_target_thickness_um"
        ][-2]
        np.savez(self.target_path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "repeats a proximal"):
            backend.validate_arthur_target_npz(self.target_path, self.row)

    def test_stage1_diagnostics_require_exact_frozen_accounting(self) -> None:
        diagnostics = {}
        for volume in backend.ARTHUR_VOLUMES:
            n_landmarks = backend.ARTHUR_LANDMARK_COUNTS[volume]
            n_raw = backend.ARTHUR_STAGE1_COUNTS[volume]
            diagnostics[volume] = {
                "n_landmarks": n_landmarks,
                "n_eye_0": n_landmarks // 2,
                "n_eye_1": n_landmarks - n_landmarks // 2,
                "median_lens_tip_distance_um": 1.0,
                "median_landmark_mesh_distance_um": 1.0,
                "n_raw_patches": n_raw,
                "stage1_failure_counts": {
                    "patch_support": n_landmarks - n_raw,
                    "distal_layer_support": 0,
                },
            }
        backend.validate_arthur_stage1_diagnostics(diagnostics)
        diagnostics["20231107"]["n_raw_patches"] -= 1
        with self.assertRaisesRegex(backend.ContractError, "frozen landmark/Stage-1"):
            backend.validate_arthur_stage1_diagnostics(diagnostics)


class HashContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.file = self.root / "artifact.bin"
        self.file.write_bytes(b"abc")
        self.binding = {
            "sha256": hashlib.sha256(b"abc").hexdigest(),
            "size_bytes": 3,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_hash_binding(self) -> None:
        self.assertEqual(
            backend.validate_file_binding(self.root, "artifact.bin", self.binding, "test"),
            self.file,
        )

    def test_rejects_tamper(self) -> None:
        self.file.write_bytes(b"abd")
        with self.assertRaisesRegex(backend.ContractError, "SHA-256 mismatch"):
            backend.validate_file_binding(self.root, "artifact.bin", self.binding, "test")

    def test_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(backend.ContractError, "Unsafe"):
            backend.validate_file_binding(self.root, "../artifact.bin", self.binding, "test")

    def test_rejects_symlink(self) -> None:
        link = self.root / "link.bin"
        link.symlink_to(self.file)
        with self.assertRaisesRegex(backend.ContractError, "Symlink"):
            backend.validate_file_binding(self.root, "link.bin", self.binding, "test")


class MaikeInputProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eye_id = next(iter(backend.EXPECTED_EYES))
        self.provenance, self.input_hashes = valid_maike_embedded_input_chain(
            self.eye_id
        )

    def validate(self) -> None:
        backend.validate_maike_input_provenance(
            self.provenance,
            self.input_hashes,
            self.eye_id,
            backend.EXPECTED_EYES[self.eye_id],
            "fixture",
        )

    def test_accepts_exact_embedded_chain(self) -> None:
        self.validate()

    def test_rejects_extra_input_inventory_key(self) -> None:
        self.input_hashes["unreviewed_target"] = {
            "path": "/bad",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
        with self.assertRaisesRegex(backend.ContractError, "exactly"):
            self.validate()

    def test_rejects_self_consistent_replacement_source_archive(self) -> None:
        replacement = dict(self.provenance["mask_source_provenance"]["source_archive"])
        replacement["sha256"] = "f" * 64
        self.provenance["mask_source_provenance"]["source_archive"] = replacement
        self.provenance["mask_source_provenance"]["archive_sha256"] = replacement["sha256"]
        self.provenance["seed_source_provenance"]["input_hashes"]["source_archive"] = replacement
        self.input_hashes["mask_provenance"].update(
            backend._embedded_pretty_json_binding(
                self.provenance["mask_source_provenance"], "mask"
            )
        )
        self.provenance["seed_source_provenance"]["input_hashes"][
            "mask_provenance"
        ].update(self.input_hashes["mask_provenance"])
        self.input_hashes["seed_provenance"].update(
            backend._embedded_pretty_json_binding(
                self.provenance["seed_source_provenance"], "seed"
            )
        )
        with self.assertRaisesRegex(backend.ContractError, "frozen TIFF ZIP"):
            self.validate()

    def test_rejects_seed_role_tamper_even_when_rehashed(self) -> None:
        self.provenance["seed_source_provenance"]["seed_role"] = "target_informed"
        self.input_hashes["seed_provenance"].update(
            backend._embedded_pretty_json_binding(
                self.provenance["seed_source_provenance"], "seed"
            )
        )
        with self.assertRaisesRegex(backend.ContractError, "seed_role"):
            self.validate()

    def test_rejects_embedded_mask_json_that_does_not_match_bound_bytes(self) -> None:
        self.provenance["mask_source_provenance"]["created_utc"] = "tampered"
        with self.assertRaisesRegex(backend.ContractError, "bound JSON bytes"):
            self.validate()

    def test_validates_exact_maike_producer_code_inventory(self) -> None:
        repo = MODULE_PATH.parents[2]
        bindings = {}
        for basename, relpath in backend.MAIKE_PRODUCER_CODE_FILES.items():
            path = repo / relpath
            bindings[basename] = {
                "sha256": backend.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        backend.validate_maike_producer_hashes(repo, bindings, "producer")
        bindings["extract_lens_surfaces.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(backend.ContractError, "hash mismatch|SHA-256 mismatch"):
            backend.validate_maike_producer_hashes(repo, bindings, "producer")


class GeometryAndTableTests(unittest.TestCase):
    def test_valid_table_and_invariants(self) -> None:
        table = backend.validate_lens_table(synthetic_table(), label="synthetic", expected_eye="eye", expected_rows=25)
        result = backend.invariant_position_features(table, "synthetic")
        self.assertTrue(np.isfinite(result.loc[:, backend.POSITION_FEATURE_NAMES].to_numpy()).all())

    def test_invariants_survive_rotation_and_reflection(self) -> None:
        original = backend.invariant_position_features(
            backend.validate_lens_table(synthetic_table(), label="a"), "a"
        )
        transformed = synthetic_table()
        xy = transformed[["position_u_um", "position_v_um"]].to_numpy()
        theta = 0.73
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = xy @ rotation.T
        xy[:, 1] *= -1
        transformed[["position_u_um", "position_v_um"]] = xy
        result = backend.invariant_position_features(
            backend.validate_lens_table(transformed, label="b"), "b"
        )
        np.testing.assert_allclose(
            original.loc[:, backend.POSITION_FEATURE_NAMES],
            result.loc[:, backend.POSITION_FEATURE_NAMES],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rejects_truncated_index_range(self) -> None:
        table = synthetic_table(25)
        table.loc[24, "lens_index"] = 30
        with self.assertRaisesRegex(backend.ContractError, "complete 0..24"):
            backend.validate_lens_table(table, label="bad", expected_rows=25)

    def test_rejects_ambiguous_boolean(self) -> None:
        table = synthetic_table()
        table["distal_qc"] = table["distal_qc"].astype(object)
        table.loc[0, "distal_qc"] = "yes"
        with self.assertRaisesRegex(backend.ContractError, "strict boolean"):
            backend.validate_lens_table(table, label="bad")

    def test_rejects_predictor_nan(self) -> None:
        table = synthetic_table()
        table.loc[0, "distal_scale_um"] = np.nan
        with self.assertRaisesRegex(backend.ContractError, "nonfinite predictor"):
            backend.validate_lens_table(table, label="bad")

    def test_rejects_reversed_curvature_eigenvalues(self) -> None:
        table = synthetic_table()
        table.loc[0, "distal_curvature_eigenvalue_1"] = 1.0
        with self.assertRaisesRegex(backend.ContractError, "not ordered"):
            backend.validate_lens_table(table, label="bad")

    def test_target_qc_cannot_rescue_unresolvable_target(self) -> None:
        table = synthetic_table()
        table.loc[0, "target_resolvable"] = False
        with self.assertRaisesRegex(backend.ContractError, "target_qc true"):
            backend.validate_lens_table(table, label="bad")

    def test_target_qc_is_exact_raw_q05_and_rmse_sensitivity(self) -> None:
        table = synthetic_table()
        table.loc[0, "target_q05_raw_thickness_um"] = -0.01
        with self.assertRaisesRegex(backend.ContractError, "target_qc is not exactly"):
            backend.validate_lens_table(table, label="bad")
        table.loc[0, "target_qc"] = False
        backend.validate_lens_table(table, label="good")

    def test_target_resolvable_requires_distal_frame(self) -> None:
        table = synthetic_table()
        table.loc[0, "distal_qc"] = False
        table.loc[0, "target_qc"] = False
        with self.assertRaisesRegex(backend.ContractError, "target_resolvable true outside"):
            backend.validate_lens_table(table, label="bad")

    def test_nonpositive_depth_does_not_redefine_structural_resolvability(self) -> None:
        table = synthetic_table()
        table.loc[0, "target_depth_um"] = -0.1
        table.loc[0, "target_q05_raw_thickness_um"] = -1.0
        table.loc[0, "target_qc"] = False
        validated = backend.validate_lens_table(table, label="structural")
        self.assertTrue(bool(validated.loc[0, "target_resolvable"]))
        with self.assertRaisesRegex(backend.ContractError, "whole-run metric-validity gate failed"):
            backend.validate_metric_denominator_gate(validated, "primary")

    def test_metric_denominator_gate_reports_full_primary_count(self) -> None:
        table = backend.validate_lens_table(synthetic_table(), label="valid")
        result = backend.validate_metric_denominator_gate(table, "primary")
        self.assertEqual(result["n_primary_rows"], len(table))
        self.assertEqual(result["n_finite_positive_target_depth"], len(table))
        self.assertEqual(result["n_invalid"], 0)
        self.assertTrue(result["passed"])

    def test_central_flag_is_recomputed(self) -> None:
        table = backend.validate_lens_table(synthetic_table(), label="bad")
        table.loc[12, "central"] = False
        with self.assertRaisesRegex(backend.ContractError, "central flags"):
            backend.invariant_position_features(table, "bad")

    def test_position_feature_reference_excludes_failed_lens(self) -> None:
        table = synthetic_table()
        table.loc[0, "distal_qc"] = False
        table.loc[0, "target_resolvable"] = False
        table.loc[0, "target_qc"] = False
        table.loc[0, ["position_u_um", "position_v_um"]] = [999999, 999999]
        validated = backend.validate_lens_table(table, label="test")
        result = backend.invariant_position_features(validated, "test")
        self.assertTrue(result.loc[0, list(backend.POSITION_FEATURE_NAMES)].isna().all())
        self.assertLess(result.loc[1:, "position_pairwise_q90_um"].max(), 100)

    def test_component_gate_accepts_exact_ninety_nine_percent(self) -> None:
        table = backend.validate_lens_table(synthetic_table(), label="test")
        backend.validate_maike_component_table(with_component_fields(table), "test")

    def test_component_gate_cannot_be_used_to_rescue_target(self) -> None:
        table = with_component_fields(
            backend.validate_lens_table(synthetic_table(), label="test")
        )
        table.loc[0, "main_component_size"] = 98
        table.loc[0, "component_removed_size"] = 2
        table.loc[0, "main_component_fraction"] = 0.98
        table.loc[0, "partition_target_resolvable"] = False
        with self.assertRaisesRegex(backend.ContractError, "resolves a target"):
            backend.validate_maike_component_table(table, "test")

    def test_component_gate_does_not_control_distal_eligibility(self) -> None:
        table = with_component_fields(
            backend.validate_lens_table(synthetic_table(), label="test")
        )
        table.loc[0, "main_component_size"] = 98
        table.loc[0, "component_removed_size"] = 2
        table.loc[0, "main_component_fraction"] = 0.98
        table.loc[0, "partition_target_resolvable"] = False
        table.loc[0, "target_resolvable"] = False
        table.loc[0, "target_qc"] = False
        # Distal-QC remains true: the component criterion is target-only.
        result = backend.validate_maike_component_table(table, "test")
        self.assertTrue(bool(result.loc[0, "distal_qc"]))


class TargetAndModelTests(unittest.TestCase):
    def _source_and_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        source_pieces = []
        for number, volume in enumerate(backend.ARTHUR_VOLUMES):
            table = synthetic_table(25, eye_id=f"{volume}_eye")
            table[["position_u_um", "position_v_um"]] *= 1.0 + number * 0.04
            table["distal_scale_um"] += number * 0.13
            table["volume"] = volume
            source_pieces.append(
                backend.invariant_position_features(
                    backend.validate_lens_table(table, label=volume, require_volume=True),
                    volume,
                )
            )
        test_pieces = []
        for number, eye_id in enumerate(backend.EXPECTED_EYES):
            table = synthetic_table(25, eye_id=eye_id)
            table[["position_u_um", "position_v_um"]] *= 1.0 + number * 0.005
            table["distal_scale_um"] += number * 0.01
            test_pieces.append(
                backend.invariant_position_features(
                    backend.validate_lens_table(table, label=eye_id), eye_id
                )
            )
        return pd.concat(source_pieces, ignore_index=True), pd.concat(test_pieces, ignore_index=True)

    def test_central_symmetrization(self) -> None:
        table = synthetic_table()
        central_index = int(np.flatnonzero(table["central"])[0])
        table.loc[central_index, list(backend.TARGET_COLUMNS)] = [10, 1, 2, 3, 4, 5]
        targets = backend.symmetrized_targets(table)
        self.assertEqual(targets[central_index].tolist(), [10, 0, 0, 4, 0, 4])

    def test_noncentral_target_keeps_reflection_odd_terms(self) -> None:
        table = synthetic_table()
        index = int(np.flatnonzero(~table["central"])[0])
        table.loc[index, list(backend.TARGET_COLUMNS)] = [10, 1, 2, 3, 4, 5]
        targets = backend.symmetrized_targets(table)
        self.assertEqual(targets[index].tolist(), [10, 1, 2, 3, 4, 5])

    def test_model_zeroes_reflection_odd_predictions(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(30, 4))
        target = rng.normal(size=(30, 6))
        groups = np.repeat(["a", "b", "c"], 10)
        model = backend.fit_weighted_ridge(x, target, groups, 1.0)
        prediction = model.predict_even(x)
        np.testing.assert_array_equal(prediction[:, [2, 4]], 0.0)

    def test_equal_volume_weighting_resists_within_volume_duplication(self) -> None:
        groups = np.array(["a", "b", "c"])
        weights = backend.equal_group_weights(groups)
        self.assertTrue(np.allclose(weights, 1 / 3))
        duplicated = np.array(["a"] * 100 + ["b"] * 2 + ["c"])
        weights = backend.equal_group_weights(duplicated)
        totals = [weights[duplicated == group].sum() for group in ("a", "b", "c")]
        np.testing.assert_allclose(totals, [1 / 3] * 3)

    def test_model_is_invariant_to_exact_within_volume_duplication(self) -> None:
        x = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 2.0], [4.0, 0.0], [5.0, 4.0], [6.0, 5.0]])
        y = np.column_stack([x[:, 0] + i * x[:, 1] for i in range(6)])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        first = backend.fit_weighted_ridge(x, y, groups, 0.1)
        take = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 5])
        second = backend.fit_weighted_ridge(x[take], y[take], groups[take], 0.1)
        np.testing.assert_allclose(first.predict_even(x), second.predict_even(x), atol=1e-12)

    def test_full_target_penalty_includes_unpredicted_terms(self) -> None:
        table = synthetic_table(25)
        table["central"] = False
        table["target_c0"] = 10.0
        table["target_c1"] = 0.0
        table["target_c2"] = 4.0
        table["target_c3"] = 0.0
        table["target_c4"] = 2.0
        table["target_c5"] = 0.0
        prediction = np.zeros((25, 6))
        prediction[:, 0] = 10.0
        error = backend.normalized_grid_mae(prediction, table)
        self.assertTrue(np.all(error > 0))

    def test_weighted_ridge_rejects_zero_variance_feature(self) -> None:
        x = np.ones((6, 2))
        y = np.ones((6, 6))
        with self.assertRaisesRegex(backend.ContractError, "zero weighted variance"):
            backend.fit_weighted_ridge(x, y, ["a", "a", "b", "b", "c", "c"], 1.0)

    def test_alpha_selection_uses_all_three_holdouts(self) -> None:
        pieces = []
        for number, volume in enumerate(backend.ARTHUR_VOLUMES):
            table = synthetic_table(25, eye_id=f"{volume}_eye")
            table[["position_u_um", "position_v_um"]] *= 1.0 + number * 0.05
            table["volume"] = volume
            table = backend.invariant_position_features(
                backend.validate_lens_table(table, label=volume, require_volume=True), volume
            )
            # Make all model features finite and nonconstant across the pooled set.
            table["distal_scale_um"] += number * 0.1
            pieces.append(table)
        source = pd.concat(pieces, ignore_index=True)
        alpha, audit = backend.select_alpha_leave_one_volume_out(source, backend.CONTROL_FEATURE_NAMES)
        self.assertIn(alpha, backend.RIDGE_ALPHAS)
        self.assertEqual(len(audit), len(backend.RIDGE_ALPHAS) * 4)
        self.assertEqual(set(audit["heldout_volume"]), {*backend.ARTHUR_VOLUMES, "equal_volume_mean"})

    def test_maike_targets_do_not_affect_primary_predictions(self) -> None:
        source, maike = self._source_and_test()
        first = backend.run_primary_models(source, maike)["predictions"]
        altered = maike.copy()
        altered.loc[:, list(backend.TARGET_COLUMNS)] += 1_000_000.0
        second = backend.run_primary_models(source, altered)["predictions"]
        for method in first:
            np.testing.assert_array_equal(first[method], second[method])

    def test_target_qc_flags_cannot_change_primary_cohort_or_predictions(self) -> None:
        source, maike = self._source_and_test()
        first = backend.run_primary_models(source, maike)
        altered = maike.copy()
        changed = altered.groupby("eye_id").head(3).index
        altered.loc[changed, "target_qc"] = False
        altered.loc[changed, "target_q05_raw_thickness_um"] = -0.1
        second = backend.run_primary_models(source, altered)
        self.assertEqual(
            list(zip(first["test_primary"].eye_id, first["test_primary"].lens_index)),
            list(zip(second["test_primary"].eye_id, second["test_primary"].lens_index)),
        )
        for method in first["predictions"]:
            np.testing.assert_array_equal(first["predictions"][method], second["predictions"][method])

    def test_target_qc_sensitivity_refits_on_strict_source_cohort(self) -> None:
        source, maike = self._source_and_test()
        source_drop = source.index[::7]
        test_drop = maike.groupby("eye_id").head(1).index
        source.loc[source_drop, "target_qc"] = False
        source.loc[source_drop, "target_q05_raw_thickness_um"] = -0.1
        maike.loc[test_drop, "target_qc"] = False
        maike.loc[test_drop, "target_q05_raw_thickness_um"] = -0.1
        result = backend.run_target_qc_sensitivity_models(source, maike)
        self.assertLess(len(result["source"]), len(source))
        self.assertLess(len(result["test"]), len(maike))
        self.assertEqual(set(result["predictions"]), {
            "position_scale_control", "position_scale_plus_distal_shape"
        })

    def test_nested_maike_alpha_selection_uses_only_other_eleven_animals(self) -> None:
        cohort = synthetic_maike_cohort()
        outer = next(iter(backend.EXPECTED_EYES))
        training = cohort[cohort["eye_id"] != outer]
        alpha, audit = backend.select_alpha_nested_maike_loao(
            training,
            backend.CONTROL_FEATURE_NAMES,
            outer_heldout_animal=outer,
        )
        self.assertIn(alpha, backend.RIDGE_ALPHAS)
        self.assertEqual(len(audit), len(backend.RIDGE_ALPHAS) * 12)
        self.assertEqual(set(audit["outer_heldout_animal"]), {outer})
        self.assertNotIn(outer, set(audit["inner_heldout_animal"]))
        self.assertEqual(
            set(audit["inner_heldout_animal"]),
            (set(backend.EXPECTED_EYES) - {outer}) | {"equal_animal_mean"},
        )
        with self.assertRaisesRegex(backend.ContractError, "other eleven"):
            backend.select_alpha_nested_maike_loao(
                cohort,
                backend.CONTROL_FEATURE_NAMES,
                outer_heldout_animal=outer,
            )

    def test_nested_maike_outer_prediction_is_blind_to_outer_targets(self) -> None:
        cohort = synthetic_maike_cohort()
        outer = next(iter(backend.EXPECTED_EYES))

        def frozen_selector(training, feature_names, *, outer_heldout_animal):
            return 1.0, pd.DataFrame(
                [
                    {
                        "model": "shape"
                        if tuple(feature_names) == backend.SHAPE_FEATURE_NAMES
                        else "control",
                        "outer_heldout_animal": outer_heldout_animal,
                        "inner_heldout_animal": "unit_test_frozen",
                        "alpha": 1.0,
                        "inner_heldout_median_81pt_normalized_mae": 0.0,
                    }
                ]
            )

        with mock.patch.object(
            backend, "select_alpha_nested_maike_loao", side_effect=frozen_selector
        ):
            first = backend.run_within_maike_nested_loao_secondary(cohort)
            altered = cohort.copy()
            altered.loc[
                altered["eye_id"] == outer, list(backend.TARGET_COLUMNS)
            ] += 1_000_000.0
            second = backend.run_within_maike_nested_loao_secondary(altered)
        outer_rows = cohort["eye_id"].astype(str).to_numpy() == outer
        for method in first["predictions"]:
            np.testing.assert_array_equal(
                first["predictions"][method][outer_rows],
                second["predictions"][method][outer_rows],
            )

    def test_secondary_summary_has_no_confirmatory_rescue_rule(self) -> None:
        rows = []
        for eye_id in backend.EXPECTED_EYES:
            for method, value in (
                ("position_scale_control", 1.0),
                ("position_scale_plus_distal_shape", 0.9),
            ):
                rows.append(
                    {
                        "eye_id": eye_id,
                        "lens_index": 0,
                        "method": method,
                        "cohort_primary": True,
                        "smoothed_81pt_mae_um": value * 10.0,
                        "smoothed_81pt_normalized_mae": value,
                    }
                )
        comparison, result = backend.aggregate_within_maike_secondary(
            pd.DataFrame(rows)
        )
        self.assertEqual(len(comparison), 12)
        self.assertIsNone(result["confirmatory_pass_fail_rule"])
        self.assertNotIn("passes_frozen_primary_rule", result)
        self.assertIn("cannot_modify_rescue", result["decision_role"])


class GateTests(unittest.TestCase):
    def test_resolution_gate_uses_fixed_expected_denominator(self) -> None:
        eighty = synthetic_table(80)
        self.assertEqual(backend.require_primary_resolution_gate(eighty, 100, "eye"), 80)
        seventy_nine = synthetic_table(79)
        with self.assertRaisesRegex(backend.ContractError, "80% gate"):
            backend.require_primary_resolution_gate(seventy_nine, 100, "eye")

    def test_valid_fixed_point(self) -> None:
        backend.validate_fixed_point(valid_fixed_point(), "fp")

    def test_fixed_point_rejects_readdition(self) -> None:
        value = valid_fixed_point()
        value["eligible_counts"] = [100, 95, 96, 96]
        with self.assertRaisesRegex(backend.ContractError, "not monotone"):
            backend.validate_fixed_point(value, "fp")

    def test_fixed_point_rejects_unstable_terminal(self) -> None:
        value = valid_fixed_point()
        value["eligible_counts"] = [100, 95, 94, 93]
        with self.assertRaisesRegex(backend.ContractError, "stable terminal"):
            backend.validate_fixed_point(value, "fp")

    def test_valid_frame_audit(self) -> None:
        backend.validate_frame_audit(valid_frame_audit(), "eye", "audit")

    def test_valid_partition_evidence(self) -> None:
        backend.validate_partition_evidence(
            {
                "source_foreground_voxel_count": 100,
                "assigned_voxel_count": 100,
                "assigned_unique_voxel_count": 100,
                "unassigned_foreground_voxel_count": 0,
                "multiply_assigned_voxel_count": 0,
                "exact_partition": True,
                "candidate_seeds_per_voxel": 1,
            },
            "partition",
        )

    def test_partition_rejects_unassigned_foreground(self) -> None:
        value = {
            "source_foreground_voxel_count": 100,
            "assigned_voxel_count": 99,
            "assigned_unique_voxel_count": 99,
            "unassigned_foreground_voxel_count": 1,
            "multiply_assigned_voxel_count": 0,
            "exact_partition": True,
            "candidate_seeds_per_voxel": 1,
        }
        with self.assertRaisesRegex(backend.ContractError, "unassigned"):
            backend.validate_partition_evidence(value, "partition")

    def test_frame_audit_recomputes_gate(self) -> None:
        audit = valid_frame_audit()
        audit["perturbations"]["ninety_percent_subsample"]["outward_axis_angle_p95_deg"] = 2.0001
        with self.assertRaisesRegex(backend.ContractError, "exceeds"):
            backend.validate_frame_audit(audit, "eye", "audit")

    def test_frame_audit_rejects_missing_perturbation(self) -> None:
        audit = valid_frame_audit()
        audit["perturbations"].pop("exhaustive_leave_one_origin_out")
        with self.assertRaisesRegex(backend.ContractError, "differ"):
            backend.validate_frame_audit(audit, "eye", "audit")

    def test_eye_comparison_treats_tie_as_nonwin(self) -> None:
        rows = []
        for eye_index, eye_id in enumerate(backend.EXPECTED_EYES):
            control = 1.0
            shape = 0.9 if eye_index < 10 else (1.0 if eye_index == 10 else 1.1)
            for method, value in (
                ("position_scale_control", control),
                ("position_scale_plus_distal_shape", shape),
            ):
                rows.append(
                    {
                        "eye_id": eye_id,
                        "lens_index": 0,
                        "method": method,
                        "smoothed_81pt_mae_um": value * 10.0,
                        "smoothed_81pt_normalized_mae": value,
                        "cohort_primary": True,
                    }
                )
        comparison, result = backend._aggregate_eye_comparison(
            pd.DataFrame(rows), "cohort_primary"
        )
        self.assertEqual((result["wins"], result["losses"], result["ties_nonwins"]), (10, 1, 1))
        self.assertTrue(result["passes_frozen_primary_rule"])
        self.assertEqual(set(comparison["species"]), {
            "Drosophila simulans", "Drosophila mauritiana"
        })
        self.assertEqual(set(comparison["sex"]), {"female", "male"})
        self.assertIn("control_median_smoothed_81pt_mae_um", comparison)
        descriptive = backend.species_sex_descriptive_table(comparison)
        self.assertEqual(len(descriptive), 8)
        self.assertEqual(
            set(descriptive["stratum_type"]), {"species", "sex", "species_x_sex"}
        )
        cells = descriptive[descriptive["stratum_type"] == "species_x_sex"]
        self.assertTrue((cells["n_animals"] == 3).all())
        self.assertTrue(
            all(len(json.loads(value)) == 3 for value in cells["named_eye_control_minus_shape_json"])
        )
        self.assertFalse(
            any(
                "p_value" in column.lower() or "pass_rule" in column.lower()
                for column in descriptive
            )
        )

    def test_internal_equal_volume_template_is_reported_for_every_eye(self) -> None:
        rows = []
        for eye_id in backend.EXPECTED_EYES:
            for method, value in (
                ("position_scale_control", 1.0),
                ("position_scale_plus_distal_shape", 0.9),
                ("equal_volume_source_template", 1.1),
            ):
                rows.append(
                    {
                        "eye_id": eye_id,
                        "method": method,
                        "cohort_primary": True,
                        "smoothed_81pt_mae_um": value * 10,
                        "smoothed_81pt_normalized_mae": value,
                        "raw_unsmoothed_available": True,
                        "raw_unsmoothed_mae_um": value * 11,
                        "raw_unsmoothed_normalized_mae": value * 1.1,
                    }
                )
        per_eye = backend.per_eye_method_descriptive_table(
            pd.DataFrame(rows), "cohort_primary"
        )
        self.assertEqual(len(per_eye), 36)
        summary = backend.summarize_internal_methods(per_eye)
        self.assertEqual(
            set(summary["methods"]),
            {
                "position_scale_control",
                "position_scale_plus_distal_shape",
                "equal_volume_source_template",
            },
        )
        self.assertEqual(summary["inference_role"], "does_not_modify_the_nested_primary_pass_rule")


class AttestationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"
        self.repo = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        (self.root / "instance_qc_visual_sample").mkdir()
        (self.repo / "experiments/maike-modern-ground-truth").mkdir(parents=True)
        self.paths = {
            "completion": self.root / "completion.json",
            "provenance": self.root / "provenance.json",
            "lens_summary": self.root / "lens_summary.csv",
            "distal_qc_sampling": self.root / "distal_qc_sampling.csv",
            "sample_manifest": self.root / "instance_qc_visual_sample/sample_manifest.json",
            "review_json": self.root / "instance_qc_review.json",
            "renderer_code": self.repo / "experiments/maike-modern-ground-truth/render_instance_qc_sample.py",
            "attester_code": self.repo / "experiments/maike-modern-ground-truth/attest_instance_qc.py",
        }
        for key, path in self.paths.items():
            path.write_text(f"{key}\n", encoding="utf-8")
        render_root = self.root / "instance_qc_visual_sample/renders"
        render_root.mkdir()
        samples = []
        decisions = []
        attested_reviewed = []
        ordinal = 0
        for radial in range(4):
            for scale in range(4):
                for repeat in range(2):
                    render_path = render_root / f"sample_{ordinal:02d}.png"
                    render_path.write_bytes(f"render {ordinal}".encode())
                    render_binding = {
                        "relative_path": f"renders/{render_path.name}",
                        "sha256": backend.sha256_file(render_path),
                        "size_bytes": render_path.stat().st_size,
                    }
                    seed_id = f"seed-{ordinal}"
                    samples.append(
                        {
                            "ordinal": ordinal,
                            "lens_index": ordinal,
                            "seed_id": seed_id,
                            "radial_stratum": radial,
                            "scale_stratum": scale,
                            "render": render_binding,
                        }
                    )
                    decisions.append(
                        {
                            "lens_index": ordinal,
                            "seed_id": seed_id,
                            "decision": "pass",
                            "notes": "",
                        }
                    )
                    attested_reviewed.append(
                        {
                            "ordinal": ordinal,
                            "lens_index": ordinal,
                            "seed_id": seed_id,
                            "radial_stratum": radial,
                            "scale_stratum": scale,
                            "decision": "pass",
                            "render_sha256": render_binding["sha256"],
                        }
                    )
                    ordinal += 1
        coverage = {f"r{r}_s{s}": 2 for r in range(4) for s in range(4)}
        sample_object = {
            "schema_version": "experiment63.instance-qc-sample.v1",
            "eye_id": "eye",
            "review_scope": "stratified_sample_only",
            "all_instances_manually_reviewed": False,
            "n_selected": 32,
            "cell_counts": coverage,
            "outcome_blindness": {
                "sampling_table_exact_field_allowlist": [
                    "eye_id", "lens_index", "seed_id", "distal_eligible",
                    "position_u_um", "position_v_um", "distal_scale_um",
                    "instance_relpath", "sealed_distal_relpath",
                ],
                "fitted_lens_npz_opened": False,
                "proximal_target_prediction_error_or_model_data_opened": False,
            },
            "samples": samples,
        }
        self.paths["sample_manifest"].write_text(json.dumps(sample_object), encoding="utf-8")
        review_object = {
            "schema_version": "experiment63.instance-qc-review.v1",
            "eye_id": "eye",
            "review_scope": "stratified_sample_only",
            "review_mode": "human",
            "reviewer_id": "reviewer",
            "reviewed_at_utc": "2026-09-04T00:00:00Z",
            "sample_manifest_sha256": backend.sha256_file(self.paths["sample_manifest"]),
            "decisions": decisions,
        }
        self.paths["review_json"].write_text(json.dumps(review_object), encoding="utf-8")
        self.partition = {
            "source_foreground_voxel_count": 100,
            "assigned_voxel_count": 100,
            "assigned_unique_voxel_count": 100,
            "unassigned_foreground_voxel_count": 0,
            "multiply_assigned_voxel_count": 0,
            "exact_partition": True,
            "candidate_seeds_per_voxel": 1,
        }
        relative = {
            "completion": "completion.json",
            "provenance": "provenance.json",
            "lens_summary": "lens_summary.csv",
            "distal_qc_sampling": "distal_qc_sampling.csv",
            "sample_manifest": "instance_qc_visual_sample/sample_manifest.json",
            "review_json": "instance_qc_review.json",
            "renderer_code": "experiments/maike-modern-ground-truth/render_instance_qc_sample.py",
            "attester_code": "experiments/maike-modern-ground-truth/attest_instance_qc.py",
        }
        bindings = {}
        for key, path in self.paths.items():
            bindings[key] = {
                "relative_path": relative[key],
                "sha256": backend.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        self.attestation = {
            "schema_version": backend.ATTESTATION_SCHEMA,
            "eye_id": "eye",
            "review_scope": "stratified_sample_only",
            "review_mode": "human",
            "all_instances_manually_reviewed": False,
            "stratified_sample_visual_qc_passed": True,
            "n_reviewed": 32,
            "technical_inventory_complete": True,
            "artifact_hash_verification_passed": True,
            "partition_exact": True,
            "partition_evidence": self.partition,
            "bindings": bindings,
            "technical_inventory": {
                "complete": True,
                "n_expected": 32,
                "n_inventory_rows": 32,
                "lens_indices_complete": True,
                "seed_ids_unique": True,
                "one_unique_instance_artifact_per_row_including_empty": True,
                "n_empty_assignment_rows": 2,
                "empty_assignment_rows_permitted": True,
                "artifact_point_counts_match_summary": True,
                "n_distal_qc_eligible": 30,
            },
            "artifact_verification": {
                "passed": True,
                "n_completion_manifest_artifacts_verified": 99,
                "provenance_manifest_identical_to_completion": True,
                "fitted_lens_archives_hash_verified_but_not_opened": True,
            },
            "sample_coverage": coverage,
            "reviewed_samples": attested_reviewed,
        }
        (self.root / "instance_qc_attestation.json").write_text(
            json.dumps(self.attestation), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self) -> None:
        (self.root / "instance_qc_attestation.json").write_text(
            json.dumps(self.attestation), encoding="utf-8"
        )

    def test_accepts_empty_rows_when_explicitly_inventory_bound(self) -> None:
        backend.validate_attestation(self.root, self.repo, "eye", 32, self.partition, 30)

    def test_rejects_wrong_stratum_count(self) -> None:
        self.attestation["sample_coverage"]["r0_s0"] = 1
        self._write()
        with self.assertRaisesRegex(backend.ContractError, "exactly two"):
            backend.validate_attestation(self.root, self.repo, "eye", 32, self.partition, 30)

    def test_rejects_bound_file_tamper(self) -> None:
        self.paths["review_json"].write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(backend.ContractError, "SHA-256 mismatch|Size mismatch"):
            backend.validate_attestation(self.root, self.repo, "eye", 32, self.partition, 30)

    def test_rejects_mismatched_distal_count(self) -> None:
        self.attestation["technical_inventory"]["n_distal_qc_eligible"] = 29
        self._write()
        with self.assertRaisesRegex(backend.ContractError, "distal-QC count"):
            backend.validate_attestation(self.root, self.repo, "eye", 32, self.partition, 30)


class LensArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "lens.npz"
        self.row = synthetic_table(25).iloc[0].copy()
        coefficients = self.row.loc[list(backend.TARGET_COLUMNS)].to_numpy(float)
        raw_xy = backend.CANONICAL_GRID_XY[:30].copy()
        raw_thickness = backend._evaluate_coefficients_on_xy(coefficients, raw_xy)
        self.row["target_support"] = 30
        self.row["target_depth_um"] = float(np.median(raw_thickness))
        self.row["target_q05_raw_thickness_um"] = float(np.quantile(raw_thickness, 0.05))
        self.row["target_rmse_um"] = 0.0
        config_text = backend.canonical_json(valid_stage2_config())
        np.savez(
            self.path,
            schema_version=np.array("experiment63.lens.v2"),
            lens_index=np.array(int(self.row["lens_index"]), dtype=np.int64),
            distal_points_xyz_um=np.zeros((30, 3), dtype=np.float64),
            proximal_points_xyz_um=np.zeros((30, 3), dtype=np.float64),
            canonical_grid_xy=backend.CANONICAL_GRID_XY,
            target_smoothed_thickness_um=backend.CANONICAL_DESIGN @ coefficients,
            raw_target_xy_normalized=raw_xy,
            raw_target_thickness_um=raw_thickness,
            target_coefficients_c0_c5=coefficients,
            distal_frame_origin_xyz_um=np.zeros(3, dtype=np.float64),
            distal_frame_u_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            distal_frame_v_xyz=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            distal_frame_outward_xyz=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            distal_coefficients_normalized=np.zeros(6, dtype=np.float64),
            config_json=np.array(config_text),
            config_sha256=np.array(backend.sha256_text(config_text)),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_validates_raw_and_smoothed_geometry(self) -> None:
        arrays = backend.validate_lens_artifact(self.path, self.row)
        self.assertEqual(arrays["raw_target_thickness_um"].shape, (30,))

    def test_rejects_grid_reordering(self) -> None:
        with np.load(self.path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["canonical_grid_xy"] = arrays["canonical_grid_xy"][::-1]
        np.savez(self.path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "grid mismatch"):
            backend.validate_lens_artifact(self.path, self.row)

    def test_rejects_smoothed_target_tamper(self) -> None:
        with np.load(self.path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["target_smoothed_thickness_um"] = arrays["target_smoothed_thickness_um"] + 1
        np.savez(self.path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "exact structural flag"):
            backend.validate_lens_artifact(self.path, self.row)

    def test_rejects_extra_fitted_lens_array(self) -> None:
        with np.load(self.path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["posthoc_model_error"] = np.array(0.0)
        np.savez(self.path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "exact v2 contract"):
            backend.validate_lens_artifact(self.path, self.row)

    def test_rejects_rank_deficient_raw_target_geometry(self) -> None:
        with np.load(self.path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["raw_target_xy_normalized"] = np.zeros((30, 2), dtype=np.float64)
        np.savez(self.path, **arrays)
        with self.assertRaisesRegex(backend.ContractError, "exact structural flag"):
            backend.validate_lens_artifact(self.path, self.row)


class ProvenanceSchemaTests(unittest.TestCase):
    def test_arthur_loader_reports_missing_immediately_indexed_keys_as_contract_errors(self) -> None:
        required = {
            "schema_version", "status", "analysis_scope", "isolation_basis",
            "oracle_stage1_scope", "table_sha256", "table_size_bytes", "n_rows", "volumes",
            "threshold_config", "threshold_config_sha256", "pipeline_config",
            "pipeline_config_sha256", "counts", "counts_by_volume", "cohorts",
            "exclusion_reason_counts", "target_observation_contract", "fixed_points", "frame_audits",
            "stage1_diagnostics",
            "input_files", "output_artifacts", "git", "code_sha256", "eyemap",
            "biological_independence", "coordinate_calibration", "created_utc",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table_path = root / "table.csv"
            table_path.write_text("placeholder\n", encoding="utf-8")
            provenance_path = root / "provenance.json"
            for missing in (
                "pipeline_config", "pipeline_config_sha256", "counts",
                "stage1_diagnostics", "target_observation_contract",
            ):
                payload = {key: None for key in required - {missing}}
                provenance_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(backend.ContractError, missing):
                        backend.load_arthur_source_bundle(
                            table_path, provenance_path, root, "0" * 40
                        )

    def test_maike_loader_requires_identity_independence_counts_and_cohort_contract(self) -> None:
        required = {
            "schema_version", "status", "eye_id", "species", "sex",
            "biological_independence", "analysis_scope", "isolation_basis",
            "threshold_config", "threshold_config_sha256", "pipeline_config",
            "pipeline_config_sha256", "predictor_pipeline_config",
            "sealed_config_sha256", "n_expected", "n_rows", "counts",
            "contiguous_indices", "index_range", "instance_segmentation_validated",
            "target_cohort_definitions", "partition_evidence", "fixed_point",
            "sealed_distal_stage1_manifest", "input_hashes", "git", "created_utc",
            "producer_implementation_hashes", "output_manifest",
        }
        eye_id = next(iter(backend.EXPECTED_EYES))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for missing in (
                "species", "biological_independence", "counts", "target_cohort_definitions",
                "producer_implementation_hashes",
            ):
                payload = {key: None for key in required - {missing}}
                (root / "completion.json").write_text(json.dumps(payload), encoding="utf-8")
                (root / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(backend.ContractError, missing):
                        backend.load_maike_eye_bundle(root, eye_id, "0" * 40, root)


class ExecutionSafetyTests(unittest.TestCase):
    def test_immutable_cli_has_no_external_supporting_prediction_hook(self) -> None:
        arguments = [
            "--repo", ".", "--expected-commit", "0" * 40,
            "--arthur-table", "a.csv", "--arthur-provenance", "a.json",
            "--maike-root", "m", "--output-dir", "o",
            "--supporting-predictions", "unfrozen.csv",
        ]
        with mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            backend.build_parser().parse_args(arguments)

    def test_output_directory_is_exclusive_before_any_input_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already"
            output.mkdir()
            with self.assertRaisesRegex(backend.ContractError, "already exists"):
                backend.execute_frozen_primary(
                    repo=Path(temporary),
                    expected_commit="0" * 40,
                    arthur_table=Path(temporary) / "missing.csv",
                    arthur_provenance=Path(temporary) / "missing.json",
                    maike_root=Path(temporary),
                    output_dir=output,
                )

    def test_execution_rejects_duplicate_or_missing_animal_independence_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = synthetic_table(25, eye_id="Arthur")
            source["volume"] = backend.ARTHUR_VOLUMES[0]
            maike_side_effect = []
            for eye_id in backend.EXPECTED_EYES:
                maike_side_effect.append(
                    (
                        synthetic_table(1, eye_id=eye_id),
                        {
                            "eye_id": eye_id,
                            "animal_id": "duplicated-animal",
                            "biological_independence": {"independent_unit": "animal"},
                        },
                    )
                )
            with (
                mock.patch.object(backend, "require_frozen_git", return_value="0" * 40),
                mock.patch.object(
                    backend,
                    "load_arthur_source_bundle",
                    return_value=(source, {}),
                ),
                mock.patch.object(
                    backend,
                    "load_maike_eye_bundle",
                    side_effect=maike_side_effect,
                ),
            ):
                with self.assertRaisesRegex(backend.ContractError, "exactly 12 distinct animals"):
                    backend.execute_frozen_primary(
                        repo=root,
                        expected_commit="0" * 40,
                        arthur_table=root / "arthur.csv",
                        arthur_provenance=root / "arthur.json",
                        maike_root=root,
                        output_dir=root / "output",
                    )

    def test_cli_requires_explicit_primary_switch(self) -> None:
        with mock.patch.object(sys, "stderr"):
            result = backend.main(
                [
                    "--repo", ".",
                    "--expected-commit", "0" * 40,
                    "--arthur-table", "a.csv",
                    "--arthur-provenance", "a.json",
                    "--maike-root", "m",
                    "--output-dir", "o",
                ]
            )
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()

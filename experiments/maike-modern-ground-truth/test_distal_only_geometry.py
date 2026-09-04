from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_distal_frame_stability as stability
import distal_only_geometry as geometry
import experiment_63_primary_backend as backend


def _sealed_config() -> dict[str, object]:
    return {
        "analysis_scope": "conditional_on_oracle_distal_surface_localization",
        "isolation_basis": "stage2_reads_only_sha256_sealed_distal_artifacts",
        "predictor_pipeline_config": {
            "fixed_point_policy": "monotone_drop_only_no_reentry",
            "original_spacing_um": [0.325, 0.325, 0.325],
            "distal_split": "largest_26_component_boundary_deterministic_1d_two_means",
            "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
        },
        "threshold_config": dict(geometry.DEFAULT_CONFIG),
    }


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tangent_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def _cap_points(normal: np.ndarray, cap_radius: float = 5.0, noisy: bool = False) -> np.ndarray:
    u, v = _tangent_axes(normal)
    coordinates = np.linspace(-cap_radius, cap_radius, 9)
    xy = np.array(
        [(x, y) for x in coordinates for y in coordinates if x * x + y * y <= cap_radius**2],
        dtype=np.float64,
    )
    height = -0.025 * np.sum(xy * xy, axis=1)
    if noisy:
        height = height + np.where(np.arange(len(height)) % 2, 4.0, -4.0)
    centre = 100.0 * normal
    return centre + xy[:, :1] * u + xy[:, 1:] * v + height[:, None] * normal


def _synthetic_eye(side: int = 5) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    values = np.linspace(-0.25, 0.25, side)
    for lens_index, (x, y) in enumerate((x, y) for x in values for y in values):
        normal = np.array([x, y, 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        records.append(
            {
                "lens_index": lens_index,
                "points_xyz_um": _cap_points(normal),
                "stage1_eligible": True,
            }
        )
    return records


def _mark_sealed(records: list[dict[str, object]]) -> list[dict[str, object]]:
    config_json = _canonical_json(_sealed_config())
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    sealed = []
    for record in records:
        item = dict(record)
        item.update(
            {
                "sealed_distal_artifact": True,
                "config": dict(geometry.DEFAULT_CONFIG),
                "sealed_config": _sealed_config(),
                "config_json": config_json,
                "config_sha256": config_hash,
                "artifact_path": f"/sealed/lens-{record['lens_index']}.npz",
                "artifact_sha256": f"{int(record['lens_index']):064x}",
            }
        )
        sealed.append(item)
    return sealed


class DistalCapFitTests(unittest.TestCase):
    def test_fits_exact_quadratic(self) -> None:
        normal = np.array([0.1, -0.2, 1.0])
        normal /= np.linalg.norm(normal)
        fit = geometry.fit_distal_cap(_cap_points(normal))
        self.assertGreaterEqual(fit["support"], 25)
        self.assertGreater(fit["scale_um"], 3.0)
        self.assertLess(fit["quadratic_rmse_um"], 1.0e-10)
        self.assertTrue(np.all(np.isfinite(fit["curvature_eigenvalues"])))
        np.testing.assert_allclose(np.cross(fit["u"], fit["v"]), fit["w"], atol=1.0e-12)

    def test_invalid_points_rejected(self) -> None:
        with self.assertRaises(geometry.DistalGeometryError):
            geometry.fit_distal_cap(np.zeros((4, 2)))
        invalid = np.zeros((25, 3))
        invalid[0, 0] = np.nan
        with self.assertRaises(geometry.DistalGeometryError):
            geometry.fit_distal_cap(invalid)

    def test_public_robust_fit_has_normalized_design_contract(self) -> None:
        x = np.linspace(-0.8, 0.8, 9)
        y = np.repeat(np.linspace(-0.7, 0.7, 7), 9)
        x = np.tile(x, 7)
        expected = np.array([2.0, 0.3, -0.2, 0.5, 0.1, -0.4])
        design = np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))
        fit = geometry.fit_robust_quadratic(x, y, design @ expected)
        np.testing.assert_allclose(fit["beta"], expected, atol=1.0e-12)
        self.assertLess(fit["rmse"], 1.0e-12)
        self.assertEqual(fit["support"], len(x))


class SealedArtifactTests(unittest.TestCase):
    def _write_artifact(self, path: Path, config_hash: str | None = None) -> str:
        config_json = _canonical_json(_sealed_config())
        correct_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        np.savez(
            path,
            schema_version=np.asarray(geometry.SEALED_DISTAL_SCHEMA_VERSION),
            lens_index=np.asarray(7, dtype=np.int64),
            points_zyx=np.asarray([[2, 3, 4], [5, 6, 7]], dtype=np.int32),
            spacing_um=np.asarray([0.5, 1.0, 2.0], dtype=np.float64),
            config_json=np.asarray(config_json),
            config_sha256=np.asarray(config_hash or correct_hash),
        )
        return correct_hash

    def test_loader_validates_and_converts_zyx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.npz"
            config_hash = self._write_artifact(path)
            record = geometry.load_sealed_distal(path, expected_config_sha256=config_hash)
        np.testing.assert_allclose(record["points_xyz_um"][0], [8.0, 3.0, 1.0])
        self.assertEqual(record["lens_index"], 7)
        self.assertTrue(record["sealed_distal_artifact"])

    def test_loader_accepts_explicit_empty_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.npz"
            config_json = _canonical_json(_sealed_config())
            config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            np.savez(
                path,
                schema_version=np.asarray(geometry.SEALED_DISTAL_SCHEMA_VERSION),
                lens_index=np.asarray(8, dtype=np.int64),
                points_zyx=np.empty((0, 3), dtype=np.int32),
                spacing_um=np.asarray([0.325, 0.325, 0.325], dtype=np.float64),
                config_json=np.asarray(config_json),
                config_sha256=np.asarray(config_hash),
            )
            record = geometry.load_sealed_distal(path)
        self.assertEqual(record["points_xyz_um"].shape, (0, 3))

    def test_loader_rejects_hash_mismatch_and_extra_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-hash.npz"
            self._write_artifact(path, config_hash="0" * 64)
            with self.assertRaisesRegex(geometry.DistalGeometryError, "config_sha256"):
                geometry.load_sealed_distal(path)

            extra = Path(directory) / "extra.npz"
            config_json = _canonical_json(_sealed_config())
            config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            np.savez(
                extra,
                schema_version=np.asarray(geometry.SEALED_DISTAL_SCHEMA_VERSION),
                lens_index=np.asarray(0, dtype=np.int64),
                points_zyx=np.zeros((25, 3), dtype=np.int32),
                spacing_um=np.ones(3, dtype=np.float64),
                config_json=np.asarray(config_json),
                config_sha256=np.asarray(config_hash),
                hidden_target=np.zeros(1),
            )
            with self.assertRaisesRegex(geometry.DistalGeometryError, "keys differ"):
                geometry.load_sealed_distal(extra)

    def test_loader_rejects_target_bearing_sealed_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.npz"
            sealed_config = _sealed_config()
            pipeline = dict(sealed_config["predictor_pipeline_config"])
            pipeline["distal_split"] = "target_informed_boundary"
            sealed_config["predictor_pipeline_config"] = pipeline
            config_json = _canonical_json(sealed_config)
            config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            np.savez(
                path,
                schema_version=np.asarray(geometry.SEALED_DISTAL_SCHEMA_VERSION),
                lens_index=np.asarray(0, dtype=np.int64),
                points_zyx=np.zeros((25, 3), dtype=np.int32),
                spacing_um=np.asarray([0.325, 0.325, 0.325], dtype=np.float64),
                config_json=np.asarray(config_json),
                config_sha256=np.asarray(config_hash),
            )
            with self.assertRaisesRegex(geometry.DistalGeometryError, "forbidden"):
                geometry.load_sealed_distal(path)


class FixedPointTests(unittest.TestCase):
    def test_drop_only_qc_and_exact_diagnostics(self) -> None:
        records = _synthetic_eye()
        low_support = dict(records[0])
        low_support["points_xyz_um"] = np.asarray(low_support["points_xyz_um"])[:20]
        records[0] = low_support

        small = dict(records[1])
        points = np.asarray(small["points_xyz_um"])
        centre = np.mean(points, axis=0)
        small["points_xyz_um"] = centre + 0.2 * (points - centre)
        records[1] = small

        normal = np.asarray(records[2]["points_xyz_um"])[0]
        normal /= np.linalg.norm(normal)
        noisy = dict(records[2])
        noisy["points_xyz_um"] = _cap_points(normal, noisy=True)
        records[2] = noisy

        result = geometry.run_monotone_fixed_point(records)
        self.assertTrue(result["converged"])
        self.assertEqual(result["readded_count"], 0)
        self.assertEqual(result["iterations"], len(result["iteration_diagnostics"]))
        self.assertEqual(result["eligible_counts"][-1], result["eligible_counts"][-2])
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    result["eligible_counts"], result["eligible_counts"][1:]
                )
            )
        )
        self.assertNotIn(0, result["eligible_indices"])
        self.assertNotIn(1, result["eligible_indices"])
        self.assertNotIn(2, result["eligible_indices"])
        reasons = {row["lens_index"]: row["drop_reasons"] for row in result["per_lens"]}
        self.assertIn("support_below_minimum", reasons[0])
        self.assertIn("scale_below_minimum", reasons[1])
        self.assertIn("quadratic_rmse_above_maximum", reasons[2])

    def test_position_descriptors_are_rotation_and_reflection_invariant(self) -> None:
        records = _synthetic_eye()
        baseline = geometry.derive_distal_only_eye_geometry(records)
        positions = np.asarray([row["position_2d_um"] for row in baseline["per_lens"]])
        pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        expected_nn = float(np.median(np.min(pairwise, axis=1)))
        self.assertAlmostEqual(baseline["median_nearest_neighbour_um"], expected_nn)
        self.assertAlmostEqual(baseline["central_threshold_um"], 0.5 * expected_nn)
        np.testing.assert_allclose(
            [row["eccentricity_um"] for row in baseline["per_lens"]],
            np.linalg.norm(positions, axis=1),
        )
        orthogonal = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        translated = []
        for record in records:
            transformed = np.asarray(record["points_xyz_um"]) @ orthogonal.T + np.array(
                [17.0, -8.0, 2.0]
            )
            translated.append({**record, "points_xyz_um": transformed})
        reflected = geometry.derive_distal_only_eye_geometry(translated)
        first = np.asarray([row["position_features"] for row in baseline["per_lens"]])
        second = np.asarray([row["position_features"] for row in reflected["per_lens"]])
        np.testing.assert_allclose(first, second, rtol=1.0e-9, atol=1.0e-9)


class StabilityAuditTests(unittest.TestCase):
    def test_audit_is_sealed_only_and_emits_frozen_gate_key(self) -> None:
        records = _synthetic_eye(side=6)
        with self.assertRaisesRegex(geometry.DistalGeometryError, "sealed"):
            stability.audit_frame_stability(records, eye_id="test-eye")
        sealed = _mark_sealed(records)
        result = stability.audit_frame_stability(sealed, eye_id="test-eye")
        self.assertEqual(result["schema_version"], "experiment63.distal-frame-audit.v1")
        self.assertTrue(result["target_blind"])
        self.assertEqual(set(result["perturbations"]), set(stability.PERTURBATION_KEYS))
        self.assertIn("n_numerical_fallback_outside_central", result["metrics"])
        self.assertIn("zero_numerical_fallback_outside_central", result["gates"])
        self.assertTrue(result["gate_passed"])
        backend.validate_frame_audit(result, "test-eye", "synthetic audit")
        self.assertEqual(
            result["input_artifacts"][0]["relative_path"], "sealed_distal/lens_000000.npz"
        )
        self.assertNotIn("path", result["input_artifacts"][0])

        empty = dict(sealed[0])
        empty.update(
            {
                "lens_index": 999,
                "lens_id": 999,
                "points_xyz_um": np.empty((0, 3), dtype=np.float64),
                "artifact_path": "/sealed/lens-999.npz",
                "artifact_sha256": "f" * 64,
            }
        )
        with_empty = stability.audit_frame_stability(sealed + [empty], eye_id="test-eye")
        self.assertEqual(with_empty["n_input_artifacts"], 37)
        self.assertEqual(with_empty["n_distal_qc_eligible"], 36)
        self.assertEqual(with_empty["gate_passed"], result["gate_passed"])
        self.assertNotIn("all_sealed_distal_artifacts_pass_qc", with_empty["gates"])


if __name__ == "__main__":
    unittest.main()

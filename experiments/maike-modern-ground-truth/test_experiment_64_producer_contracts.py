from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import distal_only_geometry as geometry
import experiment_64_extract_lens_surfaces as maike64
import experiment_64_prepare_arthur_source_table as arthur64
import experiment_64_robust_distal_core as robust_core
import experiment_64_technical_metrics as technical_metrics


def _quadratic_cap() -> np.ndarray:
    axis = np.linspace(-5.0, 5.0, 25)
    xy = np.asarray(
        [(x, y) for x in axis for y in axis if x * x + y * y <= 25.0],
        dtype=np.float64,
    )
    z = 0.03 * xy[:, 0] ** 2 + 0.04 * xy[:, 1] ** 2 + 0.01 * xy[:, 0]
    return np.column_stack((xy, z))


class Experiment64TechnicalMetricTests(unittest.TestCase):
    def test_arthur_metrics_are_target_free_and_do_not_use_connectivity(self) -> None:
        points = _quadratic_cap()
        fitted = geometry.fit_distal_cap(points)
        metrics = technical_metrics.arthur_final_distal_coherence_metrics(
            points, fitted
        )
        self.assertGreaterEqual(metrics["distal_fit_support"], 25)
        self.assertLess(metrics["distal_fit_rmse_um"], 1.0e-3)
        self.assertFalse(metrics["connectivity_gate_applicable"])
        self.assertNotIn("coherence_graph_radius_um", metrics)
        self.assertTrue(np.isfinite(metrics["coherence_margin"]))
        self.assertFalse(any("target" in key for key in metrics))

    def test_maike_metrics_use_final_q90_voxel_connectivity_and_four_margins(self) -> None:
        yx = np.asarray(
            [(y, x) for y in range(11, 20) for x in range(11, 20)],
            dtype=np.int32,
        )
        points_zyx = np.column_stack(
            (np.full(len(yx), 30, dtype=np.int32), yx)
        )
        spacing = np.ones(3, dtype=np.float64)
        points_xyz = points_zyx[:, ::-1].astype(np.float64)
        fitted = geometry.fit_distal_cap(points_xyz)
        metrics = technical_metrics.maike_final_distal_coherence_metrics(
            points_zyx, spacing, fitted
        )
        self.assertGreaterEqual(metrics["distal_fit_support"], 25)
        self.assertEqual(metrics["distal_fit_26_component_count"], 1)
        self.assertEqual(metrics["distal_fit_26_largest_component_fraction"], 1.0)
        self.assertTrue(metrics["maike_final_fit_gate_pass"])
        self.assertEqual(metrics["maike_final_fit_gate_reasons"], [])
        expected = min(
            metrics["coherence_support_margin"],
            metrics["coherence_rmse_margin"],
            metrics["coherence_lcc_margin"],
            metrics["coherence_p99_over_scale_margin"],
        )
        self.assertEqual(metrics["coherence_margin"], expected)

    def test_metric_config_is_exact_and_hash_bound(self) -> None:
        encoded = technical_metrics.canonical_technical_coherence_config_json()
        self.assertEqual(
            json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")),
            encoded,
        )
        self.assertEqual(len(technical_metrics.technical_coherence_config_sha256()), 64)
        changed = dict(technical_metrics.TECHNICAL_COHERENCE_CONFIG)
        changed["fit_largest_component_fraction_min"] = 0.98
        with self.assertRaises(technical_metrics.TechnicalMetricError):
            technical_metrics.canonical_technical_coherence_config_json(changed)


class Experiment64ArthurAdapterTests(unittest.TestCase):
    def test_arthur_adapter_seals_only_the_shared_core(self) -> None:
        raw = _quadratic_cap()
        tail = np.column_stack(
            (
                np.linspace(6.0, 14.0, 40),
                np.zeros(40),
                np.linspace(1.0, 3.0, 40),
            )
        )
        raw = np.unique(np.vstack((raw, tail)), axis=0)
        config = robust_core.normalise_robust_core_config()
        selected = robust_core.select_robust_distal_core(raw, config)
        core, status, reasons, diagnostics = arthur64._robust_core_or_empty(raw, config)
        self.assertEqual(status, "pass")
        self.assertEqual(reasons, [])
        np.testing.assert_array_equal(core, selected["retained_points_xyz_um"])

        threshold = dict(geometry.DEFAULT_CONFIG)
        sealed_config = arthur64._sealed_config(threshold, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.npz"
            record = arthur64._write_sealed_core(
                path,
                volume="20240530",
                eye_id=0,
                lens_index=7,
                raw_support=len(raw),
                points_xyz_um=core,
                diagnostics=diagnostics,
                sealed_config=sealed_config,
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(
                    set(archive.files),
                    {
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
                    },
                )
                np.testing.assert_array_equal(archive["points_xyz_um"], core)
                self.assertEqual(int(archive["raw_distal_support"]), len(raw))
            self.assertTrue(record["sealed_distal_artifact"])
            self.assertTrue(record["stage1_eligible"])

    def test_invalid_candidate_is_recorded_not_relaxed(self) -> None:
        config = robust_core.normalise_robust_core_config()
        core, status, reasons, diagnostics = arthur64._robust_core_or_empty(
            _quadratic_cap()[:20], config
        )
        self.assertEqual(core.shape, (0, 3))
        self.assertEqual(status, "ineligible")
        self.assertTrue(reasons)
        self.assertEqual(diagnostics["status"], "ineligible")

    def test_technical_table_has_no_target_fields(self) -> None:
        self.assertFalse(any(field.startswith("target_") for field in arthur64.TECHNICAL_FIELDS))
        self.assertTrue(any(field.startswith("target_") for field in arthur64.TARGET_FIELDS))
        self.assertEqual(
            arthur64.robust_core_config_sha256(),
            robust_core.robust_core_config_sha256(),
        )

    def test_arthur_targets_cross_bind_technical_and_sealed_core_configs(self) -> None:
        technical_config = arthur64._sealed_config(
            dict(geometry.DEFAULT_CONFIG), robust_core.normalise_robust_core_config()
        )
        technical_json = arthur64._canonical_json(technical_config)
        technical_sha = arthur64._sha256_bytes(technical_json.encode("utf-8"))
        outcome = arthur64._outcome_config(technical_sha)
        self.assertEqual(outcome["technical_config_sha256"], technical_sha)
        self.assertEqual(
            arthur64.TARGET_KEYS,
            frozenset(
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
            ),
        )

    def test_arthur_outcome_work_is_after_the_technical_eye_stage(self) -> None:
        self.assertNotIn("fit_source_target", inspect.getsource(arthur64._process_eye))
        self.assertIn(
            "fit_source_target", inspect.getsource(arthur64._write_pending_targets)
        )


class Experiment64CrossAdapterContractTests(unittest.TestCase):
    def test_adapters_share_core_hash_and_isolation_basis(self) -> None:
        self.assertEqual(arthur64.ISOLATION_BASIS, maike64.ISOLATION_BASIS)
        self.assertEqual(
            arthur64.robust_core_config_sha256(),
            maike64.robust_distal_core.robust_core_config_sha256(),
        )
        self.assertIn("raw_distal_points_zyx", maike64.INSTANCE_KEYS)
        self.assertEqual(
            maike64.SEALED_DISTAL_KEYS,
            frozenset(
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
            ),
        )

    def test_maike_technical_and_target_tables_are_separate(self) -> None:
        self.assertFalse(
            any(field.startswith("target_") for field in maike64.TECHNICAL_FIELDS)
        )
        self.assertTrue(
            any(field.startswith("target_") for field in maike64.TARGET_FIELDS)
        )

    def test_maike_sealed_core_rejects_integral_float_scalar_ids(self) -> None:
        yx = np.asarray(
            [(y, x) for y in range(11, 20) for x in range(11, 20)],
            dtype=np.int32,
        )
        raw = np.column_stack((np.full(len(yx), 30, dtype=np.int32), yx))
        thresholds, pipeline, robust = maike64._normalise_configs(None, None)
        _, sealed_config = maike64._technical_configs(
            thresholds, pipeline, robust
        )
        core, status, _, diagnostics = maike64._robust_core_or_empty(
            raw,
            np.asarray(pipeline["original_spacing_um"], dtype=np.float64),
            robust,
        )
        self.assertEqual(status, "pass")
        base_payload = maike64._sealed_core_payload(
            7,
            core,
            pipeline["original_spacing_um"],
            sealed_config,
            raw_distal_support=len(raw),
            robust_core_diagnostics=diagnostics,
        )
        expected_hash = str(base_payload["config_sha256"].item())
        with tempfile.TemporaryDirectory() as directory:
            for member in ("lens_index", "raw_distal_support"):
                payload = dict(base_payload)
                payload[member] = np.asarray(float(payload[member].item()))
                path = Path(directory) / f"float_{member}.npz"
                maike64._atomic_savez(path, **payload)
                with self.subTest(member=member), self.assertRaises(
                    maike64.ExtractionError
                ):
                    maike64.load_sealed_distal_core(
                        path, expected_config_sha256=expected_hash
                    )


if __name__ == "__main__":
    unittest.main()

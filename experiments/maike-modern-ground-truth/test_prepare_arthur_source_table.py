from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = _load("prepare_arthur_source_table", HERE / "prepare_arthur_source_table.py")
maike_extractor = _load("extract_lens_surfaces", HERE / "extract_lens_surfaces.py")
experiment57 = _load(
    "experiment_57_outer_only_validation",
    HERE.parent / "arthur-modern-ground-truth" / "experiment_57_outer_only_validation.py",
)


def _disk_points(origin: np.ndarray, z: float, radius: float, n: int) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radial = radius * (0.55 + 0.4 * ((np.arange(n) % 7) / 6.0))
    return origin + np.column_stack((radial * np.cos(angle), radial * np.sin(angle), np.full(n, z)))


class OracleStageOneTests(unittest.TestCase):
    def test_stage_one_ignores_scale_and_target_support(self) -> None:
        first = np.array([0.0, 0.0, 0.0])
        second = np.array([100.0, 0.0, 0.0])
        surfaces = [
            np.vstack(
                [
                    _disk_points(first, 3.0, 20.0, 30),
                    _disk_points(first, -3.0, 20.0, 20),
                ]
            ),
            np.vstack(
                [
                    _disk_points(second, 3.0, 5.0, 30),
                    _disk_points(second, -3.0, 5.0, 30),
                ]
            ),
        ]
        lenses = np.vstack((first, second))
        tips = lenses - np.array([0.0, 0.0, 10.0])
        records, diagnostics = experiment57.prepare_oracle_split_records(
            surfaces, lenses, tips
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(diagnostics["n_raw_patches"], 2)
        self.assertNotIn("_failure_counts", diagnostics)
        self.assertFalse(records[0]["target_support"])
        self.assertTrue(records[1]["target_support"])
        # The deliberately oversized first cap remains in Stage 1: scale QC
        # belongs to the later distal-only fixed point.
        self.assertGreater(np.quantile(np.linalg.norm(records[0]["outer"][:, :2], axis=1), 0.9), 13.0)
        for record in records:
            self.assertGreater(np.median(record["outer"][:, 2]), np.median(record["inner"][:, 2]))
            self.assertNotIn("outer_rmse", record)
            self.assertNotIn("target_valid", record)

    def test_stage_one_rejects_nonfinite_input(self) -> None:
        surface = _disk_points(np.zeros(3), 2.0, 5.0, 30)
        surfaces = [surface.copy(), surface + np.array([100.0, 0.0, 0.0])]
        lenses = np.array([[0.0, 0.0, np.nan], [100.0, 0.0, 0.0]])
        tips = np.array([[0.0, 0.0, -10.0], [100.0, 0.0, -10.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            experiment57.prepare_oracle_split_records(surfaces, lenses, tips)


class TargetConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        axis_um = np.arange(-4.0, 4.0 + 0.65 / 2.0, 0.65)
        xx, yy = np.meshgrid(axis_um, axis_um)
        keep = xx * xx + yy * yy <= 4.5**2
        self.scale = 5.0
        self.xy = np.column_stack((xx[keep], yy[keep])) / self.scale
        self.distal_beta = np.array([1.0, 0.2, -0.1, 0.15, 0.03, 0.12])
        self.target_beta = np.array([4.0, 0.1, -0.05, 0.2, 0.01, 0.18])
        self.geometry = {
            "origin": np.zeros(3),
            "u": np.array([1.0, 0.0, 0.0]),
            "v": np.array([0.0, 1.0, 0.0]),
            "w": np.array([0.0, 0.0, 1.0]),
            "scale_um": self.scale,
            "quadratic_beta_normalized": self.distal_beta,
        }
        distal_z = source.quadratic_design(self.xy) @ self.distal_beta
        self.distal = np.column_stack(
            (self.xy[:, 0] * self.scale, self.xy[:, 1] * self.scale, distal_z)
        )

    def proximal_surface_for_target(
        self, target_beta: np.ndarray, *, count: int | None = None
    ) -> np.ndarray:
        xy = self.xy if count is None else self.xy[:count]
        distal_z = source.quadratic_design(xy) @ self.distal_beta
        thickness = source.quadratic_design(xy) @ target_beta
        return np.column_stack(
            (
                xy[:, 0] * self.scale,
                xy[:, 1] * self.scale,
                distal_z - thickness,
            )
        )

    def test_recovers_positive_unsymmetrized_target(self) -> None:
        target = source.fit_source_target(
            self.distal,
            self.proximal_surface_for_target(self.target_beta),
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertTrue(target["target_resolvable"])
        self.assertTrue(target["target_qc"])
        np.testing.assert_allclose(target["target_coefficients"], self.target_beta, atol=1e-9)
        self.assertEqual(target["canonical_grid_xy"].shape, (81, 2))
        self.assertEqual(target["target_support"], len(self.xy))

    def test_qc_does_not_change_target_resolvability(self) -> None:
        # Negative only near the centre: median thickness remains positive,
        # while the raw q05 sensitivity criterion fails.
        negative = np.array([-0.5, 0.0, 0.0, 8.0, 0.0, 8.0])
        target = source.fit_source_target(
            self.distal,
            self.proximal_surface_for_target(negative),
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertTrue(target["target_resolvable"])
        self.assertFalse(target["target_qc"])
        self.assertIn("target_q05_raw_thickness_not_positive", target["target_qc_reasons"])

    def test_nonpositive_median_remains_structurally_resolvable(self) -> None:
        negative = self.target_beta.copy()
        negative[0] = -10.0
        target = source.fit_source_target(
            self.distal,
            self.proximal_surface_for_target(negative),
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertTrue(target["target_resolvable"])
        self.assertFalse(target["target_qc"])
        self.assertIn("target_q05_raw_thickness_not_positive", target["target_qc_reasons"])

    def test_support_is_measured_in_final_distal_q90_frame(self) -> None:
        target = source.fit_source_target(
            self.distal,
            self.proximal_surface_for_target(self.target_beta, count=24),
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertFalse(target["target_resolvable"])
        self.assertEqual(target["target_support"], 24)
        self.assertEqual(len(target["raw_target_xy_normalized"]), 24)
        self.assertEqual(len(target["raw_target_thickness_um"]), 24)
        self.assertIn(
            "target_support_below_minimum", target["target_resolvability_reasons"]
        )
        self.assertIn("target_support_below_minimum", target["target_qc_reasons"])

    def test_target_support_counts_unique_surface_vertices(self) -> None:
        proximal = self.proximal_surface_for_target(self.target_beta)
        target = source.fit_source_target(
            self.distal,
            np.vstack((proximal, proximal[:10])),
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertTrue(target["target_resolvable"])
        self.assertEqual(target["target_support"], len(proximal))

    def test_high_rmse_is_sensitivity_qc_not_resolvability(self) -> None:
        thick = self.target_beta.copy()
        thick[0] = 30.0
        proximal = self.proximal_surface_for_target(thick)
        proximal[:, 2] += 8.0 * np.sin(np.arange(len(proximal), dtype=float))
        target = source.fit_source_target(
            self.distal,
            proximal,
            self.geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        self.assertTrue(target["target_resolvable"])
        self.assertGreater(target["target_q05_raw_thickness_um"], 0.0)
        self.assertGreater(target["target_rmse_um"], 2.5)
        self.assertFalse(target["target_qc"])
        self.assertIn("target_rmse_above_maximum", target["target_qc_reasons"])

    def test_surface_and_binary_adapters_share_target_representation(self) -> None:
        spacing = 0.325
        lateral = [
            (x, y)
            for x in range(-8, 9, 2)
            for y in range(-8, 9, 2)
            if x * x + y * y <= 64
        ]
        component_zyx = np.asarray(
            [(z, y, x) for x, y in lateral for z in (0, 4)], dtype=np.int32
        )
        complete_xyz = component_zyx[:, ::-1].astype(np.float64) * spacing
        scale = 3.0
        distal_beta = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        xy = complete_xyz[::2, :2] / scale
        distal_xyz = np.column_stack(
            (complete_xyz[::2, :2], source.quadratic_design(xy) @ distal_beta)
        )
        arthur_geometry = {
            "origin": np.zeros(3),
            "u": np.array([1.0, 0.0, 0.0]),
            "v": np.array([0.0, 1.0, 0.0]),
            "w": np.array([0.0, 0.0, 1.0]),
            "scale_um": scale,
            "quadratic_beta_normalized": distal_beta,
        }
        maike_geometry = {
            "origin_xyz_um": arthur_geometry["origin"],
            "u_axis_xyz": arthur_geometry["u"],
            "v_axis_xyz": arthur_geometry["v"],
            "outward_axis_xyz": arthur_geometry["w"],
            "distal_scale_um": scale,
            "quadratic_beta_normalized": distal_beta,
        }
        maike_config = {
            **maike_extractor.DEFAULT_THRESHOLD_CONFIG,
            **maike_extractor.DEFAULT_PIPELINE_CONFIG,
        }
        arthur = source.fit_source_target(
            distal_xyz,
            complete_xyz[::2],
            arthur_geometry,
            source.distal_geometry.DEFAULT_CONFIG,
        )
        maike = maike_extractor._extract_target(
            component_zyx,
            distal_xyz,
            maike_geometry,
            maike_config,
            robust_quadratic_fit=source.distal_geometry.fit_robust_quadratic,
        )
        self.assertEqual(arthur["target_support"], maike["target_support"])
        self.assertEqual(arthur["target_resolvable"], maike["target_resolvable"])
        self.assertEqual(arthur["target_qc"], maike["target_qc"])
        np.testing.assert_allclose(
            arthur["raw_target_xy_normalized"], maike["raw_target_xy_normalized"]
        )
        np.testing.assert_allclose(
            arthur["raw_target_thickness_um"], maike["raw_target_thickness_um"]
        )
        np.testing.assert_allclose(
            arthur["target_coefficients"], maike["target_coefficients"]
        )
        self.assertEqual(
            source.TARGET_CONFIG["target_support_unit"],
            "unique_proximal_mesh_vertices",
        )
        self.assertEqual(
            source.TARGET_CONFIG["target_observation_type"],
            "oracle_split_surface_mesh",
        )


class SealedSourceArtifactTests(unittest.TestCase):
    def test_round_trip_is_distal_only_and_hash_bound(self) -> None:
        points = _disk_points(np.zeros(3), 2.0, 5.0, 30)
        supplied = np.vstack((points[::-1], points[:3]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "distal.npz"
            record = source.write_sealed_distal(
                path,
                volume="20240530",
                eye_id=1,
                lens_index=42,
                points_xyz_um=supplied,
                threshold_config=source.distal_geometry.DEFAULT_CONFIG,
            )
            self.assertTrue(record["sealed_distal_artifact"])
            self.assertEqual(record["lens_index"], 42)
            np.testing.assert_array_equal(record["points_xyz_um"], np.unique(points, axis=0))
            self.assertNotIn("proximal", " ".join(record).lower())
            self.assertNotIn("target", " ".join(record).lower())
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(
                    set(archive.files),
                    {
                        "schema_version",
                        "volume",
                        "eye_id",
                        "lens_index",
                        "points_xyz_um",
                        "config_json",
                        "config_sha256",
                    },
                )

    def test_expected_counts_are_an_explicit_regression_gate(self) -> None:
        rows = []
        for index in range(source.EXPECTED_SOURCE_COUNTS["stage1"]):
            rows.append(
                {
                    "distal_qc": index < source.EXPECTED_SOURCE_COUNTS["distal_qc"],
                    # Target counts are measured diagnostics, not values that
                    # the implementation is allowed to force.
                    "target_resolvable": index < 4_875,
                    "target_qc": index < 4_874,
                }
            )
        counts = source._validate_expected_counts(rows)
        self.assertEqual(
            {key: counts[key] for key in source.EXPECTED_SOURCE_COUNTS},
            source.EXPECTED_SOURCE_COUNTS,
        )
        self.assertEqual(counts["target_resolvable"], 4_875)
        self.assertEqual(counts["target_qc"], 4_874)
        rows[-1]["distal_qc"] = True
        with self.assertRaisesRegex(source.SourcePreparationError, "counts changed"):
            source._validate_expected_counts(rows)


class ManifestTests(unittest.TestCase):
    def test_manifest_requires_exact_three_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {}
            for name in ("lens", "tip", "rdata"):
                path = root / name
                path.write_bytes(b"x")
                common[name if name != "lens" else "lens_mesh"] = str(path)
            # Correct the two differently named keys explicitly.
            common["tip_mesh"] = common.pop("tip")
            rows = [
                {"volume": volume, **common}
                for volume in ("20231107", "20240530")
            ]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"volumes": rows}), encoding="utf-8")
            with self.assertRaisesRegex(source.SourcePreparationError, "exactly"):
                source.load_manifest(manifest)

    def test_manifest_rejects_same_named_files_with_wrong_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for volume in source.VOLUMES:
                row = {"volume": volume}
                for role, identity in source.EXPECTED_INPUT_IDENTITIES[volume].items():
                    path = root / identity["name"]
                    path.write_bytes(f"not-the-frozen-{volume}-{role}".encode())
                    row[role] = str(path)
                rows.append(row)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"volumes": rows}), encoding="utf-8")
            with self.assertRaisesRegex(source.SourcePreparationError, "identity differs"):
                source.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()

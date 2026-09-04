from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import extract_lens_surfaces as subject


class LargestComponentTests(unittest.TestCase):
    def test_largest_component_is_lexicographically_sorted(self) -> None:
        points = np.asarray(
            [[8, 8, 8], [0, 0, 1], [0, 0, 0], [8, 8, 9]], dtype=np.int32
        )
        component, sizes = subject.deterministic_largest_component(points)
        np.testing.assert_array_equal(component, [[0, 0, 0], [0, 0, 1]])
        self.assertEqual(sizes, [2, 2])

    def test_exact_ninety_nine_percent_passes(self) -> None:
        self.assertTrue(subject.target_component_fraction_passes(99, 100))
        self.assertFalse(subject.target_component_fraction_passes(98, 100))

    def test_disconnected_specks_do_not_change_sealed_distal_bytes(self) -> None:
        main = np.argwhere(np.ones((5, 5, 5), dtype=bool)).astype(np.int32)
        with_specks = np.vstack([main, [[20, 20, 20], [22, 22, 22]]]).astype(np.int32)
        clean_component, _ = subject.deterministic_largest_component(main)
        speck_component, _ = subject.deterministic_largest_component(with_specks)
        np.testing.assert_array_equal(clean_component, speck_component)
        centre = np.asarray([2.0, 2.0, 2.0])
        axis = np.asarray([1.0, 0.0, 0.0])
        clean_cap, _ = subject.localize_distal_cap(clean_component, centre, axis)
        speck_cap, _ = subject.localize_distal_cap(speck_component, centre, axis)
        np.testing.assert_array_equal(clean_cap, speck_cap)
        payload_a = subject._sealed_payload(
            7, clean_cap, [0.325] * 3, subject.DEFAULT_THRESHOLD_CONFIG
        )
        payload_b = subject._sealed_payload(
            7, speck_cap, [0.325] * 3, subject.DEFAULT_THRESHOLD_CONFIG
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            subject._atomic_savez(first, **payload_a)
            subject._atomic_savez(second, **payload_b)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )


class PartitionTests(unittest.TestCase):
    def test_voronoi_assignment_is_lossless_and_single_candidate(self) -> None:
        mask = np.zeros((4, 4, 8), dtype=np.uint8)
        mask[1:3, 1:3, 1:7] = 1
        seeds = [
            subject.Seed(0, "a", np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
            subject.Seed(1, "b", np.asarray([1.0, 1.0, 6.0]), np.asarray([1.0, 0.0, 0.0])),
        ]
        instances, evidence = subject._partition_foreground(mask, seeds, slab_depth=2)
        self.assertEqual(sum(map(len, instances)), int(mask.sum()))
        self.assertTrue(evidence["exact_partition"])
        self.assertEqual(evidence["candidate_seeds_per_voxel"], 1)
        union = {tuple(row) for instance in instances for row in instance.tolist()}
        self.assertEqual(len(union), int(mask.sum()))

    def test_voronoi_tie_goes_to_lowest_lens_index(self) -> None:
        mask = np.zeros((1, 1, 3), dtype=np.uint8)
        mask[0, 0, :] = 1
        seeds = [
            subject.Seed(0, "a", np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])),
            subject.Seed(1, "b", np.asarray([0.0, 0.0, 2.0]), np.asarray([1.0, 0.0, 0.0])),
        ]
        instances, _ = subject._partition_foreground(mask, seeds, slab_depth=1)
        self.assertIn((0, 0, 1), {tuple(row) for row in instances[0].tolist()})
        self.assertNotIn((0, 0, 1), {tuple(row) for row in instances[1].tolist()})


class ProvenanceBoundaryTests(unittest.TestCase):
    def test_schema_less_seed_alias_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / "seeds.csv"
            seeds.write_text("lens_index\n0\n", encoding="utf-8")
            sidecar = root / "seeds.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "seed_role": (
                            "oracle_correspondence_and_distal_localization_stage1_only"
                        ),
                        "output_sha256": subject.sha256_file(seeds),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(subject.ExtractionError, "mapped-seed provenance"):
                subject._validate_seed_provenance(
                    sidecar,
                    seeds,
                    eye_id="M3_M_36_01",
                )

    def test_legacy_mask_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / "mask.npy"
            np.save(mask, np.ones((2, 2, 2), dtype=np.uint8), allow_pickle=False)
            sidecar = root / "mask.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": "experiment63-maike-mask-v1",
                        "array_sha256": subject.sha256_file(mask),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(subject.ExtractionError, "prepared mask/provenance"):
                subject._validate_mask(mask, sidecar)

    def test_staging_inventory_rejects_hidden_atomic_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "expected.txt").write_text("bound", encoding="utf-8")
            (staging / ".expected.txt.conflict").write_text("copy", encoding="utf-8")
            with self.assertRaisesRegex(subject.ExtractionError, "extra="):
                subject._require_exact_staging_files(staging, {"expected.txt"})

    def test_seed_provenance_must_bind_the_extractor_mask_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / "mask.npy"
            sidecar = root / "mask.json"
            mask.write_bytes(b"mask bytes")
            sidecar.write_bytes(b"sidecar bytes")
            provenance = {
                "input_hashes": {
                    "mask_npy": {
                        "sha256": subject.sha256_file(mask),
                        "size_bytes": mask.stat().st_size,
                    },
                    "mask_provenance": {
                        "sha256": "0" * 64,
                        "size_bytes": sidecar.stat().st_size,
                    },
                }
            }
            with self.assertRaisesRegex(
                subject.ExtractionError, "exact artifact bound by seed provenance"
            ):
                subject._require_seed_mask_cross_binding(
                    provenance,
                    mask_path=mask,
                    mask_provenance_path=sidecar,
                )
            provenance["input_hashes"]["mask_provenance"]["sha256"] = (
                subject.sha256_file(sidecar)
            )
            subject._require_seed_mask_cross_binding(
                provenance,
                mask_path=mask,
                mask_provenance_path=sidecar,
            )


class GridTests(unittest.TestCase):
    def test_experiment_57_grid_has_exact_contract(self) -> None:
        grid = subject._canonical_grid()
        self.assertEqual(grid.shape, (81, 2))
        self.assertTrue(np.all(np.sum(grid * grid, axis=1) <= 0.65**2 + 1e-12))


class TargetCohortTests(unittest.TestCase):
    @staticmethod
    def _geometry(distal_intercept_um: float) -> dict[str, object]:
        return {
            "origin_xyz_um": np.zeros(3, dtype=np.float64),
            "u_axis_xyz": np.asarray([1.0, 0.0, 0.0]),
            "v_axis_xyz": np.asarray([0.0, 1.0, 0.0]),
            "outward_axis_xyz": np.asarray([0.0, 0.0, 1.0]),
            "distal_scale_um": 3.25,
            "quadratic_beta_normalized": np.asarray(
                [distal_intercept_um, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
        }

    @staticmethod
    def _component(proximal_z: object = 0) -> np.ndarray:
        points: list[list[int]] = []
        for y in range(-2, 3):
            for x in range(-2, 3):
                z = int(proximal_z(x, y)) if callable(proximal_z) else int(proximal_z)
                points.extend([[z, y, x], [z + 2, y, x]])
        return np.asarray(points, dtype=np.int32)

    @staticmethod
    def _config(**updates: object) -> dict[str, object]:
        result = {
            **subject.DEFAULT_THRESHOLD_CONFIG,
            **subject.DEFAULT_PIPELINE_CONFIG,
        }
        result.update(updates)
        return result

    def test_spanning_bin_exact_tie_uses_physical_xyz_order(self) -> None:
        points_xyz = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]]
        )
        local_xyz = np.asarray(
            [[0.10, 0.10, 0.0], [-0.10, -0.10, 0.0], [0.0, 0.0, 0.650]]
        )
        selected = subject.select_spanning_lateral_bin_minima(
            points_xyz,
            local_xyz,
            scale_um=1.0,
            lateral_bin_um=0.325,
            min_axial_span_um=0.650,
        )
        np.testing.assert_array_equal(selected, [1])

    def test_negative_raw_q05_is_primary_resolvable_but_fails_sensitivity(self) -> None:
        result = subject._extract_target(
            self._component(
                lambda x, y: 2 if y == -2 and x in {-2, -1} else 0
            ),
            np.asarray([[0.0, 0.0, 1.0]]),
            self._geometry(0.5),
            self._config(),
        )
        self.assertTrue(result["target_resolvable"])
        self.assertFalse(result["target_qc"])
        self.assertLess(result["target_q05_raw_thickness_um"], 0.0)
        self.assertGreater(result["target_depth_um"], 0.0)
        self.assertEqual(result["target_resolvability_reasons"], [])
        self.assertIn("target_q05_raw_thickness_not_positive", result["target_qc_reasons"])

    def test_excessive_target_rmse_only_fails_sensitivity(self) -> None:
        result = subject._extract_target(
            self._component(lambda x, y: (x + y) & 1),
            np.asarray([[0.0, 0.0, 1.0]]),
            self._geometry(10.0),
            self._config(target_fit_rmse_max_um=0.01),
        )
        self.assertTrue(result["target_resolvable"])
        self.assertFalse(result["target_qc"])
        self.assertGreater(result["target_rmse_um"], 0.01)
        self.assertEqual(result["target_resolvability_reasons"], [])
        self.assertIn("target_rmse_above_maximum", result["target_qc_reasons"])

    def test_structurally_under_supported_target_is_not_resolvable(self) -> None:
        component = self._component()[:20]
        result = subject._extract_target(
            component,
            np.asarray([[0.0, 0.0, 1.0]]),
            self._geometry(10.0),
            self._config(),
        )
        self.assertFalse(result["target_resolvable"])
        self.assertFalse(result["target_qc"])
        self.assertTrue(math.isnan(result["target_q05_raw_thickness_um"]))
        self.assertIn(
            "target_support_below_minimum", result["target_resolvability_reasons"]
        )


class BundleIntegrationTests(unittest.TestCase):
    def test_atomic_bundle_has_matching_complete_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mask = np.zeros((80, 80, 80), dtype=np.uint8)
            eye_centre = np.asarray([10.0, 40.0, 40.0])
            centres: list[np.ndarray] = []
            zz, yy, xx = np.ogrid[:80, :80, :80]
            for ring, theta in enumerate((0.25, 0.50, 0.72)):
                for angle_index in range(6):
                    phi = 2.0 * math.pi * angle_index / 6.0 + ring * 0.13
                    centre = eye_centre + 45.0 * np.asarray(
                        [
                            math.cos(theta),
                            math.sin(theta) * math.sin(phi),
                            math.sin(theta) * math.cos(phi),
                        ]
                    )
                    centre = np.rint(centre).astype(np.int64)
                    centres.append(centre.astype(np.float64))
                    mask[
                        (zz - centre[0]) ** 2
                        + (yy - centre[1]) ** 2
                        + (xx - centre[2]) ** 2
                        <= 4**2
                    ] = 1
            mask_path = root / "mask.npy"
            np.save(mask_path, mask, allow_pickle=False)
            source_archive = root / "source.zip"
            source_archive.write_bytes(b"synthetic archive binding")
            mask_sha = subject.sha256_file(mask_path)
            data_sha = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
            archive_sha = subject.sha256_file(source_archive)
            mask_provenance = root / "mask.json"
            mask_provenance.write_text(
                json.dumps(
                    {
                        "schema_version": "maike-mask-provenance-v2",
                        "eye_id": "synthetic",
                        "axis_order": "zyx",
                        "spacing_um": [0.325] * 3,
                        "array_sha256": mask_sha,
                        "npy_sha256": mask_sha,
                        "array_data_sha256": data_sha,
                        "archive_sha256": archive_sha,
                        "source_archive": {
                            "path": str(source_archive),
                            "size_bytes": source_archive.stat().st_size,
                            "sha256": archive_sha,
                        },
                        "output": {
                            "path": str(mask_path),
                            "shape_zyx": list(mask.shape),
                            "sha256": mask_sha,
                            "data_sha256": data_sha,
                            "foreground_voxels": int(mask.sum()),
                        },
                    }
                ),
                encoding="utf-8",
            )
            seed_path = root / "seeds.csv"
            fields = [
                "lens_index", "seed_id", "source_z", "source_y", "source_x",
                "axis_source_z", "axis_source_y", "axis_source_x",
            ]
            with seed_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, centre in enumerate(centres):
                    axis = centre - eye_centre
                    axis /= np.linalg.norm(axis)
                    writer.writerow(
                        {
                            "lens_index": index,
                            "seed_id": f"s{index}",
                            "source_z": centre[0], "source_y": centre[1], "source_x": centre[2],
                            "axis_source_z": axis[0], "axis_source_y": axis[1], "axis_source_x": axis[2],
                        }
                    )
            seed_provenance = root / "seeds.json"
            seed_provenance.write_text(
                json.dumps(
                    {
                        "seed_role": "oracle_correspondence_and_distal_localization_stage1_only",
                        "seed_csv_sha256": subject.sha256_file(seed_path),
                    }
                ),
                encoding="utf-8",
            )
            thresholds = dict(subject.DEFAULT_THRESHOLD_CONFIG)
            thresholds.update(
                {
                    "distal_scale_min_um": 0.1,
                    "distal_scale_max_um": 10.0,
                    "min_sealed_distal_cap_points": 6,
                    "distal_fit_rmse_max_um": 100.0,
                    "quadratic_design_condition_max": 1.0e12,
                }
            )
            output = root / "bundle"
            validated_seed_provenance = {
                "eye_id": "synthetic",
                "n_expected": len(centres),
                "n_rows": len(centres),
                "candidate_seeds_per_voxel": 1,
                "seed_role": (
                    "oracle_correspondence_and_distal_localization_stage1_only"
                ),
                "input_hashes": {
                    "mask_npy": {
                        "sha256": subject.sha256_file(mask_path),
                        "size_bytes": mask_path.stat().st_size,
                    },
                    "mask_provenance": {
                        "sha256": subject.sha256_file(mask_provenance),
                        "size_bytes": mask_provenance.stat().st_size,
                    },
                },
            }
            with mock.patch.object(
                subject,
                "_validate_mask",
                return_value=(
                    np.load(mask_path, mmap_mode="r", allow_pickle=False),
                    json.loads(mask_provenance.read_text(encoding="utf-8")),
                ),
            ), mock.patch.object(
                subject,
                "_validate_seed_provenance",
                return_value=validated_seed_provenance,
            ):
                subject.build_eye_bundle(
                    mask_path=mask_path,
                    mask_provenance_path=mask_provenance,
                    seeds_path=seed_path,
                    seed_provenance_path=seed_provenance,
                    eye_id="synthetic",
                    species="test",
                    sex="F",
                    output_path=output,
                    threshold_config=thresholds,
                    allow_dirty=True,
                )
            completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "complete")
            self.assertTrue(completion["contiguous_indices"])
            self.assertEqual(completion["n_rows"], len(centres))
            self.assertTrue(completion["partition_evidence"]["exact_partition"])
            self.assertEqual(completion["output_manifest"], provenance["output_manifest"])
            self.assertEqual(len(completion["sealed_distal_stage1_manifest"]), len(centres))
            self.assertFalse(completion["instance_segmentation_validated"])
            self.assertEqual(
                set(completion["counts"]),
                {"stage1", "distal_qc", "target_resolvable", "target_qc"},
            )
            self.assertEqual(completion["counts"]["stage1"], len(centres))
            self.assertEqual(
                set(completion["input_hashes"]),
                {"mask_npy", "mask_provenance", "seed_csv", "seed_provenance"},
            )
            for key, path in {
                "mask_npy": mask_path,
                "mask_provenance": mask_provenance,
                "seed_csv": seed_path,
                "seed_provenance": seed_provenance,
            }.items():
                binding = completion["input_hashes"][key]
                self.assertEqual(binding["path"], str(path.resolve()))
                self.assertEqual(binding["size_bytes"], path.stat().st_size)
                self.assertEqual(binding["sha256"], subject.sha256_file(path))
            self.assertEqual(
                provenance["mask_source_provenance"],
                json.loads(mask_provenance.read_text(encoding="utf-8")),
            )
            self.assertEqual(
                provenance["seed_source_provenance"],
                validated_seed_provenance,
            )
            expected_manifest_paths = {
                "lens_summary.csv",
                "distal_qc_sampling.csv",
                "distal_frame_audit.json",
                *{
                    f"instances/lens_{index:06d}.npz"
                    for index in range(len(centres))
                },
                *{
                    f"sealed_distal/lens_{index:06d}.npz"
                    for index in range(len(centres))
                },
                *{
                    f"lenses/lens_{index:06d}.npz"
                    for index in range(len(centres))
                },
            }
            self.assertEqual(set(completion["output_manifest"]), expected_manifest_paths)
            for relative, binding in completion["output_manifest"].items():
                artifact = output / relative
                self.assertEqual(binding["size_bytes"], artifact.stat().st_size)
                self.assertEqual(binding["sha256"], subject.sha256_file(artifact))
            self.assertNotIn("completion.json", completion["output_manifest"])
            self.assertNotIn("provenance.json", completion["output_manifest"])


if __name__ == "__main__":
    unittest.main()

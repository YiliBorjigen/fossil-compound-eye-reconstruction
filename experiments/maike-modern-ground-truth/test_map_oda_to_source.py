from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import map_oda_to_source as mapper
import prepare_maike_masks as preparer
from map_oda_to_source import (
    CoordinateMappingError,
    SEED_ROLE,
    derive_oda_transform,
    load_and_validate_seed_provenance,
    load_transform,
    map_oda_to_source,
    oda_to_raw_um,
    oda_to_source_vox,
    raw_um_to_oda,
)
from prepare_maike_masks import prepare_mask_archive, sha256_file


_SYNTHETIC_SOURCE_CONTRACTS: dict[str, dict] = {}


def _tiff_bytes(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(buffer, format="TIFF")
    return buffer.getvalue()


def _write_source_bundle(root: Path, eye_id: str = "test_eye") -> tuple[Path, Path, Path]:
    archive = root / "source.zip"
    volume = np.zeros((8, 9, 10), dtype=np.uint8)
    # The non-zero crop origin and asymmetric target coordinates catch both
    # source-axis reversal and failure to subtract the tight-crop origin.
    volume[1, 2, 3] = 1
    volume[6, 7, 8] = 1
    volume[3, 5, 7] = 1
    volume[4, 6, 8] = 1
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for z_index, values in enumerate(volume, start=20):
            output.writestr(f"eye/slice{z_index:04d}.tif", _tiff_bytes(values))
    mask, provenance = root / "mask.npy", root / "mask.json"
    contract = {
        "name": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "slice_count": len(volume),
        "uncropped_shape_zyx": list(volume.shape),
    }
    _SYNTHETIC_SOURCE_CONTRACTS[eye_id] = contract
    with mock.patch.dict(preparer.EXPECTED_SOURCE_ARCHIVES, {eye_id: contract}):
        prepare_mask_archive(archive, mask, provenance, eye_id=eye_id)
    return archive, mask, provenance


def _proper_rotation() -> np.ndarray:
    return np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _raw_for_source(source_xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(source_xyz, dtype=np.float64) - 1.5) * 0.325


def _write_oda_inputs(
    root: Path,
    source_points_xyz: list[np.ndarray],
    *,
    eye_id: str = "test_eye",
    transform_updates: dict | None = None,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    center = np.zeros(3, dtype=np.float64)
    rotation = _proper_rotation()
    oda_csv = root / "oda.csv"
    with oda_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["label", "x", "y", "z", "anatomical_axis"]
        )
        writer.writeheader()
        for label, source in enumerate(source_points_xyz, start=11):
            raw = _raw_for_source(source)
            oda = raw_um_to_oda(raw, center, rotation)
            inward_oda = (-raw / np.linalg.norm(raw)) @ rotation
            writer.writerow(
                {
                    "label": label,
                    "x": format(oda[0], ".17g"),
                    "y": format(oda[1], ".17g"),
                    "z": format(oda[2], ".17g"),
                    "anatomical_axis": repr(inward_oda.tolist()),
                }
            )
    transform = {
        "schema_version": mapper.TRANSFORM_SCHEMA_VERSION,
        "eye_id": eye_id,
        "oda_commit": mapper.ODA_COMMIT,
        "derivation": "replayed_public_binned_stack_stage_1_and_2",
        "sphere_center_xyz_um": center.tolist(),
        "rotation": rotation.tolist(),
        "source_spacing_um": 0.325,
        "binned_spacing_um": 1.3,
        "bin_factor": 4,
        "source_block_center_offset_vox": 1.5,
        "input_axis_direction": "toward_eye_center",
        "oda_csv_sha256": sha256_file(oda_csv),
        "mask_npy_sha256": sha256_file(root / "mask.npy"),
        "mask_provenance_sha256": sha256_file(root / "mask.json"),
        "expected_foreground_hits": len(source_points_xyz),
    }
    if transform_updates:
        transform.update(transform_updates)
    transform_path = root / "transform.json"
    transform_path.write_text(json.dumps(transform), encoding="utf-8")
    return oda_csv, transform_path, center, rotation


def _map_synthetic(*args, **kwargs):
    """Exercise mapping math while strict artifact replay is tested separately."""

    with mock.patch.dict(
        preparer.EXPECTED_SOURCE_ARCHIVES, _SYNTHETIC_SOURCE_CONTRACTS
    ), mock.patch.object(mapper, "_validate_derived_transform_artifacts"):
        return map_oda_to_source(*args, **kwargs)


def _write_public_archive(root: Path, stack: Path) -> Path:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w", compression=zipfile.ZIP_DEFLATED) as inner:
        for path in sorted(stack.iterdir()):
            if path.is_file():
                inner.writestr(f"{stack.name}/{path.name}", path.read_bytes())
    outer_path = root / "fig3_share.zip"
    with zipfile.ZipFile(outer_path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr(mapper.FIG3_INNER_MEMBER, inner_buffer.getvalue())
    return outer_path


@contextmanager
def _synthetic_public_constants(archive: Path):
    with zipfile.ZipFile(archive, "r") as outer:
        inner_bytes = outer.read(mapper.FIG3_INNER_MEMBER)
    with mock.patch.multiple(
        mapper,
        FIG3_SHARE_SIZE_BYTES=archive.stat().st_size,
        FIG3_SHARE_MD5=hashlib.md5(
            archive.read_bytes(), usedforsecurity=False
        ).hexdigest(),
        FIG3_SHARE_SHA256=sha256_file(archive),
        FIG3_STACKS_ZIP_SHA256=hashlib.sha256(inner_bytes).hexdigest(),
    ), mock.patch.dict(
        preparer.EXPECTED_SOURCE_ARCHIVES, _SYNTHETIC_SOURCE_CONTRACTS
    ):
        yield


class MapOdaToSourceTests(unittest.TestCase):
    def test_real_m3_m36_strict_mapping_when_available(self) -> None:
        scratch_root = Path(__file__).resolve().parents[2].parent
        mapping_root = scratch_root / "maike_oda_mappings_v3" / "M3_M_36_01"
        if not (mapping_root / "seeds.json").is_file():
            self.skipTest("external finalized M3_M_36 mapping is unavailable")
        result = load_and_validate_seed_provenance(
            mapping_root / "seeds.json",
            seed_csv_path=mapping_root / "seeds.csv",
            eye_id="M3_M_36_01",
        )
        self.assertEqual(result["n_rows"], 970)
        self.assertEqual(result["n_foreground_hits"], 970)
        self.assertEqual(result["seed_role"], SEED_ROLE)
        self.assertEqual(
            result["seed_csv_sha256"],
            "4513627223bd6242025b9739be3736665f6dbfef9c8d0f1c7dfdcddf1d3bdd82",
        )

    def test_real_batch_manifest_seals_11369_hits_when_available(self) -> None:
        scratch_root = Path(__file__).resolve().parents[2].parent
        manifest_path = scratch_root / "maike_oda_mappings_v3" / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest("external finalized 12-eye mapping manifest is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], mapper.BATCH_SCHEMA_VERSION)
        self.assertEqual(manifest["n_eyes"], 12)
        self.assertEqual(manifest["total_expected_rows"], 11_369)
        self.assertEqual(manifest["total_mapped_rows"], 11_369)
        self.assertEqual(manifest["total_foreground_hits"], 11_369)
        self.assertEqual(
            {entry["eye_id"]: entry["foreground_hits"] for entry in manifest["entries"]},
            mapper.EXPECTED_ODA_COUNTS,
        )

    def test_derives_and_hash_binds_the_exact_oda_stage_2_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, mask, mask_provenance = _write_source_bundle(root)
            stack = root / "tiffs_test_eye_eye_lenses_binned"
            stack.mkdir()
            rows, columns = np.indices((20, 21))
            for stack_index in range(18):
                values = np.where(
                    (7 * rows + 11 * columns + 13 * stack_index) % 17 < 9,
                    200,
                    0,
                ).astype(np.uint8)
                Image.fromarray(values).save(
                    stack / f"slice{stack_index:04d}.tif", format="TIFF"
                )
            (stack / "_compound_eye_data.h5").write_bytes(b"")
            oda_csv = stack / "ommatidial_data.csv"
            with oda_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["label", "x", "y", "z", "anatomical_axis"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "label": 1,
                        "x": 1,
                        "y": 2,
                        "z": 3,
                        "anatomical_axis": "[-1, 0, 0]",
                    }
                )
            transform_path = root / "transform.json"
            public_archive = _write_public_archive(root, stack)

            with _synthetic_public_constants(public_archive):
                with self.assertRaisesRegex(CoordinateMappingError, "archive is required"):
                    derive_oda_transform(
                        stack,
                        transform_path,
                        eye_id="test_eye",
                        oda_csv_path=oda_csv,
                        mask_path=mask,
                        mask_provenance_path=mask_provenance,
                    )
                original_csv = oda_csv.read_bytes()
                oda_csv.write_bytes(original_csv + b"substitution")
                with self.assertRaisesRegex(CoordinateMappingError, "differs"):
                    derive_oda_transform(
                        stack,
                        transform_path,
                        eye_id="test_eye",
                        oda_csv_path=oda_csv,
                        mask_path=mask,
                        mask_provenance_path=mask_provenance,
                        public_archive_path=public_archive,
                    )
                oda_csv.write_bytes(original_csv)
                result = derive_oda_transform(
                    stack,
                    transform_path,
                    eye_id="test_eye",
                    oda_csv_path=oda_csv,
                    mask_path=mask,
                    mask_provenance_path=mask_provenance,
                    public_archive_path=public_archive,
                )

            self.assertEqual(result["oda_commit"][:7], "55684a9")
            self.assertEqual(result["oda_binned_stack"]["file_count"], 18)
            self.assertEqual(result["oda_binned_stack"]["image_shape_yx"], [20, 21])
            self.assertEqual(
                result["oda_binned_stack"]["prefilter_threshold_inclusive"], 128
            )
            self.assertEqual(result["oda_binned_stack"]["sphere_sample_stride"], 100)
            self.assertEqual(result["public_h5"]["status"], "zero_byte_public_placeholder")
            np.testing.assert_allclose(
                result["sphere_center_xyz_um"],
                [9.603465046488944, 11.711381727019155, 11.792144548338442],
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result["rotation"],
                [
                    [0.1888063388408169, -0.9820143412463621, 0.0],
                    [0.06963275455520676, 0.013387895572160014, -0.9974828538602573],
                    [0.9795424676381218, 0.1883310856938448, 0.07090808314146371],
                ],
                rtol=0.0,
                atol=1e-12,
            )
            with _synthetic_public_constants(public_archive):
                loaded, _, _, _ = load_transform(transform_path, eye_id="test_eye")
                self.assertEqual(loaded, result)

                for mutation, expected in [
                    (("schema_version", None), "schema must be exactly"),
                    (("public_archive", None), "public archive binding"),
                    (("oda_csv_sha256", None), "ODA CSV"),
                    (("sphere_center_xyz_um", [0.0, 0.0, 0.0]), "does not replay"),
                    (("rotation", np.eye(3).tolist()), "does not replay"),
                ]:
                    with self.subTest(mutation=mutation):
                        changed = dict(result)
                        key, replacement = mutation
                        if replacement is None:
                            changed.pop(key)
                        else:
                            changed[key] = replacement
                        transform_path.write_text(json.dumps(changed), encoding="utf-8")
                        with self.assertRaisesRegex(CoordinateMappingError, expected):
                            load_transform(transform_path, eye_id="test_eye")
                transform_path.write_text(json.dumps(result), encoding="utf-8")

                first_tiff = sorted(stack.glob("*.tif"))[0]
                first_tiff.write_bytes(first_tiff.read_bytes() + b"tamper")
                with self.assertRaisesRegex(CoordinateMappingError, "does not match"):
                    load_transform(transform_path, eye_id="test_eye")

    def test_exact_contract_and_hash_bound_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, mask, mask_provenance = _write_source_bundle(root)
            oda_csv, transform, center, rotation = _write_oda_inputs(
                root,
                [np.asarray([3.0, 5.0, 7.0]), np.asarray([4.0, 6.0, 8.0])],
            )
            output_csv, output_json = root / "seeds.csv", root / "seeds.json"

            result = _map_synthetic(
                oda_csv,
                transform,
                mask,
                mask_provenance,
                output_csv,
                output_json,
                eye_id="test_eye",
                expected_count=2,
            )

            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["lens_index"] for row in rows], ["0", "1"])
            self.assertEqual([row["seed_id"] for row in rows], ["11", "12"])
            self.assertEqual(
                [float(rows[0][key]) for key in ("source_x", "source_y", "source_z")],
                [4.0, 3.0, 2.0],
            )
            raw_axis_zyx = _raw_for_source(np.asarray([3.0, 5.0, 7.0]))
            expected_axis_zyx = raw_axis_zyx / np.linalg.norm(raw_axis_zyx)
            np.testing.assert_allclose(
                [float(rows[0][key]) for key in
                 ("axis_source_z", "axis_source_y", "axis_source_x")],
                expected_axis_zyx,
                rtol=0.0,
                atol=1e-14,
            )
            self.assertEqual(result["seed_role"], SEED_ROLE)
            self.assertEqual(result["n_expected"], 2)
            self.assertEqual(result["n_rows"], 2)
            self.assertEqual(result["seed_csv_sha256"], sha256_file(output_csv))
            self.assertEqual(json.loads(output_json.read_text()), result)
            self.assertTrue(result["checks"]["all_rounded_centres_on_foreground"])
            self.assertEqual(result["coordinate_contract"]["bin_factor"], 4)

            with mock.patch.dict(mapper.EXPECTED_ODA_COUNTS, {"test_eye": 2}), mock.patch.object(
                mapper, "_validate_derived_transform_artifacts"
            ):
                validated = load_and_validate_seed_provenance(
                    output_json, seed_csv_path=output_csv, eye_id="test_eye"
                )
            self.assertEqual(validated, result)

            downgraded = dict(result)
            downgraded["schema_version"] = "legacy"
            output_json.write_text(json.dumps(downgraded), encoding="utf-8")
            with mock.patch.dict(mapper.EXPECTED_ODA_COUNTS, {"test_eye": 2}):
                with self.assertRaisesRegex(CoordinateMappingError, "schema_version"):
                    load_and_validate_seed_provenance(
                        output_json, seed_csv_path=output_csv, eye_id="test_eye"
                    )

            source = np.asarray([[2.0, 3.0, 4.0]])
            raw = _raw_for_source(source)
            oda = raw_um_to_oda(raw, center, rotation)
            np.testing.assert_allclose(oda_to_raw_um(oda, center, rotation), raw)
            np.testing.assert_allclose(oda_to_source_vox(oda, center, rotation), source)

    def test_rejects_truncated_frozen_eye_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, mask, mask_provenance = _write_source_bundle(root, eye_id="M3_M_36_01")
            oda_csv, transform, _, _ = _write_oda_inputs(
                root,
                [np.asarray([3.0, 5.0, 7.0])],
                eye_id="M3_M_36_01",
            )
            with self.assertRaisesRegex(CoordinateMappingError, "expected exactly 970"):
                _map_synthetic(
                    oda_csv,
                    transform,
                    mask,
                    mask_provenance,
                    root / "seeds.csv",
                    root / "seeds.json",
                    eye_id="M3_M_36_01",
                )

    def test_rejects_bad_rotation_or_frozen_constants(self) -> None:
        for update, expected in [
            ({"rotation": [[1, 0, 0], [0, 2, 0], [0, 0, 1]]}, "proper orthonormal"),
            ({"bin_factor": 2}, "bin factor"),
        ]:
            with self.subTest(update=update), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, mask, mask_provenance = _write_source_bundle(root)
                oda_csv, transform, _, _ = _write_oda_inputs(
                    root, [np.asarray([3.0, 5.0, 7.0])], transform_updates=update
                )
                with self.assertRaisesRegex(CoordinateMappingError, expected):
                    _map_synthetic(
                        oda_csv,
                        transform,
                        mask,
                        mask_provenance,
                        root / "seeds.csv",
                        root / "seeds.json",
                        eye_id="test_eye",
                        expected_count=1,
                    )

    def test_rejects_input_hash_mismatch_and_eye_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, mask, mask_provenance = _write_source_bundle(root)
            oda_csv, transform, _, _ = _write_oda_inputs(
                root,
                [np.asarray([3.0, 5.0, 7.0])],
                transform_updates={"oda_csv_sha256": "0" * 64},
            )
            with self.assertRaisesRegex(CoordinateMappingError, "binding"):
                _map_synthetic(
                    oda_csv,
                    transform,
                    mask,
                    mask_provenance,
                    root / "seeds.csv",
                    root / "seeds.json",
                    eye_id="test_eye",
                    expected_count=1,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, mask, mask_provenance = _write_source_bundle(root)
            oda_csv, transform, _, _ = _write_oda_inputs(
                root, [np.asarray([3.0, 5.0, 7.0])], eye_id="different"
            )
            with self.assertRaisesRegex(CoordinateMappingError, "transform eye_id"):
                _map_synthetic(
                    oda_csv,
                    transform,
                    mask,
                    mask_provenance,
                    root / "seeds.csv",
                    root / "seeds.json",
                    eye_id="test_eye",
                    expected_count=1,
                )

    def test_rejects_seed_off_foreground_and_wrong_axis_direction(self) -> None:
        cases = [
            (np.asarray([3.0, 5.0, 6.0]), {}, "foreground"),
            (
                np.asarray([3.0, 5.0, 7.0]),
                {"input_axis_direction": "away_from_eye_center"},
                "input axis direction",
            ),
        ]
        for point, update, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, mask, mask_provenance = _write_source_bundle(root)
                oda_csv, transform, _, _ = _write_oda_inputs(
                    root, [point], transform_updates=update
                )
                with self.assertRaisesRegex(CoordinateMappingError, expected):
                    _map_synthetic(
                        oda_csv,
                        transform,
                        mask,
                        mask_provenance,
                        root / "seeds.csv",
                        root / "seeds.json",
                        eye_id="test_eye",
                        expected_count=1,
                    )

    def test_rejects_duplicate_labels_and_malformed_axis(self) -> None:
        for axis_value, labels, expected in [
            ("[1, 2]", [1], "length-3"),
            ("[1, 2, 3]", [1, 1], "duplicate"),
        ]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, mask, mask_provenance = _write_source_bundle(root)
                oda_csv, transform, _, _ = _write_oda_inputs(
                    root,
                    [np.asarray([3.0, 5.0, 7.0]) for _ in labels],
                )
                with oda_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["label", "x", "y", "z", "anatomical_axis"],
                    )
                    writer.writeheader()
                    for label in labels:
                        writer.writerow(
                            {
                                "label": label,
                                "x": 0.1625,
                                "y": 0.1625,
                                "z": 0.1625,
                                "anatomical_axis": axis_value,
                            }
                        )
                transform_value = json.loads(transform.read_text(encoding="utf-8"))
                transform_value["oda_csv_sha256"] = sha256_file(oda_csv)
                transform.write_text(json.dumps(transform_value), encoding="utf-8")
                with self.assertRaisesRegex(CoordinateMappingError, expected):
                    _map_synthetic(
                        oda_csv,
                        transform,
                        mask,
                        mask_provenance,
                        root / "seeds.csv",
                        root / "seeds.json",
                        eye_id="test_eye",
                        expected_count=len(labels),
                    )


if __name__ == "__main__":
    unittest.main()

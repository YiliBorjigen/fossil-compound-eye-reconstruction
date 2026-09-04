from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import prepare_maike_masks as preparer
from prepare_maike_masks import (
    EXPECTED_SOURCE_ARCHIVES,
    MaskPreparationError,
    infer_eye_id,
    load_and_validate_mask_provenance,
    prepare_mask_archive,
)


def _tiff_bytes(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(buffer, format="TIFF")
    return buffer.getvalue()


def _write_archive(
    path: Path,
    slices: list[tuple[int, np.ndarray]],
    *,
    extra_member: tuple[str, bytes] | None = None,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, values in slices:
            archive.writestr(f"nested/eye_lenses{index:04d}.tif", _tiff_bytes(values))
        if extra_member is not None:
            archive.writestr(*extra_member)


@contextmanager
def _synthetic_archive_contract(archive: Path, eye_id: str):
    with zipfile.ZipFile(archive, "r") as source:
        tiff_infos = [
            info
            for info in source.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in {".tif", ".tiff"}
        ]
        with source.open(tiff_infos[0]) as stream, Image.open(stream) as image:
            height, width = np.asarray(image).shape
    contract = {
        "name": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": preparer.sha256_file(archive),
        "slice_count": len(tiff_infos),
        "uncropped_shape_zyx": [len(tiff_infos), int(height), int(width)],
    }
    with mock.patch.dict(preparer.EXPECTED_SOURCE_ARCHIVES, {eye_id: contract}):
        yield contract


def _prepare_synthetic(
    archive: Path,
    output: Path,
    sidecar: Path,
    *,
    eye_id: str | None = None,
    **kwargs,
) -> dict:
    resolved_eye_id = eye_id or infer_eye_id(archive)
    with _synthetic_archive_contract(archive, resolved_eye_id):
        return prepare_mask_archive(
            archive, output, sidecar, eye_id=eye_id, **kwargs
        )


class PrepareMaikeMasksTests(unittest.TestCase):
    def test_uploaded_m3_m_36_pilot_matches_frozen_reference_when_available(self) -> None:
        """Seal the real pilot facts without making the external data a CI fixture."""

        scratch_root = Path(__file__).resolve().parents[2].parent
        archive = (
            scratch_root
            / "upload"
            / "tiffs_M3_M_36_01_eye_lenses-20260903T135121Z-1-001.zip"
        )
        mask = scratch_root / "maike_pilot_mask" / "M3_M_36_01.mask.uint8.npy"
        sidecar = scratch_root / "maike_pilot_mask" / "M3_M_36_01.mask.json"
        if not all(path.is_file() for path in (archive, mask, sidecar)):
            self.skipTest("Maike's external M3_M_36_01 pilot bundle is not available")

        result = load_and_validate_mask_provenance(
            sidecar, mask_path=mask, archive_path=archive
        )
        self.assertEqual(result["uncropped"]["shape_zyx"], [684, 1659, 1569])
        self.assertEqual(result["crop"]["origin_zyx"], [0, 168, 186])
        self.assertEqual(result["crop"]["upper_inclusive_zyx"], [682, 1497, 1384])
        self.assertEqual(result["crop"]["shape_zyx"], [683, 1330, 1199])
        self.assertEqual(result["output"]["foreground_voxels"], 36_009_405)
        self.assertEqual(result["output"]["size_bytes"], 1_089_159_738)
        self.assertEqual(
            result["array_sha256"],
            "3dbcc43a95c8e41b05fa6363671c633690aa518628d9c9e156ee4430de521114",
        )
        self.assertEqual(
            result["array_data_sha256"],
            "2abbb6be6cb67a8c1638aa82f5231508ae750f415e337c5127ff1a8e6abcce95",
        )

    def test_builds_tight_lossless_crop_and_hash_bound_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "tiffs_M3_F_24_01_eye_lenses-20260903T135137Z-1-001.zip"
            slices = []
            expected = np.zeros((4, 6, 7), dtype=np.uint8)
            expected[1, 2:4, 1:5] = 1
            expected[2, 1:5, 2:6] = 1
            for z_index, values in enumerate(expected, start=18):
                slices.append((z_index, values))
            _write_archive(archive, slices)
            output = root / "mask.npy"
            sidecar = root / "mask.json"

            result = _prepare_synthetic(archive, output, sidecar)

            observed = np.load(output, allow_pickle=False)
            np.testing.assert_array_equal(observed, expected[1:3, 1:5, 1:6])
            self.assertEqual(observed.dtype, np.uint8)
            self.assertEqual(result["crop"]["origin_zyx"], [1, 1, 1])
            self.assertEqual(result["crop"]["shape_zyx"], [2, 4, 5])
            self.assertEqual(result["uncropped"]["shape_zyx"], [4, 6, 7])
            self.assertEqual(result["source_slices"]["first_numeric_index"], 18)
            self.assertEqual(result["source_slices"]["last_numeric_index"], 21)
            self.assertEqual(result["spacing_um"], [0.325, 0.325, 0.325])
            self.assertEqual(result["eye_id"], "M3_F_24_01")
            self.assertEqual(
                result["array_data_sha256"],
                hashlib.sha256(observed.tobytes(order="C")).hexdigest(),
            )
            self.assertEqual(json.loads(sidecar.read_text()), result)
            with _synthetic_archive_contract(archive, "M3_F_24_01"):
                validated = load_and_validate_mask_provenance(sidecar)
            self.assertEqual(validated["array_sha256"], result["array_sha256"])
            self.assertEqual(
                list(root.glob(".mask.npy*")),
                [],
                "large NPY staging files must not remain beside the published mask",
            )

    def test_rejects_non_contiguous_numeric_slices_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            binary = np.asarray([[0, 1], [0, 0]], dtype=np.uint8)
            _write_archive(archive, [(7, binary), (9, binary)])
            output, sidecar = root / "mask.npy", root / "mask.json"
            with self.assertRaisesRegex(MaskPreparationError, "not contiguous"):
                _prepare_synthetic(archive, output, sidecar, eye_id="bad")
            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())

    def test_rejects_nonbinary_or_heterogeneous_slices(self) -> None:
        cases = [
            (
                [(1, np.asarray([[0, 2]], dtype=np.uint8))],
                "outside binary",
            ),
            (
                [
                    (1, np.asarray([[0, 1]], dtype=np.uint8)),
                    (2, np.asarray([[0], [1]], dtype=np.uint8)),
                ],
                "not homogeneous",
            ),
        ]
        for slices, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "bad.zip"
                _write_archive(archive, slices)
                with self.assertRaisesRegex(MaskPreparationError, message):
                    _prepare_synthetic(
                        archive, root / "mask.npy", root / "mask.json", eye_id="bad"
                    )

    def test_rejects_non_tiff_members_and_empty_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            _write_archive(
                archive,
                [(1, np.asarray([[0, 1]], dtype=np.uint8))],
                extra_member=("notes.txt", b"not part of a TIFF stack"),
            )
            with self.assertRaisesRegex(MaskPreparationError, "non-TIFF"):
                _prepare_synthetic(
                    archive, root / "mask.npy", root / "mask.json", eye_id="bad"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "empty.zip"
            _write_archive(archive, [(1, np.zeros((2, 2), dtype=np.uint8))])
            with self.assertRaisesRegex(MaskPreparationError, "no foreground"):
                _prepare_synthetic(
                    archive, root / "mask.npy", root / "mask.json", eye_id="bad"
                )

    def test_requires_exact_original_spacing_and_protects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "good.zip"
            _write_archive(archive, [(1, np.asarray([[0, 1]], dtype=np.uint8))])
            with self.assertRaisesRegex(MaskPreparationError, "exactly 0.325"):
                _prepare_synthetic(
                    archive,
                    root / "mask.npy",
                    root / "mask.json",
                    eye_id="test",
                    spacing_um=1.3,
                )
            output = root / "mask.npy"
            output.write_bytes(b"owned")
            with self.assertRaisesRegex(MaskPreparationError, "already exists"):
                _prepare_synthetic(archive, output, root / "mask.json", eye_id="test")
            self.assertEqual(output.read_bytes(), b"owned")

    def test_validation_detects_mask_and_archive_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "good.zip"
            _write_archive(archive, [(1, np.asarray([[0, 1]], dtype=np.uint8))])
            output, sidecar = root / "mask.npy", root / "mask.json"
            with _synthetic_archive_contract(archive, "test"):
                prepare_mask_archive(archive, output, sidecar, eye_id="test")

                provenance = json.loads(sidecar.read_text(encoding="utf-8"))
                provenance["output"]["foreground_voxels"] += 1
                sidecar.write_text(json.dumps(provenance), encoding="utf-8")
                with self.assertRaisesRegex(MaskPreparationError, "foreground count"):
                    load_and_validate_mask_provenance(sidecar)

                prepare_mask_archive(
                    archive, output, sidecar, eye_id="test", overwrite=True
                )
                provenance = json.loads(sidecar.read_text(encoding="utf-8"))
                provenance["output"]["size_bytes"] += 1
                sidecar.write_text(json.dumps(provenance), encoding="utf-8")
                with self.assertRaisesRegex(MaskPreparationError, "mask NPY does not match"):
                    load_and_validate_mask_provenance(sidecar)

                prepare_mask_archive(
                    archive, output, sidecar, eye_id="test", overwrite=True
                )
                output.write_bytes(output.read_bytes() + b"tamper")
                with self.assertRaisesRegex(MaskPreparationError, "does not match"):
                    load_and_validate_mask_provenance(sidecar)

                prepare_mask_archive(
                    archive, output, sidecar, eye_id="test", overwrite=True
                )
                with archive.open("ab") as handle:
                    handle.write(b"tamper")
                with self.assertRaisesRegex(MaskPreparationError, "archive"):
                    load_and_validate_mask_provenance(sidecar)

    def test_rejects_self_consistent_substitute_for_a_frozen_eye(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / EXPECTED_SOURCE_ARCHIVES["M3_F_24_01"]["name"]
            _write_archive(archive, [(1, np.asarray([[0, 1]], dtype=np.uint8))])
            with self.assertRaisesRegex(MaskPreparationError, "frozen contract"):
                prepare_mask_archive(
                    archive,
                    root / "mask.npy",
                    root / "mask.json",
                    eye_id="M3_F_24_01",
                )

    def test_all_frozen_source_contracts_and_real_archives_when_available(self) -> None:
        canonical = json.dumps(
            EXPECTED_SOURCE_ARCHIVES, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(len(EXPECTED_SOURCE_ARCHIVES), 12)
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "75a9c3c71e39e64a2effa6b50ffc7733ee7ad92e1a4e39730ddc973574cc7aad",
        )

        scratch_root = Path(__file__).resolve().parents[2].parent
        upload_root = scratch_root / "upload"
        archives = [upload_root / value["name"] for value in EXPECTED_SOURCE_ARCHIVES.values()]
        if not all(path.is_file() for path in archives):
            self.skipTest("the 12 external Maike source archives are unavailable")
        for eye_id, contract in EXPECTED_SOURCE_ARCHIVES.items():
            archive = upload_root / contract["name"]
            self.assertEqual(archive.stat().st_size, contract["size_bytes"], eye_id)
            self.assertEqual(preparer.sha256_file(archive), contract["sha256"], eye_id)
            with zipfile.ZipFile(archive, "r") as source:
                members = [
                    info
                    for info in source.infolist()
                    if not info.is_dir()
                    and Path(info.filename).suffix.lower() in {".tif", ".tiff"}
                ]
                self.assertEqual(len(members), contract["slice_count"], eye_id)
                with source.open(members[0]) as stream, Image.open(stream) as image:
                    self.assertEqual(
                        [len(members), image.height, image.width],
                        contract["uncropped_shape_zyx"],
                        eye_id,
                    )

    def test_eye_id_inference_is_strict(self) -> None:
        self.assertEqual(
            infer_eye_id(
                Path("tiffs_RED3_25_M_28_eye_lenses-20260903T135127Z-1-001.zip")
            ),
            "RED3_25_M_28",
        )
        with self.assertRaises(MaskPreparationError):
            infer_eye_id(Path("unknown.zip"))


if __name__ == "__main__":
    unittest.main()

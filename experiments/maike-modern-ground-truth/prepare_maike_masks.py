#!/usr/bin/env python3
"""Validate Maike Kittelmann's TIFF archives and build cropped binary masks.

The supplied stacks contain one numbered TIFF per original 0.325 micrometre
slice.  This module never expands a complete eye in memory: it validates the
archive once to find the occupied bounding box, then writes that box directly
to an atomic, memory-mapped ``uint8`` NPY file.  The v2 sidecar binds both the
source archive and the exact NPY bytes used by Experiment 63.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "maike-mask-provenance-v2"
ORIGINAL_SPACING_UM = 0.325
AXIS_ORDER = "zyx"
_TIFF_SUFFIXES = {".tif", ".tiff"}
_TRAILING_INTEGER = re.compile(r"(\d+)$")

# Frozen identities of the twelve archives supplied by Maike Kittelmann.  A
# self-consistent replacement ZIP is not an admissible Experiment 63 input.
EXPECTED_SOURCE_ARCHIVES: dict[str, dict[str, Any]] = {
    "M3_F_24_01": {"name": "tiffs_M3_F_24_01_eye_lenses-20260903T135137Z-1-001.zip", "size_bytes": 8_656_592, "sha256": "a1af8a4b3f04f9b22f4f416854189072b14dd59b13b6708048d4766580e4a638", "slice_count": 864, "uncropped_shape_zyx": [864, 1635, 1740]},
    "M3_F_28_03": {"name": "tiffs_M3_F_28_03_eye_lenses-20260903T135112Z-1-001.zip", "size_bytes": 7_929_219, "sha256": "d18763929208c1fff12ada55124e1f489cd420ec66f101ebb9309656fce38ffa", "slice_count": 875, "uncropped_shape_zyx": [875, 1470, 1914]},
    "M3_F_35_03": {"name": "tiffs_M3_F_35_03_eye_lenses-20260903T135109Z-1-001.zip", "size_bytes": 8_363_277, "sha256": "6a205f1d691e88829861fd56d16269b4c6867d1ae7bd3f14ca558279b18ccef0", "slice_count": 1031, "uncropped_shape_zyx": [1031, 1425, 1602]},
    "M3_M_26_01": {"name": "tiffs_M3_M_26_01_eye_lenses-20260903T135105Z-1-001.zip", "size_bytes": 6_172_316, "sha256": "69b0206668e185d333f5fcdf8302a85cec048e4bb39ad93cd54001d5ec415f37", "slice_count": 694, "uncropped_shape_zyx": [694, 1575, 1689]},
    "M3_M_32_01": {"name": "tiffs_M3_M_32_01_eye_lenses-20260903T135052Z-1-001.zip", "size_bytes": 6_382_619, "sha256": "adc4aa40945bcba20956679262b211444ec307391361f775e15ab4555e797d51", "slice_count": 714, "uncropped_shape_zyx": [714, 1587, 1485]},
    "M3_M_36_01": {"name": "tiffs_M3_M_36_01_eye_lenses-20260903T135121Z-1-001.zip", "size_bytes": 6_859_529, "sha256": "024cb4ba7de8c56a98fa96bf881b67d94cdb49a2d0c43160cfdffe8b9a30f2bc", "slice_count": 684, "uncropped_shape_zyx": [684, 1659, 1569]},
    "RED3_25_F_36": {"name": "tiffs_RED3_25_F_36_eye_lenses-20260903T135103Z-1-001.zip", "size_bytes": 11_412_517, "sha256": "9208899f9c2481ebbe43b760b3efe6dfb60cfeb8b3b7fe156c8ee7884feed495", "slice_count": 1158, "uncropped_shape_zyx": [1158, 1794, 2004]},
    "RED3_25_F_37": {"name": "tiffs_RED3_25_F_37_eye_lenses-20260903T135100Z-1-001.zip", "size_bytes": 9_907_754, "sha256": "1d230daea5bacacc92f5ab5aec4da13c1d4b10e7a2aedc554ed3baf7cc0e3012", "slice_count": 1172, "uncropped_shape_zyx": [1172, 1449, 2079]},
    "RED3_25_F_38": {"name": "tiffs_RED3_25_F_38_eye_lenses-20260903T135116Z-1-001.zip", "size_bytes": 10_740_551, "sha256": "b735f6767029f512ba094a7ffeb4c1d46a3cc04039022890cce1f54b4e8a6b57", "slice_count": 1135, "uncropped_shape_zyx": [1135, 1845, 1542]},
    "RED3_25_M_26": {"name": "tiffs_RED3_25_M_26_eye_lenses-20260903T135055Z-1-001.zip", "size_bytes": 8_599_547, "sha256": "9c1ab44205610e940617e19249df8c5364a0b88115ff736b3c6a3952bd63b3de", "slice_count": 1040, "uncropped_shape_zyx": [1040, 1620, 1560]},
    "RED3_25_M_27": {"name": "tiffs_RED3_25_M_27_eye_lenses-20260903T135132Z-1-001.zip", "size_bytes": 9_110_763, "sha256": "5997775e6366fc4c268eb1cc65e467b101cc07ecaad2c0d8d4363bd16729d4ef", "slice_count": 1252, "uncropped_shape_zyx": [1252, 1458, 1365]},
    "RED3_25_M_28": {"name": "tiffs_RED3_25_M_28_eye_lenses-20260903T135127Z-1-001.zip", "size_bytes": 8_047_098, "sha256": "d3492e58cde9a6182729f0c9803fcc368f3bb7debdeb798d6e633f03febc40ce", "slice_count": 1283, "uncropped_shape_zyx": [1283, 1062, 1698]},
}


class MaskPreparationError(RuntimeError):
    """Raised when a source archive or a generated mask fails validation."""


@dataclass(frozen=True)
class SliceMember:
    index: int
    name: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without reading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_names_sha256(members: Sequence[SliceMember]) -> str:
    payload = "".join(f"{member.index}\t{member.name}\n" for member in members)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _numeric_tiff_members(archive: zipfile.ZipFile) -> list[SliceMember]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise MaskPreparationError("ZIP contains duplicate member names")

    members: list[SliceMember] = []
    for info in infos:
        if info.is_dir():
            continue
        suffix = Path(info.filename).suffix.lower()
        if suffix not in _TIFF_SUFFIXES:
            raise MaskPreparationError(
                f"ZIP contains a non-TIFF file: {info.filename!r}"
            )
        if info.flag_bits & 0x1:
            raise MaskPreparationError(f"encrypted TIFF is not supported: {info.filename!r}")
        match = _TRAILING_INTEGER.search(Path(info.filename).stem)
        if match is None:
            raise MaskPreparationError(
                f"TIFF filename lacks a trailing numeric slice index: {info.filename!r}"
            )
        members.append(SliceMember(index=int(match.group(1)), name=info.filename))

    if not members:
        raise MaskPreparationError("ZIP contains no TIFF slices")
    members.sort(key=lambda member: (member.index, member.name))
    indices = [member.index for member in members]
    if len(indices) != len(set(indices)):
        raise MaskPreparationError("numeric TIFF slice indices are not unique")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected).difference(indices))
        preview = ", ".join(str(value) for value in missing[:10])
        raise MaskPreparationError(f"numeric TIFF slices are not contiguous; missing {preview}")
    return members


def _read_binary_slice(archive: zipfile.ZipFile, member: SliceMember) -> np.ndarray:
    try:
        with archive.open(member.name, "r") as stream:
            with Image.open(stream) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise MaskPreparationError(
                        f"TIFF must contain one frame: {member.name!r}"
                    )
                image.load()
                values = np.asarray(image)
    except (OSError, UnidentifiedImageError, zipfile.BadZipFile) as exc:
        raise MaskPreparationError(f"cannot decode TIFF {member.name!r}: {exc}") from exc

    if values.ndim != 2:
        raise MaskPreparationError(
            f"TIFF must be a single-channel 2-D image: {member.name!r}"
        )
    if np.any((values != 0) & (values != 1)):
        invalid = np.unique(values[(values != 0) & (values != 1)])
        preview = ", ".join(str(value) for value in invalid[:8])
        raise MaskPreparationError(
            f"TIFF contains values outside binary {{0,1}}: {member.name!r} ({preview})"
        )
    return np.asarray(values, dtype=np.uint8)


def _validate_spacing(spacing_um: float) -> None:
    if not math.isfinite(spacing_um) or not math.isclose(
        spacing_um, ORIGINAL_SPACING_UM, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MaskPreparationError(
            f"source spacing must be exactly {ORIGINAL_SPACING_UM} um isotropic"
        )


def infer_eye_id(archive_path: Path) -> str:
    """Infer the stable eye identifier from a supplied archive filename."""

    stem = archive_path.stem
    match = re.fullmatch(
        r"tiffs_(?P<eye>.+?)_eye_lenses(?:-\d{8}T\d{6}Z(?:-\d+-\d+)?)?",
        stem,
    )
    if match is None:
        raise MaskPreparationError(
            "cannot infer eye ID; use --eye-id for an archive outside the supplied naming scheme"
        )
    eye_id = match.group("eye")
    if not eye_id or any(character in eye_id for character in "/\\"):
        raise MaskPreparationError("invalid inferred eye ID")
    return eye_id


def _scan_archive(
    archive_path: Path,
) -> tuple[
    list[SliceMember],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    int,
    list[int],
]:
    """Validate all slices and return shape, occupied bounds, and counts."""

    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise MaskPreparationError(f"cannot open ZIP {archive_path}: {exc}") from exc

    with archive:
        members = _numeric_tiff_members(archive)
        height: int | None = None
        width: int | None = None
        lower = np.asarray([len(members), np.iinfo(np.int64).max, np.iinfo(np.int64).max])
        upper = np.asarray([-1, -1, -1])
        foreground_voxels = 0
        observed_values: set[int] = set()

        for z_position, member in enumerate(members):
            values = _read_binary_slice(archive, member)
            if height is None:
                height, width = (int(values.shape[0]), int(values.shape[1]))
            elif values.shape != (height, width):
                raise MaskPreparationError(
                    "TIFF dimensions are not homogeneous: "
                    f"{member.name!r} is {values.shape}, expected {(height, width)}"
                )
            if np.any(values == 0):
                observed_values.add(0)
            positions = np.argwhere(values == 1)
            if positions.size:
                observed_values.add(1)
                foreground_voxels += int(len(positions))
                lower = np.minimum(
                    lower,
                    np.asarray(
                        [z_position, int(positions[:, 0].min()), int(positions[:, 1].min())]
                    ),
                )
                upper = np.maximum(
                    upper,
                    np.asarray(
                        [z_position, int(positions[:, 0].max()), int(positions[:, 1].max())]
                    ),
                )

    if height is None or width is None:  # guarded by _numeric_tiff_members
        raise MaskPreparationError("ZIP contains no readable TIFF slices")
    if foreground_voxels == 0:
        raise MaskPreparationError("binary stack contains no foreground voxels")
    uncropped_shape = (len(members), height, width)
    return (
        members,
        (height, width),
        uncropped_shape,
        (int(lower[0]), int(lower[1]), int(lower[2])),
        (int(upper[0]), int(upper[1]), int(upper[2])),
        foreground_voxels,
        sorted(observed_values),
    )


def prepare_mask_archive(
    archive_path: Path,
    output_path: Path,
    provenance_path: Path,
    *,
    eye_id: str | None = None,
    spacing_um: float = ORIGINAL_SPACING_UM,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one cropped memory-mapped mask and its hash-bound v2 sidecar."""

    archive_path = Path(archive_path)
    output_path = Path(output_path)
    provenance_path = Path(provenance_path)
    _validate_spacing(float(spacing_um))
    if not archive_path.is_file():
        raise MaskPreparationError(f"source archive does not exist: {archive_path}")
    resolved_eye_id = eye_id or infer_eye_id(archive_path)
    expected_archive = EXPECTED_SOURCE_ARCHIVES.get(resolved_eye_id)
    if expected_archive is None:
        raise MaskPreparationError(
            f"{resolved_eye_id!r} is not one of the 12 frozen source archives"
        )
    if output_path.resolve() == provenance_path.resolve():
        raise MaskPreparationError("mask and provenance paths must differ")
    if not overwrite and (output_path.exists() or provenance_path.exists()):
        raise MaskPreparationError("output already exists; pass overwrite=True to replace it")

    archive_size = int(archive_path.stat().st_size)
    archive_sha256 = sha256_file(archive_path)
    if (
        archive_path.name != expected_archive["name"]
        or archive_size != expected_archive["size_bytes"]
        or archive_sha256 != expected_archive["sha256"]
    ):
        raise MaskPreparationError(
            f"source archive does not match the frozen contract for {resolved_eye_id}"
        )
    (
        members,
        image_shape,
        uncropped_shape,
        crop_origin,
        crop_upper,
        foreground_voxels,
        observed_values,
    ) = _scan_archive(archive_path)
    if (
        len(members) != expected_archive["slice_count"]
        or list(uncropped_shape) != expected_archive["uncropped_shape_zyx"]
    ):
        raise MaskPreparationError(
            f"source stack geometry does not match the frozen contract for {resolved_eye_id}"
        )
    lower = np.asarray(crop_origin, dtype=np.int64)
    upper = np.asarray(crop_upper, dtype=np.int64)
    cropped_shape_array = upper - lower + 1
    cropped_shape = tuple(int(value) for value in cropped_shape_array)
    if any(value <= 0 for value in cropped_shape):
        raise MaskPreparationError("invalid occupied bounding box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    # Large, incrementally written dot-files in the workspace may be copied by
    # workspace checkpointing while they grow.  Put the staging NPY in the
    # process temporary directory when it is on the same filesystem; rename is
    # still atomic, and no multi-gigabyte checkpoint debris is produced.
    temporary_directory = Path(tempfile.gettempdir())
    if temporary_directory.stat().st_dev != output_path.parent.stat().st_dev:
        temporary_directory = output_path.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="experiment63-maike-mask-",
        suffix=".tmp.npy",
        dir=temporary_directory,
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    temporary_path = Path(temporary_name)
    data_digest = hashlib.sha256()
    written_foreground = 0
    try:
        destination = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.uint8, shape=cropped_shape, fortran_order=False
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            for output_z, source_z in enumerate(range(int(lower[0]), int(upper[0]) + 1)):
                values = _read_binary_slice(archive, members[source_z])
                cropped = np.ascontiguousarray(
                    values[int(lower[1]) : int(upper[1]) + 1,
                           int(lower[2]) : int(upper[2]) + 1],
                    dtype=np.uint8,
                )
                destination[output_z] = cropped
                data_digest.update(cropped.tobytes(order="C"))
                written_foreground += int(np.count_nonzero(cropped))
        destination.flush()
        del destination
        if written_foreground != foreground_voxels:
            raise MaskPreparationError("cropping did not preserve every foreground voxel")
        # Bind the same immutable archive that was hashed before the scan.  A
        # concurrent replacement must not yield a mask with stale provenance.
        if int(archive_path.stat().st_size) != archive_size or sha256_file(archive_path) != archive_sha256:
            raise MaskPreparationError("source archive changed during mask preparation")
        reloaded = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
        if reloaded.shape != cropped_shape or reloaded.dtype != np.uint8:
            raise MaskPreparationError("generated NPY failed shape/dtype verification")
        del reloaded

        npy_sha256 = sha256_file(temporary_path)
        source_slice_numbers = [member.index for member in members]
        provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "eye_id": resolved_eye_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "axis_order": AXIS_ORDER,
            "spacing_um": [ORIGINAL_SPACING_UM] * 3,
            "original_spacing_um": [ORIGINAL_SPACING_UM] * 3,
            # Extraction validates this top-level digest against the complete
            # NPY file.  The raw C-order voxel digest remains separately named.
            "array_sha256": npy_sha256,
            "npy_sha256": npy_sha256,
            "array_data_sha256": data_digest.hexdigest(),
            "source_archive": {
                "path": str(archive_path.resolve()),
                "name": archive_path.name,
                "size_bytes": archive_size,
                "sha256": archive_sha256,
            },
            "archive_sha256": archive_sha256,
            "source_slices": {
                "count": len(members),
                "first_numeric_index": source_slice_numbers[0],
                "last_numeric_index": source_slice_numbers[-1],
                "numeric_indices_contiguous": True,
                "ordered_names_sha256": _canonical_names_sha256(members),
            },
            "uncropped": {
                "shape_zyx": list(uncropped_shape),
                "image_shape_yx": list(image_shape),
            },
            "crop": {
                "origin_zyx": [int(value) for value in lower],
                "upper_inclusive_zyx": [int(value) for value in upper],
                "shape_zyx": list(cropped_shape),
                "preserves_all_foreground": True,
            },
            "output": {
                "path": str(output_path.resolve()),
                "name": output_path.name,
                "format": "npy",
                "dtype": "uint8",
                "order": "C",
                "axis_order": AXIS_ORDER,
                "shape_zyx": list(cropped_shape),
                "size_bytes": int(temporary_path.stat().st_size),
                "sha256": npy_sha256,
                "data_sha256": data_digest.hexdigest(),
                "foreground_voxels": foreground_voxels,
                "observed_values": observed_values,
            },
            "validation": {
                "binary_values_only": True,
                "homogeneous_dimensions": True,
                "numeric_slices_contiguous": True,
                "spacing_is_original_0_325_um_isotropic": True,
                "atomic_memory_mapped_write": True,
            },
        }

        # Publish the mask first and the sidecar last.  A consumer requires both
        # and re-hashes the mask, so an interrupted pair cannot be accepted.
        os.replace(temporary_path, output_path)
        try:
            _atomic_json(provenance_path, provenance)
        except BaseException:
            # Do not leave a newly published, provenance-free mask on failure.
            # Never remove a pre-existing output when overwrite was requested.
            if not overwrite:
                try:
                    output_path.unlink()
                except FileNotFoundError:
                    pass
            raise
        return provenance
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _require_int_list(value: Any, length: int, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise MaskPreparationError(f"{field} must be a length-{length} JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise MaskPreparationError(f"{field} must contain integers")
    return value


def load_and_validate_mask_provenance(
    provenance_path: Path,
    *,
    mask_path: Path | None = None,
    archive_path: Path | None = None,
    verify_archive: bool = True,
) -> dict[str, Any]:
    """Load a v2 sidecar and verify every locally available bound artifact."""

    provenance_path = Path(provenance_path)
    try:
        value = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaskPreparationError(f"cannot read provenance {provenance_path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise MaskPreparationError("mask provenance is not the required v2 schema")
    if value.get("axis_order") != AXIS_ORDER:
        raise MaskPreparationError("mask provenance axis order must be zyx")
    if value.get("spacing_um") != [ORIGINAL_SPACING_UM] * 3:
        raise MaskPreparationError("mask provenance spacing is not 0.325 um isotropic")
    if value.get("original_spacing_um") != [ORIGINAL_SPACING_UM] * 3:
        raise MaskPreparationError(
            "mask provenance original spacing is not 0.325 um isotropic"
        )

    eye_id = value.get("eye_id")
    if not isinstance(eye_id, str) or eye_id not in EXPECTED_SOURCE_ARCHIVES:
        raise MaskPreparationError("mask provenance eye_id is not a frozen source eye")
    expected_archive = EXPECTED_SOURCE_ARCHIVES[eye_id]

    source_slices = value.get("source_slices")
    uncropped = value.get("uncropped")
    crop = value.get("crop")
    if not isinstance(source_slices, dict):
        raise MaskPreparationError("mask provenance lacks source-slice metadata")
    if not isinstance(uncropped, dict) or not isinstance(crop, dict):
        raise MaskPreparationError("mask provenance lacks crop geometry")
    uncropped_shape = _require_int_list(
        uncropped.get("shape_zyx"), 3, "uncropped.shape_zyx"
    )
    image_shape = _require_int_list(
        uncropped.get("image_shape_yx"), 2, "uncropped.image_shape_yx"
    )
    crop_origin = _require_int_list(crop.get("origin_zyx"), 3, "crop.origin_zyx")
    crop_upper = _require_int_list(
        crop.get("upper_inclusive_zyx"), 3, "crop.upper_inclusive_zyx"
    )
    crop_shape = _require_int_list(crop.get("shape_zyx"), 3, "crop.shape_zyx")
    slice_count = source_slices.get("count")
    first_slice = source_slices.get("first_numeric_index")
    last_slice = source_slices.get("last_numeric_index")
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (slice_count, first_slice, last_slice)
    ):
        raise MaskPreparationError("source-slice counts/indices must be JSON integers")
    if (
        source_slices.get("numeric_indices_contiguous") is not True
        or last_slice - first_slice + 1 != slice_count
        or not isinstance(source_slices.get("ordered_names_sha256"), str)
        or len(source_slices["ordered_names_sha256"]) != 64
        or uncropped_shape != expected_archive["uncropped_shape_zyx"]
        or image_shape != uncropped_shape[1:]
        or slice_count != expected_archive["slice_count"]
        or crop.get("preserves_all_foreground") is not True
        or any(lower < 0 for lower in crop_origin)
        or any(upper < lower for lower, upper in zip(crop_origin, crop_upper, strict=True))
        or any(
            upper >= full
            for upper, full in zip(crop_upper, uncropped_shape, strict=True)
        )
        or [
            upper - lower + 1
            for lower, upper in zip(crop_origin, crop_upper, strict=True)
        ]
        != crop_shape
    ):
        raise MaskPreparationError("mask provenance source/crop geometry is inconsistent")

    output = value.get("output")
    if not isinstance(output, dict):
        raise MaskPreparationError("mask provenance lacks output metadata")
    shape = _require_int_list(output.get("shape_zyx"), 3, "output.shape_zyx")
    if (
        any(dimension <= 0 for dimension in shape)
        or shape != crop_shape
        or output.get("format") != "npy"
        or output.get("dtype") != "uint8"
        or output.get("order") != "C"
        or output.get("axis_order") != AXIS_ORDER
        or output.get("observed_values") != [0, 1]
    ):
        raise MaskPreparationError("mask provenance output metadata is inconsistent")
    recorded_mask_hash = value.get("array_sha256")
    if not isinstance(recorded_mask_hash, str) or recorded_mask_hash != output.get("sha256"):
        raise MaskPreparationError("mask provenance contains inconsistent NPY digests")
    if value.get("npy_sha256") != recorded_mask_hash:
        raise MaskPreparationError("mask provenance npy_sha256 disagrees with array_sha256")

    if mask_path is None:
        recorded_path = output.get("path")
        if not isinstance(recorded_path, str):
            raise MaskPreparationError("mask provenance lacks output path")
        mask_path = Path(recorded_path)
    mask_path = Path(mask_path)
    recorded_output_path = output.get("path")
    recorded_output_size = output.get("size_bytes")
    if (
        not isinstance(recorded_output_path, str)
        or output.get("name") != mask_path.name
        or isinstance(recorded_output_size, bool)
        or not isinstance(recorded_output_size, int)
        or not mask_path.is_file()
        or int(mask_path.stat().st_size) != recorded_output_size
        or sha256_file(mask_path) != recorded_mask_hash
    ):
        raise MaskPreparationError("mask NPY does not match its provenance SHA-256")
    array = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.dtype != np.uint8 or list(array.shape) != shape:
        raise MaskPreparationError("mask NPY shape/dtype does not match provenance")
    foreground_voxels = 0
    data_digest = hashlib.sha256()
    for z_index in range(array.shape[0]):
        slab = np.ascontiguousarray(array[z_index])
        if np.any((slab != 0) & (slab != 1)):
            raise MaskPreparationError("mask NPY is not binary {0,1}")
        foreground_voxels += int(np.count_nonzero(slab))
        data_digest.update(slab.tobytes(order="C"))
    recorded_foreground = output.get("foreground_voxels")
    if (
        isinstance(recorded_foreground, bool)
        or not isinstance(recorded_foreground, int)
        or recorded_foreground <= 0
        or foreground_voxels != recorded_foreground
    ):
        raise MaskPreparationError(
            "mask NPY foreground count does not match provenance"
        )
    recorded_data_hash = value.get("array_data_sha256")
    if (
        not isinstance(recorded_data_hash, str)
        or recorded_data_hash != output.get("data_sha256")
        or data_digest.hexdigest() != recorded_data_hash
    ):
        raise MaskPreparationError("mask NPY data digest does not match provenance")
    del array

    source = value.get("source_archive")
    if not isinstance(source, dict):
        raise MaskPreparationError("mask provenance lacks source archive binding")
    if (
        not isinstance(source.get("path"), str)
        or source.get("name") != expected_archive["name"]
        or source.get("size_bytes") != expected_archive["size_bytes"]
        or source.get("sha256") != expected_archive["sha256"]
    ):
        raise MaskPreparationError("mask provenance violates the frozen source contract")
    recorded_archive_hash = source.get("sha256")
    if value.get("archive_sha256") != recorded_archive_hash:
        raise MaskPreparationError("archive digests are inconsistent in mask provenance")
    if verify_archive:
        if archive_path is None:
            recorded_path = source.get("path")
            if not isinstance(recorded_path, str):
                raise MaskPreparationError("mask provenance lacks source archive path")
            archive_path = Path(recorded_path)
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise MaskPreparationError("bound source archive is unavailable")
        if archive_path.name != expected_archive["name"]:
            raise MaskPreparationError("source archive basename does not match frozen contract")
        if int(archive_path.stat().st_size) != source.get("size_bytes"):
            raise MaskPreparationError("source archive size does not match provenance")
        if sha256_file(archive_path) != recorded_archive_hash:
            raise MaskPreparationError("source archive does not match provenance SHA-256")
    validation = value.get("validation")
    required_validation = {
        "binary_values_only": True,
        "homogeneous_dimensions": True,
        "numeric_slices_contiguous": True,
        "spacing_is_original_0_325_um_isotropic": True,
        "atomic_memory_mapped_write": True,
    }
    if validation != required_validation:
        raise MaskPreparationError("mask provenance validation gates are incomplete")
    return value


def _resolve_outputs(
    archive: Path,
    output_dir: Path,
    explicit_eye_id: str | None,
) -> tuple[str, Path, Path]:
    eye_id = explicit_eye_id or infer_eye_id(archive)
    return (
        eye_id,
        output_dir / f"{eye_id}.mask.uint8.npy",
        output_dir / f"{eye_id}.mask.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path, help="source TIFF ZIP archive(s)")
    parser.add_argument(
        "--archive", dest="named_archives", action="append", type=Path, default=[],
        help="source TIFF ZIP (repeatable alternative to positional archives)",
    )
    parser.add_argument("--output-dir", "--out-dir", type=Path)
    parser.add_argument("--output", type=Path, help="NPY path; only valid for one archive")
    parser.add_argument("--provenance", type=Path, help="JSON path; only valid for one archive")
    parser.add_argument("--eye-id", help="explicit eye ID; only valid for one archive")
    parser.add_argument("--spacing-um", type=float, default=ORIGINAL_SPACING_UM)
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    archives = list(arguments.archives) + list(arguments.named_archives)
    if not archives:
        raise SystemExit("at least one TIFF ZIP archive is required")
    if len(archives) > 1 and any(
        value is not None for value in (arguments.output, arguments.provenance, arguments.eye_id)
    ):
        raise SystemExit("--output, --provenance, and --eye-id require exactly one archive")
    if arguments.output is not None or arguments.provenance is not None:
        if arguments.output is None or arguments.provenance is None:
            raise SystemExit("--output and --provenance must be supplied together")
        jobs = [(arguments.eye_id or infer_eye_id(archives[0]), arguments.output, arguments.provenance)]
    else:
        if arguments.output_dir is None:
            raise SystemExit("--output-dir is required unless explicit output paths are supplied")
        jobs = [
            _resolve_outputs(archive, arguments.output_dir, arguments.eye_id)
            for archive in archives
        ]

    for archive, (eye_id, output, provenance) in zip(archives, jobs, strict=True):
        result = prepare_mask_archive(
            archive,
            output,
            provenance,
            eye_id=eye_id,
            spacing_um=arguments.spacing_um,
            overwrite=arguments.force,
        )
        shape = result["output"]["shape_zyx"]
        print(f"{eye_id}: {shape} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

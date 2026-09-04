#!/usr/bin/env python3
"""Map published ODA lens centres and axes into Maike's source-mask voxels.

ODA stores physical columns named ``(x,y,z)`` after centring and rotating the
point cloud.  For a CT stack those columns originate from NumPy
``(slice,row,column)``, so they correspond directly to source ``(z,y,x)``.
Experiment 63 uses the exact inverse transform and four-fold voxel-centre
relation, then subtracts the validated tight-crop origin used by the mask:

``p_oda = (p_raw_um - sphere_center) @ rotation``
``q_binned = (p_oda @ rotation.T + sphere_center) / 1.3``
``q_source = 4 * q_binned + 1.5``
``q_mask = q_source - crop_origin``

The ODA full-lens centre and anatomical axis are sealed as Stage-1 oracle
localisation inputs.  They are not Stage-2 prediction features.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from prepare_maike_masks import (
    MaskPreparationError,
    load_and_validate_mask_provenance,
    sha256_file,
)


SCHEMA_VERSION = "experiment63-oda-source-map-v3"
TRANSFORM_SCHEMA_VERSION = "experiment63-oda-transform-v1"
SEED_ROLE = "oracle_correspondence_and_distal_localization_stage1_only"
ODA_COMMIT = "55684a97fb32a95f24d17eaf04c49253c98fee27"
SOURCE_SPACING_UM = 0.325
BIN_FACTOR = 4
BINNED_SPACING_UM = 1.3
SOURCE_BLOCK_CENTRE_OFFSET_VOX = 1.5
ODA_POINT_SAMPLE_STRIDE = 100
ODA_PREFILTER_THRESHOLD = 128
FIG3_SHARE_SIZE_BYTES = 25_767_536
FIG3_SHARE_MD5 = "81bab98c1e6c6aa1fc152132cb5d4c66"
FIG3_SHARE_SHA256 = "c4703afb44e3a67fc1565bf613b06e2948b9a12e1887c72d32137590a998a095"
FIG3_INNER_MEMBER = "share/stacks.zip"
FIG3_STACKS_ZIP_SHA256 = "c6904f692aead34d58896ba99eb8bb74254379902cd05926ea50f5cfbfb92cf8"
BATCH_SCHEMA_VERSION = "experiment63-oda-source-map-batch-v1"
ROUNDTRIP_TOLERANCE_UM = 1.0e-9
ORTHOGONALITY_TOLERANCE = 1.0e-10

# Frozen counts from the twelve published ODA tables.  Keeping the denominator
# here prevents a truncated table from silently becoming a smaller experiment.
EXPECTED_ODA_COUNTS: dict[str, int] = {
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

OUTPUT_FIELDS = [
    "lens_index",
    "seed_id",
    "source_z",
    "source_y",
    "source_x",
    "axis_source_z",
    "axis_source_y",
    "axis_source_x",
    "oda_x_um",
    "oda_y_um",
    "oda_z_um",
    "axis_oda_x",
    "axis_oda_y",
    "axis_oda_z",
]


class CoordinateMappingError(RuntimeError):
    """Raised when an input binding or coordinate invariant fails."""


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CoordinateMappingError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (OSError, json.JSONDecodeError, CoordinateMappingError) as exc:
        raise CoordinateMappingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoordinateMappingError(f"JSON root must be an object: {path}")
    return value


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


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise CoordinateMappingError("cannot serialize a non-finite coordinate")
    return format(float(value), ".17g")


def _atomic_seed_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_FIELDS, extrasaction="raise", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _unused_path(parent: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
    os.close(descriptor)
    os.unlink(name)
    return Path(name)


def _publish_file_pair(
    staged_csv: Path,
    output_csv: Path,
    staged_json: Path,
    output_json: Path,
) -> None:
    """Publish a CSV/JSON pair with rollback of any pre-existing pair."""

    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for output in (output_csv, output_json):
            if output.exists():
                backup = _unused_path(
                    output.parent,
                    prefix=f".{output.name}.",
                    suffix=".rollback",
                )
                os.replace(output, backup)
                backups[output] = backup
        os.replace(staged_csv, output_csv)
        published.append(output_csv)
        os.replace(staged_json, output_json)
        published.append(output_json)
    except BaseException:
        for output in reversed(published):
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        for output, backup in backups.items():
            if backup.exists():
                os.replace(backup, output)
        raise
    finally:
        for staged in (staged_csv, staged_json):
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass


def _rotation_matrix(theta: float, axis: int) -> np.ndarray:
    """Reproduce ODA ``rotate``'s row-vector rotation matrices exactly."""

    cosine = np.cos(theta)
    sine = np.sin(theta)
    if axis == 0:
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    if axis == 2:
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise CoordinateMappingError("only ODA rotation axes 0 and 2 are used here")


def _ordered_tiff_paths(stack_path: Path) -> list[Path]:
    """Return the same lexicographic TIFF order used by ODA ``Stack``."""

    stack_path = Path(stack_path)
    if not stack_path.is_dir():
        raise CoordinateMappingError(f"ODA binned-stack directory does not exist: {stack_path}")
    paths = sorted(
        path
        for path in stack_path.iterdir()
        if path.is_file() and path.name.endswith(".tif")
    )
    if not paths:
        raise CoordinateMappingError("ODA binned-stack directory contains no .tif files")
    return paths


def _manifest_digest(files: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{item['name']}\t{item['size_bytes']}\t{item['sha256']}\n" for item in files
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_stack_files(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    files = [
        {
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return files, _manifest_digest(files)


def _md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: Any, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _derive_oda_geometry(tiff_paths: Sequence[Path]) -> dict[str, Any]:
    """Replay the deterministic ODA import, sphere fit, and two rotations."""

    sampled_chunks: list[np.ndarray] = []
    point_count = 0
    image_shape: tuple[int, int] | None = None
    for stack_index, path in enumerate(tiff_paths):
        try:
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise CoordinateMappingError(
                        f"ODA binned TIFF must contain one frame: {path.name!r}"
                    )
                image.load()
                values = np.asarray(image)
        except (OSError, UnidentifiedImageError) as exc:
            raise CoordinateMappingError(
                f"cannot decode ODA binned TIFF {path.name!r}: {exc}"
            ) from exc
        if values.ndim != 2 or values.dtype != np.uint8:
            raise CoordinateMappingError(
                f"ODA binned TIFF must be 2-D uint8: {path.name!r}"
            )
        if image_shape is None:
            image_shape = (int(values.shape[0]), int(values.shape[1]))
        elif values.shape != image_shape:
            raise CoordinateMappingError("ODA binned TIFF dimensions are not homogeneous")

        # ODA materialises thresholded points in this same order.  Its HDF5
        # chunks contain 1,000 points and it takes [::100] within each chunk;
        # because 1,000 is divisible by 100, this is exactly the global
        # point-index rule below (including the final partial chunk).
        positions = np.argwhere(values >= ODA_PREFILTER_THRESHOLD)
        if len(positions):
            global_indices = point_count + np.arange(len(positions), dtype=np.int64)
            selected = positions[global_indices % ODA_POINT_SAMPLE_STRIDE == 0]
            if len(selected):
                sampled_chunks.append(
                    np.column_stack(
                        [
                            np.full(len(selected), stack_index * BINNED_SPACING_UM),
                            selected[:, 0] * BINNED_SPACING_UM,
                            selected[:, 1] * BINNED_SPACING_UM,
                        ]
                    ).astype(np.float64, copy=False)
                )
            point_count += len(positions)

    if image_shape is None:
        raise CoordinateMappingError("ODA binned stack contains no readable TIFFs")
    if not sampled_chunks:
        raise CoordinateMappingError("ODA binned stack contains no voxels at threshold 128")
    sample = np.concatenate(sampled_chunks)
    if len(sample) < 4:
        raise CoordinateMappingError("ODA sphere sample has fewer than four points")

    outcome = np.sum(sample**2, axis=1)[:, np.newaxis]
    coefficients = np.ones((len(sample), 4), dtype=np.float64)
    coefficients[:, :3] = sample * 2.0
    solution, sum_sq_residuals, rank, singular_values = np.linalg.lstsq(
        coefficients, outcome, rcond=None
    )
    if int(rank) != 4:
        raise CoordinateMappingError("ODA sphere fit is rank deficient")
    sphere_center = solution[:-1, 0]
    centred_sample = sample - sphere_center[np.newaxis, :]
    centre_of_mass = centred_sample.mean(axis=0)
    angle_axis_0 = float(np.arctan2(centre_of_mass[2], centre_of_mass[1]))
    rotation_axis_0 = _rotation_matrix(angle_axis_0, axis=0)
    rotated_centre_of_mass = centre_of_mass @ rotation_axis_0
    angle_axis_2 = float(
        np.arctan2(rotated_centre_of_mass[1], rotated_centre_of_mass[0])
    )
    rotation = rotation_axis_0 @ _rotation_matrix(angle_axis_2, axis=2)
    orthogonality_error = float(
        np.max(np.abs(rotation @ rotation.T - np.eye(3)))
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > ORTHOGONALITY_TOLERANCE or not math.isclose(
        determinant, 1.0, rel_tol=0.0, abs_tol=ORTHOGONALITY_TOLERANCE
    ):
        raise CoordinateMappingError("derived ODA rotation is not proper orthonormal")
    return {
        "image_shape": image_shape,
        "point_count": point_count,
        "sample_count": len(sample),
        "sphere_center": sphere_center,
        "rotation": rotation,
        "angle_axis_0": angle_axis_0,
        "angle_axis_2": angle_axis_2,
        "sum_sq_residuals": sum_sq_residuals,
        "rank": int(rank),
        "singular_values": singular_values,
        "centre_of_mass": centre_of_mass,
        "orthogonality_error": orthogonality_error,
        "determinant": determinant,
    }


def _verify_public_archive_eye(
    public_archive_path: Path,
    *,
    eye_id: str,
    stack_path: Path,
    oda_csv_path: Path,
    h5_path: Path,
) -> dict[str, Any]:
    """Bind selected extracted inputs byte-for-byte to Figshare's nested ZIP."""

    public_archive_path = Path(public_archive_path)
    if not public_archive_path.is_file():
        raise CoordinateMappingError(
            f"public Fig. 3 archive does not exist: {public_archive_path}"
        )
    archive_size = int(public_archive_path.stat().st_size)
    archive_sha256 = sha256_file(public_archive_path)
    archive_md5 = _md5_file(public_archive_path)
    if (
        archive_size != FIG3_SHARE_SIZE_BYTES
        or archive_sha256 != FIG3_SHARE_SHA256
        or archive_md5 != FIG3_SHARE_MD5
    ):
        raise CoordinateMappingError("public archive is not the frozen Fig. 3 share")

    try:
        with zipfile.ZipFile(public_archive_path, "r") as outer:
            if len(outer.namelist()) != len(set(outer.namelist())):
                raise CoordinateMappingError("public archive contains duplicate names")
            inner_info = outer.getinfo(FIG3_INNER_MEMBER)
            inner_bytes = outer.read(inner_info)
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CoordinateMappingError(f"cannot read frozen Fig. 3 archive: {exc}") from exc
    inner_sha256 = hashlib.sha256(inner_bytes).hexdigest()
    if inner_sha256 != FIG3_STACKS_ZIP_SHA256:
        raise CoordinateMappingError("nested public stacks.zip has the wrong SHA-256")

    canonical_directory = f"tiffs_{eye_id}_eye_lenses_binned"
    if stack_path.name != canonical_directory:
        raise CoordinateMappingError(
            f"ODA stack directory must be named exactly {canonical_directory!r}"
        )
    if oda_csv_path.name != "ommatidial_data.csv" or h5_path.name != "_compound_eye_data.h5":
        raise CoordinateMappingError("public CSV/H5 inputs do not use canonical names")
    prefix = f"{canonical_directory}/"
    tiff_paths = _ordered_tiff_paths(stack_path)
    local_inputs = [*tiff_paths, oda_csv_path, h5_path]
    required_names = [f"{prefix}{path.name}" for path in local_inputs]
    selected_entries: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner:
            names = inner.namelist()
            if len(names) != len(set(names)):
                raise CoordinateMappingError("nested public stacks.zip has duplicate names")
            archived_tiffs = sorted(
                name for name in names if name.startswith(prefix) and name.endswith(".tif")
            )
            expected_tiffs = [f"{prefix}{path.name}" for path in tiff_paths]
            if archived_tiffs != expected_tiffs:
                raise CoordinateMappingError(
                    "extracted ODA TIFF names do not match nested public stacks.zip"
                )
            for local_path, member_name in zip(local_inputs, required_names, strict=True):
                info = inner.getinfo(member_name)
                with inner.open(info, "r") as handle:
                    archived_sha256 = _sha256_stream(handle)
                if (
                    not local_path.is_file()
                    or int(local_path.stat().st_size) != info.file_size
                    or sha256_file(local_path) != archived_sha256
                ):
                    raise CoordinateMappingError(
                        f"extracted public input differs from {member_name!r}"
                    )
                selected_entries.append(
                    {
                        "name": member_name,
                        "size_bytes": int(info.file_size),
                        "sha256": archived_sha256,
                    }
                )
    except (KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CoordinateMappingError(f"cannot validate nested public payload: {exc}") from exc

    return {
        "path": str(public_archive_path.resolve()),
        "name": public_archive_path.name,
        "size_bytes": archive_size,
        "sha256": archive_sha256,
        "md5": archive_md5,
        "inner_stacks_zip": {
            "member": FIG3_INNER_MEMBER,
            "size_bytes": len(inner_bytes),
            "sha256": inner_sha256,
            "selected_eye_prefix": prefix,
            "selected_entries_manifest_sha256": _manifest_digest(selected_entries),
            "selected_entries": selected_entries,
        },
    }


def derive_oda_transform(
    stack_path: Path,
    output_path: Path,
    *,
    eye_id: str,
    oda_csv_path: Path,
    mask_path: Path,
    mask_provenance_path: Path,
    public_archive_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rebuild ODA's saved centring/rotation from its public binned TIFFs.

    The public Fig. 3 archive contains zero-byte H5 placeholders, so the
    transform must be reproduced from the still-present Stage-1 input.  This
    follows ODA commit ``55684a9`` literally: lexicographic TIFF order,
    ``prefiltered=True``'s uint8 threshold of 128, point order
    ``(stack_index,row,column)``, the global ``points[::100]`` sphere sample,
    ordinary least squares sphere fit, and the two centre-of-mass rotations.
    """

    stack_path = Path(stack_path)
    output_path = Path(output_path)
    oda_csv_path = Path(oda_csv_path)
    mask_path = Path(mask_path)
    mask_provenance_path = Path(mask_provenance_path)
    public_archive_path = (
        None if public_archive_path is None else Path(public_archive_path)
    )
    if not oda_csv_path.is_file():
        raise CoordinateMappingError(f"published ODA CSV does not exist: {oda_csv_path}")
    try:
        mask_provenance = load_and_validate_mask_provenance(
            mask_provenance_path, mask_path=mask_path, verify_archive=True
        )
    except MaskPreparationError as exc:
        raise CoordinateMappingError(f"invalid source-mask provenance: {exc}") from exc
    if mask_provenance.get("eye_id") != eye_id:
        raise CoordinateMappingError("source-mask eye_id does not match transform eye")
    crop = mask_provenance.get("crop")
    if not isinstance(crop, dict) or not isinstance(crop.get("origin_zyx"), list):
        raise CoordinateMappingError("source-mask provenance lacks crop origin")
    if public_archive_path is None:
        raise CoordinateMappingError(
            "the frozen public Fig. 3 archive is required to derive a transform"
        )
    protected_inputs = {
        oda_csv_path.resolve(),
        mask_path.resolve(),
        mask_provenance_path.resolve(),
        public_archive_path.resolve(),
    }
    if output_path.resolve() in protected_inputs:
        raise CoordinateMappingError("transform output collides with an input artifact")
    if output_path.exists() and not overwrite:
        raise CoordinateMappingError("transform output already exists; pass overwrite=True")

    tiff_paths = _ordered_tiff_paths(stack_path)
    try:
        with oda_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            published_row_count = sum(1 for _ in csv.DictReader(handle))
    except OSError as exc:
        raise CoordinateMappingError(f"cannot read published ODA CSV: {exc}") from exc
    expected_count = EXPECTED_ODA_COUNTS.get(eye_id)
    if expected_count is not None and published_row_count != expected_count:
        raise CoordinateMappingError(
            f"published ODA table for {eye_id} has {published_row_count} rows; "
            f"expected exactly {expected_count}"
        )
    geometry = _derive_oda_geometry(tiff_paths)
    image_shape = geometry["image_shape"]
    point_count = geometry["point_count"]
    sphere_center = geometry["sphere_center"]
    rotation = geometry["rotation"]
    angle_axis_0 = geometry["angle_axis_0"]
    angle_axis_2 = geometry["angle_axis_2"]
    orthogonality_error = geometry["orthogonality_error"]
    determinant = geometry["determinant"]
    stack_files, stack_manifest_sha256 = _hash_stack_files(tiff_paths)
    h5_path = stack_path / "_compound_eye_data.h5"
    if not h5_path.is_file() or h5_path.stat().st_size != 0:
        raise CoordinateMappingError(
            "public _compound_eye_data.h5 must be the zero-byte Figshare placeholder"
        )
    archive_binding = _verify_public_archive_eye(
        public_archive_path,
        eye_id=eye_id,
        stack_path=stack_path,
        oda_csv_path=oda_csv_path,
        h5_path=h5_path,
    )
    h5_binding: dict[str, Any] = {
        "path": str(h5_path.resolve()),
        "name": h5_path.name,
        "size_bytes": 0,
        "sha256": sha256_file(h5_path),
        "used_for_transform": False,
        "status": "zero_byte_public_placeholder",
    }

    result: dict[str, Any] = {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eye_id": eye_id,
        "oda_commit": ODA_COMMIT,
        "derivation": "replayed_public_binned_stack_stage_1_and_2",
        "sphere_center_xyz_um": sphere_center.tolist(),
        "rotation": rotation.tolist(),
        "rotation_angles_radians": {
            "axis_0": angle_axis_0,
            "axis_2": angle_axis_2,
        },
        "source_spacing_um": SOURCE_SPACING_UM,
        "binned_spacing_um": BINNED_SPACING_UM,
        "bin_factor": BIN_FACTOR,
        "source_block_center_offset_vox": SOURCE_BLOCK_CENTRE_OFFSET_VOX,
        "input_axis_direction": "toward_eye_center",
        "oda_csv_sha256": sha256_file(oda_csv_path),
        "mask_npy_sha256": sha256_file(mask_path),
        "mask_provenance_sha256": sha256_file(mask_provenance_path),
        "expected_oda_rows": expected_count,
        "published_oda_rows": published_row_count,
        "expected_foreground_hits": expected_count,
        "oda_csv": {
            "path": str(oda_csv_path.resolve()),
            "name": oda_csv_path.name,
            "size_bytes": int(oda_csv_path.stat().st_size),
            "sha256": sha256_file(oda_csv_path),
        },
        "oda_binned_stack": {
            "path": str(stack_path.resolve()),
            "file_count": len(tiff_paths),
            "image_shape_yx": list(image_shape),
            "dtype": "uint8",
            "ordered_lexicographically": True,
            "spacing_um": [BINNED_SPACING_UM] * 3,
            "prefilter_threshold_inclusive": ODA_PREFILTER_THRESHOLD,
            "point_coordinate_order": "stack_index,row,column",
            "point_count": point_count,
            "sphere_sample_stride": ODA_POINT_SAMPLE_STRIDE,
            "sphere_sample_count": geometry["sample_count"],
            "sphere_sample_rule": (
                "points.iter_chunks()[::100]; equivalent to global points[::100] "
                "because ODA chunks contain 1000 points"
            ),
            "ordered_files_manifest_sha256": stack_manifest_sha256,
            "files": stack_files,
        },
        "public_archive": archive_binding,
        "public_h5": h5_binding,
        "source_mask": {
            "path": str(mask_path.resolve()),
            "size_bytes": int(mask_path.stat().st_size),
            "sha256": sha256_file(mask_path),
            "provenance_path": str(mask_provenance_path.resolve()),
            "provenance_size_bytes": int(mask_provenance_path.stat().st_size),
            "provenance_sha256": sha256_file(mask_provenance_path),
            "source_archive": mask_provenance["source_archive"],
            "crop_origin_zyx": crop["origin_zyx"],
            "crop_shape_zyx": crop["shape_zyx"],
        },
        "source_mapping": {
            "inverse_binned": (
                "q_binned = (p_oda @ rotation.T + sphere_center) / 1.3"
            ),
            "binned_to_full_source": "q_source = 4 * q_binned + 1.5",
            "full_source_to_mask": "q_mask = q_source - crop_origin",
            "source_axis_order": "z,y,x",
        },
        "sphere_fit": {
            "method": "numpy.linalg.lstsq matching ODA SphereFit",
            "design_rank": geometry["rank"],
            "sum_squared_residuals": geometry["sum_sq_residuals"].tolist(),
            "singular_values": geometry["singular_values"].tolist(),
            "sample_centre_of_mass_before_rotation": geometry["centre_of_mass"].tolist(),
            "sample_centre_of_mass_after_rotation": (
                geometry["centre_of_mass"] @ rotation
            ).tolist(),
            "orthogonality_max_abs_error": orthogonality_error,
            "determinant": determinant,
        },
        "coordinate_semantics": {
            "oda_csv_columns": ["x", "y", "z"],
            "raw_point_order": "source_z,source_y,source_x",
            "note": "ODA CTStack names stack_index,row,column as x,y,z",
        },
    }
    _atomic_json(output_path, result)
    return result


def _validate_derived_transform_artifacts(transform: Mapping[str, Any]) -> None:
    """Fail closed and replay every derived value from hash-bound public inputs."""

    if transform.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise CoordinateMappingError(
            f"transform schema must be exactly {TRANSFORM_SCHEMA_VERSION}"
        )
    eye_id = transform.get("eye_id")
    if not isinstance(eye_id, str) or not eye_id:
        raise CoordinateMappingError("derived transform lacks a valid eye_id")
    exact_top_level = {
        "oda_commit": ODA_COMMIT,
        "derivation": "replayed_public_binned_stack_stage_1_and_2",
        "source_spacing_um": SOURCE_SPACING_UM,
        "binned_spacing_um": BINNED_SPACING_UM,
        "bin_factor": BIN_FACTOR,
        "source_block_center_offset_vox": SOURCE_BLOCK_CENTRE_OFFSET_VOX,
        "input_axis_direction": "toward_eye_center",
    }
    for field, expected in exact_top_level.items():
        if transform.get(field) != expected:
            raise CoordinateMappingError(
                f"derived transform {field} must equal the frozen value {expected!r}"
            )
    expected_count = EXPECTED_ODA_COUNTS.get(eye_id)
    if expected_count is not None:
        for field in ("expected_oda_rows", "published_oda_rows", "expected_foreground_hits"):
            if transform.get(field) != expected_count:
                raise CoordinateMappingError(
                    f"derived transform {field} must equal {expected_count} for {eye_id}"
                )

    expected_mapping = {
        "inverse_binned": "q_binned = (p_oda @ rotation.T + sphere_center) / 1.3",
        "binned_to_full_source": "q_source = 4 * q_binned + 1.5",
        "full_source_to_mask": "q_mask = q_source - crop_origin",
        "source_axis_order": "z,y,x",
    }
    if transform.get("source_mapping") != expected_mapping:
        raise CoordinateMappingError("derived transform source mapping contract changed")
    semantics = transform.get("coordinate_semantics")
    if not isinstance(semantics, dict) or (
        semantics.get("oda_csv_columns") != ["x", "y", "z"]
        or semantics.get("raw_point_order") != "source_z,source_y,source_x"
    ):
        raise CoordinateMappingError("derived transform coordinate semantics changed")

    stack = transform.get("oda_binned_stack")
    if not isinstance(stack, dict) or not isinstance(stack.get("path"), str):
        raise CoordinateMappingError("derived transform lacks its binned-stack binding")
    frozen_stack_fields = {
        "dtype": "uint8",
        "ordered_lexicographically": True,
        "spacing_um": [BINNED_SPACING_UM] * 3,
        "prefilter_threshold_inclusive": ODA_PREFILTER_THRESHOLD,
        "point_coordinate_order": "stack_index,row,column",
        "sphere_sample_stride": ODA_POINT_SAMPLE_STRIDE,
        "sphere_sample_rule": (
            "points.iter_chunks()[::100]; equivalent to global points[::100] "
            "because ODA chunks contain 1000 points"
        ),
    }
    for field, expected in frozen_stack_fields.items():
        if stack.get(field) != expected:
            raise CoordinateMappingError(
                f"derived transform oda_binned_stack.{field} changed"
            )
    paths = _ordered_tiff_paths(Path(stack["path"]))
    files, manifest_sha256 = _hash_stack_files(paths)
    if files != stack.get("files") or manifest_sha256 != stack.get(
        "ordered_files_manifest_sha256"
    ):
        raise CoordinateMappingError("ODA binned stack does not match transform provenance")
    if len(paths) != stack.get("file_count"):
        raise CoordinateMappingError("ODA binned-stack file count changed")

    geometry = _derive_oda_geometry(paths)
    geometry_checks = {
        "image_shape_yx": list(geometry["image_shape"]),
        "point_count": geometry["point_count"],
        "sphere_sample_count": geometry["sample_count"],
    }
    for field, expected in geometry_checks.items():
        if stack.get(field) != expected:
            raise CoordinateMappingError(
                f"derived transform oda_binned_stack.{field} does not replay"
            )
    try:
        stored_center = np.asarray(transform["sphere_center_xyz_um"], dtype=np.float64)
        stored_rotation = np.asarray(transform["rotation"], dtype=np.float64)
        stored_angles = transform["rotation_angles_radians"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CoordinateMappingError("derived transform geometry is malformed") from exc
    if (
        stored_center.shape != (3,)
        or stored_rotation.shape != (3, 3)
        or not np.array_equal(stored_center, geometry["sphere_center"])
        or not np.array_equal(stored_rotation, geometry["rotation"])
        or not isinstance(stored_angles, dict)
        or stored_angles.get("axis_0") != geometry["angle_axis_0"]
        or stored_angles.get("axis_2") != geometry["angle_axis_2"]
    ):
        raise CoordinateMappingError("derived transform geometry does not replay from TIFFs")

    oda_csv = transform.get("oda_csv")
    if not isinstance(oda_csv, dict) or not isinstance(oda_csv.get("path"), str):
        raise CoordinateMappingError("derived transform lacks its ODA CSV binding")
    oda_csv_path = Path(oda_csv["path"])
    if (
        not oda_csv_path.is_file()
        or int(oda_csv_path.stat().st_size) != oda_csv.get("size_bytes")
        or sha256_file(oda_csv_path) != oda_csv.get("sha256")
        or transform.get("oda_csv_sha256") != oda_csv.get("sha256")
    ):
        raise CoordinateMappingError("ODA CSV does not match transform provenance")

    h5 = transform.get("public_h5")
    if not isinstance(h5, dict) or not isinstance(h5.get("path"), str):
        raise CoordinateMappingError("derived transform lacks its public H5 binding")
    h5_path = Path(h5["path"])
    if (
        not h5_path.is_file()
        or h5.get("size_bytes") != 0
        or h5_path.stat().st_size != 0
        or sha256_file(h5_path) != h5.get("sha256")
        or h5.get("used_for_transform") is not False
        or h5.get("status") != "zero_byte_public_placeholder"
    ):
        raise CoordinateMappingError("public H5 does not match transform provenance")

    public_archive = transform.get("public_archive")
    if not isinstance(public_archive, dict) or not isinstance(
        public_archive.get("path"), str
    ):
        raise CoordinateMappingError("derived transform lacks its public archive binding")
    replayed_archive = _verify_public_archive_eye(
        Path(public_archive["path"]),
        eye_id=eye_id,
        stack_path=Path(stack["path"]),
        oda_csv_path=oda_csv_path,
        h5_path=h5_path,
    )
    if replayed_archive != public_archive:
        raise CoordinateMappingError("public archive payload chain changed")

    mask = transform.get("source_mask")
    if not isinstance(mask, dict):
        raise CoordinateMappingError("derived transform lacks source-mask binding")
    mask_path = Path(str(mask.get("path", "")))
    provenance_path = Path(str(mask.get("provenance_path", "")))
    if (
        not mask_path.is_file()
        or int(mask_path.stat().st_size) != mask.get("size_bytes")
        or sha256_file(mask_path) != mask.get("sha256")
        or transform.get("mask_npy_sha256") != mask.get("sha256")
        or not provenance_path.is_file()
        or int(provenance_path.stat().st_size) != mask.get("provenance_size_bytes")
        or sha256_file(provenance_path) != mask.get("provenance_sha256")
        or transform.get("mask_provenance_sha256") != mask.get("provenance_sha256")
    ):
        raise CoordinateMappingError("source mask does not match transform provenance")
    try:
        mask_provenance = load_and_validate_mask_provenance(
            provenance_path, mask_path=mask_path, verify_archive=True
        )
    except MaskPreparationError as exc:
        raise CoordinateMappingError(f"source-mask provenance is invalid: {exc}") from exc
    crop = mask_provenance.get("crop")
    if (
        mask_provenance.get("eye_id") != eye_id
        or not isinstance(crop, dict)
        or mask.get("crop_origin_zyx") != crop.get("origin_zyx")
        or mask.get("crop_shape_zyx") != crop.get("shape_zyx")
        or mask.get("source_archive") != mask_provenance.get("source_archive")
    ):
        raise CoordinateMappingError("source-mask geometry/archive binding changed")


def raw_um_to_oda(
    raw_xyz_um: np.ndarray,
    sphere_center_xyz_um: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Apply the recorded ODA row-vector centring/rotation transform."""

    return (np.asarray(raw_xyz_um, dtype=np.float64) - sphere_center_xyz_um) @ rotation


def oda_to_raw_um(
    oda_xyz_um: np.ndarray,
    sphere_center_xyz_um: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Invert the recorded ODA transform for row-vector coordinates."""

    return np.asarray(oda_xyz_um, dtype=np.float64) @ rotation.T + sphere_center_xyz_um


def oda_to_source_vox(
    oda_xyz_um: np.ndarray,
    sphere_center_xyz_um: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Return full-source coordinates in ODA/source ``(z,y,x)`` order."""

    raw_xyz_um = oda_to_raw_um(oda_xyz_um, sphere_center_xyz_um, rotation)
    q_binned = raw_xyz_um / BINNED_SPACING_UM
    return BIN_FACTOR * q_binned + SOURCE_BLOCK_CENTRE_OFFSET_VOX


def oda_axis_to_raw(
    oda_axis_xyz: np.ndarray,
    rotation: np.ndarray,
    *,
    input_direction: str,
) -> np.ndarray:
    """Return an outward axis in ODA/source ``(z,y,x)`` order."""

    axis = np.asarray(oda_axis_xyz, dtype=np.float64) @ rotation.T
    if input_direction == "toward_eye_center":
        axis = -axis
    elif input_direction != "away_from_eye_center":
        raise CoordinateMappingError(
            "input axis direction must be toward_eye_center or away_from_eye_center"
        )
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CoordinateMappingError("ODA axis has zero or non-finite length")
    return axis / norm


def _finite_vector(value: Any, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CoordinateMappingError(f"{field} is not numeric") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise CoordinateMappingError(f"{field} must be a finite length-3 vector")
    return vector


def load_transform(path: Path, *, eye_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    """Load and validate the explicit ODA transform contract."""

    value = _read_json(path)
    if value.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise CoordinateMappingError(
            f"transform schema must be exactly {TRANSFORM_SCHEMA_VERSION}"
        )
    _validate_derived_transform_artifacts(value)
    recorded_eye = value.get("eye_id")
    if recorded_eye != eye_id:
        raise CoordinateMappingError(
            f"transform eye_id {recorded_eye!r} does not match requested {eye_id!r}"
        )
    center = _finite_vector(value.get("sphere_center_xyz_um"), "sphere centre")
    try:
        rotation = np.asarray(value["rotation"], dtype=np.float64)
    except KeyError as exc:
        raise CoordinateMappingError("transform JSON lacks rotation") from exc
    except (TypeError, ValueError) as exc:
        raise CoordinateMappingError("rotation is not numeric") from exc
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise CoordinateMappingError("rotation must be a finite 3x3 matrix")
    orthogonality_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > ORTHOGONALITY_TOLERANCE or not math.isclose(
        determinant, 1.0, rel_tol=0.0, abs_tol=ORTHOGONALITY_TOLERANCE
    ):
        raise CoordinateMappingError("rotation must be a proper orthonormal matrix")

    constants: list[tuple[str, float, str]] = [
        ("source_spacing_um", SOURCE_SPACING_UM, "source spacing"),
        ("binned_spacing_um", BINNED_SPACING_UM, "binned spacing"),
        ("bin_factor", float(BIN_FACTOR), "bin factor"),
        (
            "source_block_center_offset_vox",
            SOURCE_BLOCK_CENTRE_OFFSET_VOX,
            "source block-centre offset",
        ),
    ]
    for field, expected, description in constants:
        supplied = value.get(field)
        try:
            supplied_float = float(supplied)
        except (TypeError, ValueError) as exc:
            raise CoordinateMappingError(f"transform {description} is not numeric") from exc
        if not math.isclose(supplied_float, expected, rel_tol=0.0, abs_tol=1e-12):
            raise CoordinateMappingError(
                f"transform {description} must equal the frozen value {expected}"
            )

    direction = value.get("input_axis_direction")
    if direction != "toward_eye_center":
        raise CoordinateMappingError(
            "transform input axis direction must be toward_eye_center"
        )
    return value, center, rotation, str(direction)


def _parse_vector_cell(value: str, *, row_number: int, field: str) -> np.ndarray:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise CoordinateMappingError(
            f"row {row_number}: {field} is not a literal length-3 vector"
        ) from exc
    return _finite_vector(parsed, f"row {row_number} {field}")


def _column(names: Sequence[str], alternatives: Sequence[str]) -> str | None:
    available = set(names)
    return next((name for name in alternatives if name in available), None)


def _read_oda_rows(
    path: Path,
    *,
    axis_column: str | None,
) -> tuple[list[dict[str, Any]], str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            raw_rows = list(reader)
    except OSError as exc:
        raise CoordinateMappingError(f"cannot read ODA CSV {path}: {exc}") from exc
    if not raw_rows:
        raise CoordinateMappingError("ODA CSV has no data rows")
    if len(names) != len(set(names)):
        raise CoordinateMappingError("ODA CSV contains duplicate column names")
    x_key = _column(names, ["x", "oda_x", "x_um"])
    y_key = _column(names, ["y", "oda_y", "y_um"])
    z_key = _column(names, ["z", "oda_z", "z_um"])
    label_key = _column(names, ["label", "seed_id", "lens_id", "oda_lens_id"])
    if any(key is None for key in (x_key, y_key, z_key, label_key)):
        raise CoordinateMappingError("ODA CSV must contain x, y, z, and label columns")

    vector_key = axis_column
    if vector_key is None:
        vector_key = _column(names, ["anatomical_axis", "axis", "approx_axis"])
    separate_keys: tuple[str | None, str | None, str | None] = (None, None, None)
    if vector_key is None:
        separate_keys = (
            _column(names, ["anatomical_axis_x", "axis_x", "oda_axis_x"]),
            _column(names, ["anatomical_axis_y", "axis_y", "oda_axis_y"]),
            _column(names, ["anatomical_axis_z", "axis_z", "oda_axis_z"]),
        )
        if any(key is None for key in separate_keys):
            raise CoordinateMappingError(
                "ODA CSV must contain anatomical_axis (or separate axis x/y/z columns)"
            )
        axis_description = "/".join(str(key) for key in separate_keys)
    else:
        if vector_key not in names:
            raise CoordinateMappingError(f"requested axis column is absent: {vector_key}")
        axis_description = vector_key

    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        label = str(raw[label_key]).strip()  # type: ignore[index]
        if not label or label in seen_labels:
            raise CoordinateMappingError(f"row {row_number}: empty or duplicate ODA label")
        seen_labels.add(label)
        try:
            point = np.asarray(
                [float(raw[x_key]), float(raw[y_key]), float(raw[z_key])],  # type: ignore[index]
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise CoordinateMappingError(f"row {row_number}: invalid ODA coordinate") from exc
        if not np.all(np.isfinite(point)):
            raise CoordinateMappingError(f"row {row_number}: non-finite ODA coordinate")
        if vector_key is not None:
            axis = _parse_vector_cell(raw[vector_key], row_number=row_number, field=vector_key)
        else:
            try:
                axis = np.asarray(
                    [float(raw[key]) for key in separate_keys],  # type: ignore[index]
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise CoordinateMappingError(f"row {row_number}: invalid ODA axis") from exc
            if not np.all(np.isfinite(axis)):
                raise CoordinateMappingError(f"row {row_number}: non-finite ODA axis")
        if float(np.linalg.norm(axis)) <= 0.0:
            raise CoordinateMappingError(f"row {row_number}: zero ODA axis")
        rows.append({"label": label, "point": point, "axis": axis})
    return rows, axis_description


def _validate_transform_bindings(
    transform: Mapping[str, Any],
    *,
    oda_csv_path: Path,
    mask_path: Path,
    mask_provenance_path: Path,
) -> None:
    checks = [
        ("oda_csv_sha256", sha256_file(oda_csv_path), "ODA CSV"),
        ("mask_npy_sha256", sha256_file(mask_path), "source mask"),
        (
            "mask_provenance_sha256",
            sha256_file(mask_provenance_path),
            "mask provenance",
        ),
    ]
    for field, observed, description in checks:
        supplied = transform.get(field)
        if supplied != observed:
            raise CoordinateMappingError(
                f"transform JSON {description} binding does not match the supplied artifact"
            )


def map_oda_to_source(
    oda_csv_path: Path,
    transform_path: Path,
    mask_path: Path,
    mask_provenance_path: Path,
    output_csv_path: Path,
    output_json_path: Path,
    *,
    eye_id: str,
    expected_count: int | None = None,
    axis_column: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Map one complete ODA table and publish a hash-bound CSV/JSON pair."""

    paths = [
        Path(oda_csv_path),
        Path(transform_path),
        Path(mask_path),
        Path(mask_provenance_path),
        Path(output_csv_path),
        Path(output_json_path),
    ]
    oda_csv_path, transform_path, mask_path, mask_provenance_path, output_csv_path, output_json_path = paths
    for input_path in paths[:4]:
        if not input_path.is_file():
            raise CoordinateMappingError(f"required input does not exist: {input_path}")
    input_locations = {path.resolve() for path in paths[:4]}
    if any(path.resolve() in input_locations for path in paths[4:]):
        raise CoordinateMappingError("an output path collides with an input artifact")
    if output_csv_path.resolve() == output_json_path.resolve():
        raise CoordinateMappingError("seed CSV and provenance paths must differ")
    if not overwrite and (output_csv_path.exists() or output_json_path.exists()):
        raise CoordinateMappingError("output already exists; pass overwrite=True to replace it")

    try:
        mask_provenance = load_and_validate_mask_provenance(
            mask_provenance_path, mask_path=mask_path, verify_archive=True
        )
    except MaskPreparationError as exc:
        raise CoordinateMappingError(f"invalid source-mask provenance: {exc}") from exc
    if mask_provenance.get("eye_id") != eye_id:
        raise CoordinateMappingError("source-mask eye_id does not match requested eye")
    crop = mask_provenance.get("crop")
    uncropped = mask_provenance.get("uncropped")
    if not isinstance(crop, dict) or not isinstance(uncropped, dict):
        raise CoordinateMappingError("source-mask provenance lacks crop geometry")
    try:
        crop_origin_zyx = np.asarray(crop["origin_zyx"], dtype=np.float64)
        uncropped_shape_zyx = np.asarray(uncropped["shape_zyx"], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise CoordinateMappingError("source-mask crop geometry is malformed") from exc
    if (
        crop_origin_zyx.shape != (3,)
        or uncropped_shape_zyx.shape != (3,)
        or not np.all(np.isfinite(crop_origin_zyx))
        or np.any(crop_origin_zyx < 0.0)
        or np.any(crop_origin_zyx != np.floor(crop_origin_zyx))
        or np.any(uncropped_shape_zyx <= 0)
    ):
        raise CoordinateMappingError("source-mask crop geometry is invalid")
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)

    transform, sphere_center, rotation, input_direction = load_transform(
        transform_path, eye_id=eye_id
    )
    _validate_transform_bindings(
        transform,
        oda_csv_path=oda_csv_path,
        mask_path=mask_path,
        mask_provenance_path=mask_provenance_path,
    )
    oda_rows, axis_description = _read_oda_rows(oda_csv_path, axis_column=axis_column)

    frozen_count = EXPECTED_ODA_COUNTS.get(eye_id)
    if expected_count is None:
        expected_count = frozen_count
    elif frozen_count is not None and expected_count != frozen_count:
        raise CoordinateMappingError(
            f"expected count for {eye_id} is frozen at {frozen_count}, not {expected_count}"
        )
    if expected_count is not None and len(oda_rows) != expected_count:
        raise CoordinateMappingError(
            f"ODA table for {eye_id} has {len(oda_rows)} rows; expected exactly {expected_count}"
        )

    output_rows: list[dict[str, Any]] = []
    max_roundtrip_error = 0.0
    minimum_outward_dot = math.inf
    rounded_positions: list[tuple[int, int, int]] = []
    for lens_index, row in enumerate(oda_rows):
        oda_point = row["point"]
        raw_point = oda_to_raw_um(oda_point, sphere_center, rotation)
        roundtrip = raw_um_to_oda(raw_point, sphere_center, rotation)
        roundtrip_error = float(np.max(np.abs(roundtrip - oda_point)))
        max_roundtrip_error = max(max_roundtrip_error, roundtrip_error)
        if roundtrip_error > ROUNDTRIP_TOLERANCE_UM:
            raise CoordinateMappingError("ODA transform failed its numerical round-trip gate")

        source_full_zyx = oda_to_source_vox(oda_point, sphere_center, rotation)
        if np.any(source_full_zyx < 0.0) or np.any(
            source_full_zyx >= uncropped_shape_zyx
        ):
            raise CoordinateMappingError(
                f"ODA seed {row['label']!r} maps outside the full source stack"
            )
        source_zyx = source_full_zyx - crop_origin_zyx
        if np.any(source_zyx < 0.0) or np.any(source_zyx >= np.asarray(mask.shape)):
            raise CoordinateMappingError(
                f"ODA seed {row['label']!r} maps outside the cropped source mask"
            )
        nearest = tuple(int(value) for value in np.rint(source_zyx))
        if any(value < 0 or value >= mask.shape[axis] for axis, value in enumerate(nearest)):
            raise CoordinateMappingError(
                f"ODA seed {row['label']!r} rounds outside the cropped source mask"
            )
        if mask[nearest] != 1:
            raise CoordinateMappingError(
                f"ODA seed {row['label']!r} does not map to source-mask foreground"
            )
        rounded_positions.append(nearest)

        outward_axis_zyx = oda_axis_to_raw(
            row["axis"], rotation, input_direction=input_direction
        )
        radial = raw_point - sphere_center
        outward_dot = float(np.dot(outward_axis_zyx, radial))
        minimum_outward_dot = min(minimum_outward_dot, outward_dot)
        if not math.isfinite(outward_dot) or outward_dot <= 0.0:
            raise CoordinateMappingError(
                f"ODA seed {row['label']!r} axis is not outward after orientation"
            )
        serialized: dict[str, Any] = {
            "lens_index": lens_index,
            "seed_id": row["label"],
            "source_z": _format_float(source_zyx[0]),
            "source_y": _format_float(source_zyx[1]),
            "source_x": _format_float(source_zyx[2]),
            "axis_source_z": _format_float(outward_axis_zyx[0]),
            "axis_source_y": _format_float(outward_axis_zyx[1]),
            "axis_source_x": _format_float(outward_axis_zyx[2]),
            "oda_x_um": _format_float(oda_point[0]),
            "oda_y_um": _format_float(oda_point[1]),
            "oda_z_um": _format_float(oda_point[2]),
            "axis_oda_x": _format_float(row["axis"][0]),
            "axis_oda_y": _format_float(row["axis"][1]),
            "axis_oda_z": _format_float(row["axis"][2]),
        }
        output_rows.append(serialized)

    if len(set(rounded_positions)) != len(rounded_positions):
        raise CoordinateMappingError("multiple ODA seeds round to the same source voxel")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    staged_csv = _unused_path(
        output_csv_path.parent,
        prefix=f".{output_csv_path.name}.",
        suffix=".staged",
    )
    staged_json = _unused_path(
        output_json_path.parent,
        prefix=f".{output_json_path.name}.",
        suffix=".staged",
    )
    try:
        _atomic_seed_csv(staged_csv, output_rows)
        seed_csv_sha256 = sha256_file(staged_csv)
        provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "eye_id": eye_id,
            "seed_role": SEED_ROLE,
            "seed_csv_sha256": seed_csv_sha256,
            "csv_sha256": seed_csv_sha256,
            "n_expected": expected_count,
            "n_rows": len(output_rows),
            "n_foreground_hits": len(output_rows),
            "lens_index_range": [0, len(output_rows) - 1],
            "candidate_seeds_per_voxel": 1,
            "input_axis_column": axis_description,
            "input_axis_direction": input_direction,
            "output_axis_direction": "away_from_eye_center",
            "coordinate_contract": {
                "oda_csv_column_order": "x,y,z",
                "oda_raw_and_full_source_order": "z,y,x",
                "coordinate_order_out": "cropped_source_z,y,x",
                "forward_oda": "p_oda = (p_raw_um - sphere_center) @ rotation",
                "inverse_binned": "q_binned = (p_oda @ rotation.T + sphere_center) / 1.3",
                "binned_to_source": "q_source = 4 * q_binned + 1.5",
                "full_source_to_mask": "q_mask = q_source - crop_origin",
                "crop_origin_zyx": crop_origin_zyx.astype(int).tolist(),
                "source_spacing_um": SOURCE_SPACING_UM,
                "binned_spacing_um": BINNED_SPACING_UM,
                "bin_factor": BIN_FACTOR,
                "source_block_center_offset_vox": SOURCE_BLOCK_CENTRE_OFFSET_VOX,
            },
            "transform": {
                "sphere_center_xyz_um": sphere_center.tolist(),
                "rotation": rotation.tolist(),
                "orthogonality_max_abs_error": float(
                    np.max(np.abs(rotation @ rotation.T - np.eye(3)))
                ),
                "determinant": float(np.linalg.det(rotation)),
                "maximum_roundtrip_error_um": max_roundtrip_error,
            },
            "checks": {
                "all_expected_rows_present": expected_count is None
                or len(output_rows) == expected_count,
                "all_centres_inside_source_mask": True,
                "all_rounded_centres_on_foreground": True,
                "rounded_centres_unique": True,
                "all_axes_finite_unit_and_outward": True,
                "minimum_outward_axis_radial_dot_um": minimum_outward_dot,
                "roundtrip_tolerance_um": ROUNDTRIP_TOLERANCE_UM,
            },
            "input_hashes": {
                "oda_csv": {
                    "path": str(oda_csv_path.resolve()),
                    "size_bytes": int(oda_csv_path.stat().st_size),
                    "sha256": sha256_file(oda_csv_path),
                },
                "transform_json": {
                    "path": str(transform_path.resolve()),
                    "size_bytes": int(transform_path.stat().st_size),
                    "sha256": sha256_file(transform_path),
                },
                "mask_npy": {
                    "path": str(mask_path.resolve()),
                    "size_bytes": int(mask_path.stat().st_size),
                    "sha256": sha256_file(mask_path),
                },
                "mask_provenance": {
                    "path": str(mask_provenance_path.resolve()),
                    "size_bytes": int(mask_provenance_path.stat().st_size),
                    "sha256": sha256_file(mask_provenance_path),
                },
                "source_archive": mask_provenance["source_archive"],
            },
            "output": {
                "path": str(output_csv_path.resolve()),
                "name": output_csv_path.name,
                "size_bytes": int(staged_csv.stat().st_size),
                "sha256": seed_csv_sha256,
                "fields": OUTPUT_FIELDS,
            },
        }
        _atomic_json(staged_json, provenance)
        _publish_file_pair(staged_csv, output_csv_path, staged_json, output_json_path)
        return provenance
    finally:
        for staged in (staged_csv, staged_json):
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
        del mask


def load_and_validate_seed_provenance(
    provenance_path: Path,
    *,
    seed_csv_path: Path,
    eye_id: str,
) -> dict[str, Any]:
    """Validate an exact v3 oracle-seed pair and its complete provenance chain."""

    provenance_path = Path(provenance_path)
    seed_csv_path = Path(seed_csv_path)
    value = _read_json(provenance_path)
    expected_count = EXPECTED_ODA_COUNTS.get(eye_id)
    if expected_count is None:
        raise CoordinateMappingError(f"{eye_id!r} is not one of the 12 frozen eyes")
    exact_fields = {
        "schema_version": SCHEMA_VERSION,
        "eye_id": eye_id,
        "seed_role": SEED_ROLE,
        "n_expected": expected_count,
        "n_rows": expected_count,
        "n_foreground_hits": expected_count,
        "lens_index_range": [0, expected_count - 1],
        "candidate_seeds_per_voxel": 1,
        "input_axis_direction": "toward_eye_center",
        "output_axis_direction": "away_from_eye_center",
    }
    for field, expected in exact_fields.items():
        if value.get(field) != expected:
            raise CoordinateMappingError(
                f"seed provenance {field} must equal {expected!r}"
            )
    checks = value.get("checks")
    required_true_checks = (
        "all_expected_rows_present",
        "all_centres_inside_source_mask",
        "all_rounded_centres_on_foreground",
        "rounded_centres_unique",
        "all_axes_finite_unit_and_outward",
    )
    if not isinstance(checks, dict) or any(
        checks.get(field) is not True for field in required_true_checks
    ):
        raise CoordinateMappingError("seed provenance success checks are incomplete")
    if checks.get("roundtrip_tolerance_um") != ROUNDTRIP_TOLERANCE_UM:
        raise CoordinateMappingError("seed provenance round-trip tolerance changed")

    output = value.get("output")
    if not isinstance(output, dict):
        raise CoordinateMappingError("seed provenance lacks output binding")
    recorded_csv_hash = value.get("seed_csv_sha256")
    if (
        not seed_csv_path.is_file()
        or output.get("path") != str(seed_csv_path.resolve())
        or output.get("name") != seed_csv_path.name
        or output.get("size_bytes") != seed_csv_path.stat().st_size
        or output.get("fields") != OUTPUT_FIELDS
        or output.get("sha256") != recorded_csv_hash
        or value.get("csv_sha256") != recorded_csv_hash
        or sha256_file(seed_csv_path) != recorded_csv_hash
    ):
        raise CoordinateMappingError("seed CSV does not match its v3 provenance")

    input_hashes = value.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise CoordinateMappingError("seed provenance lacks input hash bindings")
    bound_paths: dict[str, Path] = {}
    for field in ("oda_csv", "transform_json", "mask_npy", "mask_provenance"):
        binding = input_hashes.get(field)
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise CoordinateMappingError(f"seed provenance lacks {field} binding")
        path = Path(binding["path"])
        if (
            not path.is_file()
            or binding.get("size_bytes") != path.stat().st_size
            or binding.get("sha256") != sha256_file(path)
        ):
            raise CoordinateMappingError(
                f"seed provenance {field} does not match its bound artifact"
            )
        bound_paths[field] = path

    transform, center, rotation, direction = load_transform(
        bound_paths["transform_json"], eye_id=eye_id
    )
    if direction != "toward_eye_center":  # load_transform is already fail closed
        raise CoordinateMappingError("seed transform has the wrong axis direction")
    _validate_transform_bindings(
        transform,
        oda_csv_path=bound_paths["oda_csv"],
        mask_path=bound_paths["mask_npy"],
        mask_provenance_path=bound_paths["mask_provenance"],
    )
    mask_provenance = _read_json(bound_paths["mask_provenance"])
    if input_hashes.get("source_archive") != mask_provenance.get("source_archive"):
        raise CoordinateMappingError("seed provenance source archive binding changed")
    crop = mask_provenance.get("crop")
    if not isinstance(crop, dict):
        raise CoordinateMappingError("seed mask provenance lacks crop geometry")
    crop_origin = np.asarray(crop.get("origin_zyx"), dtype=np.float64)
    coordinate_contract = value.get("coordinate_contract")
    required_contract = {
        "oda_csv_column_order": "x,y,z",
        "oda_raw_and_full_source_order": "z,y,x",
        "coordinate_order_out": "cropped_source_z,y,x",
        "forward_oda": "p_oda = (p_raw_um - sphere_center) @ rotation",
        "inverse_binned": "q_binned = (p_oda @ rotation.T + sphere_center) / 1.3",
        "binned_to_source": "q_source = 4 * q_binned + 1.5",
        "full_source_to_mask": "q_mask = q_source - crop_origin",
        "crop_origin_zyx": crop.get("origin_zyx"),
        "source_spacing_um": SOURCE_SPACING_UM,
        "binned_spacing_um": BINNED_SPACING_UM,
        "bin_factor": BIN_FACTOR,
        "source_block_center_offset_vox": SOURCE_BLOCK_CENTRE_OFFSET_VOX,
    }
    if coordinate_contract != required_contract:
        raise CoordinateMappingError("seed provenance coordinate contract changed")
    if transform.get("expected_foreground_hits") != expected_count:
        raise CoordinateMappingError("transform foreground-hit denominator changed")

    try:
        with seed_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OUTPUT_FIELDS:
                raise CoordinateMappingError("seed CSV columns changed")
            rows = list(reader)
    except OSError as exc:
        raise CoordinateMappingError(f"cannot read seed CSV: {exc}") from exc
    if len(rows) != expected_count:
        raise CoordinateMappingError("seed CSV row count changed")
    mask = np.load(bound_paths["mask_npy"], mmap_mode="r", allow_pickle=False)
    rounded_positions: set[tuple[int, int, int]] = set()
    labels: set[str] = set()
    numeric_fields = OUTPUT_FIELDS[2:]
    for lens_index, row in enumerate(rows):
        if row.get("lens_index") != str(lens_index):
            raise CoordinateMappingError("seed CSV lens_index sequence changed")
        label = row.get("seed_id", "")
        if not label or label in labels:
            raise CoordinateMappingError("seed CSV contains an empty/duplicate seed_id")
        labels.add(label)
        try:
            numbers = {field: float(row[field]) for field in numeric_fields}
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinateMappingError("seed CSV contains a non-numeric value") from exc
        if not all(math.isfinite(number) for number in numbers.values()):
            raise CoordinateMappingError("seed CSV contains a non-finite value")
        source = np.asarray(
            [numbers["source_z"], numbers["source_y"], numbers["source_x"]]
        )
        nearest = tuple(int(number) for number in np.rint(source))
        if (
            any(number < 0 or number >= mask.shape[axis] for axis, number in enumerate(nearest))
            or mask[nearest] != 1
            or nearest in rounded_positions
        ):
            raise CoordinateMappingError("seed CSV foreground/uniqueness gate failed")
        rounded_positions.add(nearest)
        oda_point = np.asarray(
            [numbers["oda_x_um"], numbers["oda_y_um"], numbers["oda_z_um"]]
        )
        expected_source = oda_to_source_vox(oda_point, center, rotation) - crop_origin
        if not np.allclose(source, expected_source, rtol=0.0, atol=ROUNDTRIP_TOLERANCE_UM):
            raise CoordinateMappingError("seed CSV coordinates do not replay")
        oda_axis = np.asarray(
            [numbers["axis_oda_x"], numbers["axis_oda_y"], numbers["axis_oda_z"]]
        )
        source_axis = np.asarray(
            [numbers["axis_source_z"], numbers["axis_source_y"], numbers["axis_source_x"]]
        )
        expected_axis = oda_axis_to_raw(
            oda_axis, rotation, input_direction="toward_eye_center"
        )
        if not np.allclose(source_axis, expected_axis, rtol=0.0, atol=1.0e-12):
            raise CoordinateMappingError("seed CSV axes do not replay")
        raw_point = oda_to_raw_um(oda_point, center, rotation)
        if float(np.dot(source_axis, raw_point - center)) <= 0.0:
            raise CoordinateMappingError("seed CSV contains a non-outward axis")
    del mask
    return value


def map_all_oda_to_source(
    public_root: Path,
    public_archive_path: Path,
    mask_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Derive and map all twelve frozen public eyes, then seal one manifest."""

    public_root = Path(public_root)
    public_archive_path = Path(public_archive_path)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    if not public_root.is_dir() or not mask_dir.is_dir():
        raise CoordinateMappingError("batch public-root and mask-dir must exist")
    if output_dir.resolve() in {public_root.resolve(), mask_dir.resolve()}:
        raise CoordinateMappingError("batch output directory must differ from every input")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise CoordinateMappingError("batch manifest already exists; pass overwrite=True")

    # Full preflight: no output is written until all 12 public payload chains and
    # all 12 source-mask/archive bindings have passed.
    inputs: list[dict[str, Path | str]] = []
    for eye_id in EXPECTED_ODA_COUNTS:
        stack_path = public_root / f"tiffs_{eye_id}_eye_lenses_binned"
        oda_csv_path = stack_path / "ommatidial_data.csv"
        h5_path = stack_path / "_compound_eye_data.h5"
        mask_path = mask_dir / f"{eye_id}.mask.uint8.npy"
        mask_provenance_path = mask_dir / f"{eye_id}.mask.json"
        for path in (oda_csv_path, h5_path, mask_path, mask_provenance_path):
            if not path.is_file():
                raise CoordinateMappingError(f"batch input does not exist: {path}")
        _verify_public_archive_eye(
            public_archive_path,
            eye_id=eye_id,
            stack_path=stack_path,
            oda_csv_path=oda_csv_path,
            h5_path=h5_path,
        )
        try:
            mask_provenance = load_and_validate_mask_provenance(
                mask_provenance_path, mask_path=mask_path, verify_archive=True
            )
        except MaskPreparationError as exc:
            raise CoordinateMappingError(
                f"invalid batch source mask for {eye_id}: {exc}"
            ) from exc
        if mask_provenance.get("eye_id") != eye_id:
            raise CoordinateMappingError(f"batch source-mask eye mismatch for {eye_id}")
        inputs.append(
            {
                "eye_id": eye_id,
                "stack": stack_path,
                "oda_csv": oda_csv_path,
                "mask": mask_path,
                "mask_provenance": mask_provenance_path,
            }
        )

    entries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in inputs:
        eye_id = str(item["eye_id"])
        eye_output = output_dir / eye_id
        transform_path = eye_output / "transform.json"
        seed_csv_path = eye_output / "seeds.csv"
        seed_json_path = eye_output / "seeds.json"
        transform = derive_oda_transform(
            Path(item["stack"]),
            transform_path,
            eye_id=eye_id,
            oda_csv_path=Path(item["oda_csv"]),
            mask_path=Path(item["mask"]),
            mask_provenance_path=Path(item["mask_provenance"]),
            public_archive_path=public_archive_path,
            overwrite=overwrite,
        )
        mapped = map_oda_to_source(
            Path(item["oda_csv"]),
            transform_path,
            Path(item["mask"]),
            Path(item["mask_provenance"]),
            seed_csv_path,
            seed_json_path,
            eye_id=eye_id,
            expected_count=EXPECTED_ODA_COUNTS[eye_id],
            overwrite=overwrite,
        )
        if (
            mapped["n_rows"] != EXPECTED_ODA_COUNTS[eye_id]
            or mapped["n_foreground_hits"] != EXPECTED_ODA_COUNTS[eye_id]
        ):
            raise CoordinateMappingError(f"incomplete mapped output for {eye_id}")
        entries.append(
            {
                "eye_id": eye_id,
                "expected_rows": EXPECTED_ODA_COUNTS[eye_id],
                "mapped_rows": mapped["n_rows"],
                "foreground_hits": mapped["n_foreground_hits"],
                "published_csv_sha256": transform["oda_csv_sha256"],
                "source_mask_sha256": transform["mask_npy_sha256"],
                "crop_origin_zyx": transform["source_mask"]["crop_origin_zyx"],
                "sphere_center_xyz_um": transform["sphere_center_xyz_um"],
                "rotation": transform["rotation"],
                "transform_json": {
                    "path": f"{eye_id}/transform.json",
                    "sha256": sha256_file(transform_path),
                },
                "seed_csv": {
                    "path": f"{eye_id}/seeds.csv",
                    "sha256": sha256_file(seed_csv_path),
                },
                "seed_provenance": {
                    "path": f"{eye_id}/seeds.json",
                    "sha256": sha256_file(seed_json_path),
                },
            }
        )

    total = sum(int(entry["mapped_rows"]) for entry in entries)
    total_hits = sum(int(entry["foreground_hits"]) for entry in entries)
    expected_total = sum(EXPECTED_ODA_COUNTS.values())
    if len(entries) != 12 or total != expected_total or total_hits != expected_total:
        raise CoordinateMappingError("batch did not produce the frozen 12-eye denominator")
    manifest: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oda_commit": ODA_COMMIT,
        "seed_role": SEED_ROLE,
        "public_archive": {
            "path": str(public_archive_path.resolve()),
            "size_bytes": FIG3_SHARE_SIZE_BYTES,
            "sha256": FIG3_SHARE_SHA256,
            "md5": FIG3_SHARE_MD5,
            "inner_member": FIG3_INNER_MEMBER,
            "inner_sha256": FIG3_STACKS_ZIP_SHA256,
        },
        "n_eyes": len(entries),
        "expected_eye_ids": list(EXPECTED_ODA_COUNTS),
        "total_expected_rows": expected_total,
        "total_mapped_rows": total,
        "total_foreground_hits": total_hits,
        "all_centres_on_source_foreground": True,
        "entries": entries,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oda-csv", type=Path)
    parser.add_argument(
        "--oda-stack",
        type=Path,
        help="public binned TIFF directory; derive --transform before mapping",
    )
    parser.add_argument(
        "--public-archive",
        type=Path,
        help="frozen official fig3_share.zip (required for transform derivation)",
    )
    parser.add_argument(
        "--transform", "--transform-json", type=Path,
        help="validated transform input, or output when --oda-stack is supplied",
    )
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-provenance", type=Path)
    parser.add_argument("--eye-id")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", "--provenance", type=Path)
    parser.add_argument(
        "--batch-public-root",
        type=Path,
        help="run all 12 eyes from the extracted public stacks under this directory",
    )
    parser.add_argument("--batch-mask-dir", type=Path)
    parser.add_argument("--batch-output-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--axis-column")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    batch_values = (
        arguments.batch_public_root,
        arguments.batch_mask_dir,
        arguments.batch_output_dir,
    )
    if any(value is not None for value in batch_values):
        if not all(value is not None for value in batch_values) or arguments.public_archive is None:
            parser.error(
                "batch mode requires --batch-public-root, --batch-mask-dir, "
                "--batch-output-dir, and --public-archive"
            )
        result = map_all_oda_to_source(
            arguments.batch_public_root,
            arguments.public_archive,
            arguments.batch_mask_dir,
            arguments.batch_output_dir,
            overwrite=arguments.force,
        )
        print(
            f"mapped {result['n_eyes']} eyes / {result['total_mapped_rows']} "
            f"published ODA seeds -> {arguments.batch_output_dir}"
        )
        return 0
    required_single = {
        "--oda-csv": arguments.oda_csv,
        "--transform": arguments.transform,
        "--mask": arguments.mask,
        "--mask-provenance": arguments.mask_provenance,
        "--eye-id": arguments.eye_id,
        "--output-csv": arguments.output_csv,
        "--output-json": arguments.output_json,
    }
    missing = [name for name, value in required_single.items() if value is None]
    if missing:
        parser.error("single-eye mode requires " + ", ".join(missing))
    if arguments.public_archive is not None and arguments.oda_stack is None:
        raise SystemExit("--public-archive requires --oda-stack")
    if arguments.oda_stack is not None:
        if arguments.public_archive is None:
            parser.error("--oda-stack requires the frozen --public-archive")
        transform = derive_oda_transform(
            arguments.oda_stack,
            arguments.transform,
            eye_id=arguments.eye_id,
            oda_csv_path=arguments.oda_csv,
            mask_path=arguments.mask,
            mask_provenance_path=arguments.mask_provenance,
            public_archive_path=arguments.public_archive,
            overwrite=arguments.force,
        )
        print(
            f"{transform['eye_id']}: derived ODA transform from "
            f"{transform['oda_binned_stack']['file_count']} public TIFFs"
        )
    result = map_oda_to_source(
        arguments.oda_csv,
        arguments.transform,
        arguments.mask,
        arguments.mask_provenance,
        arguments.output_csv,
        arguments.output_json,
        eye_id=arguments.eye_id,
        expected_count=arguments.expected_count,
        axis_column=arguments.axis_column,
        overwrite=arguments.force,
    )
    print(
        f"{result['eye_id']}: mapped {result['n_rows']} ODA seeds -> "
        f"{arguments.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

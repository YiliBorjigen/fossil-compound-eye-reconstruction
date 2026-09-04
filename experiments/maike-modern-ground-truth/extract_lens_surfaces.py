#!/usr/bin/env python3
"""Build the leakage-isolated Experiment 63 per-eye analysis bundle.

The program deliberately has two stages.  Stage 1 may use the supplied ODA
centre/axis to assign each binary lens voxel and to identify its distal cap.
It then writes a deliberately narrow, hash-bound distal artifact.  Stage 2
re-opens only those artifacts; all eye geometry and model inputs are therefore
derived from distal points alone.  Proximal points are attached afterwards as
held-out targets.

This is a conditional/oracle localisation experiment.  It is not an automatic
outer-surface detector and must not be described as one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


BUNDLE_SCHEMA = "experiment63-eye-bundle-v2"
INSTANCE_SCHEMA = "experiment63.instance.v2"
SEALED_DISTAL_SCHEMA = "experiment63.sealed-distal.v2"
LENS_SCHEMA = "experiment63.lens.v2"
ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
ISOLATION_BASIS = "stage2_reads_only_sha256_sealed_distal_artifacts"
NPZ_COMPRESSION_LEVEL = 1

DEFAULT_THRESHOLD_CONFIG: dict[str, Any] = {
    "distal_scale_statistic": "q90_radius_um",
    "distal_scale_min_um": 3.0,
    "distal_scale_max_um": 13.0,
    "min_sealed_distal_cap_points": 25,
    "distal_fit_rmse_max_um": 2.5,
    "quadratic_design_condition_max": 1.0e6,
    "fixed_point_max_iterations": 20,
    "lateral_bin_um": 0.325,
    "min_axial_span_um": 0.650,
    "connectivity": 26,
    "candidate_seeds_per_voxel": 1,
}

DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "fixed_point_policy": "monotone_drop_only_no_reentry",
    "target_main_component_fraction_min": 0.99,
    "target_min_support": 25,
    "target_fit_rmse_max_um": 2.5,
    "target_q05_thickness_min_um_exclusive": 0.0,
    "original_spacing_um": [0.325, 0.325, 0.325],
    "distal_split": "largest_26_component_boundary_deterministic_1d_two_means",
    "canonical_grid": "experiment57_disk_radius_0.65_step_0.13",
}


class ExtractionError(RuntimeError):
    """Raised when an input or an invariant of the frozen pipeline fails."""


@dataclass(frozen=True)
class Seed:
    lens_index: int
    seed_id: str
    centre_zyx: np.ndarray
    outward_axis_zyx: np.ndarray


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}


def _require_exact_staging_files(staging: Path, expected: set[str]) -> None:
    actual = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ExtractionError(
            f"bundle staging inventory differs from contract; missing={missing}, extra={extra}"
        )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: ("true" if value else "false") if isinstance(value, (bool, np.bool_)) else value
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        # numpy.savez embeds the wall-clock time in ZIP headers.  Fixed headers
        # make identical sealed distal content byte-identical and hash-stable.
        with open(temporary, "wb") as raw_handle:
            with zipfile.ZipFile(
                raw_handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=NPZ_COMPRESSION_LEVEL,
                strict_timestamps=True,
            ) as archive:
                for key in sorted(arrays):
                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer, np.asanyarray(arrays[key]), allow_pickle=False
                    )
                    info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(
                        info,
                        buffer.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=NPZ_COMPRESSION_LEVEL,
                    )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtractionError(f"JSON root must be an object: {path}")
    return value


def _find_column(names: Iterable[str], alternatives: Sequence[str]) -> str | None:
    available = set(names)
    for candidate in alternatives:
        if candidate in available:
            return candidate
    return None


def load_seeds(path: Path) -> list[Seed]:
    """Load the mapper CSV and require one explicit, outward-oriented axis/seed."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        names = reader.fieldnames or []
    if not rows:
        raise ExtractionError("seed CSV has no rows")

    index_key = _find_column(names, ["lens_index", "seed_index"])
    id_key = _find_column(names, ["seed_id", "lens_id", "oda_lens_id"])
    z_key = _find_column(names, ["source_z", "seed_source_z", "q_source_z"])
    y_key = _find_column(names, ["source_y", "seed_source_y", "q_source_y"])
    x_key = _find_column(names, ["source_x", "seed_source_x", "q_source_x"])
    az_key = _find_column(names, ["axis_source_z", "outward_axis_source_z"])
    ay_key = _find_column(names, ["axis_source_y", "outward_axis_source_y"])
    ax_key = _find_column(names, ["axis_source_x", "outward_axis_source_x"])
    required = [index_key, z_key, y_key, x_key, az_key, ay_key, ax_key]
    if any(value is None for value in required):
        raise ExtractionError(
            "seed CSV must contain lens_index, source_[zyx], and axis_source_[zyx] columns"
        )

    result: list[Seed] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            index = int(row[index_key])  # type: ignore[index]
            centre = np.asarray(
                [float(row[z_key]), float(row[y_key]), float(row[x_key])],  # type: ignore[index]
                dtype=np.float64,
            )
            axis = np.asarray(
                [float(row[az_key]), float(row[ay_key]), float(row[ax_key])],  # type: ignore[index]
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ExtractionError(f"invalid seed CSV value on row {row_number}") from exc
        if index != row_number - 2:
            raise ExtractionError("lens_index must be the complete ordered range 0..n-1")
        if not np.all(np.isfinite(centre)) or not np.all(np.isfinite(axis)):
            raise ExtractionError(f"non-finite seed/axis on row {row_number}")
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            raise ExtractionError(f"zero ODA axis on row {row_number}")
        result.append(
            Seed(
                lens_index=index,
                seed_id=(row[id_key] if id_key else str(index)),
                centre_zyx=centre,
                outward_axis_zyx=axis / norm,
            )
        )
    centres = np.stack([seed.centre_zyx for seed in result])
    if np.unique(centres, axis=0).shape[0] != len(result):
        raise ExtractionError("duplicate source seed centres are not allowed")
    return result


def _validate_seed_provenance(
    path: Path,
    seed_csv: Path,
    *,
    eye_id: str,
) -> dict[str, Any]:
    """Delegate to the mapper's complete v3 artifact-chain validator."""

    try:
        from map_oda_to_source import (
            CoordinateMappingError,
            load_and_validate_seed_provenance,
        )
    except ImportError:  # supports explicit file-based imports in tests/tools
        import importlib.util

        module_path = Path(__file__).with_name("map_oda_to_source.py")
        spec = importlib.util.spec_from_file_location("map_oda_to_source", module_path)
        if spec is None or spec.loader is None:
            raise ExtractionError("cannot load map_oda_to_source.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CoordinateMappingError = module.CoordinateMappingError
        load_and_validate_seed_provenance = module.load_and_validate_seed_provenance
    try:
        return load_and_validate_seed_provenance(
            path,
            seed_csv_path=seed_csv,
            eye_id=eye_id,
        )
    except CoordinateMappingError as exc:
        raise ExtractionError(f"invalid mapped-seed provenance: {exc}") from exc


def _validate_mask(mask_path: Path, provenance_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from prepare_maike_masks import (
            MaskPreparationError,
            load_and_validate_mask_provenance,
        )
    except ImportError:  # supports explicit file-based imports in tests/tools
        import importlib.util

        module_path = Path(__file__).with_name("prepare_maike_masks.py")
        spec = importlib.util.spec_from_file_location("prepare_maike_masks", module_path)
        if spec is None or spec.loader is None:
            raise ExtractionError("cannot load prepare_maike_masks.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MaskPreparationError = module.MaskPreparationError
        load_and_validate_mask_provenance = module.load_and_validate_mask_provenance
    try:
        provenance = load_and_validate_mask_provenance(
            provenance_path,
            mask_path=mask_path,
            verify_archive=True,
        )
    except MaskPreparationError as exc:
        raise ExtractionError(f"invalid prepared mask/provenance: {exc}") from exc
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    return mask, provenance


def _require_seed_mask_cross_binding(
    seed_provenance: Mapping[str, Any],
    *,
    mask_path: Path,
    mask_provenance_path: Path,
) -> None:
    """Require the mapper and extractor to consume the same mask pair bytes."""

    input_hashes = seed_provenance.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ExtractionError("seed provenance lacks input_hashes")
    for key, path in (
        ("mask_npy", Path(mask_path)),
        ("mask_provenance", Path(mask_provenance_path)),
    ):
        binding = input_hashes.get(key)
        observed = _artifact_entry(path)
        if (
            not isinstance(binding, dict)
            or binding.get("sha256") != observed["sha256"]
            or binding.get("size_bytes") != observed["size_bytes"]
        ):
            raise ExtractionError(
                f"extractor {key} is not the exact artifact bound by seed provenance"
            )


def deterministic_largest_component(points_zyx: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the deterministic largest 26-connected component and all sizes.

    Equal-sized components are resolved by the lexicographically smallest
    occupied ``(z,y,x)`` voxel.  Returned coordinates are lexicographically
    sorted so adding/removing disconnected specks cannot alter serialized caps.
    """

    points = np.asarray(points_zyx, dtype=np.int32)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ExtractionError("component points must have shape (n,3)")
    if len(points) == 0:
        return points.copy(), []
    low = points.min(axis=0)
    high = points.max(axis=0)
    shape = tuple((high - low + 1).astype(int))
    local = np.zeros(shape, dtype=bool)
    shifted = points - low
    local[tuple(shifted.T)] = True
    labels, number = ndimage.label(local, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel(), minlength=number + 1)[1:]
    candidates: list[tuple[int, tuple[int, int, int], int]] = []
    for label_value, count in enumerate(counts, start=1):
        coords = np.argwhere(labels == label_value) + low
        lex_min = tuple(coords[np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))[0]].tolist())
        candidates.append((-int(count), lex_min, label_value))
    candidates.sort()
    chosen = candidates[0][2]
    component = np.argwhere(labels == chosen).astype(np.int32) + low
    order = np.lexsort((component[:, 2], component[:, 1], component[:, 0]))
    return component[order], sorted((int(v) for v in counts), reverse=True)


def target_component_fraction_passes(
    main_component_size: int, full_assigned_size: int, threshold: float = 0.99
) -> bool:
    """Evaluate the target-only component gate, exactly at the frozen boundary."""

    if full_assigned_size <= 0 or main_component_size < 0:
        return False
    if threshold == 0.99:
        return main_component_size * 100 >= full_assigned_size * 99
    return main_component_size / full_assigned_size >= threshold


def _deterministic_two_means(values: np.ndarray, max_iterations: int = 100) -> tuple[np.ndarray, tuple[float, float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ExtractionError("two-means requires at least two finite projections")
    centres = np.asarray([float(values.min()), float(values.max())], dtype=np.float64)
    if centres[0] == centres[1]:
        raise ExtractionError("cannot split a zero-span boundary")
    labels = np.zeros(len(values), dtype=np.int8)
    for _ in range(max_iterations):
        midpoint = float((centres[0] + centres[1]) / 2.0)
        new_labels = (values > midpoint).astype(np.int8)  # ties deterministically enter low group
        if not np.any(new_labels == 0) or not np.any(new_labels == 1):
            raise ExtractionError("degenerate boundary split")
        new_centres = np.asarray(
            [values[new_labels == 0].mean(), values[new_labels == 1].mean()], dtype=np.float64
        )
        if np.array_equal(new_labels, labels) and np.array_equal(new_centres, centres):
            labels = new_labels
            centres = new_centres
            break
        labels = new_labels
        if np.allclose(new_centres, centres, rtol=0.0, atol=1e-12):
            centres = new_centres
            break
        centres = new_centres
    if centres[0] > centres[1]:
        labels = 1 - labels
        centres = centres[::-1]
    return labels, (float(centres[0]), float(centres[1]))


def localize_distal_cap(
    component_points_zyx: np.ndarray,
    centre_zyx: np.ndarray,
    outward_axis_zyx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use only the Stage-1 oracle centre/axis to select the outward boundary mode."""

    points = np.asarray(component_points_zyx, dtype=np.int32)
    if len(points) == 0:
        return points.copy(), {"status": "empty_component"}
    low = points.min(axis=0)
    high = points.max(axis=0)
    local = np.zeros(tuple((high - low + 1).astype(int)), dtype=bool)
    local[tuple((points - low).T)] = True
    eroded = ndimage.binary_erosion(
        local,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
        border_value=0,
    )
    boundary = np.argwhere(local & ~eroded).astype(np.int32) + low
    projections = (boundary.astype(np.float64) - centre_zyx) @ outward_axis_zyx
    try:
        labels, centres = _deterministic_two_means(projections)
    except ExtractionError as exc:
        return np.empty((0, 3), dtype=np.int32), {
            "status": "split_failed",
            "reason": str(exc),
            "boundary_support": int(len(boundary)),
        }
    distal = boundary[labels == 1]
    order = np.lexsort((distal[:, 2], distal[:, 1], distal[:, 0]))
    return distal[order].astype(np.int32, copy=False), {
        "status": "ok",
        "boundary_support": int(len(boundary)),
        "distal_support": int(len(distal)),
        "projection_cluster_centres_vox": list(centres),
    }


def _partition_foreground(
    mask: np.ndarray,
    seeds: Sequence[Seed],
    *,
    slab_depth: int = 8,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Losslessly assign each foreground voxel to its one nearest source seed."""

    seed_points = np.stack([seed.centre_zyx for seed in seeds])
    shape = np.asarray(mask.shape, dtype=np.float64)
    for seed in seeds:
        if np.any(seed.centre_zyx < 0) or np.any(seed.centre_zyx >= shape):
            raise ExtractionError(f"seed {seed.lens_index} falls outside mask")
        nearest = np.rint(seed.centre_zyx).astype(int)
        if int(mask[tuple(nearest)]) != 1:
            raise ExtractionError(f"seed {seed.lens_index} is not inside foreground")
    tree = cKDTree(seed_points)

    def nearest_labels(coordinates: np.ndarray) -> np.ndarray:
        if len(seeds) == 1:
            return np.zeros(len(coordinates), dtype=np.int64)
        distances, candidates = tree.query(
            coordinates.astype(np.float64), k=2, workers=1
        )
        labels = np.asarray(candidates[:, 0], dtype=np.int64)
        tied_rows = np.flatnonzero(
            np.isclose(distances[:, 0], distances[:, 1], rtol=0.0, atol=1e-12)
        )
        for row in tied_rows:
            point = coordinates[row].astype(np.float64)
            radius = float(distances[row, 0]) + 1e-12
            neighbours = np.asarray(tree.query_ball_point(point, radius), dtype=np.int64)
            squared = np.sum((seed_points[neighbours] - point) ** 2, axis=1)
            minimum = float(squared.min())
            labels[row] = int(neighbours[np.isclose(squared, minimum, rtol=0.0, atol=1e-12)].min())
        return labels

    counts = np.zeros(len(seeds), dtype=np.int64)
    source_foreground = 0
    for start in range(0, mask.shape[0], slab_depth):
        coordinates = np.argwhere(np.asarray(mask[start : start + slab_depth]) == 1).astype(np.int32)
        if len(coordinates) == 0:
            continue
        coordinates[:, 0] += start
        labels = nearest_labels(coordinates)
        counts += np.bincount(labels, minlength=len(seeds)).astype(np.int64)
        source_foreground += int(len(coordinates))

    offsets = np.concatenate(([0], np.cumsum(counts)))
    storage = np.empty((source_foreground, 3), dtype=np.int32)
    cursors = offsets[:-1].copy()
    for start in range(0, mask.shape[0], slab_depth):
        coordinates = np.argwhere(np.asarray(mask[start : start + slab_depth]) == 1).astype(np.int32)
        if len(coordinates) == 0:
            continue
        coordinates[:, 0] += start
        labels = nearest_labels(coordinates)
        order = np.argsort(labels, kind="stable")
        ordered_labels = labels[order]
        ordered_coordinates = coordinates[order]
        unique, begin, group_counts = np.unique(ordered_labels, return_index=True, return_counts=True)
        for label_value, group_begin, group_count in zip(unique, begin, group_counts, strict=True):
            cursor = int(cursors[label_value])
            storage[cursor : cursor + group_count] = ordered_coordinates[
                group_begin : group_begin + group_count
            ]
            cursors[label_value] += group_count
    if not np.array_equal(cursors, offsets[1:]):
        raise ExtractionError("internal partition fill-count mismatch")
    instances = [storage[offsets[i] : offsets[i + 1]].copy() for i in range(len(seeds))]
    evidence = {
        "source_foreground_voxel_count": int(source_foreground),
        "assigned_voxel_count": int(sum(len(value) for value in instances)),
        "assigned_unique_voxel_count": int(sum(len(value) for value in instances)),
        "unassigned_foreground_voxel_count": 0,
        "multiply_assigned_voxel_count": 0,
        "exact_partition": bool(sum(len(value) for value in instances) == source_foreground),
        "candidate_seeds_per_voxel": 1,
    }
    if not evidence["exact_partition"]:
        raise ExtractionError("centre-Voronoi assignment was not a lossless partition")
    return instances, evidence


def _sealed_payload(
    lens_index: int,
    points_zyx: np.ndarray,
    spacing_um: Sequence[float],
    threshold_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    config_json = _canonical_json(threshold_config)
    return {
        "schema_version": np.asarray(SEALED_DISTAL_SCHEMA),
        "lens_index": np.asarray(lens_index, dtype=np.int64),
        "points_zyx": np.asarray(points_zyx, dtype=np.int32),
        "spacing_um": np.asarray(spacing_um, dtype=np.float64),
        "config_json": np.asarray(config_json),
        "config_sha256": np.asarray(_sha256_bytes(config_json.encode("utf-8"))),
    }


def _load_geometry_module() -> Any:
    try:
        import distal_only_geometry as geometry
    except ImportError:  # allows import through an explicit package/file loader
        import importlib.util

        module_path = Path(__file__).with_name("distal_only_geometry.py")
        spec = importlib.util.spec_from_file_location("distal_only_geometry", module_path)
        if spec is None or spec.loader is None:
            raise ExtractionError("cannot load distal_only_geometry.py")
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
    return geometry


def _load_frame_audit_module() -> Any:
    import importlib.util

    module_path = Path(__file__).with_name("audit_distal_frame_stability.py")
    spec = importlib.util.spec_from_file_location("audit_distal_frame_stability", module_path)
    if spec is None or spec.loader is None:
        raise ExtractionError("cannot load audit_distal_frame_stability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_grid() -> np.ndarray:
    axis = np.arange(-0.65, 0.65 + 0.13 / 2.0, 0.13, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    inside = grid_x * grid_x + grid_y * grid_y <= 0.65 * 0.65 + 1e-12
    result = np.column_stack([grid_x[inside], grid_y[inside]])
    if result.shape != (81, 2):
        raise ExtractionError("canonical Experiment 57 grid is not 81 points")
    return result


def _quadratic_design(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    return np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])


def select_spanning_lateral_bin_minima(
    points_xyz_um: np.ndarray,
    local_xyz_um: np.ndarray,
    *,
    scale_um: float,
    lateral_bin_um: float,
    min_axial_span_um: float,
) -> np.ndarray:
    """Select one inward axial extreme per spanning bin, tied in physical XYZ.

    The explicit physical-coordinate tie break matches the Arthur adapter even
    though Maike's source array itself is indexed in ZYX order.
    """

    points = np.asarray(points_xyz_um, dtype=np.float64)
    local = np.asarray(local_xyz_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or local.shape != points.shape:
        raise ExtractionError("target-bin points/local coordinates must both have shape (n,3)")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(local)):
        raise ExtractionError("target-bin coordinates must be finite")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (scale_um, lateral_bin_um, min_axial_span_um)
    ):
        raise ExtractionError("target-bin scale, width, and axial span must be positive")
    bins = np.floor(local[:, :2] / lateral_bin_um + 0.5).astype(np.int64)
    within_cap = np.linalg.norm(local[:, :2], axis=1) <= scale_um + 1e-12
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(within_cap):
        pair = bins[index]
        groups[(int(pair[0]), int(pair[1]))].append(int(index))
    selected: list[int] = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        axial = local[indices, 2]
        if float(axial.max() - axial.min()) + 1e-12 < min_axial_span_um:
            continue
        minimum = float(axial.min())
        tied = indices[np.flatnonzero(np.isclose(axial, minimum, rtol=0.0, atol=1e-12))]
        candidate_points_xyz = points[tied]
        choice_order = np.lexsort(
            (
                candidate_points_xyz[:, 2],
                candidate_points_xyz[:, 1],
                candidate_points_xyz[:, 0],
            )
        )
        selected.append(int(tied[choice_order[0]]))
    return np.asarray(selected, dtype=np.int64)


def _extract_target(
    component_zyx: np.ndarray,
    distal_xyz_um: np.ndarray,
    geometry: Mapping[str, Any],
    threshold_config: Mapping[str, Any],
    robust_quadratic_fit: Any | None = None,
) -> dict[str, Any]:
    """Attach a proximal target only after distal-only frame construction."""

    empty = {
        "target_resolvable": False,
        "target_qc": False,
        "target_resolvability_reasons": [],
        "target_qc_reasons": [],
        "proximal_points_xyz_um": np.empty((0, 3), dtype=np.float64),
        "raw_target_xy_normalized": np.empty((0, 2), dtype=np.float64),
        "raw_target_thickness_um": np.empty((0,), dtype=np.float64),
        "target_coefficients": np.full(6, np.nan, dtype=np.float64),
        "target_depth_um": math.nan,
        "target_support": 0,
        "target_q05_raw_thickness_um": math.nan,
        "target_rmse_um": math.nan,
    }
    if len(component_zyx) == 0 or len(distal_xyz_um) == 0:
        empty["target_resolvability_reasons"] = ["empty_component_or_distal"]
        empty["target_qc_reasons"] = ["empty_component_or_distal"]
        return empty
    origin = np.asarray(geometry["origin_xyz_um"], dtype=np.float64)
    u_axis = np.asarray(geometry["u_axis_xyz"], dtype=np.float64)
    v_axis = np.asarray(geometry["v_axis_xyz"], dtype=np.float64)
    w_axis = np.asarray(geometry["outward_axis_xyz"], dtype=np.float64)
    scale = float(geometry["distal_scale_um"])
    if not np.isfinite(scale) or scale <= 0:
        empty["target_resolvability_reasons"] = ["invalid_distal_scale"]
        empty["target_qc_reasons"] = ["invalid_distal_scale"]
        return empty
    spacing_xyz = np.asarray(threshold_config["original_spacing_um"], dtype=np.float64)[::-1]
    all_xyz = component_zyx[:, ::-1].astype(np.float64) * spacing_xyz
    relative = all_xyz - origin
    local = np.column_stack([relative @ u_axis, relative @ v_axis, relative @ w_axis])
    distal_relative = distal_xyz_um - origin
    distal_local = np.column_stack(
        [distal_relative @ u_axis, distal_relative @ v_axis, distal_relative @ w_axis]
    )
    distal_xy_norm = distal_local[:, :2] / scale
    stored_distal_coefficients = np.asarray(
        geometry.get("quadratic_beta_normalized", np.full(6, np.nan)),
        dtype=np.float64,
    )
    if stored_distal_coefficients.shape == (6,) and np.all(
        np.isfinite(stored_distal_coefficients)
    ):
        distal_coefficients = stored_distal_coefficients
    else:
        distal_within_cap = np.linalg.norm(distal_xy_norm, axis=1) <= 1.0 + 1e-12
        try:
            distal_coefficients, *_ = np.linalg.lstsq(
                _quadratic_design(distal_xy_norm[distal_within_cap]),
                distal_local[distal_within_cap, 2],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            empty["target_resolvability_reasons"] = ["distal_refit_failed"]
            empty["target_qc_reasons"] = ["distal_refit_failed"]
            return empty

    selected = select_spanning_lateral_bin_minima(
        all_xyz,
        local,
        scale_um=scale,
        lateral_bin_um=float(threshold_config["lateral_bin_um"]),
        min_axial_span_um=float(threshold_config["min_axial_span_um"]),
    )
    if len(selected) == 0:
        empty["target_resolvability_reasons"] = ["no_spanning_lateral_bins"]
        empty["target_qc_reasons"] = ["no_spanning_lateral_bins"]
        return empty

    proximal_xyz = all_xyz[selected]
    proximal_local = local[selected]
    xy_norm = proximal_local[:, :2] / scale
    distal_z_at_target = _quadratic_design(xy_norm) @ distal_coefficients
    thickness = distal_z_at_target - proximal_local[:, 2]
    resolvability_reasons: list[str] = []
    sensitivity_reasons: list[str] = []
    support = int(len(thickness))
    if support < int(threshold_config["target_min_support"]):
        resolvability_reasons.append("target_support_below_minimum")
    if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(xy_norm)):
        resolvability_reasons.append("nonfinite_target")
    if support:
        q05 = float(np.quantile(thickness, 0.05))
        target_depth = float(np.median(thickness))
        if not q05 > float(threshold_config["target_q05_thickness_min_um_exclusive"]):
            sensitivity_reasons.append("target_q05_raw_thickness_not_positive")
    else:
        q05 = math.nan
        target_depth = math.nan
    coefficients = np.full(6, np.nan, dtype=np.float64)
    rmse = math.nan
    if support >= 6 and np.all(np.isfinite(thickness)) and np.all(np.isfinite(xy_norm)):
        design = _quadratic_design(xy_norm)
        try:
            if robust_quadratic_fit is None:
                proximal_coefficients, *_ = np.linalg.lstsq(
                    design, proximal_local[:, 2], rcond=None
                )
            else:
                geometry_config = {
                    key: threshold_config[key] for key in DEFAULT_THRESHOLD_CONFIG
                }
                shared_fit = robust_quadratic_fit(
                    xy_norm[:, 0],
                    xy_norm[:, 1],
                    proximal_local[:, 2],
                    config=geometry_config,
                )
                proximal_coefficients = np.asarray(
                    shared_fit["coefficients"], dtype=np.float64
                )
            coefficients = distal_coefficients - proximal_coefficients
            # Algebraically this is the proximal-surface residual.  Keeping it
            # in thickness form makes the stored target convention explicit.
            residual = thickness - design @ coefficients
            rmse = float(np.sqrt(np.mean(residual * residual)))
        except (KeyError, ValueError, np.linalg.LinAlgError):
            resolvability_reasons.append("target_fit_failed")
        else:
            if coefficients.shape != (6,) or not np.all(np.isfinite(coefficients)) or not math.isfinite(rmse):
                resolvability_reasons.append("target_fit_failed")
            elif rmse > float(threshold_config["target_fit_rmse_max_um"]):
                sensitivity_reasons.append("target_rmse_above_maximum")
    else:
        resolvability_reasons.append("target_fit_unavailable")
    resolvability_reasons = sorted(set(resolvability_reasons))
    qc_reasons = sorted(set(resolvability_reasons + sensitivity_reasons))
    return {
        "target_resolvable": not resolvability_reasons,
        "target_qc": not qc_reasons,
        "target_resolvability_reasons": resolvability_reasons,
        "target_qc_reasons": qc_reasons,
        "proximal_points_xyz_um": proximal_xyz,
        "raw_target_xy_normalized": xy_norm,
        "raw_target_thickness_um": thickness,
        "target_coefficients": coefficients,
        "target_depth_um": target_depth,
        "target_q05_raw_thickness_um": q05 if not resolvability_reasons else math.nan,
        "target_support": support,
        "target_rmse_um": rmse,
    }


def _git_identity(repository_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExtractionError(f"cannot establish git identity: {exc}") from exc
    return {"commit": commit, "dirty": dirty}


def _normalise_geometry_result(
    result: Any, n: int, max_iterations: int
) -> tuple[set[int], dict[str, Any], dict[int, dict[str, Any]]]:
    """Accept the stable geometry API while keeping serialization strict."""

    if not isinstance(result, Mapping):
        raise ExtractionError("distal-only fixed-point result must be a mapping")
    eligible = {int(value) for value in result.get("eligible_indices", [])}
    iterations = int(result.get("iterations", 0))
    eligible_counts = [int(value) for value in result.get("eligible_counts", [])]
    per_lens_raw = result.get("per_lens", {})
    if isinstance(per_lens_raw, Sequence) and not isinstance(per_lens_raw, (str, bytes)):
        per_lens = {}
        for row in per_lens_raw:
            value = dict(row)
            metrics = value.pop("metrics", {})
            if isinstance(metrics, Mapping):
                value.update(dict(metrics))
            if "reasons" not in value:
                value["reasons"] = value.get(
                    "drop_reasons", value.get("distal_qc_reasons", [])
                )
            per_lens[int(row["lens_index"])] = value
    elif isinstance(per_lens_raw, Mapping):
        per_lens = {int(key): dict(value) for key, value in per_lens_raw.items()}
    else:
        raise ExtractionError("invalid per_lens fixed-point result")
    if any(value < 0 or value >= n for value in eligible):
        raise ExtractionError("fixed-point returned an out-of-range lens index")
    if not bool(result.get("converged", False)):
        raise ExtractionError("distal-only drop-only fixed point did not converge")
    if iterations < 0 or iterations > max_iterations:
        raise ExtractionError("invalid fixed-point iteration count")
    if not eligible_counts:
        raise ExtractionError("fixed-point must report eligible_counts")
    if any(right > left for left, right in zip(eligible_counts, eligible_counts[1:])):
        raise ExtractionError("fixed-point eligible counts are not monotone")
    if iterations == 0:
        if len(eligible_counts) != 1:
            raise ExtractionError("zero-iteration fixed point must report one eligible count")
    elif len(eligible_counts) < 2 or eligible_counts[-1] != eligible_counts[-2]:
        raise ExtractionError("fixed point must end with repeated stable eligible count")
    if int(result.get("readded_count", 0)) != 0:
        raise ExtractionError("drop-only fixed point reported lens re-entry")
    fixed_point = {
        "converged": True,
        "max_iterations": int(max_iterations),
        "iterations": iterations,
        "eligible_counts": eligible_counts,
        "readded_count": 0,
    }
    return eligible, fixed_point, per_lens


def _canonical_geometry_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the shared geometry module's array-rich field names."""

    row = dict(value)
    position = np.asarray(
        row.get("position_2d_um", [row.get("position_u_um", np.nan), row.get("position_v_um", np.nan)]),
        dtype=np.float64,
    )
    curvature = np.asarray(row.get("curvature_eigenvalues", [np.nan, np.nan]), dtype=np.float64)
    return {
        **row,
        "origin_xyz_um": np.asarray(
            row.get("origin_xyz_um", row.get("origin", [np.nan] * 3)), dtype=np.float64
        ),
        "u_axis_xyz": np.asarray(row.get("u_axis_xyz", row.get("u", [np.nan] * 3)), dtype=np.float64),
        "v_axis_xyz": np.asarray(row.get("v_axis_xyz", row.get("v", [np.nan] * 3)), dtype=np.float64),
        "outward_axis_xyz": np.asarray(
            row.get("outward_axis_xyz", row.get("outward", row.get("w", [np.nan] * 3))),
            dtype=np.float64,
        ),
        "position_u_um": float(position[0]) if position.shape == (2,) else math.nan,
        "position_v_um": float(position[1]) if position.shape == (2,) else math.nan,
        "distal_scale_um": float(row.get("distal_scale_um", row.get("scale_um", math.nan))),
        "distal_gradient_magnitude": float(
            row.get("distal_gradient_magnitude", row.get("gradient_magnitude", math.nan))
        ),
        "distal_curvature_eigenvalue_1": float(
            row.get(
                "distal_curvature_eigenvalue_1",
                curvature[0] if curvature.shape == (2,) else math.nan,
            )
        ),
        "distal_curvature_eigenvalue_2": float(
            row.get(
                "distal_curvature_eigenvalue_2",
                curvature[1] if curvature.shape == (2,) else math.nan,
            )
        ),
        "distal_normalized_fit_residual": float(
            row.get("distal_normalized_fit_residual", row.get("normalised_rmse", math.nan))
        ),
    }


def build_eye_bundle(
    *,
    mask_path: Path,
    mask_provenance_path: Path,
    seeds_path: Path,
    seed_provenance_path: Path,
    eye_id: str,
    species: str,
    sex: str,
    output_path: Path,
    repository_root: Path | None = None,
    threshold_config: Mapping[str, Any] | None = None,
    pipeline_config: Mapping[str, Any] | None = None,
    allow_dirty: bool = False,
) -> Path:
    """Build one complete bundle in a sibling staging directory, then publish."""

    thresholds = dict(DEFAULT_THRESHOLD_CONFIG)
    if threshold_config is not None:
        thresholds.update(dict(threshold_config))
    pipeline = dict(DEFAULT_PIPELINE_CONFIG)
    if pipeline_config is not None:
        pipeline.update(dict(pipeline_config))
    runtime_config = {**thresholds, **pipeline}
    predictor_pipeline_config = {
        key: pipeline[key]
        for key in (
            "fixed_point_policy",
            "original_spacing_um",
            "distal_split",
            "canonical_grid",
        )
    }
    sealed_config = {
        "analysis_scope": ANALYSIS_SCOPE,
        "isolation_basis": ISOLATION_BASIS,
        "predictor_pipeline_config": predictor_pipeline_config,
        "threshold_config": thresholds,
    }
    config_json = _canonical_json(sealed_config)
    config_sha = _sha256_bytes(config_json.encode("utf-8"))
    threshold_sha = _sha256_bytes(_canonical_json(thresholds).encode("utf-8"))
    pipeline_sha = _sha256_bytes(_canonical_json(pipeline).encode("utf-8"))
    mask, mask_provenance = _validate_mask(mask_path, mask_provenance_path)
    seed_provenance = _validate_seed_provenance(
        seed_provenance_path,
        seeds_path,
        eye_id=eye_id,
    )
    _require_seed_mask_cross_binding(
        seed_provenance,
        mask_path=mask_path,
        mask_provenance_path=mask_provenance_path,
    )
    seeds = load_seeds(seeds_path)
    if mask_provenance.get("eye_id") not in (None, eye_id):
        raise ExtractionError("mask provenance eye_id differs from the requested eye")
    if seed_provenance.get("eye_id") not in (None, eye_id):
        raise ExtractionError("seed provenance eye_id differs from the requested eye")
    for count_key in ("n_expected", "n_rows"):
        recorded_count = seed_provenance.get(count_key)
        if recorded_count is not None:
            if isinstance(recorded_count, bool) or not isinstance(recorded_count, int):
                raise ExtractionError(
                    f"seed provenance {count_key} must be a JSON integer"
                )
            if recorded_count != len(seeds):
                raise ExtractionError(f"seed provenance {count_key} differs from the CSV")
    candidate_count = seed_provenance.get("candidate_seeds_per_voxel", 1)
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != 1
    ):
        raise ExtractionError("seed provenance changed candidate_seeds_per_voxel")
    repository_root = repository_root or Path(__file__).resolve().parents[2]
    git = _git_identity(repository_root)
    if git["dirty"] and not allow_dirty:
        raise ExtractionError("repository is dirty; freeze the implementation before extraction")
    if output_path.exists():
        raise ExtractionError(f"output already exists: {output_path}")
    implementation_directory = Path(__file__).resolve().parent
    implementation_names = (
        "prepare_maike_masks.py",
        "map_oda_to_source.py",
        "extract_lens_surfaces.py",
        "distal_only_geometry.py",
        "audit_distal_frame_stability.py",
    )
    implementation_hashes = {
        name: _artifact_entry(implementation_directory / name)
        for name in implementation_names
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging.", dir=output_path.parent))
    try:
        (staging / "instances").mkdir()
        (staging / "sealed_distal").mkdir()
        (staging / "lenses").mkdir()
        instances, partition = _partition_foreground(mask, seeds)

        stage1_rows: list[dict[str, Any]] = []
        components: dict[int, np.ndarray] = {}
        sealed_paths: list[Path] = []
        spacing = np.asarray(pipeline["original_spacing_um"], dtype=np.float64)
        for seed, assigned in zip(seeds, instances, strict=True):
            component, component_sizes = deterministic_largest_component(assigned)
            components[seed.lens_index] = component
            fraction = float(len(component) / len(assigned)) if len(assigned) else 0.0
            distal, split = localize_distal_cap(
                component, seed.centre_zyx, seed.outward_axis_zyx
            )
            instance_relpath = f"instances/lens_{seed.lens_index:06d}.npz"
            sealed_relpath = f"sealed_distal/lens_{seed.lens_index:06d}.npz"
            _atomic_savez(
                staging / instance_relpath,
                schema_version=np.asarray(INSTANCE_SCHEMA),
                lens_index=np.asarray(seed.lens_index, dtype=np.int64),
                full_assigned_points_zyx=np.asarray(assigned, dtype=np.int32),
                main_component_points_zyx=np.asarray(component, dtype=np.int32),
                component_sizes_descending=np.asarray(component_sizes, dtype=np.int64),
                spacing_um=spacing,
                seed_source_zyx=np.asarray(seed.centre_zyx, dtype=np.float64),
                oda_axis_source_zyx=np.asarray(seed.outward_axis_zyx, dtype=np.float64),
                config_json=np.asarray(config_json),
                config_sha256=np.asarray(config_sha),
            )
            _atomic_savez(
                staging / sealed_relpath,
                **_sealed_payload(seed.lens_index, distal, spacing, sealed_config),
            )
            sealed_paths.append(staging / sealed_relpath)
            stage1_rows.append(
                {
                    "lens_index": seed.lens_index,
                    "seed_id": seed.seed_id,
                    "assignment_status": "ok" if len(assigned) else "empty_assignment",
                    "full_assigned_size": int(len(assigned)),
                    "main_component_size": int(len(component)),
                    "component_removed_size": int(len(assigned) - len(component)),
                    "main_component_fraction": fraction,
                    "component_sizes": component_sizes,
                    "component_sizes_json": _canonical_json(component_sizes),
                    "partition_target_resolvable": target_component_fraction_passes(
                        len(component),
                        len(assigned),
                        float(pipeline["target_main_component_fraction_min"]),
                    ),
                    "distal_initial_support": int(len(distal)),
                    "stage1_status": split.get("status"),
                    "stage1_detail": split,
                    "instance_relpath": instance_relpath,
                    "sealed_distal_relpath": sealed_relpath,
                }
            )

        # Stage 2: explicitly discard the oracle structures and reopen only the
        # sealed artifacts.  The geometry function is required to verify their
        # hashes/config before deriving any frame or position.
        del instances
        sealed_stage1_manifest = {
            path.relative_to(staging).as_posix(): _artifact_entry(path)
            for path in sorted(sealed_paths)
        }
        geometry_module = _load_geometry_module()
        records = []
        for sealed_path in sealed_paths:
            relative = sealed_path.relative_to(staging).as_posix()
            if _artifact_entry(sealed_path) != sealed_stage1_manifest[relative]:
                raise ExtractionError(f"sealed distal artifact changed before Stage 2: {relative}")
            loaded = geometry_module.load_sealed_distal(
                sealed_path, expected_config_sha256=config_sha
            )
            records.append(loaded)
        record_by_index = {int(record["lens_index"]): record for record in records}
        fixed_result = geometry_module.run_monotone_fixed_point(records, config=thresholds)
        eligible, fixed_point_metadata, per_lens_qc = _normalise_geometry_result(
            fixed_result, len(seeds), int(thresholds["fixed_point_max_iterations"])
        )
        eligible_records_for_geometry = [record_by_index[index] for index in sorted(eligible)]
        geometry_result = geometry_module.derive_distal_only_eye_geometry(
            eligible_records_for_geometry, config=thresholds
        )
        if isinstance(geometry_result, Mapping) and "per_lens" in geometry_result:
            per_lens_geometry_raw = geometry_result["per_lens"]
        else:
            per_lens_geometry_raw = geometry_result
        if isinstance(per_lens_geometry_raw, Mapping):
            per_lens_geometry = {
                int(key): _canonical_geometry_entry(value)
                for key, value in per_lens_geometry_raw.items()
            }
        else:
            per_lens_geometry = {
                int(row["lens_index"]): _canonical_geometry_entry(row)
                for row in per_lens_geometry_raw
            }
        audit_module = _load_frame_audit_module()
        frame_audit = audit_module.run_frame_stability_audit(
            records, eye_id=eye_id, config=thresholds
        )
        if frame_audit.get("schema_version") != "experiment63.distal-frame-audit.v1":
            raise ExtractionError("frame audit returned an unexpected schema")
        if frame_audit.get("eye_id") != eye_id:
            raise ExtractionError("frame audit returned the wrong eye_id")
        _atomic_json(staging / "distal_frame_audit.json", frame_audit)

        rows: list[dict[str, Any]] = []
        sampling_rows: list[dict[str, Any]] = []
        canonical_grid = _canonical_grid()
        for base in stage1_rows:
            index = int(base["lens_index"])
            geom = dict(per_lens_geometry.get(index, {}))
            qc = dict(per_lens_qc.get(index, {}))
            is_eligible = index in eligible
            distal_points_xyz = np.asarray(
                record_by_index[index].get("points_xyz_um", np.empty((0, 3))), dtype=np.float64
            )
            target = _extract_target(
                components[index],
                distal_points_xyz,
                geom,
                runtime_config,
                robust_quadratic_fit=geometry_module.fit_robust_quadratic,
            ) if geom else {
                "target_resolvable": False,
                "target_qc": False,
                "target_resolvability_reasons": ["no_distal_geometry"],
                "target_qc_reasons": ["no_distal_geometry"],
                "proximal_points_xyz_um": np.empty((0, 3), dtype=np.float64),
                "raw_target_xy_normalized": np.empty((0, 2), dtype=np.float64),
                "raw_target_thickness_um": np.empty((0,), dtype=np.float64),
                "target_coefficients": np.full(6, np.nan),
                "target_depth_um": math.nan,
                "target_support": 0,
                "target_q05_raw_thickness_um": math.nan,
                "target_rmse_um": math.nan,
            }
            if not bool(base["partition_target_resolvable"]):
                target["target_resolvable"] = False
                target["target_qc"] = False
                target["target_resolvability_reasons"] = sorted(
                    set(target.get("target_resolvability_reasons", []))
                    | {"main_component_fraction_below_0.99"}
                )
                target["target_qc_reasons"] = sorted(
                    set(target.get("target_qc_reasons", [])) | {"main_component_fraction_below_0.99"}
                )
                target["target_q05_raw_thickness_um"] = math.nan
            q05_raw = float(target["target_q05_raw_thickness_um"])
            target_rmse = float(target["target_rmse_um"])
            expected_target_qc = bool(
                target["target_resolvable"]
                and math.isfinite(q05_raw)
                and q05_raw
                > float(pipeline["target_q05_thickness_min_um_exclusive"])
                and math.isfinite(target_rmse)
                and target_rmse <= float(pipeline["target_fit_rmse_max_um"])
            )
            if bool(target["target_qc"]) != expected_target_qc:
                raise ExtractionError(
                    f"target_qc equivalence failed for lens {index}"
                )
            coefficients = np.asarray(target["target_coefficients"], dtype=np.float64)
            smoothed = _quadratic_design(canonical_grid) @ coefficients
            lens_relpath = f"lenses/lens_{index:06d}.npz"
            _atomic_savez(
                staging / lens_relpath,
                schema_version=np.asarray(LENS_SCHEMA),
                lens_index=np.asarray(index, dtype=np.int64),
                distal_points_xyz_um=distal_points_xyz,
                proximal_points_xyz_um=np.asarray(target["proximal_points_xyz_um"], dtype=np.float64),
                canonical_grid_xy=canonical_grid,
                target_smoothed_thickness_um=smoothed.astype(np.float64),
                raw_target_xy_normalized=np.asarray(target["raw_target_xy_normalized"], dtype=np.float64),
                raw_target_thickness_um=np.asarray(target["raw_target_thickness_um"], dtype=np.float64),
                target_coefficients_c0_c5=coefficients,
                distal_frame_origin_xyz_um=np.asarray(geom.get("origin_xyz_um", [np.nan] * 3), dtype=np.float64),
                distal_frame_u_xyz=np.asarray(geom.get("u_axis_xyz", [np.nan] * 3), dtype=np.float64),
                distal_frame_v_xyz=np.asarray(geom.get("v_axis_xyz", [np.nan] * 3), dtype=np.float64),
                distal_frame_outward_xyz=np.asarray(geom.get("outward_axis_xyz", [np.nan] * 3), dtype=np.float64),
                distal_coefficients_normalized=np.asarray(
                    geom.get("quadratic_beta_normalized", [np.nan] * 6), dtype=np.float64
                ),
                config_json=np.asarray(config_json),
                config_sha256=np.asarray(config_sha),
            )

            target_reasons = target.get("target_qc_reasons", [])
            target_resolvability_reasons = target.get(
                "target_resolvability_reasons", []
            )
            distal_reasons = qc.get("reasons", qc.get("distal_qc_reasons", []))
            row: dict[str, Any] = {
                **{key: value for key, value in base.items() if key not in {"component_sizes", "stage1_detail"}},
                "eye_id": eye_id,
                "species": species,
                "sex": sex,
                "distal_qc": bool(is_eligible),
                "distal_eligible": bool(is_eligible),
                "distal_qc_reasons": "|".join(map(str, distal_reasons)),
                "target_resolvable": bool(target["target_resolvable"]),
                "target_resolvability_reasons": "|".join(
                    map(str, target_resolvability_reasons)
                ),
                "target_qc": bool(target["target_qc"]),
                "target_qc_reasons": "|".join(map(str, target_reasons)),
                "central": bool(geom.get("central", False)),
                "position_u_um": geom.get("position_u_um", math.nan),
                "position_v_um": geom.get("position_v_um", math.nan),
                "distal_scale_um": geom.get("distal_scale_um", qc.get("distal_scale_um", math.nan)),
                "distal_gradient_magnitude": geom.get("distal_gradient_magnitude", qc.get("distal_gradient_magnitude", math.nan)),
                "distal_curvature_eigenvalue_1": geom.get("distal_curvature_eigenvalue_1", qc.get("distal_curvature_eigenvalue_1", math.nan)),
                "distal_curvature_eigenvalue_2": geom.get("distal_curvature_eigenvalue_2", qc.get("distal_curvature_eigenvalue_2", math.nan)),
                "distal_normalized_fit_residual": geom.get("distal_normalized_fit_residual", qc.get("distal_normalized_fit_residual", math.nan)),
                "target_depth_um": target["target_depth_um"],
                "target_support": int(target["target_support"]),
                "target_q05_raw_thickness_um": target[
                    "target_q05_raw_thickness_um"
                ],
                "target_rmse_um": target["target_rmse_um"],
                "lens_relpath": lens_relpath,
            }
            for coefficient_index in range(6):
                row[f"target_c{coefficient_index}"] = coefficients[coefficient_index]
            rows.append(row)
            sampling_rows.append(
                {
                    "eye_id": eye_id,
                    "lens_index": index,
                    "seed_id": base["seed_id"],
                    "distal_eligible": bool(is_eligible),
                    "position_u_um": row["position_u_um"],
                    "position_v_um": row["position_v_um"],
                    "distal_scale_um": row["distal_scale_um"],
                    "instance_relpath": base["instance_relpath"],
                    "sealed_distal_relpath": base["sealed_distal_relpath"],
                }
            )

        summary_fields = [
            "lens_index", "seed_id", "eye_id", "species", "sex", "assignment_status",
            "full_assigned_size", "main_component_size", "component_removed_size",
            "main_component_fraction", "component_sizes_json", "partition_target_resolvable", "distal_initial_support",
            "stage1_status", "distal_qc", "distal_eligible", "distal_qc_reasons",
            "target_resolvable", "target_resolvability_reasons", "target_qc",
            "target_qc_reasons", "central",
            "position_u_um", "position_v_um", "distal_scale_um", "distal_gradient_magnitude",
            "distal_curvature_eigenvalue_1", "distal_curvature_eigenvalue_2",
            "distal_normalized_fit_residual", "target_depth_um", "target_support",
            "target_q05_raw_thickness_um", "target_rmse_um", "target_c0", "target_c1",
            "target_c2", "target_c3",
            "target_c4", "target_c5", "instance_relpath", "sealed_distal_relpath", "lens_relpath",
        ]
        sampling_fields = [
            "eye_id", "lens_index", "seed_id", "distal_eligible", "position_u_um",
            "position_v_um", "distal_scale_um", "instance_relpath", "sealed_distal_relpath",
        ]
        _atomic_csv(staging / "lens_summary.csv", rows, summary_fields)
        _atomic_csv(staging / "distal_qc_sampling.csv", sampling_rows, sampling_fields)

        expected_artifact_paths = {
            "lens_summary.csv",
            "distal_qc_sampling.csv",
            "distal_frame_audit.json",
            *{
                f"instances/lens_{index:06d}.npz" for index in range(len(seeds))
            },
            *{
                f"sealed_distal/lens_{index:06d}.npz" for index in range(len(seeds))
            },
            *{f"lenses/lens_{index:06d}.npz" for index in range(len(seeds))},
        }
        _require_exact_staging_files(staging, expected_artifact_paths)
        artifacts = {
            relative: _artifact_entry(staging / relative)
            for relative in sorted(expected_artifact_paths)
        }
        input_hashes = {
            "mask_npy": {"path": str(mask_path.resolve()), **_artifact_entry(mask_path)},
            "mask_provenance": {
                "path": str(mask_provenance_path.resolve()),
                **_artifact_entry(mask_provenance_path),
            },
            "seed_csv": {"path": str(seeds_path.resolve()), **_artifact_entry(seeds_path)},
            "seed_provenance": {
                "path": str(seed_provenance_path.resolve()),
                **_artifact_entry(seed_provenance_path),
            },
        }
        counts = {
            "stage1": len(rows),
            "distal_qc": sum(bool(row["distal_qc"]) for row in rows),
            "target_resolvable": sum(
                bool(row["target_resolvable"]) for row in rows
            ),
            "target_qc": sum(bool(row["target_qc"]) for row in rows),
        }
        implementation_hashes_after = {
            name: _artifact_entry(implementation_directory / name)
            for name in implementation_names
        }
        if implementation_hashes_after != implementation_hashes:
            raise ExtractionError("producer implementation changed during extraction")
        common = {
            "schema_version": BUNDLE_SCHEMA,
            "status": "complete",
            "eye_id": eye_id,
            "species": species,
            "sex": sex,
            "biological_independence": {
                "independent_unit": "animal",
                "animal_id": eye_id,
                "one_eye_per_animal_in_validation": True,
                "source_basis": "one supplied eye stack per uniquely named fly",
            },
            "analysis_scope": ANALYSIS_SCOPE,
            "isolation_basis": ISOLATION_BASIS,
            "threshold_config": thresholds,
            "threshold_config_sha256": threshold_sha,
            "pipeline_config": pipeline,
            "pipeline_config_sha256": pipeline_sha,
            "predictor_pipeline_config": predictor_pipeline_config,
            "sealed_config_sha256": config_sha,
            "n_expected": len(seeds),
            "n_rows": len(rows),
            "counts": counts,
            "contiguous_indices": True,
            "index_range": [0, len(seeds) - 1],
            "instance_segmentation_validated": False,
            "target_cohort_definitions": {
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
            },
            "partition_evidence": partition,
            "fixed_point": fixed_point_metadata,
            "sealed_distal_stage1_manifest": sealed_stage1_manifest,
            "input_hashes": input_hashes,
            "git": git,
            "producer_implementation_hashes": implementation_hashes,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        provenance = {
            **common,
            "mask_source_provenance": mask_provenance,
            "seed_source_provenance": seed_provenance,
            "output_manifest": artifacts,
        }
        _atomic_json(staging / "provenance.json", provenance)
        completion = {
            **common,
            "output_manifest": artifacts,
        }
        _atomic_json(staging / "completion.json", completion)
        _require_exact_staging_files(
            staging,
            expected_artifact_paths | {"completion.json", "provenance.json"},
        )
        os.replace(staging, output_path)
        return output_path
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="prepared uint8 NPY mask")
    parser.add_argument("--provenance", type=Path, required=True, help="v2 mask provenance JSON")
    parser.add_argument("--seeds", type=Path, required=True, help="mapped ODA seed CSV")
    parser.add_argument("--seed-provenance", type=Path, required=True)
    parser.add_argument("--eye-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--sex", required=True, choices=["F", "M"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development/pilot only; immutable primary outcome runs must not use this",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    result = build_eye_bundle(
        mask_path=arguments.mask,
        mask_provenance_path=arguments.provenance,
        seeds_path=arguments.seeds,
        seed_provenance_path=arguments.seed_provenance,
        eye_id=arguments.eye_id,
        species=arguments.species,
        sex=arguments.sex,
        output_path=arguments.output,
        repository_root=arguments.repository_root,
        allow_dirty=arguments.allow_dirty,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Portable annotation storage and label export for the cone GUI."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_path(nodes: list[dict[str, float]]) -> list[dict[str, float]]:
    if not nodes:
        return []
    ordered = sorted(nodes, key=lambda node: node["depth"])
    depth = np.asarray([node["depth"] for node in ordered], dtype=float)
    row = np.asarray([node["row"] for node in ordered], dtype=float)
    column = np.asarray([node["column"] for node in ordered], dtype=float)
    if len(depth) == 1:
        return [dict(ordered[0])]
    target = np.arange(int(np.ceil(depth.min())), int(np.floor(depth.max())) + 1)
    if len(depth) >= 3:
        row_fit = PchipInterpolator(depth, row)(target)
        column_fit = PchipInterpolator(depth, column)(target)
    else:
        row_fit = np.interp(target, depth, row)
        column_fit = np.interp(target, depth, column)
    return [
        {"depth": int(value), "row": float(r), "column": float(c)}
        for value, r, c in zip(target, row_fit, column_fit)
    ]


def new_project(volume_path: Path, shape: tuple[int, ...], voxel_size: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_volume": str(volume_path.resolve()),
        "source_filename": volume_path.name,
        "source_sha256": sha256(volume_path),
        "shape_uvd": list(shape),
        "voxel_size_micrometres": voxel_size,
        "cones": [],
        "background_points": [],
        "annotation_scope": (
            "manual centre-line control points and explicit background scribbles; "
            "not full anatomical cone masks"
        ),
    }


def save_project(project: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialisable = json.loads(json.dumps(project))
    for cone in serialisable["cones"]:
        cone["dense_path"] = dense_path(cone.get("nodes", []))
    path = output_dir / "annotations.json"
    path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
    return path


def load_project(path: Path) -> dict:
    project = json.loads(path.read_text(encoding="utf-8"))
    if project.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported annotation schema")
    for cone in project.get("cones", []):
        cone.pop("dense_path", None)
    return project


def draw_disk(array: np.ndarray, row: float, column: float, depth: int, radius: float, value: int) -> None:
    if not 0 <= depth < array.shape[2]:
        return
    r0 = max(0, int(np.floor(row - radius)))
    r1 = min(array.shape[0], int(np.ceil(row + radius + 1)))
    c0 = max(0, int(np.floor(column - radius)))
    c1 = min(array.shape[1], int(np.ceil(column + radius + 1)))
    rr, cc = np.ogrid[r0:r1, c0:c1]
    keep = (rr - row) ** 2 + (cc - column) ** 2 <= radius**2
    target = array[r0:r1, c0:c1, depth]
    target[keep] = value


def export_training_labels(project: dict, output_dir: Path) -> dict[str, Path]:
    shape = tuple(int(value) for value in project["shape_uvd"])
    labels = np.zeros(shape, dtype=np.uint8)
    csv_rows: list[dict[str, float | int]] = []
    for cone in project.get("cones", []):
        radius = float(cone.get("radius_voxels", 2.0))
        for point in dense_path(cone.get("nodes", [])):
            draw_disk(
                labels,
                float(point["row"]),
                float(point["column"]),
                int(point["depth"]),
                max(1.0, radius * 0.5),
                1,
            )
            csv_rows.append(
                {
                    "cone_id": int(cone["id"]),
                    "depth": int(point["depth"]),
                    "row": float(point["row"]),
                    "column": float(point["column"]),
                    "radius_voxels": radius,
                }
            )
    for point in project.get("background_points", []):
        draw_disk(
            labels,
            float(point["row"]),
            float(point["column"]),
            int(point["depth"]),
            float(point.get("radius_voxels", 2.0)),
            2,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "training_scribbles.npy"
    np.save(label_path, labels)
    csv_path = output_dir / "dense_centrelines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["cone_id", "depth", "row", "column", "radius_voxels"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return {"labels": label_path, "centrelines": csv_path}

"""Shared, testable helpers for blinded Asaphus boundary annotation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PACK_VERSION = "1.0"
ANNOTATION_VERSION = "1.0"
STRUCTURE_CLASSES = (
    "proximal lens or prism boundary",
    "cone-like internal structure",
    "dissolution or mineral-replacement front",
    "other or uncertain structure",
    "no visible boundary",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Case:
    case_id: str
    path: Path
    intensity: np.ndarray
    u_vox: np.ndarray
    v_vox: np.ndarray
    depth_vox: np.ndarray


def load_pack(folder: Path) -> tuple[dict, list[Case]]:
    folder = folder.resolve()
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pack_version") != PACK_VERSION:
        raise ValueError(
            f"Unsupported pack version: {manifest.get('pack_version')}"
        )
    cases = []
    for entry in manifest.get("cases", []):
        path = folder / entry["file"]
        with np.load(path, allow_pickle=False) as data:
            intensity = data["intensity"].astype(np.float32, copy=False)
            u_vox = data["u_vox"].astype(float, copy=False)
            v_vox = data["v_vox"].astype(float, copy=False)
            depth_vox = data["depth_vox"].astype(float, copy=False)
        expected = (len(u_vox), len(v_vox), len(depth_vox))
        if intensity.shape != expected:
            raise ValueError(
                f"{path.name}: intensity {intensity.shape}, expected {expected}"
            )
        cases.append(Case(entry["case_id"], path, intensity, u_vox, v_vox,
                          depth_vox))
    if not cases:
        raise ValueError("The annotation pack contains no cases")
    return manifest, cases


def empty_annotation(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "visibility": "uncertain",
        "structure_class": "other or uncertain structure",
        "confidence": 1,
        "notes": "",
        "u_depth_points": [],
        "v_depth_points": [],
    }


def load_annotations(path: Path, case_ids: list[str]) -> dict[str, dict]:
    records = {case_id: empty_annotation(case_id) for case_id in case_ids}
    if not path.exists():
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("annotation_version") != ANNOTATION_VERSION:
        raise ValueError(
            f"Unsupported annotation version: {payload.get('annotation_version')}"
        )
    for row in payload.get("annotations", []):
        case_id = row.get("case_id")
        if case_id in records:
            records[case_id].update(row)
    return records


def save_annotations(
    path: Path,
    pack_manifest: dict,
    annotator_id: str,
    records: dict[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotation_version": ANNOTATION_VERSION,
        "pack_sha256": pack_manifest.get("source_dataset_sha256"),
        "pack_id": pack_manifest.get("pack_id"),
        "annotator_id": annotator_id.strip(),
        "annotations": [records[key] for key in sorted(records)],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    fields = [
        "case_id", "visibility", "structure_class", "confidence", "notes",
        "view", "lateral_vox", "depth_vox", "depth_um",
    ]
    spacing = float(pack_manifest["voxel_spacing_um"])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case_id in sorted(records):
            row = records[case_id]
            points = []
            for view, field in (("u-depth", "u_depth_points"),
                                ("v-depth", "v_depth_points")):
                for point in row.get(field, []):
                    points.append((view, point))
            if not points:
                points = [("", {"lateral_vox": "", "depth_vox": ""})]
            for view, point in points:
                depth = point.get("depth_vox", "")
                writer.writerow({
                    "case_id": case_id,
                    "visibility": row.get("visibility", ""),
                    "structure_class": row.get("structure_class", ""),
                    "confidence": row.get("confidence", ""),
                    "notes": row.get("notes", ""),
                    "view": view,
                    "lateral_vox": point.get("lateral_vox", ""),
                    "depth_vox": depth,
                    "depth_um": "" if depth == "" else float(depth) * spacing,
                })


def point_from_canvas(
    x: float,
    y: float,
    width: int,
    height: int,
    lateral: np.ndarray,
    depth: np.ndarray,
) -> dict:
    lateral_value = float(np.interp(x, [0, max(width - 1, 1)],
                                    [lateral[0], lateral[-1]]))
    depth_value = float(np.interp(y, [0, max(height - 1, 1)],
                                  [depth[0], depth[-1]]))
    return {"lateral_vox": lateral_value, "depth_vox": depth_value}


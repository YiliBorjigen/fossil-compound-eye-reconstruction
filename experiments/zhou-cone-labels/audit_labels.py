#!/usr/bin/env python3
"""Audit the author-supplied InSegtCone labels without altering them.

This is geometric input QC, not anatomical validation or a reconstruction test.
Coordinates are TIFF row, column and zero-based slice; distances are voxels.
Dependencies: numpy, scipy, Pillow. No extraction of the full TIFF archive.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage


EXPECTED_SHA256 = "4637a27d46d5df9c467936effaabdccd7b16e3b8c002d12eda684dd872a490d3"
SOURCE_COMMIT = "da283cac7818e72f62e9a2f645e2ab93fdb29d27"


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def quantify(region, label_id, points, shape):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    sub = np.zeros(tuple(hi - lo + 1), dtype=bool)
    sub[tuple((points - lo).T)] = True
    cc, ncc = ndimage.label(sub, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(cc.ravel())[1:]
    largest = int(sizes.max())
    centred = points - points.mean(axis=0)
    eig, vec = np.linalg.eigh(centred.T @ centred / len(points))
    axis = vec[:, -1]
    # Axis is undirected: fix only a reproducible sign, not inward/outward.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    span = float(np.ptp(centred @ axis))
    ratio = float(eig[-1] / max(eig[-2], 1e-12))
    border = bool(np.any(lo == 0) or np.any(hi == np.asarray(shape) - 1))
    # Fixed permissive morphology screen. Passing does not certify a cone.
    screen = len(points) >= 50 and largest / len(points) >= 0.95 and ratio >= 3 and not border
    return {
        "region": region, "label_id": int(label_id), "voxels": len(points),
        "components_26": int(ncc), "largest_component_fraction": largest / len(points),
        "centroid_row": float(points[:, 0].mean()),
        "centroid_column": float(points[:, 1].mean()),
        "centroid_slice": float(points[:, 2].mean()),
        "axis_row": float(axis[0]), "axis_column": float(axis[1]),
        "axis_slice": float(axis[2]), "pca_variance_ratio": ratio,
        "axis_span_voxels": span, "touches_volume_border": border,
        "passes_morphology_screen": bool(screen),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive differs from the audited author source; check provenance first")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, regions, sparse = [], [], {}
    with zipfile.ZipFile(args.archive) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if len(members) != 3294:
            raise ValueError("Expected 9 regions, each with 366 TIFF slices")
        for region in range(1, 10):
            locs, ids = [], []
            for depth in range(366):
                name = f"autoSegmentedLabelsPnapi/coneLable{region}/{depth + 1:03d}.tiff"
                with Image.open(io.BytesIO(z.read(name))) as im:
                    a = np.array(im)
                if a.shape != (368, 496) or a.dtype != np.uint16:
                    raise ValueError(f"Unexpected TIFF format: {name}: {a.shape}, {a.dtype}")
                r, c = np.nonzero(a)
                locs.append(np.column_stack((r, c, np.full(len(r), depth))).astype(np.int32))
                ids.append(a[r, c])
            points, labels = np.concatenate(locs), np.concatenate(ids)
            # Z-major sparse indices are unique and already sorted by loading order.
            linear = np.ravel_multi_index((points[:, 2], points[:, 0], points[:, 1]), (366, 368, 496))
            sparse[region] = (linear, labels)
            order = np.argsort(labels, kind="stable")
            values, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
            regrows = []
            for value, start, count in zip(values, starts, counts):
                regrows.append(quantify(region, value, points[order[start:start + count]], (368, 496, 366)))
            rows.extend(regrows)
            region_summary = {
                "region": region, "objects": len(regrows), "labelled_voxels": len(labels),
                "fragmented_objects": sum(r["components_26"] > 1 for r in regrows),
                "largest_component_below_95pct": sum(r["largest_component_fraction"] < 0.95 for r in regrows),
                "objects_below_50_voxels": sum(r["voxels"] < 50 for r in regrows),
                "border_objects": sum(r["touches_volume_border"] for r in regrows),
                "passes_morphology_screen": sum(r["passes_morphology_screen"] for r in regrows),
                "median_voxels": float(np.median([r["voxels"] for r in regrows])),
                "median_axis_span_voxels": float(np.median([r["axis_span_voxels"] for r in regrows])),
            }
            regions.append(region_summary)
            print(json.dumps(region_summary), flush=True)
    sizes = {(r["region"], r["label_id"]): r["voxels"] for r in rows}
    pairs = []
    for a in range(1, 10):
        for b in range(a + 1, 10):
            common, ia, ib = np.intersect1d(sparse[a][0], sparse[b][0], assume_unique=True, return_indices=True)
            if not len(common):
                continue
            ids_a, ids_b = sparse[a][1][ia], sparse[b][1][ib]
            encoded = ids_a.astype(np.int64) * 65536 + ids_b
            codes, counts = np.unique(encoded, return_counts=True)
            for code, count in zip(codes, counts):
                la, lb = int(code // 65536), int(code % 65536)
                sa, sb = sizes[a, la], sizes[b, lb]
                pairs.append({"region_a": a, "label_a": la, "region_b": b, "label_b": lb,
                              "overlap_voxels": int(count), "intersection_over_union": int(count) / (sa + sb - int(count)),
                              "overlap_fraction_smaller": int(count) / min(sa, sb)})
    candidate_duplicate_pairs = [p for p in pairs if p["overlap_fraction_smaller"] >= 0.5]
    duplicate_objects = set()
    for pair in candidate_duplicate_pairs:
        duplicate_objects.add((pair["region_a"], pair["label_a"]))
        duplicate_objects.add((pair["region_b"], pair["label_b"]))
    for row in rows:
        row["candidate_cross_region_duplicate"] = (row["region"], row["label_id"]) in duplicate_objects
    summary = {
        "source_url": f"https://github.com/zhoutunhe/InSegtCone/blob/{SOURCE_COMMIT}/data/autoSegmentedLabelsPnapi.zip",
        "source_commit": SOURCE_COMMIT, "archive_sha256": digest,
        "archive_bytes": args.archive.stat().st_size,
        "tiff_file_bytes_sum": sum(i.file_size for i in members),
        "tiff_files": len(members), "shape_row_column_slice": [368, 496, 366],
        "coordinate_units": "voxels; registration and physical spacing not independently checked",
        "screen": {"minimum_voxels": 50, "minimum_largest_26_connected_fraction": 0.95,
                   "minimum_pca_largest_to_second_variance_ratio": 3, "exclude_volume_border": True},
        "total_region_label_objects": len(rows),
        "total_labelled_voxel_assignments": sum(r["voxels"] for r in rows),
        "objects_with_multiple_26_connected_components": sum(r["components_26"] > 1 for r in rows),
        "objects_largest_component_below_95pct": sum(r["largest_component_fraction"] < 0.95 for r in rows),
        "passes_morphology_screen": sum(r["passes_morphology_screen"] for r in rows),
        "cross_region_overlapping_label_pairs": len(pairs),
        "candidate_duplicate_pairs_at_half_smaller_object_overlap": len(candidate_duplicate_pairs),
        "objects_in_candidate_duplicate_pairs": len(duplicate_objects),
        "screened_objects_without_candidate_cross_region_duplicate": sum(
            r["passes_morphology_screen"] and not r["candidate_cross_region_duplicate"] for r in rows),
        "regions": regions,
        "status": "Geometric input audit only; no anatomical validation or reconstruction accuracy result",
    }
    write_csv(args.output / "objects.csv", rows)
    write_csv(args.output / "regions.csv", regions)
    write_csv(args.output / "overlapping_labels.csv", pairs)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "regions"}), flush=True)


if __name__ == "__main__":
    main()

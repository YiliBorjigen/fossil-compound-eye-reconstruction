#!/usr/bin/env python3
"""Audit the two supplied Imaris lens-layer WRLs without anatomical relabelling.

Reads the restricted shared-Coordinate, triangular IndexedFaceSet format used
by these files. Preserves connectivity. Does not reconstruct missing data,
identify individual lenses, or assign anatomical identity to the shell walls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def read_mesh(path):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if data.count(b"Coordinate {") != 1:
        raise ValueError("Expected one shared Coordinate table")
    # This importer intentionally fails instead of ignoring non-identity transforms.
    offset = 0
    while (offset := data.find(b"Transform {", offset)) >= 0:
        block = data[offset:data.index(b"}", offset)]
        for field, expected in [(b"translation", [0, 0, 0]),
                                (b"rotation", [0, 0, 1, 0]),
                                (b"scale", [1, 1, 1]),
                                (b"scaleOrientation", [0, 0, 1, 0]),
                                (b"center", [0, 0, 0])]:
            match = re.search(rb"\b" + field + rb"\s+([0-9eE.+ \-]+)", block)
            if match and not np.array_equal(np.fromstring(match[1], sep=" "), expected):
                raise ValueError("Unsupported non-identity transform")
        offset += len(b"Transform {")
    start = data.index(b"point [") + len(b"point [")
    stop = data.index(b"]", start)
    points = np.fromstring(data[start:stop].replace(b",", b" "), sep=" ").reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Non-finite coordinates")
    face_sets, offset = [], 0
    while (offset := data.find(b"coordIndex [", offset)) >= 0:
        start = offset + len(b"coordIndex [")
        stop = data.index(b"]", start)
        values = np.fromstring(data[start:stop].replace(b",", b" "),
                               sep=" ", dtype=np.int64).reshape(-1, 4)
        if not np.all(values[:, 3] == -1):
            raise ValueError("Expected triangles separated by -1")
        faces = values[:, :3].copy()
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("Out-of-range vertex index")
        face_sets.append(faces)
        offset = stop
    if len(face_sets) != 2:
        raise ValueError("Expected two eye face sets")
    return points, face_sets, digest


def topology(points, faces):
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    code = np.minimum(edges[:, 0], edges[:, 1]) * len(points) + np.maximum(edges[:, 0], edges[:, 1])
    unique, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=signs, minlength=len(unique))
    a, b = unique // len(points), unique % len(points)
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)),
                       shape=(len(points), len(points))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    triangles = points[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    return {
        "vertices": len(points), "triangles": len(faces),
        "connected_components": int(n_components),
        "largest_component_vertices": int(sizes.max()),
        "largest_component_fraction": float(sizes.max() / len(points)),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "inconsistent_two_face_edges": int(np.sum((counts == 2) & (winding != 0))),
        "degenerate_triangles": int(np.sum(area <= 1e-12)),
        "triangle_area_q05_q50_q95_coordinate_squared": np.quantile(area, [.05, .5, .95]).tolist(),
    }


def section_segments(local, faces):
    """Exact triangle intersection with y=0, returned in local x,z coordinates.

    Coplanar triangles and exact vertex hits are excluded and counted. These
    sections are visual diagnostics; PCA axes are not biological lens axes.
    """
    heights = local[:, 1][faces]
    active = (heights.min(axis=1) < 0) & (heights.max(axis=1) > 0)
    vertex_hits = np.any(heights == 0, axis=1)
    tri = local[faces[active & ~vertex_hits]]
    hits = np.full((len(tri), 2, 2), np.nan)
    counts = np.zeros(len(tri), dtype=int)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        crosses = tri[:, a, 1] * tri[:, b, 1] < 0
        ids = np.flatnonzero(crosses)
        fraction = -tri[ids, a, 1] / (tri[ids, b, 1] - tri[ids, a, 1])
        hit = tri[ids, a] + fraction[:, None] * (tri[ids, b] - tri[ids, a])
        hits[ids, counts[ids]] = hit[:, [0, 2]]
        counts[ids] += 1
    if np.any(counts != 2):
        raise ValueError("Unexpected section intersection count")
    return hits, int(np.sum(vertex_hits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous audit run.
    names = ["mesh_audit.json", "mesh_sections.csv", "mesh_sections.png"]
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("Use a new output directory; prior results are preserved")
    result, sections = [], []
    figure, axes = plt.subplots(len(args.meshes), 2, squeeze=False,
                                figsize=(10, 4.2 * len(args.meshes)))
    for row, path in enumerate(args.meshes):
        points, face_sets, digest = read_mesh(path)
        entry = {"filename": path.name, "sha256": digest, "file_bytes": path.stat().st_size,
                 "coordinate_table_vertices": len(points), "face_sets": []}
        print("Parsed", path.name, flush=True)
        for eye, global_faces in enumerate(face_sets):
            used, inverse = np.unique(global_faces, return_inverse=True)
            pts, faces = points[used], inverse.reshape(-1, 3)
            stats = topology(pts, faces)
            centre = pts.mean(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov((pts - centre).T))
            basis = eigenvectors[:, ::-1]
            for col in range(3):
                if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
                    basis[:, col] *= -1
            local = (pts - centre) @ basis
            segments, skipped = section_segments(local, faces)
            stats.update({"eye_face_set": eye, "pca_centre": centre.tolist(),
                          "pca_basis_columns": basis.tolist(),
                          "section_segments": len(segments),
                          "section_vertex_hit_triangles_excluded": skipped})
            entry["face_sets"].append(stats)
            for segment_id, pair in enumerate(segments):
                sections.append([path.name, eye, segment_id, *pair.ravel().tolist()])
            ax = axes[row, eye]
            ax.add_collection(LineCollection(segments, color="#333333", linewidth=.7))
            ax.autoscale()
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{path.name[:8]} · eye face set {eye}")
            ax.set_xlabel("PCA section x (mesh coordinate units)")
            ax.set_ylabel("PCA section z (mesh coordinate units)")
            ax.spines[["top", "right"]].set_visible(False)
            print(json.dumps(stats), flush=True)
        result.append(entry)
    figure.suptitle("Sections of supplied lens-layer meshes\nActual triangle intersections; anatomical boundaries unverified", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(args.output / "mesh_sections.png", dpi=160)
    plt.close(figure)
    with (args.output / "mesh_sections.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "eye_face_set", "segment", "x0", "z0", "x1", "z1"])
        writer.writerows(sections)
    with (args.output / "mesh_audit.json").open("x") as handle:
        json.dump({"scope": "Topology and geometric section audit, not anatomical validation",
                   "units": "WRL coordinate units; no physical-unit declaration in supplied format",
                   "files": result}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

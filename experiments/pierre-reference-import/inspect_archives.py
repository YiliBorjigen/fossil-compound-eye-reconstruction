#!/usr/bin/env python3
"""Inspect Pierre Tichit's source archives without changing or expanding them.

Dependencies: numpy, scipy, h5py, Pillow. No MATLAB code is executed.
This verifies the file interface, not biological accuracy or reconstruction.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat


def usable(name):
    return not name.startswith("__MACOSX/") and not PurePosixPath(name).name.startswith(".")


def inspect(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as z:
        members = [i for i in z.infolist() if not i.is_dir() and usable(i.filename)]
        names = [i.filename for i in members]
        gathered_path, = [n for n in names if n.endswith("/GatheredConeStruct.mat")]
        trusted_path, = [n for n in names if n.endswith("/trustedConeCenters.mat")]
        groups = loadmat(io.BytesIO(z.read(gathered_path)), simplify_cells=True)["truc"]
        with h5py.File(io.BytesIO(z.read(trusted_path))) as h:
            flags = h["trustedConeCenters"][()].ravel()
        if not np.all(np.isin(flags, [0, 1])):
            raise ValueError("Expected the supplied binary selection vector")
        parent = str(PurePosixPath(gathered_path).parent)
        folders = sorted({str(PurePosixPath(n).parent) for n in names
                          if n.startswith(parent + "/coneLable") and n.endswith(".tiff")})
        if len(folders) != len(groups):
            raise ValueError("MAT region count does not match TIFF folder count")
        regions = []
        for group in groups:
            mask = int(group["MaskInd"])
            xyz = np.asarray(group["Sub"], dtype=np.int64)
            indices = np.asarray(group["Ind"], dtype=np.int64)
            if xyz.shape != (len(indices), 3):
                raise ValueError("Unexpected sparse coordinate format")
            # Observed alphabetic order; validated against sampled TIFF voxels.
            folder = folders[mask - 1]
            slices = {int(PurePosixPath(n).stem): n for n in names
                      if str(PurePosixPath(n).parent) == folder and n.endswith(".tiff")}
            sample_depth = int(np.median(xyz[:, 2]))
            with Image.open(io.BytesIO(z.read(slices[sample_depth]))) as im:
                image = np.array(im)
            height, width = image.shape
            calculated = xyz[:, 0] + (xyz[:, 1] - 1) * width + (xyz[:, 2] - 1) * width * height
            index_ok = bool(np.array_equal(calculated, indices))
            sample = xyz[xyz[:, 2] == sample_depth]
            values = image[sample[:, 1] - 1, sample[:, 0] - 1]
            matched = int(np.count_nonzero(values))
            regions.append({
                "saved_mask_index": mask, "tiff_folder": folder,
                "tiff_shape_row_column": [height, width], "tiff_slices": len(slices),
                "sparse_voxels": len(indices), "all_linear_indices_match_xyz": index_ok,
                "sample_tiff": slices[sample_depth], "sample_sparse_voxels": len(sample),
                "sample_sparse_voxels_nonzero_in_tiff": matched,
                "sample_all_match": matched == len(sample),
            })
        stacks = []
        for folder in sorted({str(PurePosixPath(n).parent) for n in names}):
            tiffs = [n for n in names if str(PurePosixPath(n).parent) == folder
                     and PurePosixPath(n).suffix.lower() in [".tif", ".tiff"]]
            if not tiffs or "coneLable" in folder:
                continue
            sample = sorted(tiffs)[len(tiffs) // 2]
            with Image.open(io.BytesIO(z.read(sample))) as im:
                image = np.array(im)
            stacks.append({"folder": folder, "tiff_files": len(tiffs),
                           "sample_file": sample, "sample_shape_row_column": list(image.shape),
                           "sample_dtype": str(image.dtype), "sample_values": np.unique(image).tolist()})
        info_headers = {}
        for name in names:
            if name.endswith(".info"):
                lines = z.read(name).decode(errors="replace").splitlines()
                info_headers[name] = [l for l in lines if l.strip()][:6]
        optional_grid = None
        grids = [n for n in names if n.endswith("/CCorientationOnDistanceBasedGrid.mat")]
        if grids:
            with h5py.File(io.BytesIO(z.read(grids[0]))) as h:
                data = h["CCorientationOnDistanceBasedGrid"][()].T
            optional_grid = {
                "path": grids[0], "shape_rows_columns": list(data.shape),
                "finite_values_per_column": np.isfinite(data).sum(axis=0).tolist(),
                "fully_finite_rows": int(np.all(np.isfinite(data), axis=1).sum()),
                "status": "Not a usable complete position/direction table if any required column is entirely missing",
            }
        return {
            "archive": path.name, "archive_sha256": digest, "archive_bytes": path.stat().st_size,
            "source_files": len(members), "uncompressed_file_bytes": sum(i.file_size for i in members),
            "gathered_path": gathered_path,
            "gathered_meaning": "Ind contains 1-based linear voxel indices, Sub contains 1-based x,y,z coordinates; neither gives individual cone IDs",
            "selection_path": trusted_path, "selection_entries": len(flags),
            "selection_ones": int(np.count_nonzero(flags)),
            "selection_meaning": "Saved keep/discard vector; cone ordering and downstream cleaning not verified. Do not interpret ones as final unique cone count",
            "region_folder_order": "alphabetical, validated at one sampled slice per region",
            "regions": regions, "other_label_stacks": stacks,
            "amira_info_headers": info_headers, "optional_orientation_grid": optional_grid,
            "matlab_scripts_present": [n for n in names if n.lower().endswith(".m")],
            "status": "Source import checks only. No new anatomical validation or prediction accuracy claimed",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [inspect(path) for path in args.archives]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"archives": reports}, indent=2) + "\n")
    for report in reports:
        print(json.dumps({"archive": report["archive"], "regions": len(report["regions"]),
                          "selection_entries": report["selection_entries"],
                          "selection_ones": report["selection_ones"],
                          "all_index_checks_pass": all(r["all_linear_indices_match_xyz"] for r in report["regions"]),
                          "all_sample_checks_pass": all(r["sample_all_match"] for r in report["regions"])}))


if __name__ == "__main__":
    main()

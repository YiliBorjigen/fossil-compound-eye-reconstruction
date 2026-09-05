#!/usr/bin/env python3
"""Link supplied native masks to Fig. 3's documented calibration.

Read ZIPs directly; execute no supplied code and load no pickle objects.
Compare three complete binned planes per eye, reproducing the authors'
per-slice integer rounding before summation across four slices.
"""
import argparse
import ast
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def number(name):
    return int(re.search(r"(\d+)\.tif$", name, re.I).group(1))


def tiffs(archive, prefix=""):
    return sorted((n for n in archive.namelist()
                   if n.startswith(prefix) and n.lower().endswith(".tif")),
                  key=number)


def read_image(archive, name):
    with Image.open(io.BytesIO(archive.read(name))) as im:
        return np.array(im)


def constants(source):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("PIXEL_SIZE", "BIN_SIZE", "BIN_LENGTH")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fig3_zip", type=Path)
    parser.add_argument("native_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier calibration evidence")
    rows = []
    with ZipFile(args.fig3_zip) as outer:
        sources = {name: outer.read("share/" + name)
                   for name in ("README.md", "analysis.py", "plot&stats.py", "rescale_imgs.py")}
        for name in ("analysis.py", "plot&stats.py"):
            assert constants(sources[name].decode("utf-8-sig")) == {"PIXEL_SIZE": .325, "BIN_SIZE": 4}
        assert constants(sources["rescale_imgs.py"].decode("utf-8-sig"))["BIN_LENGTH"] == 4
        with ZipFile(io.BytesIO(outer.read("share/stacks.zip"))) as binned:
            folders = sorted({n.split("/")[0] for n in tiffs(binned)})
            for folder in folders:
                candidates = sorted(args.native_directory.glob(folder.removesuffix("_binned") + "-*.zip"))
                if not candidates:
                    raise FileNotFoundError(folder)
                hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
                assert len(hashes) == 1, "Conflicting copies for " + folder
                native_path = candidates[0]
                bn = tiffs(binned, folder + "/")
                checks = []
                with ZipFile(native_path) as native:
                    nn = tiffs(native)
                    assert len(bn) == len(nn) // 4
                    assert [Path(n).name for n in bn] == [Path(n).name for n in nn[3::4]]
                    for idx in sorted({len(bn)//4, len(bn)//2, 3*len(bn)//4}):
                        terms = []
                        for name in nn[4*idx:4*idx+4]:
                            data = read_image(native, name)
                            assert np.isin(data, [0, 1]).all()
                            h, w = np.array(data.shape) // 4
                            reduced = data[:4*h, :4*w].reshape(h, 4, w, 4).sum((1, 3)).astype(float)
                            terms.append((reduced / 64 * 255).astype("uint8"))
                        expected = np.stack(terms).sum(0).astype("uint8")
                        observed = read_image(binned, bn[idx])
                        assert expected.shape == observed.shape
                        different = int(np.count_nonzero(expected != observed))
                        checks.append({"binned_member": bn[idx], "native_members": nn[4*idx:4*idx+4],
                                       "shape_yx": list(expected.shape), "compared_pixels": int(expected.size),
                                       "foreground_pixels": int(np.count_nonzero(observed)),
                                       "different_pixels": different})
                    rows.append({"specimen_folder": folder, "native_archive": native_path.name,
                                 "native_sha256": next(iter(hashes)), "native_slices": len(nn),
                                 "binned_slices": len(bn), "all_output_names_match": True,
                                 "sampled_planes": checks})
                    print(folder, "different pixels:", sum(c["different_pixels"] for c in checks), flush=True)
    matched = all(c["different_pixels"] == 0 for row in rows for c in row["sampled_planes"])
    result = {"fig3_archive_sha256": hashlib.sha256(args.fig3_zip.read_bytes()).hexdigest(),
              "source_member_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in sources.items()},
              "all_sampled_pixels_match": matched,
              "calibration_basis": "Author code, with specimen linkage through filenames, bin counts and sampled pixel identity",
              "native_spacing_xyz_um": [.325, .325, .325] if matched else None,
              "fig3_binning_xyz": [4, 4, 4], "fig3_binned_spacing_xyz_um": [1.3, 1.3, 1.3],
              "unit_note": "analysis.py comments mm; plot&stats.py explicitly comments um and labels physical outputs in micrometres. The mm comment is inconsistent with those outputs and the published 325 nm scan scale.",
              "scope": "Three full binned planes per eye; not an exhaustive voxel comparison or independent instrument calibration",
              "specimens": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"specimens": len(rows), "all_sampled_pixels_match": matched,
                      "compared_pixels": sum(c["compared_pixels"] for row in rows for c in row["sampled_planes"])}))
    if not matched:
        raise SystemExit("Calibration linkage needs review")


if __name__ == "__main__":
    main()

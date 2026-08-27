#!/usr/bin/env python3
"""Prepare the MorphoSource Apis CT stack for Experiment 44.

The source TIFF stack is written to a Fortran-ordered NumPy memmap whose
coordinate order matches the accompanying NIfTI label: (x, y, z).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def nifti_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        header = handle.read(352)
    little = struct.unpack("<i", header[:4])[0] == 348
    endian = "<" if little else ">"
    dimensions = struct.unpack(endian + "8h", header[40:56])
    return tuple(int(value) for value in dimensions[1 : dimensions[0] + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tiffs = sorted(args.data_dir.glob("**/RAW/*.tif"))
    if not tiffs:
        raise FileNotFoundError("No TIFF slices found beneath the data directory")

    label_path = args.data_dir / "60185_AM_F_manLabel.nii"
    label_shape = nifti_shape(label_path)

    with Image.open(tiffs[0]) as image:
        slice_shape = (image.height, image.width)
        image_mode = image.mode

    expected = slice_shape + (len(tiffs),)
    if label_shape != expected:
        raise ValueError(f"CT shape {expected} does not match label {label_shape}")
    if image_mode != "L":
        raise ValueError(f"Expected 8-bit greyscale TIFFs, found mode {image_mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    volume = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.uint8,
        shape=expected,
        fortran_order=True,
    )
    for z, path in enumerate(tiffs):
        with Image.open(path) as image:
            volume[:, :, z] = np.asarray(image, dtype=np.uint8)
    volume.flush()

    summary = {
        "source": "MorphoSource media 000396182",
        "specimen": "Apis mellifera LU:3_14:AM_F_5",
        "shape_xyz": list(expected),
        "dtype": "uint8",
        "reported_acquisition_voxel_size_micrometres": 1.6,
        "voxel_calibration_note": (
            "Tichit et al. (2022) report 1.6 micrometre voxels for the bee scans. "
            "The supplied NIfTI label header stores unit voxel dimensions."
        ),
        "slice_count": len(tiffs),
        "first_slice": tiffs[0].name,
        "last_slice": tiffs[-1].name,
        "label_file": label_path.name,
        "label_values": {
            "3": "full eye volume (published InSegtCone input label)",
            "7": "external corneal/lens surface (published InSegtCone setting)",
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

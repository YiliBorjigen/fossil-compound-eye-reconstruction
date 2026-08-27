#!/usr/bin/env python3
"""Convert the public Pieris TIFF stack to a calibrated 8-bit NumPy volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower-percentile", type=float, default=0.1)
    parser.add_argument("--upper-percentile", type=float, default=99.9)
    args = parser.parse_args()

    tiffs = sorted(args.raw_dir.rglob("*.tif"))
    if not tiffs:
        raise FileNotFoundError("No TIFF files found")

    with Image.open(tiffs[0]) as image:
        first = np.asarray(image)
    shape = (first.shape[0], first.shape[1], len(tiffs))
    if first.dtype.kind != "u" or first.dtype.itemsize != 2:
        raise ValueError(f"Expected unsigned 16-bit input, found {first.dtype}")

    sample_indices = np.unique(np.linspace(0, len(tiffs) - 1, 64).round().astype(int))
    sampled = []
    for index in sample_indices:
        with Image.open(tiffs[index]) as image:
            array = np.asarray(image)
        if array.shape != first.shape or array.dtype != first.dtype:
            raise ValueError(f"Inconsistent TIFF at {tiffs[index]}")
        sampled.append(array.ravel())
    sample = np.concatenate(sampled)
    lower, upper = np.percentile(
        sample, (args.lower_percentile, args.upper_percentile)
    )
    if not upper > lower:
        raise ValueError("Invalid intensity calibration")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    volume = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=np.uint8,
        shape=shape,
        fortran_order=True,
    )
    for z, path in enumerate(tiffs):
        with Image.open(path) as image:
            raw = np.asarray(image, dtype=np.float32)
        scaled = np.clip((raw - lower) * (255.0 / (upper - lower)), 0, 255)
        volume[:, :, z] = np.rint(scaled).astype(np.uint8)
    volume.flush()

    metadata = {
        "source": "MorphoSource media 000397558",
        "specimen": "Pieris napi Pnapi_halfhead",
        "shape_xyz": list(shape),
        "input_dtype": str(first.dtype),
        "output_dtype": "uint8",
        "reported_voxel_size_micrometres": 1.08,
        "normalisation": {
            "sampled_slices": sample_indices.tolist(),
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "lower_value": float(lower),
            "upper_value": float(upper),
        },
        "first_slice": tiffs[0].name,
        "last_slice": tiffs[-1].name,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()


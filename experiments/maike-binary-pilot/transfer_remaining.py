#!/usr/bin/env python3
"""Apply the saved M26 model to ten unused eyes; never fit or tune a model.

See REMAINING_EYES_PROTOCOL.md. Existing files and outputs are never overwritten.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import run_pilot as pilot
from transfer_m32 import predict

SPECIMENS = ["M3_M_36_01", "M3_F_24_01", "M3_F_28_03", "M3_F_35_03",
             "RED3_25_M_26", "RED3_25_M_27", "RED3_25_M_28",
             "RED3_25_F_36", "RED3_25_F_37", "RED3_25_F_38"]
MODEL_SHA = "be753b7367db4a4aa8f08bdd739245ea6d8e85d4967404a8c99dad8e3e694cb2"
METHODS = ["outer_ellipsoid", "frozen_training_template", "frozen_outer_geometry"]
SPACING_UM = .325


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels(archive, name):
    with Image.open(io.BytesIO(archive.read(name))) as im:
        array = np.asarray(im)
        if array.ndim != 2 or array.dtype != np.uint8 or array.max() > 1:
            raise ValueError("Expected a single-plane uint8 binary mask")
        return array.copy()


def choose_crop(archive, names):
    # Only the last occupied y coordinate is used. No thickness/inner data.
    zs = np.arange(0, len(names), 16)
    maps = []
    for z in zs:
        array = pixels(archive, names[z])
        height, width = array.shape
        occupied = array[:, ::4].any(axis=0)
        distal = height - 1 - np.argmax(array[::-1, ::4], axis=0)
        maps.append(np.where(occupied, distal, np.nan))
    outer = np.asarray(maps)
    occupied = np.isfinite(outer)
    weights = ndi.gaussian_filter(occupied.astype(float), (2, 8))
    smooth = ndi.gaussian_filter(np.nan_to_num(outer), (2, 8)) / np.maximum(weights, 1e-9)
    eligible = occupied & (ndi.distance_transform_edt(occupied, sampling=(16, 4)) > 50)
    if not eligible.any() or len(names) < 321 or width < 450 or height < 300:
        raise ValueError("No suitable outer-envelope crop")
    iz, ix = np.unravel_index(np.argmax(np.where(eligible, smooth, -np.inf)), outer.shape)
    centre_z = int(np.clip(zs[iz], 160, len(names) - 161))
    centre_x = int(np.clip(4 * ix, 225, width - 225))
    x0, x1 = centre_x - 225, centre_x + 225
    z_region = np.abs(zs - centre_z) <= 170
    x_region = (4 * np.arange(outer.shape[1]) >= x0) & (4 * np.arange(outer.shape[1]) < x1)
    local_max = float(np.nanmax(outer[np.ix_(z_region, x_region)]))
    y1 = int(np.clip(np.ceil(local_max) + 33, 300, height))
    return {"y": [y1 - 300, y1], "x": [x0, x1],
            "seed_z": [centre_z - 140, centre_z + 140],
            "anchor_zxy": [int(zs[iz]), int(4 * ix), float(smooth[iz, ix])],
            "source_shape_zyx": [len(names), height, width]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_directory", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "transfer-results/frozen_training_model.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if digest(args.model) != MODEL_SHA:
        raise ValueError("The original frozen model hash does not match")
    frozen = json.loads(args.model.read_text())
    model = {key: np.asarray(frozen[key]) for key in
             ["mean", "std", "target_mean", "coefficients", "template"]}
    args.output.mkdir(parents=True)
    run = {"model_sha256": MODEL_SHA, "script_sha256": digest(Path(__file__)),
           "spacing_um_xyz": [SPACING_UM] * 3, "specimens": SPECIMENS,
           "training_specimen": "M3_M_26_01", "earlier_test_excluded": "M3_M_32_01",
           "crop_rule": "Outer-envelope rule in REMAINING_EYES_PROTOCOL.md; no manual adjustments",
           "results": []}
    summary = []
    for specimen in SPECIMENS:
        entry = {"specimen": specimen, "species": "D. simulans" if specimen.startswith("M3") else "D. mauritiana",
                 "sex": "male" if "_M_" in specimen else "female"}
        try:
            paths = sorted(args.archive_directory.glob("tiffs_" + specimen + "_eye_lenses*.zip"))
            if not paths:
                raise FileNotFoundError("Missing source archive: " + specimen)
            hashes = {digest(p) for p in paths}
            if len(hashes) != 1:
                raise ValueError("Conflicting copies of " + specimen)
            entry.update(archive=paths[0].name, archive_sha256=hashes.pop())
            folder = args.output / specimen
            folder.mkdir()
            with zipfile.ZipFile(paths[0]) as archive:
                names, ids = pilot.members(archive)
                crop = choose_crop(archive, names)
                entry.update(crop=crop, first_source_slice=ids[0])
                # Freeze the outer-defined crop on disk before target extraction.
                (folder / "outer_crop.json").write_text(json.dumps(entry, indent=2) + "\n")
                y0, y1 = crop["y"]
                x0, x1 = crop["x"]
                volume = np.empty((len(names), 300, 450), dtype=np.uint8)
                for z, name in enumerate(names):
                    array = pixels(archive, name)
                    if list(array.shape) != crop["source_shape_zyx"][1:]:
                        raise ValueError("Inconsistent TIFF dimensions")
                    volume[z] = array[y0:y1, x0:x1]
            pilot.SEED_Z = tuple(crop["seed_z"])
            pilot.BLOCK_EDGES = np.r_[np.linspace(*crop["seed_z"], 5).astype(int)[:-1], crop["seed_z"][1] + 1]
            records, outer, relief, gx, gz = pilot.records_from_volume(volume)
            del volume, outer, relief
            rows = []
            predictions = []
            for record in records:
                outer_fields = {k: v for k, v in record.items() if k not in ["inner", "target_valid"]}
                predicted = predict(model, outer_fields)
                for key, value in predict(model, record).items():
                    np.testing.assert_array_equal(value, predicted[key])
                predictions.append(predicted)
                for method in METHODS:
                    error = float(np.mean(np.abs(predicted[method] - record["inner"]))) if record["target_valid"] else None
                    rows.append({"specimen": specimen, "lens_id": record["id"], "z": record["z"], "x": record["x"],
                                 "radius_voxels": record["radius"], "method": method, "target_valid": record["target_valid"],
                                 "mae_voxels": error, "mae_um": error * SPACING_UM if error is not None else None,
                                 "nonpositive_thickness_fraction": float(np.mean(predicted[method] >= record["outer"]))})
            pilot.write_csv(folder / "per_facet.csv", rows)
            scorable = sum(r["target_valid"] for r in records)
            entry.update(status="scored" if scorable else "no_scorable_targets", candidates=len(records), scorable=scorable,
                         target_fields_removed_prediction_invariance="passed")
            eye = []
            for method in METHODS:
                errors = np.asarray([r["mae_um"] for r in rows if r["method"] == method and r["target_valid"]])
                result = {"specimen": specimen, "species": entry["species"], "sex": entry["sex"],
                          "method": method, "candidates": len(records), "scorable": scorable,
                          "median_patch_mae_um": float(np.median(errors)) if errors.size else None,
                          "p90_patch_mae_um": float(np.quantile(errors, .9)) if errors.size else None}
                summary.append(result)
                eye.append(result)
            valid = [i for i, r in enumerate(records) if r["target_valid"]]
            paired = [np.mean(np.abs(predictions[i][METHODS[1]] - records[i]["inner"])) -
                      np.mean(np.abs(predictions[i][METHODS[2]] - records[i]["inner"])) for i in valid]
            entry.update(metrics=eye, geometry_better_patches=sum(v > 0 for v in paired),
                         median_paired_improvement_um=float(np.median(paired) * SPACING_UM) if paired else None)
        except zipfile.BadZipFile as error:
            entry.update(status="input_unreadable", reason=str(error))
        except (ValueError, FileNotFoundError) as error:
            entry.update(status="failed", reason=str(error))
        if "geometry_better_patches" in entry:
            entry["geometry_better_patches"] = int(entry["geometry_better_patches"])
        run["results"].append(entry)
        (args.output / (specimen + "_manifest.json")).write_text(json.dumps(entry, indent=2) + "\n")
        print(json.dumps({k: entry[k] for k in ["specimen", "status", "candidates", "scorable", "metrics", "reason"] if k in entry}), flush=True)
    if digest(args.model) != MODEL_SHA:
        raise AssertionError("Model changed during evaluation")
    if summary:
        pilot.write_csv(args.output / "eye_summary.csv", summary)
    (args.output / "manifest.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()

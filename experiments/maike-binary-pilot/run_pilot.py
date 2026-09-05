#!/usr/bin/env python3
"""Bounded M3_M_26_01 pilot on supplied binary corneal-lens TIFFs.

Predict raw segmented proximal height grids from distal geometry and a training
prior. Coordinates are voxel indices; physical voxel spacing is not provided.
This is one-crop development, not independent-animal or fossil validation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Chosen from outer-surface localisation before reconstruction scoring.
Y_CROP = (1050, 1350)
X_CROP = (750, 1200)
SEED_Z = (260, 540)
SEED_X = (35, 415)
RELIEF_SIGMAS = (2, 30)
PEAK_WINDOW = 25
PEAK_RELIEF_MIN = 1.5
BLOCK_EDGES = np.array([260, 330, 400, 470, 541])
TRAIN_GUARD = 30
RIDGE_ALPHA = 10.0


def members(archive):
    items = [n for n in archive.namelist() if not n.endswith("/")]
    if any(not re.search(r"\d+\.tif$", n, re.I) for n in items):
        raise ValueError("Expected numbered TIFF members only")
    items.sort(key=lambda n: int(re.search(r"(\d+)\.tif$", n, re.I)[1]))
    ids = [int(re.search(r"(\d+)\.tif$", n, re.I)[1]) for n in items]
    if not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
        raise ValueError("Missing or duplicate slice numbers")
    return items, ids


def read_crop(path):
    if "M3_M_26_01" not in path.name:
        raise ValueError("This pilot's crop/orientation is defined only for M3_M_26_01")
    with zipfile.ZipFile(path) as archive:
        names, ids = members(archive)
        volume = np.empty((len(names), Y_CROP[1]-Y_CROP[0], X_CROP[1]-X_CROP[0]), np.uint8)
        for i, name in enumerate(names):
            with Image.open(io.BytesIO(archive.read(name))) as im:
                if im.mode != "P" or im.size != (1689, 1575):
                    raise ValueError("Unexpected indexed TIFF shape or mode")
                pixels = np.asarray(im)
                if pixels.max() > 1:
                    raise ValueError("Expected binary palette indices 0 and 1")
                volume[i] = pixels[Y_CROP[0]:Y_CROP[1], X_CROP[0]:X_CROP[1]]
            if i % 200 == 0:
                print(f"Decoded slice {i}/{len(names)}", flush=True)
    return volume, ids[0]


def smooth_finite(array, sigma):
    weight = ndi.gaussian_filter(np.isfinite(array).astype(float), sigma)
    return ndi.gaussian_filter(np.nan_to_num(array), sigma) / np.maximum(weight, 1e-9)


def outer_observation(volume):
    occupied = volume.any(axis=1)
    outer = (volume.shape[1] - 1 - np.argmax(volume[:, ::-1, :], axis=1)).astype(float) + .5
    outer[~occupied] = np.nan
    relief = smooth_finite(outer, RELIEF_SIGMAS[0]) - smooth_finite(outer, RELIEF_SIGMAS[1])
    eligible = occupied & (ndi.distance_transform_edt(occupied) > 25)
    candidates = np.argwhere((relief == ndi.maximum_filter(relief, size=PEAK_WINDOW))
                             & (relief > PEAK_RELIEF_MIN) & eligible)
    keep = ((candidates[:, 0] >= SEED_Z[0]) & (candidates[:, 0] <= SEED_Z[1])
            & (candidates[:, 1] >= SEED_X[0]) & (candidates[:, 1] <= SEED_X[1]))
    seeds = candidates[keep]
    if len(candidates) < 7 or len(seeds) < 20:
        raise ValueError("Insufficient outer-defined candidates")
    distances, _ = cKDTree(candidates).query(seeds, k=7)
    # Use all outer candidates, including outside the scoring region, for scale.
    radii = .5 * np.median(distances[:, 1:], axis=1)
    return outer, relief, seeds, radii


def sample(array, coords):
    return ndi.map_coordinates(array, coords, order=1, mode="constant", cval=np.nan)


def records_from_volume(volume):
    outer, relief, seeds, radii = outer_observation(volume)
    # Truth construction occurs only after candidates and scale are fixed.
    inner = np.argmax(volume, axis=1).astype(float) - .5
    continuous = np.sum(volume, axis=1) == outer - inner
    target_support = continuous & (inner > -.5) & (outer < volume.shape[1]-.5)
    inner[~target_support] = np.nan
    axis = np.linspace(-.7, .7, 11)
    xx, zz = np.meshgrid(axis, axis)
    disk = xx**2 + zz**2 <= .7**2 + 1e-12
    gx, gz = xx[disk], zz[disk]
    r2 = gx*gx + gz*gz
    ellipsoid_design = np.column_stack([np.ones(len(gx)), gx, gz, np.sqrt(1-r2)])
    poly_design = np.column_stack([np.ones(len(gx)), gx, gz, gx*gx, gx*gz, gz*gz])
    records = []
    for lens_id, ((z, x), radius) in enumerate(zip(seeds, radii)):
        coords = np.array([z+radius*gz, x+radius*gx])
        distal = sample(outer, coords)
        if not np.all(np.isfinite(distal)) or np.any(distal >= volume.shape[1]-.5):
            raise ValueError("Selected candidate lacks retained outer support")
        ellipsoid_beta = np.linalg.lstsq(ellipsoid_design, distal, rcond=None)[0]
        ellipsoid_inner = ellipsoid_design @ ellipsoid_beta - 2*ellipsoid_beta[3]*np.sqrt(1-r2)
        beta = np.linalg.lstsq(poly_design, distal, rcond=None)[0]
        rmse = float(np.sqrt(np.mean((distal-poly_design@beta)**2)))
        proximal = sample(inner, coords)
        valid = bool(np.all(np.isfinite(proximal)))
        records.append({"id":lens_id, "z":int(z), "x":int(x), "radius":float(radius),
                        "outer":distal, "inner":proximal, "target_valid":valid,
                        "features":np.r_[radius, beta[1:]/radius, rmse/radius],
                        "ellipsoid":ellipsoid_inner, "ellipsoid_c":float(ellipsoid_beta[3]),
                        "block":int(np.searchsorted(BLOCK_EDGES[1:],z,side="right"))})
    return records, outer, relief, gx, gz


def evaluate(records):
    metrics, saved, fold_counts = [], [], []
    for block in range(4):
        lo, hi = BLOCK_EDGES[block:block+2]
        train = [r for r in records if r["target_valid"] and
                 (r["z"] < lo-TRAIN_GUARD or r["z"] >= hi+TRAIN_GUARD)]
        test = [r for r in records if r["block"] == block]
        if len(train) < 10:
            raise ValueError("Too few training facets after spatial guard")
        train_ids = {r["id"] for r in train}
        assert not train_ids.intersection(r["id"] for r in test)
        features = np.vstack([r["features"] for r in train])
        mean, std = features.mean(0), features.std(0)
        std[std < 1e-12] = 1
        standardized = (features-mean)/std
        targets = np.vstack([(r["outer"]-r["inner"])/r["radius"] for r in train])
        target_mean = targets.mean(0)
        coefficients = np.linalg.solve(standardized.T@standardized + RIDGE_ALPHA*np.eye(features.shape[1]),
                                       standardized.T@(targets-target_mean))
        template = np.median(targets, axis=0)
        fold_counts.append({"block":block,"training_facets":len(train),"test_candidates":len(test),
                            "scorable_test_facets":sum(r["target_valid"] for r in test),
                            "training_ids":sorted(train_ids),"test_ids":[r["id"] for r in test]})
        for r in test:
            predicted_thickness = target_mean + ((r["features"]-mean)/std)@coefficients
            predictions = {"outer_ellipsoid":r["ellipsoid"],
                           "training_shape_template":r["outer"]-r["radius"]*template,
                           "outer_geometry_ridge":r["outer"]-r["radius"]*predicted_thickness}
            saved.append({"record":r,"predictions":predictions})
            for method,prediction in predictions.items():
                error = np.abs(prediction-r["inner"])
                metrics.append({"lens_id":r["id"],"block":block,"z":r["z"],"x":r["x"],
                                "radius_voxels":r["radius"],"target_valid":r["target_valid"],
                                "method":method,"mae_voxels":float(error.mean()) if r["target_valid"] else None,
                                "true_median_axial_thickness_voxels":float(np.median(r["outer"]-r["inner"])) if r["target_valid"] else None,
                                "predicted_nonpositive_thickness_fraction":float(np.mean(prediction >= r["outer"]))})
    summaries = []
    for method in ["outer_ellipsoid","training_shape_template","outer_geometry_ridge"]:
        for block in ["pooled",0,1,2,3]:
            group = [m for m in metrics if m["method"]==method and m["target_valid"] and (block=="pooled" or m["block"]==block)]
            errors = np.array([m["mae_voxels"] for m in group])
            summaries.append({"method":method,"block":block,"n":len(errors),
                              "median_facet_mae_voxels":float(np.median(errors)),
                              "p90_facet_mae_voxels":float(np.quantile(errors,.9))})
    return metrics, summaries, fold_counts, saved


def write_csv(path, rows):
    with path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    volume, first_slice = read_crop(args.archive)
    records, outer, relief, gx, gz = records_from_volume(volume)
    metrics, summary, folds, saved = evaluate(records)
    write_csv(args.output/"per_facet_metrics.csv",metrics)
    write_csv(args.output/"summary.csv",summary)
    metadata={"archive":args.archive.name,"sha256":hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "first_source_slice":first_slice,"crop_shape_zyx":list(volume.shape),
              "crop_yx":[list(Y_CROP),list(X_CROP)],"units":"voxel coordinates; physical spacing unknown",
              "n_candidates":len(records),"n_target_valid":sum(r["target_valid"] for r in records),
              "scope":"Exploratory four-band holdout in one crop of one supplied binary segmentation",
              "frozen_before_scoring":{"seed_z":list(SEED_Z),"seed_x":list(SEED_X),
              "relief_sigmas":list(RELIEF_SIGMAS),"peak_window":PEAK_WINDOW,"peak_min":PEAK_RELIEF_MIN,
              "block_edges":BLOCK_EDGES.tolist(),"guard_voxels":TRAIN_GUARD,"ridge_alpha":RIDGE_ALPHA},
              "folds":folds}
    (args.output/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    # Raw-derived grids stay local; code and aggregate/per-facet errors may be committed.
    np.savez_compressed(args.output/"local_surface_grids.npz",gx=gx,gz=gz,
        ids=[s["record"]["id"] for s in saved],
        outer=np.vstack([s["record"]["outer"] for s in saved]),
        inner=np.vstack([s["record"]["inner"] for s in saved]),
        **{method:np.vstack([s["predictions"][method] for s in saved]) for method in saved[0]["predictions"]})
    figure,axes=plt.subplots(1,2,figsize=(10,4.7),layout="constrained")
    axes[0].imshow(relief,cmap="gray",vmin=-6,vmax=6,origin="lower")
    for block in range(4):
        group=[r for r in records if r["block"]==block]
        axes[0].scatter([r["x"] for r in group],[r["z"] for r in group],s=17,facecolors="none",edgecolors="#e67722")
    for edge in BLOCK_EDGES: axes[0].axhline(edge,color="#e67722",lw=.8)
    axes[0].set(xlim=(0,450),ylim=(240,560),xlabel="x (crop voxels)",ylabel="z (crop voxels)",title="Outer-defined facets and held-out bands")
    # Representative picked by position, never by reconstruction error.
    representative=min(saved,key=lambda s:(s["record"]["z"]-400)**2+(s["record"]["x"]-225)**2)
    r=representative["record"]; line=np.abs(gz)<1e-9; order=np.argsort(gx[line]); q=gx[line][order]*r["radius"]
    top=r["outer"][line][order]; truth=r["inner"][line][order]
    axes[1].plot(q,top,color="black",label="Retained outer boundary")
    axes[1].plot(q,truth,color="black",ls="--",label="Hidden segmented inner boundary")
    axes[1].plot(q,representative["predictions"]["outer_geometry_ridge"][line][order],color="#e67722",label="Outer-geometry prediction")
    axes[1].plot(q,representative["predictions"]["training_shape_template"][line][order],color="#999999",label="Training shape template")
    axes[1].set(xlabel="Local x (voxels)",ylabel="Boundary y (crop voxels)",title=f"Held-out facet {r['id']} — raw mask comparison")
    axes[1].legend(frameon=False,fontsize=8)
    figure.savefig(args.output/"pilot_reconstruction.png",dpi=160)
    plt.close(figure)
    print(json.dumps({"candidates":len(records),"target_valid":metadata["n_target_valid"],
                      "pooled":[s for s in summary if s["block"]=="pooled"]},indent=2),flush=True)


if __name__=="__main__": main()

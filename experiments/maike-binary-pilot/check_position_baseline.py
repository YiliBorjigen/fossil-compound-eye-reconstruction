#!/usr/bin/env python3
"""Add a spatial-only thickness comparator without changing the initial pilot.

Thin-plate RBF smoothing is selected only inside each outer training split.
The comparator is a post-pilot diagnostic, not preregistered confirmation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    p=args.results
    if (p/"position_baseline.csv").exists():
        raise FileExistsError("Previous position comparator exists")
    manifest=json.loads((p/"manifest.json").read_text())
    rows=list(csv.DictReader((p/"per_facet_metrics.csv").open()))
    facets={int(r["lens_id"]):r for r in rows if r["method"]=="outer_geometry_ridge"}
    arrays=np.load(p/"local_surface_grids.npz")
    indices={int(lens_id):i for i,lens_id in enumerate(arrays["ids"])}
    def matrix(ids):
        positions=np.array([[float(facets[i]["z"]),float(facets[i]["x"])] for i in ids])/40.
        thickness=np.vstack([arrays["outer"][indices[i]]-arrays["inner"][indices[i]] for i in ids])
        return positions,thickness
    scores=[]; tuning=[]
    for fold in manifest["folds"]:
        train_ids=fold["training_ids"]
        test_ids=[i for i in fold["test_ids"] if facets[i]["target_valid"]=="True"]
        tx,ty=matrix(train_ids); qx,qy=matrix(test_ids)
        train_blocks=np.array([int(facets[i]["block"]) for i in train_ids])
        candidates=[]
        for smoothing in [.01,.1,1.,10.,100.]:
            losses=[]
            for held in np.unique(train_blocks):
                test=train_blocks==held; fit=~test
                if fit.sum()<6:continue
                model=RBFInterpolator(tx[fit],ty[fit],kernel="thin_plate_spline",degree=1,smoothing=smoothing)
                losses.extend(np.mean(np.abs(model(tx[test])-ty[test]),axis=1).tolist())
            candidates.append((float(np.mean(losses)),smoothing))
        _,selected=min(candidates)
        prediction=RBFInterpolator(tx,ty,kernel="thin_plate_spline",degree=1,smoothing=selected)(qx)
        tuning.append({"block":fold["block"],"selected_smoothing":selected,
                       "inner_validation":[{"mean_facet_mae":a,"smoothing":b} for a,b in candidates]})
        for lens_id,mae in zip(test_ids,np.mean(np.abs(prediction-qy),axis=1)):
            scores.append({"lens_id":lens_id,"block":fold["block"],"mae_voxels":float(mae),
                           "outer_geometry_mae_voxels":float(facets[lens_id]["mae_voxels"])})
    with (p/"position_baseline.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    summary=[]
    for block in ["pooled",0,1,2,3]:
        group=[r for r in scores if block=="pooled" or r["block"]==block]
        summary.append({"block":block,"n":len(group),
                        "position_rbf_median_mae_voxels":float(np.median([r["mae_voxels"] for r in group])),
                        "outer_geometry_median_mae_voxels":float(np.median([r["outer_geometry_mae_voxels"] for r in group])),
                        "median_paired_position_minus_geometry_voxels":float(np.median([r["mae_voxels"]-r["outer_geometry_mae_voxels"] for r in group]))})
    (p/"position_baseline_summary.json").write_text(json.dumps({"summary":summary,"tuning":tuning},indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()

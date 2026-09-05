#!/usr/bin/env python3
"""One frozen transfer from M3_M_26_01 to M3_M_32_01 in voxel coordinates.

The test crop was located from its outer envelope in a middle-slice preview,
before any target prediction scores. Predictor settings stay at pilot values.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import run_pilot as pilot

TEST_Y = (1200, 1500)
TEST_X = (375, 825)


def fit_model(records):
    records=[r for r in records if r["target_valid"]]
    features=np.vstack([r["features"] for r in records])
    mean,std=features.mean(0),features.std(0)
    std[std<1e-12]=1
    x=(features-mean)/std
    y=np.vstack([(r["outer"]-r["inner"])/r["radius"] for r in records])
    ym=y.mean(0)
    coefficients=np.linalg.solve(x.T@x+pilot.RIDGE_ALPHA*np.eye(x.shape[1]),x.T@(y-ym))
    return {"mean":mean,"std":std,"target_mean":ym,"coefficients":coefficients,
            "template":np.median(y,axis=0)}


def predict(model,record):
    # No inner grid, validity flag, or test-target statistics enter prediction.
    d=model["target_mean"]+((record["features"]-model["mean"])/model["std"])@model["coefficients"]
    return {"outer_ellipsoid":record["ellipsoid"],
            "frozen_training_template":record["outer"]-record["radius"]*model["template"],
            "frozen_outer_geometry":record["outer"]-record["radius"]*d}


def read_test(path):
    if "M3_M_32_01" not in path.name:raise ValueError("Expected M3_M_32_01 archive")
    with zipfile.ZipFile(path) as archive:
        names,ids=pilot.members(archive)
        volume=np.empty((len(names),300,450),np.uint8)
        for i,name in enumerate(names):
            with Image.open(io.BytesIO(archive.read(name))) as im:
                if im.mode!="P" or im.size!=(1485,1587):raise ValueError("Unexpected test TIFF format")
                pixels=np.asarray(im)
                if pixels.max()>1:raise ValueError("Test TIFF is not binary 0/1")
                volume[i]=pixels[TEST_Y[0]:TEST_Y[1],TEST_X[0]:TEST_X[1]]
            if i%200==0:print(f"Test slice {i}/{len(names)}",flush=True)
    return volume,ids[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_archive",type=Path)
    parser.add_argument("test_archive",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError("Use new empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    train_volume,_=pilot.read_crop(args.training_archive)
    training,*_=pilot.records_from_volume(train_volume)
    model=fit_model(training)
    frozen={k:v.tolist() for k,v in model.items()}
    frozen.update({"training_archive_sha256":hashlib.sha256(args.training_archive.read_bytes()).hexdigest(),
                   "training_candidates":len(training),"training_scorable":sum(r["target_valid"] for r in training),
                   "ridge_alpha":pilot.RIDGE_ALPHA,"target_grid_radius_fraction":.7,
                   "test_crop_yx":[list(TEST_Y),list(TEST_X)],"units":"voxel coordinates; physical spacing unknown"})
    serialized=json.dumps(frozen,sort_keys=True,indent=2)+"\n"
    (args.output/"frozen_training_model.json").write_text(serialized)
    del train_volume
    # The model file exists before test target extraction and scoring.
    test_volume,first_slice=read_test(args.test_archive)
    test,outer,relief,gx,gz=pilot.records_from_volume(test_volume)
    rows=[];predictions=[]
    for r in test:
        predicted=predict(model,r)
        masked={k:v for k,v in r.items() if k not in ["inner","target_valid"]}
        for key,value in predict(model,masked).items():
            np.testing.assert_array_equal(value,predicted[key])
        predictions.append(predicted)
        for method,prediction in predicted.items():
            rows.append({"lens_id":r["id"],"block":r["block"],"z":r["z"],"x":r["x"],
                         "radius_voxels":r["radius"],"method":method,"target_valid":r["target_valid"],
                         "mae_voxels":float(np.mean(np.abs(prediction-r["inner"]))) if r["target_valid"] else None,
                         "median_target_thickness_voxels":float(np.median(r["outer"]-r["inner"])) if r["target_valid"] else None,
                         "nonpositive_predicted_thickness_fraction":float(np.mean(prediction>=r["outer"]))})
    pilot.write_csv(args.output/"transfer_per_facet.csv",rows)
    summary=[]
    for method in predictions[0]:
        for block in ["pooled",0,1,2,3]:
            subset=[r for r in rows if r["method"]==method and r["target_valid"] and (block=="pooled" or r["block"]==block)]
            if not subset:continue
            errors=np.array([r["mae_voxels"] for r in subset])
            summary.append({"method":method,"block":block,"n":len(subset),
                            "median_facet_mae_voxels":float(np.median(errors)),
                            "p90_facet_mae_voxels":float(np.quantile(errors,.9))})
    pilot.write_csv(args.output/"transfer_summary.csv",summary)
    assert (args.output/"frozen_training_model.json").read_text()==serialized
    meta={"test_archive":args.test_archive.name,"test_archive_sha256":hashlib.sha256(args.test_archive.read_bytes()).hexdigest(),
          "training_model_sha256":hashlib.sha256(serialized.encode()).hexdigest(),
          "test_candidates":len(test),"test_scorable":sum(r["target_valid"] for r in test),
          "test_first_source_slice":first_slice,"test_crop_yx":[list(TEST_Y),list(TEST_X)],
          "test_target_fields_removed_prediction_invariance":"passed for all candidates",
          "scope":"One frozen cross-file, within-strain crop transfer. Physical spacing and independent raw anatomical verification absent."}
    (args.output/"transfer_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    np.savez_compressed(args.output/"local_transfer_grids.npz",gx=gx,gz=gz,
        outer=np.vstack([r["outer"] for r in test]),inner=np.vstack([r["inner"] for r in test]),
        **{method:np.vstack([r[method] for r in predictions]) for method in predictions[0]})
    figure,axes=plt.subplots(1,2,figsize=(10,4.7),layout="constrained")
    axes[0].imshow(relief,cmap="gray",vmin=-6,vmax=6,origin="lower")
    axes[0].scatter([r["x"] for r in test],[r["z"] for r in test],s=15,facecolors="none",edgecolors="#e67722")
    axes[0].set(xlim=(0,450),ylim=(240,560),xlabel="x (crop voxels)",ylabel="z (crop voxels)",title="M3_M_32_01 — outer-defined test facets")
    index=min(range(len(test)),key=lambda i:(test[i]["z"]-400)**2+(test[i]["x"]-225)**2)
    r=test[index];line=np.abs(gz)<1e-9;order=np.argsort(gx[line]);q=gx[line][order]*r["radius"]
    axes[1].plot(q,r["outer"][line][order],color="black",label="Retained outer boundary")
    axes[1].plot(q,r["inner"][line][order],color="black",ls="--",label="Hidden segmented inner boundary")
    axes[1].plot(q,predictions[index]["frozen_outer_geometry"][line][order],color="#e67722",label="Frozen outer-geometry prediction")
    axes[1].plot(q,predictions[index]["frozen_training_template"][line][order],color="#999999",label="Frozen training template")
    axes[1].set(xlabel="Local x (voxels)",ylabel="Boundary y (crop voxels)",title=f"Test facet {r['id']} — no test-eye inner training")
    axes[1].legend(frameon=False,fontsize=8)
    figure.savefig(args.output/"frozen_transfer.png",dpi=160)
    plt.close(figure)
    print(json.dumps({**meta,"pooled":[r for r in summary if r["block"]=="pooled"]},indent=2),flush=True)


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Show both orthogonal profiles of a position-selected test facet."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    args=parser.parse_args();p=args.results
    if (p/"transfer_two_profiles.png").exists():raise FileExistsError("Preserve previous figure")
    data=np.load(p/"local_transfer_grids.npz")
    rows=[r for r in csv.DictReader((p/"transfer_per_facet.csv").open()) if r["method"]=="frozen_outer_geometry"]
    index=min(range(len(rows)),key=lambda i:(int(rows[i]["z"])-400)**2+(int(rows[i]["x"])-225)**2)
    radius=float(rows[index]["radius_voxels"])
    figure,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    for ax,vary,fixed,name in [(axes[0],"gx","gz","x"),(axes[1],"gz","gx","z")]:
        keep=np.abs(data[fixed])<1e-9;order=np.argsort(data[vary][keep]);q=data[vary][keep][order]*radius
        origin=data["outer"][index][np.argmin(data["gx"]**2+data["gz"]**2)]
        for field,label,color,style in [("outer","Visible outer surface","black","-"),
                ("inner","Hidden inner reference","black","--"),
                ("frozen_training_template","Training template","#999999","-"),
                ("frozen_outer_geometry","Frozen geometry model","#e67722","-")]:
            ax.plot(q,data[field][index][keep][order]-origin,label=label,color=color,ls=style,lw=1.6)
        ax.set(xlabel=f"Local {name} (voxels)",ylabel="Axial y from outer centre (voxels)",title=f"{name}–y section of test facet {index}")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    figure.suptitle("Two views of the same predicted inner-surface patch",fontsize=12)
    figure.savefig(p/"transfer_two_profiles.png",dpi=160);plt.close(figure)

if __name__=="__main__":main()

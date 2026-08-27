# Manual-axis Pieris experiments

Experiments 48–49 use Li Yi's manually traced cone centre-lines from three
pre-selected *Pieris napi* regions. Experiment 48 asks whether a leave-one-cone-
out residual template improves when it is placed on the traced 3D path rather
than held on a static surface normal. Experiment 49 asks whether a simple skew
model learned in one or two regions transfers to a held-out region.

The intensity reconstructions are oracle-axis diagnostics within previously
examined data. The regional transfer test withholds the test-region paths from
model fitting, but all regions are from one specimen and one annotator. These
experiments are not an automatic segmentation result or biological replication.
The raw annotation archives and CT volume are not stored in Git.

The script records a single auditable correction: the depth-14 and depth-15
points of cone 2 are removed because the next point jumps 79.14 voxels to a
different cone; all other within-track steps are below 1.6 voxels. The original
archive is preserved unchanged.

## Results

Five tracks overlap from depths 34–50. Their median displacement is 4.87 voxels
(5.26 µm). A leave-one-cone-out template placed on an oracle-fitted straight
axis reduces median normalised MAE from 0.399 for local background to 0.282 and
wins for four of five cones. A quadratic axis also gives 0.282 and beats the
straight axis for only one cone. The current evidence therefore supports a
regional tilted axis, not necessary cone-specific curvature.

Patch 2 independently contains five clean tracks over depths 30–60. A straight
axis reduces median normalised MAE from 0.291 to 0.202 and wins for all five
cones. Patch 3 contains 12 shorter traces in several depth bands. In the two
five-cone common windows, a straight axis reduces median NMAE from 0.185 to
0.114 and from 0.268 to 0.221. Thus all three regions support the existence of
cone skew relative to the surface normal.

The cross-region result is more restrictive. A constant eye-relative skew
learned in Patch 1 reduces Patch 2 path RMSE from 5.63 to 1.86 voxels. However,
the model learned from Patches 1–2 increases held-out Patch 3 RMSE from 2.89 to
3.52 voxels and loses for all 11 usable tracks. Patch 3 reverses one tangent
component of the manual skew. The next predictor must therefore be a spatially
varying orientation field; one global tilt correction is not sufficient.

None of the uploaded preliminary random-forest masks is used. They label 53.3%,
59.8% and 95.2% of Patches 1–3 as cone, respectively, despite high internal OOB
scores. Those scores reflect the sparse, imbalanced training scribbles rather
than anatomical segmentation quality.

Run with:

```bash
python run_manual_axis_pilot.py \
  --volume /path/to/patch_1/unfolded_intensity.npy \
  --annotations /path/to/manual_annotations/annotations.json \
  --output-dir results \
  --depth-start 34 --depth-stop 50 \
  --drop-nodes 2:14,15
```

The cross-region script takes the registered whole-eye label, each patch's
`surface_xyz.npy` and `inward_normals.npy`, and the three annotation files. Run
`python run_cross_region_axis_transfer.py --help` for the explicit inputs.

# Predicting cone directions in hidden regions of one honeybee eye

The [fixed replication in bumblebee and butterfly eyes](REPLICATION.md) is now
complete. The correction improves median error in the butterfly but is slightly
worse than position alone in the bumblebee. The honeybee results below are
preserved; the three-eye evidence does not establish universal improvement.


The supplied data support a within-eye direction test against manual cone
annotations. A surface-normal predictor with a learned spatial tilt correction
has lower median error than position alone or a single rotation of the surface
normals in each of four hidden quarters. This is preliminary evidence for
interpolating missing directions when other internal directions survive in the
same eye. It does not establish reconstruction from an entirely external
surface or transfer to fossils.

## Results

All 101 manually labelled cones were scored once. Angles are unsigned angles
between axes, in degrees.

| Predictor | Median | Mean | 90th percentile |
|---|---:|---:|---:|
| Corneal surface normal | 37.88 | 36.92 | 45.94 |
| Position-only quadratic | 18.78 | 21.49 | 41.07 |
| Surface normal with one learned rotation | 18.13 | 21.53 | 39.91 |
| Surface normal with spatial tilt correction | **13.55** | **15.54** | **25.28** |

| Hidden quarter | Manual cones | Position-only median | Single-rotation median | Spatial-correction median |
|---|---:|---:|---:|---:|
| 0 | 27 | 19.47 | 18.15 | 16.08 |
| 1 | 25 | 10.77 | 15.27 | 8.28 |
| 2 | 25 | 23.86 | 23.24 | 13.55 |
| 3 | 24 | 27.74 | 20.25 | 18.14 |

The spatial correction improves on position alone for 81/101 cones and on
uncorrected normals for 93/101. These cones are observations within **one
specimen**, not 101 independent biological replicates. Errors remain substantial:
the fourth quarter has a 49.66-degree 90th-percentile error even with the spatial
correction. No model or neighbourhood size was selected by searching these
test errors.

## Registration and reference

The author supplied both voxel coordinates and transformed coordinates for
3,253 landmarks in `Results04/CleanManualConeFig.fig`. Fitting an affine
transform to alternating landmarks reproduced the remaining landmarks to less
than 1e-8 plot units. Its three singular values are approximately four, as
expected for a rotation with uniform scaling.

Applying the transform to all **124,560 voxels labelled 2** in
`60185_AM_only_CE.zip` reproduces the complete plotted corneal surface to less
than 1e-8 plot units, checked in both directions. The corneal point cloud in
`Results06/CleanManualConeFig.fig` also matches this surface. That figure
contains the **101 individual manual voxel clouds** used here. Their leading
PCA axes supply the reference directions; their largest-to-second eigenvalue
ratios range from 3.83 to 34.30.

The 4,423 automatic directions come from
`Results04/CleanConeViewingAxes.fig`. Figure line lengths are display scaling,
not cone lengths. No cleaning flags were guessed or applied to new object IDs.

## Fixed test

1. Fit local quadratic surfaces to the 256 nearest corneal voxels and derive
   their normals. Only external surface points enter these fits.
2. Locate each cone's evaluation site at the corneal point nearest its supplied
   centre. The centre supplies correspondence, not a depth or direction feature.
3. Divide the surface into four quarters using median cuts along its first two
   principal components. These cuts use external geometry only.
4. For each quarter, exclude all automatic directions in that quarter and all
   training sites within 80 plot units of it. Fit the two quadratic models to
   automatic directions in the remaining regions. Use a ridge penalty of 0.001
   per training cone, with the intercept unpenalised.
5. Score the hidden manual axes. The position model uses quadratic functions
   of surface location. The spatial correction uses the same functions to
   predict the difference between the automatic direction and surface normal.

The one-rotation control was added after the primary results to test whether
a single orientation bias could explain the gain. It uses the same exclusions
and no tuning. Baseline normal errors are stable with 64 or 1,024 neighbours:
their medians are 37.94 and 37.92 degrees, respectively.

The author's segmentation and curation happened before this holdout. They were
not rerun with the quarters hidden, so this is a test of downstream direction
prediction, not of an entire blinded segmentation pipeline. Manual annotations
can also contain errors. There is no new validation against raw greyscale CT.

The next decisive check is to repeat the fixed method in the supplied bumblebee
and butterfly eyes. That would test replication of this within-eye task.
Training on one eye and predicting another is a separate, stronger transfer
test and has not been performed here.

## Reproduce

Dependencies: `numpy`, `scipy`, `Pillow`. Put the two unchanged source archives
`Archive(3).zip` and `60185_AM_only_CE.zip` in an input directory, then run:

```bash
python experiments/pierre-reference-import/registration-test/run_holdout.py \
  /path/to/input --output /path/to/results
```

[Machine-readable results](results/results.json) include source hashes,
registration checks, settings and results by quarter.
[Per-cone errors](results/per_cone_errors.csv) use the one-based order of manual
point clouds in the supplied figure. Source voxel arrays are not redistributed.

Data supplied by Pierre Tichit. Source study: Tichit, Zhou, Kjer and colleagues,
[InSegtCone (2022)](https://doi.org/10.1186/s40850-021-00101-w).

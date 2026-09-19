# Replication in bumblebee and butterfly eyes

Follow-up: [fixed interpolation controls](INTERPOLATION.md) give lower median
errors with simple neighbour-averaged tilt corrections in all three eyes, but
larger tail errors than the quadratic correction. The original results below
are retained.

The fixed procedure improves on position alone in the honeybee and butterfly,
but does not improve the primary median-error metric in the bumblebee. The
evidence supports a useful, specimen-dependent correction rather than a
universally superior reconstruction method.

## Results

Median unsigned direction errors in degrees. Each column uses the same four
hidden quarters and the same manual reference cones within each specimen.

| Specimen | Manual cones | Surface normal | Position alone | One rotation of normals | Spatial tilt correction |
|---|---:|---:|---:|---:|---:|
| Honeybee, *Apis mellifera* | 101 | 37.88 | 18.78 | 18.13 | **13.55** |
| Bumblebee, *Bombus terrestris* | 102 | 18.71 | **11.73** | 24.91 | 12.04 |
| Butterfly, *Pieris napi* | 104 | 17.51 | 9.03 | 18.99 | **7.11** |

The tilt correction has a lower median than position alone in all four
honeybee quarters, all four butterfly quarters, and one of four bumblebee
quarters. It improves individual errors relative to position alone for 81/101,
66/104 and 42/102 cones, respectively. The bumblebee mean and 90th percentile
are slightly better with the correction (12.16 and 20.02 degrees versus 12.34
and 22.93), but that does not reverse its failure on the primary median metric.

These are **three eyes, one specimen per species**, containing 307 manual
reference cones in total. No pooled-cone significance claim or species-wide
conclusion is made. Each model learns from other regions of the same specimen;
no model coefficients were transferred from the honeybee to the other species.

## What stayed fixed

The replication imports the original `normals`, `features`, `ridge` and `angle`
functions from `run_holdout.py`. Settings remain 256 surface neighbours,
quadratic location features, ridge penalty 0.001 per training cone, four
surface-defined quarters and an 80-plot-unit exclusion buffer. All manual
cones were retained. Neither species triggered hyperparameter search or model
selection. The single-rotation control, added during the honeybee analysis,
was included before viewing either replication result.

The author figures contain every fifth corneal mask voxel in TIFF traversal
order. A transform fitted on alternating plotted points reproduces the
remaining points within 1e-8 plot units in both specimens, with singular values
approximately four. The complete masks contain 114,376 bumblebee surface voxels
and 169,458 butterfly surface voxels. Normal estimation uses these full masks,
not the lower-density plotted subsets.

The author's automatic segmentation and curation were completed before our
holdout. Upstream image processing was not rerun with regions hidden. Manual
voxel clouds provide reference axes but can contain annotation error. Known
cone centres supply correspondence to evaluation sites. These limitations from
the [honeybee test](README.md) still apply. This is not validation of missing
fossil anatomy, closed lens reconstruction or an optical-performance model.

## Novelty and publication assessment

The candidate contribution is a controlled test of how accurately missing
internal directions can be inferred when nearby anatomy survives, including
the circumstances in which external geometry adds no benefit. The current
calculations are new analyses of existing source data. The data, cone tilt,
PCA axes, surface fitting and regression are not new discoveries or new
mathematical methods.

A focused comparison of relevant prior work establishes substantial overlap:

- [Tichit et al. (2022), InSegtCone](https://doi.org/10.1186/s40850-021-00101-w)
  already segment cones in these same three eyes, derive their axes, compare
  automatic and manual annotations, and describe why surface normals can
  misrepresent internal directions. Their reported evaluation addresses
  segmentation performance rather than our buffered missing-quarter prediction.
- [Currea et al. (2023), ODA-3D](https://escholarship.org/content/qt61k2988b/qt61k2988b.pdf)
  already reconstruct ommatidial geometry from micro-CT, measure skew and
  compare optical estimates based on external approximations and internal
  anatomy. An additional software wrapper or demonstration of skew would not
  establish a distinct contribution.

This comparison does not establish that the exact prediction benchmark is
unprecedented. Do not claim a first-ever method. A broader literature review
and direct comparisons with appropriate interpolation methods remain necessary
before a novelty claim in a manuscript.

There is currently **no defensible basis to promise a high-tier publication**.
Three single-eye examples, mixed improvement over a simple baseline, and no
validated fossil application support a preliminary methods result. A stronger
paper would need a clear biological or palaeontological question, additional
independent specimens, calibrated uncertainty as anatomy is lost, and evidence
that the resulting reconstruction changes a supported scientific conclusion.
Those are conditions for a stronger contribution, not guarantees of acceptance.

The next decision should concern whether reconstruction errors are acceptable
for a specific biological question. Enlarging the software or tuning away the
bumblebee result would not answer that question.

## Reproduce

Place `77970 Bombus terrestris.zip` and `Pieris napi.zip` in an input directory.
Dependencies are `numpy`, `scipy` and `Pillow`.

```bash
python experiments/pierre-reference-import/registration-test/replicate_holdout.py \
  /path/to/input --output /path/to/replication-results
```

[Full results](replication-results/replication.json) include source hashes,
registration, fold counts and errors. Individual errors are retained for
[bumblebee](replication-results/bombus_per_cone_errors.csv) and
[butterfly](replication-results/pieris_per_cone_errors.csv).
No source voxel arrays are redistributed. Data were supplied by Pierre Tichit.

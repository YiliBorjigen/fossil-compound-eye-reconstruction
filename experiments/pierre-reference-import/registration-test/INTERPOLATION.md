# Interpolation controls for missing cone directions

A simple local average of surviving tilt corrections has lower median error
than the existing quadratic correction in all three eyes. The quadratic model
has a lower 90th-percentile error in all three. This check does not establish
an advantage for the quadratic model as a new reconstruction method.

The useful observation is narrower: visible surface normals, corrected using
surviving internal directions elsewhere in the same eye, can help predict
missing directions. The errors and their tails still matter. These are three
previously examined eyes, not a new independent validation set.

## Comparison

Median unsigned axis error in degrees, against the authors' manual cone
annotations. The same quarters, buffer, reference cones and sites are used for
every method. There is one specimen of each species.

| Method | Honeybee, 101 cones | Bumblebee, 102 cones | Butterfly, 104 cones |
|---|---:|---:|---:|
| Surface normal | 37.88 | 18.71 | 17.51 |
| Quadratic position-only model | 18.78 | 11.73 | 9.03 |
| One learned rotation of normals | 18.13 | 24.91 | 18.99 |
| Existing quadratic tilt correction | 13.55 | 12.04 | 7.11 |
| Inverse-distance average of directions | 19.45 | 22.46 | 28.02 |
| Inverse-distance average of tilt corrections | **10.99** | **10.43** | **6.79** |
| Local thin-plate spline of directions | 17.63 | 17.12 | 16.76 |
| Local thin-plate spline of tilt corrections | 17.13 | 20.54 | 18.05 |

For the inverse-distance methods, the 16 nearest retained training sites
contribute weights proportional to inverse squared distance. The correction
version averages the difference between a cone's outward unit direction and
its local surface normal, then adds the query site's normal and normalises.
It is standard neighbour averaging applied to those residual vectors.

Lower median error does not mean uniformly better predictions:

| Specimen | Quadratic correction p90 | Neighbour correction p90 | Neighbour correction: lower median in quarters |
|---|---:|---:|---:|
| Honeybee | 25.28° | 33.75° | 2/4 |
| Bumblebee | 20.02° | 25.54° | 3/4 |
| Butterfly | 14.36° | 15.59° | 3/4 |

The neighbour correction improves individual errors in 56/101 honeybee,
50/102 bumblebee and 54/104 butterfly cones relative to the quadratic
correction. We retain median error as the primary metric and report the tails
alongside it; neither method dominates every aspect of the result.

## What stayed fixed

The existing [honeybee protocol](README.md) and [replication](REPLICATION.md)
define four quarters from the visible surface's first two PCA coordinates.
Training excludes the entire test quarter and an 80 plot-unit buffer. Normals
use 256 surface neighbours. The original quadratic fits retain their ridge
penalty of 0.001 times the training count.

The added [comparator specification](interpolation-protocol.json) was written
before calculating the new comparator errors. This was a follow-up on known
specimens and known original results, not a preregistered confirmation study.
No parameter search or target-error-based setting selection was performed.

Both new interpolation rules use the same standardised two-dimensional
surface coordinates as the original quadratic fits. Each is applied both to
directions directly and to direction-minus-normal residuals. Neighbourhoods
and settings are identical within each direct/residual pair.

- Inverse-distance averaging: 16 neighbours, inverse-square distance weights.
- Local thin-plate spline: 64 neighbours, degree-one polynomial term and
  smoothing parameter 0.001, using SciPy's `RBFInterpolator`.
- Repeated training sites: average targets at exactly identical surface sites
  after selecting the training fold. Normalise final predictions.

The spline's large error tails are retained. This is evidence about this
fixed specification under large missing regions; it is not evidence that
splines are generally unsuitable. Holding out a quarter may require
extrapolation beyond the retained training sites, despite the conventional
name “interpolation” for these methods.

## Checks and limitations

All 1,228 original per-cone errors (307 cones times four original methods)
reproduced exactly, with a maximum numerical difference of 0 degrees. Every
fold assignment also matched. Source registration checks were repeated, and
archive, code and protocol hashes are recorded with the results.

The earlier limitations remain:

- Three eyes are the independent specimens; the 307 cones are not 307
  independent biological replicates.
- Each predictor learns from internal directions elsewhere in the same eye.
  A model trained on one eye and transferred to another was not tested.
- Known cone centres supply evaluation-site correspondence. Missing-facet
  detection and complete anatomical reconstruction were not tested.
- The authors' automatic segmentation and curation predate these holdouts.
  Upstream processing was not rerun with the test regions hidden.
- Manual voxel-cloud PCA axes are the reference. Their errors and uncertainty
  have not been independently calibrated against the source images.
- Angle error does not establish visual acuity, field of view, fossil validity
  or the anatomical identity of the fossil CT boundary.

## Consequence for the project

Do not build a large application around an asserted advantage of the quadratic
model. The current evidence supports keeping these small, reproducible methods
as baselines for a missing-anatomy benchmark. Any future advantage should be
tested against the neighbour correction too, with both median and tail errors
reported.

A useful next research question is when the amount and location of surviving
anatomy permit a reconstruction precise enough for a stated biological use.
That needs an error tolerance, an independent test and a declared failure
criterion. Further fitting on these three eyes cannot itself establish broad
generalisability or novelty. There is no new basis here for a high-tier
publication claim.

For the Bristol fossil question, the next decisive input remains independent
anatomical identification of the preserved boundary and a suitable second
specimen. A better directional interpolator cannot supply that evidence.

## Reproduce

Run from the repository root, with the four original input archives in
`/path/to/inputs` and an empty output directory:

```bash
python experiments/pierre-reference-import/registration-test/compare_interpolation.py \
  /path/to/inputs --output /path/to/new-output
```

Required archives are `Archive(3).zip`, `60185_AM_only_CE.zip`,
`77970 Bombus terrestris.zip` and `Pieris napi.zip`. The existing result CSVs
must be present for the regression check. No source archive is modified or
expanded into a full image stack. Dependencies are NumPy, SciPy and Pillow;
the exact versions used are recorded in
[`comparison.json`](interpolation-results/comparison.json).

The numerical result files contain error summaries and per-cone errors only.
They do not redistribute the supplied annotation coordinates or image stacks.

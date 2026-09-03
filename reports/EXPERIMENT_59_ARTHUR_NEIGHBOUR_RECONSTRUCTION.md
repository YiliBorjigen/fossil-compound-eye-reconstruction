# Experiment 59 — verified modern reconstruction from surviving neighbours

## Verdict

There is a strong alternative to outer-only fitting **when proximal surfaces
survive elsewhere in the same eye**. A fixed six-neighbour depth interpolation
plus a shared within-lens shape reconstructed contiguous 19-lens-scale missing
regions whose quadratic-smoothed central proximal targets have median axial
errors of **1.32 µm** (20231107), **0.36 µm**
(20240530) and **0.52 µm** (20240701). The method remained useful across the
large 2023 depth shift that defeated the frozen cross-volume population model.

This does not solve Lauren Sumner-Rooney's hardest case, in which only the
outer curvatures remain across the relevant eye region. It validates a
within-eye interpolation method for patchy internal loss, not outer-only
identifiability, edge extrapolation or fossil transfer.

## Why this experiment was run

Experiment 58 showed that an outer-cap ellipsoid does not reconstruct the
proximal lens surface. A population template learned on 20240701 transferred
reasonably to the similar 20240530 scan, but failed on 20231107 because its
proximal surface was much deeper. The next defensible question was whether
surviving homologous surfaces in the test eye can supply the missing depth.

Arthur Zhao supplied matched complete lens and photoreceptor-tip surface
meshes for three *Drosophila* micro-CT volumes. All three pairs are used here:
20231107, 20240530 and 20240701, with both eyes evaluated in every volume.

## Frozen spatial-loss protocol

Complete geometry and matched tips first provide the oracle layer separation
needed to construct the benchmark's retained distal observation and hidden
proximal truth. After that separation, the graph and masks are created before
target status, depth or prediction error is examined:

1. retain every lens patch that passes distal-surface support, scale and fit
   checks;
2. build a connected, length-pruned six-nearest-neighbour graph from distal-cap
   centroids only;
3. identify the eye margin from neighbour-free sectors in the distal tangent
   plane;
4. choose eight deterministic seeds per eye, each at least five graph hops
   from the margin and seven hops from every other seed;
5. hide nested graph-radius 1, 2 and 3 regions around every seed.

The resulting masks contain exactly 7, 19 and 37 lenses in every eye. The
radius-three masks are non-overlapping. Radius two is the primary analysis;
radii one and three are fixed size-sensitivity checks. There are 16 patches
per volume at each radius and 48 primary patches across six eyes.

Every proximal surface inside a test patch is removed before any neighbour,
template, ridge or harmonic model is fitted. Proximal surfaces elsewhere in
the same eye are legitimate training inputs. This makes the experiment
same-eye, transductive reconstruction, externally repeated under the same rule
across three scan volumes. It is not cross-volume prediction.

## Target availability without selection leakage

Mask membership uses distal data only. Target fields are then classified
separately:

- **target resolvable:** enough proximal support exists to fit a finite scored
  surface;
- **target QC pass:** the resolvable surface lies inside the distal cap and its
  fit RMSE is no greater than 2.5 µm.

Only target-QC surfaces outside a mask may train a predictor. Primary accuracy
is conditional on target-QC cases inside an outer-defined mask. All resolvable
targets form a prespecified sensitivity cohort; unresolvable cases remain in
the denominator and are never assigned a fabricated error.

| Volume | Distal-QC graph nodes | Finite target | Target-QC pass |
|---|---:|---:|---:|
| 20231107 | 1,607 | 1,588 | 1,528 |
| 20240530 | 1,592 | 1,592 | 1,575 |
| 20240701 | 1,700 | 1,700 | 1,679 |

All 304 lenses selected by the primary interior masks in each volume had a
finite target and passed target QC. This complete scoring coverage is partly a
consequence of the fixed five-hop margin guard; the experiment does not test
poorly supported edge damage. It also makes the all-resolvable sensitivity
cohort numerically identical to the primary cohort inside these selected
patches, so that sensitivity analysis adds no evidence about failed-QC lenses.

## Methods

The fixed primary is a prespecified modern-data adaptation of Experiment 54's
six-neighbour inverse-square depth rule. Experiment 59 uses outer-graph
geodesic distances and a pointwise median residual grid rather than Experiment
54's specimen coordinates and fitted quadratic residual, so it is not an
unchanged external replication.
For each visible training lens, its median thickness is separated from its
within-lens residual shape. Six nearest visible target-QC lenses set the hidden
lens depth by inverse-square graph-geodesic weighting; the median training
residual supplies the shared shape.

Comparators are an outer-only axisymmetric ellipsoid, a same-eye median
thickness template, same-eye ridge regression from distal curvature, the
nearest visible thickness surface, inverse-square interpolation of six full
thickness surfaces, and graph-harmonic depth interpolation plus the shared
shape. The graph-harmonic method is secondary because it was added for this
experiment rather than inherited as the primary rule.

Each reported positional MAE is the mean absolute **axial** difference between
predicted and target proximal heights at corresponding points on the canonical
central grid. The target is a robust quadratic fit to proximal vertices on a
disk extending to 0.65 times the retained distal-cap scale. The metric does not
cover the raw proximal mesh, rim or a watertight two-surface lens, and it is not
a closest-point, Hausdorff or symmetric three-dimensional surface distance.

## Primary 19-lens-scale result

| Volume | Median true depth | Ellipsoid MAE | Same-eye template MAE | Fixed six-neighbour primary MAE | Graph-harmonic MAE |
|---|---:|---:|---:|---:|---:|
| 20231107 | 29.95 µm | 28.22 µm | 2.40 µm | 1.32 µm | **1.18 µm** |
| 20240530 | 16.45 µm | 11.82 µm | 0.49 µm | 0.36 µm | **0.32 µm** |
| 20240701 | 13.68 µm | 10.16 µm | 0.67 µm | 0.52 µm | **0.51 µm** |

The six-neighbour normalized median errors are **4.55%**, **2.16%** and
**3.87%**, respectively. Median absolute central-depth errors are 1.35, 0.34
and 0.48 µm. Median proximal-surface normal errors are 8.98°, 3.96° and 7.14°,
so accurate surface position should not be confused with exact optical-axis
recovery.

At the eye level, the primary method beats the ellipsoid, same-eye template,
same-eye outer-curvature ridge and nearest-surface copy in all six eyes. The
six-neighbour full-surface interpolation is effectively tied with the primary
depth-plus-shape rule. The graph-harmonic secondary is modestly better in all
six eye summaries, but this small comparison is method-development evidence,
not independent confirmation that it is universally superior.

## Missing-region size

| Volume | 7-lens region | 19-lens region | 37-lens region |
|---|---:|---:|---:|
| 20231107 | 1.02 µm | 1.32 µm | 1.49 µm |
| 20240530 | 0.32 µm | 0.36 µm | 0.42 µm |
| 20240701 | 0.51 µm | 0.52 µm | 0.62 µm |

Error rises smoothly as the missing region grows, but the fixed neighbour
method remains below 1.5 µm median error even for the 37-lens interior patches
in all three volumes.

![Experiment 59 comparison](../experiments/arthur-modern-ground-truth/results/experiment_59_comparison.png)

## Calibrated error bands

For each primary test patch, the other seven separated patches in that eye are
pseudo-hidden in turn while the true test patch remains excluded. Their point
errors set a nested nominal 90% point-error half-width.

| Volume | Median 90% half-width | Marginal point coverage |
|---|---:|---:|
| 20231107 | 3.32 µm | 89.30% |
| 20240530 | 1.06 µm | 88.61% |
| 20240701 | 1.53 µm | 88.98% |

The pooled descriptive coverage is 88.96%, close to but below nominal. Only
721/912 lenses (79.1%) have at least 90% of their grid points covered, and only 29/48
patches reach 90% marginal point coverage; the patch range is 46.0–100%.
These are descriptive marginal same-eye point bands, not guaranteed
patch-specific surface intervals or anatomical uncertainty bounds.

## Interpretation against Lauren's goal

Lauren described the key fossil problem as loss of the internal lens surfaces
while only the outer curvatures remain, and anticipated ellipsoid fitting or a
similar reconstruction. Experiments 57–58 show directly on verified modern
lenses that ellipsoid continuation is insufficient and that an outer-only
population mapping can fail under a depth-domain shift.

Experiment 59 adds a practical route, but only for the narrower case in which
some internal surfaces remain visible nearby. It shows that same-eye anatomical
repetition can absorb the 2023 depth shift far better than a population model
trained on another volume. If every proximal surface is absent, these data do
not make the hidden surface uniquely recoverable; a taxon- and morphology-
matched prior, physical constraints and explicit uncertainty would still be
assumptions rather than recovered ground truth.

## Evidence boundary

**Supported with limits:** controlled reconstruction of quadratic-smoothed
central proximal-cap targets under synthetic contiguous interior loss in
modern *Drosophila* eyes, using surviving proximal surfaces from the same eye
and repeated across three volumes.

**Not tested or not established:** torn-edge extrapolation; an eye with no
surviving proximal surfaces; transfer between specimens, taxa or fossils;
anatomical identity of the *Asaphus* CT boundary; undeformed fossil geometry;
optical axes, ray tracing or visual function.

The evidential units are three volumes and six eyes, not 912 independent
animals. The metadata used here do not establish three independent biological
replicates. The unusually deep 20231107 target may reflect biology, specimen
handling or a segmentation-definition shift; the current files cannot choose
among those explanations.

## Reproduction and provenance

Run:

```bash
python experiments/arthur-modern-ground-truth/experiment_59_neighbor_block_validation.py \
  --manifest /path/to/local_manifest.json
```

The local manifest follows the Experiment 58 schema and points to all six
supplied WRL meshes plus the three public eyemap RData files. Supplied meshes
remain external and are not redistributed. The committed diagnostics record
their SHA-256 hashes and the complete frozen graph, mask, scoring and
calibration settings.

# Experiment 52: preserved-surface angular geometry of the *Asaphus* eye

## Question

How reproducibly can the angular geometry of the preserved *Asaphus* facet
surface be measured, and how underdetermined is any optical interpretation?

This experiment measures the fossil as preserved. It does not retrodeform the
specimen, infer the living field of view, or use the *Pieris* directions as an
anatomical model for *Asaphus*. It separates a reproducible surface measurement
from an explicitly unknown internal-axis departure.

## Preserved-surface baseline

The frozen Experiment 36 parameters recover 116 robust facets from the 3.7 µm
*Asaphus* crop. Surface normals were calculated from the low-frequency corneal
curvature, not from the local facet relief.

The preserved surface normals form an angular envelope with:

- maximum sampled span: **34.64 degrees**;
- spherical convex-hull area: **0.167 steradians**;
- median separation between adjacent surface normals: **2.31 degrees**;
- median adjacent facet spacing: **64.17 µm**.

These values describe one unilateral crop in its present fossil state. They are
not a full-eye field of view, living eye geometry, or anatomically validated
interommatidial angles.

![Asaphus preserved-surface angular sensitivity](../experiments/asaphus/results/visual_field/experiment_52_asaphus_visual_field_sensitivity.png)

## Measurement stability

Across neighbouring surface thresholds (45–70), the median change in a matched
facet normal is 0.22 degrees and the 95th percentile is 1.67 degrees. The outer
geometric measurement is therefore stable relative to the tested
intensity-threshold choices.

This precision must not be confused with anatomical accuracy. Experiment 51
showed that internal directions in a modern *Pieris* region can depart from its
surface normals by about 15 degrees. That number is not transferred to the
fossil; it motivates asking how any stated uncertainty bound affects the
geometric conclusion.

## Sensitivity to an unknown internal axis

If every unknown internal axis lies within beta degrees of its corresponding
surface normal, the triangle inequality bounds the maximum angular span within
plus or minus 2 beta of the 34.64-degree baseline.

| Allowed axis departure | Defensible maximum-span range |
| ---: | ---: |
| 5 degrees | 24.64–44.64 degrees |
| 10 degrees | 14.64–54.64 degrees |
| 15 degrees | 4.64–64.64 degrees |
| 20 degrees | 0–74.64 degrees |

The wide ranges show that the scan measures the outer surface much more
precisely than it identifies biological sight directions. External curvature
alone supports a preserved-state geometric baseline, not a precise biological
field of view.

## Interpretation

The result answers a narrow measurement question. The preserved facet field is
sufficient to map a stable surface-normal envelope and a geometric
neighbour-angle proxy. It is not sufficient to recover the original eye shape
or decide the true optical axes without taphonomic and anatomical evidence.

The next decisive evidence is therefore not another threshold adjustment. It
is a deformation constraint from more complete or bilaterally preserved
material, plus an independently validated relationship between corneal facets
and internal axes.

## Claim limits

- The analysed volume is one crop from one *Asaphus* specimen.
- Surface normals are a geometric convention, not verified optical axes.
- The crop does not cover the complete eye.
- No specimen-specific compression, shear or twisting field has been
  estimated; the living geometry is not identifiable from this crop alone.
- The 0.167 sr hull is the convex envelope of sampled normals, not a biological
  field-of-view measurement.
- The uncertainty bounds do not claim that *Asaphus* has a *Pieris*-like axis
  tilt; they show how conclusions depend on any stated bound.
- No refraction, calcite birefringence, rhabdom geometry, sensitivity or acuity
  is modelled.

## Reproducibility

`experiments/asaphus/run_visual_field_sensitivity.py` reads the corrected 3.7
µm NRRD and writes the facet-normal table, threshold audit, analytic uncertainty
bounds, summary and figure. The legacy filename retains the original
“visual-field sensitivity” wording for provenance; the report above states the
narrower interpretation. The raw CT volume is excluded from Git; its provenance
is recorded in `data/README.md`.

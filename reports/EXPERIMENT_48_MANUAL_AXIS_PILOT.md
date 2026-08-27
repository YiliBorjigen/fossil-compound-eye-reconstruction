# Experiment 48: manually traced Pieris axes

Li Yi manually followed seven bright cone cross-sections through the first
pre-selected *Pieris napi* region. The archive matches the expected 141 × 141 ×
61 volume and its SHA-256 provenance. One trajectory contained a 79.14-voxel
jump between depths 15 and 16, indicating a switch to a different cone. The two
pre-jump points were removed in a documented corrected copy; the original
archive was not changed.

Five trajectories overlap continuously between depths 34 and 50. These were
used for a leave-one-cone-out diagnostic. The intensity template for each held-
out cone was made only from the other four cones. Its placement was then tested
using a static surface normal, an oracle-fitted straight axis, an oracle-fitted
quadratic axis, the raw clicked path and a quadratic path borrowed from another
cone. The held-out manual axis is therefore supplied to the diagnostic; this is
not a deployable prediction or an independent validation.

| Placement | Median NMAE | Median correlation | Wins over background |
|---|---:|---:|---:|
| Local depth background | 0.399 | −0.000 | — |
| Static surface normal | 0.353 | 0.211 | 3/5 |
| Fitted straight axis | 0.282 | 0.731 | 4/5 |
| Fitted quadratic axis | 0.282 | 0.728 | 4/5 |
| Raw manual path | 0.274 | 0.670 | 4/5 |
| Permuted quadratic path | 0.296 | 0.417 | 4/5 |

The straight tilted axis reduces median error by 29.4% relative to the local
depth background. The five paths move by a median 4.87 voxels, or 5.26 µm,
over the evaluated depth interval. This displacement is large enough to account
for the earlier approximately 3.2 µm registration error.

Curvature is not the missing information. A quadratic axis has essentially the
same median error as a straight axis and wins against it for only one of five
cones. The quadratic fit reduces path residual from 0.459 to 0.418 voxels—a
difference of only 0.045 µm at the 1.08 µm voxel size. The lower raw-path median
does not overturn this: raw clicks contain slice-to-slice annotation jitter and
have lower median correlation than the straight fit. A path borrowed from
another cone also improves four of five cases, showing that much of the gain is
generic regional tilt rather than cone-specific curvature.

One held-out cone is worse than background under every template placement.
This local heterogeneity prevents a general reconstruction claim from five
tracks in one region.

The GUI's preliminary random-forest mask is rejected. It was trained with only
three explicit background clicks, labelled 53.3% of the volume as cone and
formed two implausibly large connected components. Its 0.986 out-of-bag score
measures the imbalanced scribbles within the same volume, not anatomical
accuracy.

The defensible conclusion is narrower and useful: **surface normals are a poor
axis model in this Pieris region; a smooth tilted axis is sufficient at the
current resolution, while cone-specific curvature remains unproven.** The next
deployable step is to estimate that smooth depth-dependent tilt from the intact
lattice without using the held-out cone's manual path, then test it on the other
two pre-selected regions.

![Manual-axis pilot](../experiments/manual-axis-pieris/results/experiment_48_manual_axis_pilot.png)

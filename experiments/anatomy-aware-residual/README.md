# Anatomy-aware residual reconstruction

Experiment 46 begins after the failed independent *Pieris* transfer. It is
method development, not a new independent validation.

The old model treated a cone as an intensity block translated from one centre
to another. This experiment separates three quantities: the local depth
profile, the cone residual beyond that profile, and the uncertainty in the
predicted cone centre. Training residuals are aligned by the local hexagonal
lattice angle and scaled by local facet spacing. A held-out target supplies
only an annulus outside the hidden four-voxel core.

`oracle_centred_residual` uses the true internal centre and is diagnostic only.
It asks how much error would remain if centre prediction were solved; it is not
a deployable reconstruction.

## Result

Across the same 62 Pieris targets and 15 spatial blocks, a local depth-only
background had median normalised MAE 0.140. The feasible anatomy-aware residual
model scored 0.162 and won only 2 of 15 blocks. Intensity normalisation,
hexagonal orientation and local scale therefore do not rescue the method while
the centre is predicted by the old affine map.

The diagnostic changes when the true centre is supplied: median error falls to
0.111, and the residual model beats background for 49 of 62 targets and 10 of
15 blocks. This is evidence that repeated residual shape exists in these
patches, but the present method cannot locate it accurately enough.

There is an additional limitation. The current target centres are peaks in one
internal CT slice, not centre-lines from segmented three-dimensional cones.
The next experiment must trace each cone through depth and estimate its axis,
length and radius before any further template fitting.

![Experiment 46 error decomposition](results/experiment_46_error_decomposition.png)

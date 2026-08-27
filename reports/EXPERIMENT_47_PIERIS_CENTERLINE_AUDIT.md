# Experiment 47: does cone-axis curvature explain the error?

Experiment 46 implicated centre and axis registration. Experiment 47 tested
two proposed explanations: unequal z sampling and slightly curved peripheral
axes.

The acquisition metadata rule out unequal voxel spacing: the Pieris stack is
sampled at 1.08 µm in x, y and z. The earlier approximately 3.2 µm value was a
lateral displacement between predicted and detected two-dimensional centres,
not a z-depth error. Isotropic voxel spacing does not guarantee isotropic image
blur, so directional sharpness remains a quality-control question, but
resampling the existing volume would not correct the observed displacement.

The curvature test is necessarily provisional. The public segmentation marks
the eye volume and external corneal surface; it does not label individual
crystalline cones. Bright periodic ridges were therefore linked through
successive slices of the unfolded CT and retained only when continuous for at
least 15 slices. Straight and quadratic trajectories were compared using five
contiguous depth holdouts.

Regions 1 and 2 showed a modest advantage for the quadratic path. Median
blocked-CV errors changed from 1.37 to 1.00 voxels and from 1.11 to 0.92 voxels,
respectively. Region 3 did not reproduce the effect (1.19 versus 1.22 voxels).
The median deviation of a quadratic path from its straight counterpart was
only approximately 0.6 µm in all three regions, far below the earlier centre
error. Sensitivity analyses across four detection thresholds and three linking
limits retained the direction in regions 1 and 2 but not consistently in
region 3.

The more serious result was continuity. Eighty, 56 and 14 candidate tracks met
the 15-slice criterion in the three regions, but only 24, 8 and 0 of them
spanned both the shallow and internal depths used previously. Slice-wise
intensity maxima therefore cannot yet provide a reliable anatomical mapping
from a corneal facet to its crystalline cone.

The defensible conclusion is narrow. A model should permit gentle curvature,
but curvature is too small to explain the present registration failure. The
next required evidence is a trained or manually validated 3D segmentation of
individual cones. Only those masks can establish centre-lines, endpoints,
lengths, radii and tilts and allow the reconstruction model to be retested.

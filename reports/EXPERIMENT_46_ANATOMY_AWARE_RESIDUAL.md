# Experiment 46: what is actually causing the cone error?

Experiment 45 showed that an average cone-shaped intensity block did not
transfer from one eye design to another. Experiment 46 asked a more useful
question: was that failure caused by the cone pattern itself, or by putting the
pattern in the wrong place?

The revised model first removes the local depth-dependent CT profile. It then
aligns training residuals by the orientation of the hexagonal lattice and
scales them by local facet spacing. The central four-voxel target is hidden;
only a non-overlapping annulus remains available for estimating the local
background.

The deployable model still fails. Across 62 targets and 15 spatial blocks,
local depth-only background gives median normalised MAE 0.140. The
anatomy-aware residual gives 0.162 and wins only two blocks. Its error becomes
worse as centre error increases (Spearman r = 0.330, p = 0.0089).

A diagnostic run supplies the true internal centre but changes nothing else.
Its median error is 0.111; it beats background for 49 of 62 targets and 10 of
15 blocks. The paired fold-level median improvement is 0.0278, although the
fold-bootstrap 95% interval crosses zero because the third region contains
only six matched targets and performs poorly.

This does not validate a reconstruction method. It locates the present
bottleneck: repeated cone residuals are visible, but the current outer-to-inner
centre map is not accurate enough to place them. Moreover, the internal targets
are two-dimensional intensity peaks rather than verified three-dimensional
cone axes. Further template tuning would be premature.

The next step is a centre-line experiment. Candidate cones should be segmented
or tracked through the unfolded depth volume, represented by axis, length,
radius and local spacing, and linked to their corneal facets. Only after that
mapping is evaluated in spatial blocks should intensity reconstruction be
revisited.

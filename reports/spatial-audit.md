# Spatial audit of the Asaphus reconstruction

The original full model reproduced the earlier result: 116 stable facets, 74
retained internal-edge maps, a median boundary depth of 51.8 micrometres, and a
median held-out error of 6.90 micrometres versus 12.76 micrometres for the
radial baseline.

That comparison was incomplete because the full model contained absolute
specimen coordinates. A nonlinear coordinate-only smoother reached 6.79
micrometres. In a matched comparison, adding the geometry features to the same
nonlinear spatial basis made median performance worse and helped only one of
five held-out spatial blocks. One thousand within-block geometry permutations
did not rescue a stable geometric contribution.

The conclusion is not that the internal CT boundary disappeared. It is that
the old improvement over a radial surface should be interpreted as spatial
interpolation, not recovery of physical lens geometry from the outside.

This audit is why the later feature search evaluates improvement over a strong
spatial-only model and why the repository does not lead with the earlier
approximately 46% headline.

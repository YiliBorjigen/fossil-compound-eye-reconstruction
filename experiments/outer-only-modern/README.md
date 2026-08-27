# Outer-only surface test on modern lenses

`16_outer_only_reconstruction_with_uncertainty.py` pairs outer and inner
surfaces from complete segmented modern lenses, then removes the inner points
from the predictor. It compares constrained geometric and regularised models
under held-out evaluation.

The median hidden-inner-surface error was 28.2% of local lens depth. The result
was useful for rejecting unconstrained ellipsoid or quadric fitting, but it was
not accurate enough to serve as fossil anatomical reconstruction.

The derived input `labeled_lens_points.npz` is not committed. This script is
retained as method provenance until that input is restored and hashed.

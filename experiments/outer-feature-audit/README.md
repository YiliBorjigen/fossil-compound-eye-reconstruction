# Outer-feature audit

This analysis asks whether the *Asaphus* boundary can be predicted from outer
surface geometry after a nonlinear specimen-position field has already been
fitted. It uses five contiguous spatial blocks and nested tuning inside each
training split.

```bash
python experiment_42_outer_feature_discovery.py \
  --volume /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --samples /path/to/reconstruction_samples.csv \
  --centers /path/to/robust_facet_centers.csv \
  --out results-local
```

The richer surface model reached 6.59 micrometres versus 6.79 micrometres for
position alone, but the small advantage was inconsistent and unsupported by
the exact block-level test (p = 1.000). Component tests also found no stable
signal for either between-facet depth or within-facet curvature.

Small summary outputs are committed. The 8.3 MB feature table and row-level
predictions are excluded.

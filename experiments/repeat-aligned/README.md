# Repeat-aligned internal signal and local gap filling

This experiment does not predict a wholly absent surface from the outer lens.
It aligns raw CT intensities from repeated facets using surface normal and
facet spacing, then asks two narrower questions:

1. Does a shared internal image-domain boundary survive facet bootstrapping and
   removal of complete spatial blocks?
2. Can a cross-facet template help fill a deliberately masked region when the
   rest of the boundary remains visible in the same facet?

```bash
python experiment_43_repeat_aligned_reconstruction.py \
  --volume /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --samples /path/to/reconstruction_samples.csv \
  --centers /path/to/robust_facet_centers.csv \
  --out results-local
```

The shared edge lay at 48.03 micrometres, with a 43.46–61.75 micrometre
facet-bootstrap 95% interval. A guarded hierarchical template improved on a
local quadratic fit for the frozen off-centre mask, but not consistently on a
flexible RBF interpolator. The initial unguarded residual model's severe
half-plane failure is preserved in the summary outputs.

## Complete missing surfaces (Experiment 54)

Experiment 54 hides every internal point from a target spatial block. It tests
ellipsoid-like fits, measured outer curvature, a repeat template, a strong
position-only control and a fixed six-neighbour anatomical prior. Model
selection and error-band calibration use only the remaining blocks.

First extract outer-surface features with Experiment 42, then run:

```bash
python experiment_54_whole_facet_reconstruction.py \
  --features /path/to/experiment_42_feature_table.csv \
  --out results-local/experiment_54
```

The six-neighbour method reached 8.10 micrometres median error when complete
spatial regions were hidden and 6.59 micrometres for exploratory isolated-facet
loss. Strictly local outer curvature did not improve the quadratic surface.
The supported interpretation is therefore reconstruction from neighbouring
homologous facets, not inference from the surviving outer curve alone.
`experiment_54_six_neighbor_surfaces.npz` contains the complete canonical-grid
prediction and calibrated error-band width for every held-out facet.

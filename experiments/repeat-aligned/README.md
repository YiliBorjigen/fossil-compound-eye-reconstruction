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

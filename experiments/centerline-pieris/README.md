# Pieris candidate centre-line audit

Experiment 47 tests whether slight centre-line curvature or anisotropic voxel
sampling explains the centre-registration error identified in Experiment 46.

The raw MorphoSource manifest reports 1.08 µm spacing in x, y and z. No
isotropic resampling is therefore applied. Equal sampling does not prove an
isotropic point-spread function, but unequal z spacing cannot explain the
current result.

The public label contains the eye volume and corneal surface, not individual
cones. This script consequently tracks repeated bright CT ridges through the
unfolded volume and calls them **candidate tracks**. It compares straight and
quadratic paths using five blocked depth holdouts and reports whether each
track spans the shallow and internal depths used in Experiment 45.

```bash
python run_centerline_audit.py \
  --patch-root /path/to/experiment_45_pieris_transfer/work \
  --output-dir results
```

The result is a constraint, not a segmentation claim. A trained or manually
validated 3D cone mask is needed before these trajectories can be identified
as anatomical crystalline-cone axes.

![Experiment 47 centre-line audit](results/experiment_47_centerline_audit.png)

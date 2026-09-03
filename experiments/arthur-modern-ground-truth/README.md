# Experiment 57 — Arthur Zhao modern ground truth

This experiment tests the question that motivated the project: can a missing
proximal lens surface be reconstructed when the distal/outer surface is the
only surviving lens geometry?

Arthur Zhao supplied complete lens-surface and photoreceptor-tip WRL meshes for
three *Drosophila* micro-CT volumes. The first strict run uses `20240701`, whose
raw lens and tip landmark CSV files are public in
[`reiserlab/eyemap_T4`](https://github.com/reiserlab/eyemap_T4). The supplied
WRL files remain external and are not redistributed by this repository.

## Leakage boundary

The complete lens mesh, annotated lens positions and matched tip positions are
used to partition the connected lens layer and label the distal and proximal
surfaces. This is ground-truth construction only. Tip coordinates, proximal
vertices and complete-lens centroids are then discarded.

Prediction receives only the retained distal mesh. Its origin, surface normal,
local frame, scale, curvature coefficients and eye centre are all re-estimated
from those retained vertices. The photoreceptor-tip mesh is parsed only for
file-integrity/provenance checks and is never passed to a predictor.

## Development split

The two eyes are held out in turn. A model trained on one whole eye predicts
the other whole eye. This blocks facet-level leakage but remains a within-volume
test; the two eyes are not independent animals. Cross-volume validation on
`20231107` and `20240530` uses raw coordinates recovered exactly by inverting
the PCA transform stored in the public processed data.

Three methods are compared:

1. an axisymmetric ellipsoid/sphere continued from the visible cap alone;
2. the opposite-eye median thickness template;
3. ridge regression from local outer-surface scale and curvature to the hidden
   proximal surface.

Run with:

```bash
python experiments/arthur-modern-ground-truth/experiment_57_outer_only_validation.py \
  --lens-mesh /path/to/20240701-PTA-surface-lens.wrl \
  --tip-mesh /path/to/20240701-PTA-surface-tip.wrl \
  --lens-csv /path/to/eyemap_T4/data/microCT/20240701_position/lens.csv \
  --tip-csv /path/to/eyemap_T4/data/microCT/20240701_position/cone.csv
```

The script writes aggregate summaries, QC/provenance metadata and a comparison
figure. Per-lens metrics are ignored by the repository and retained locally.

## Cross-volume validation

`experiment_58_cross_volume_confirmation.py` reconstructs the raw landmark
coordinates from the PCA scores and transform stored in the public RData files.
The inversion was checked against the 20240701 CSV coordinates to machine
precision. It freezes 20240701 as the training volume and scores 20231107 and
20240530 without refitting on either test volume.

The manifest is local because it contains paths to supplied, non-redistributed
meshes. Its schema is:

```json
{
  "volumes": [
    {
      "volume": "20240701",
      "lens_mesh": "/path/to/lens.wrl",
      "tip_mesh": "/path/to/tip.wrl",
      "rdata": "/path/to/20240701.RData"
    }
  ]
}
```

Run with:

```bash
python experiments/arthur-modern-ground-truth/experiment_58_cross_volume_confirmation.py \
  --manifest /path/to/local_manifest.json \
  --training-volume 20240701
```

## Contiguous same-eye loss

`experiment_59_neighbor_block_validation.py` tests the usable case in which a
contiguous group of proximal surfaces is missing but other proximal surfaces
remain visible in the same eye. After oracle separation of the connected lens
layer, all graph construction and mask selection use retained distal-cap
centroids only. Target availability, QC, depth and error cannot affect the
masks.

Each eye contributes eight separated seeds and nested graph-radius 1, 2 and 3
masks containing 7, 19 and 37 lenses. The fixed primary adapts the earlier
six-neighbour inverse-square depth rule; same-eye templates, outer curvature,
nearest/full-surface interpolation, an ellipsoid and graph-harmonic depth are
comparators. The score is axial error against a robust-quadratic proximal fit
on the central canonical disk, not whole-mesh or watertight-lens distance.

Run all three supplied pairs with:

```bash
python experiments/arthur-modern-ground-truth/experiment_59_neighbor_block_validation.py \
  --manifest /path/to/local_manifest.json
```

The aggregate result files and full interpretation are in the
[Experiment 59 report](../../reports/EXPERIMENT_59_ARTHUR_NEIGHBOUR_RECONSTRUCTION.md).

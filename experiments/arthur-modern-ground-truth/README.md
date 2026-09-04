# Experiment 57 — Arthur Zhao modern ground truth

This experiment tests the question that motivated the project: can a missing
proximal lens surface be reconstructed when the distal/outer surface is the
only surviving lens geometry?

Arthur Zhao supplied complete lens-surface and photoreceptor-tip WRL meshes for
three *Drosophila* micro-CT volumes. These are three separate whole-head scans
from different 6--7-day-old female animals; the bilateral eyes are nested
within animal and are not six independent specimens. Here, independent means
different animals, not different laboratories, studies or acquisition
workflows. The animal and bilateral-eye interpretation follows the
[source paper](https://doi.org/10.1038/s41586-025-09276-5) and its pinned
analysis code. The first strict run uses `20240701`, whose
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

Experiment 63 exposes this oracle operation as
`prepare_oracle_split_records()`. It returns the raw layer split after only the
25-vertex distal-support gate, before any frame, scale, curvature, residual or
target QC. The separate source adapter then hash-seals the distal caps and
reopens only those artifacts for its drop-only q90 central-cap fit. Proximal
support and quality are evaluated later in the final distal-only frame and can
never change distal membership.

Arthur's target observation operator is necessarily source-specific. These
files are sparse, already extracted distal and proximal **surface meshes**, so
the source adapter fits the unique supplied proximal vertices inside the final
distal-q90 domain. The Maike stacks are filled binary lens volumes and instead
select one minimum-axial proximal voxel from each 0.325-µm lateral bin that
spans at least 0.650 µm. Applying that filled-volume rule literally to the
Arthur boundary meshes left the entire `20231107` target cohort below the
25-bin support gate, confirming that mesh tessellation alignment is not a
meaningful proxy for filled axial columns.

The latent response is still common to both adapters: positive
distal-minus-proximal thickness is represented by the same unsymmetrized
six-coefficient robust quadratic in the final distal-only q90 frame and
evaluated on the same 81-point grid. The different observation operators are
an explicit representation/domain shift. Consequently, a failed external
transfer cannot by itself distinguish biological mismatch from mesh-versus-
binary-volume representation mismatch.

## Development split

The two eyes are held out in turn. A model trained on one whole eye predicts
the other whole eye. This blocks facet-level leakage but remains a within-volume
test; the two eyes are not independent animals. Cross-volume validation on
`20231107` and `20240530` uses raw coordinates recovered exactly by inverting
the PCA transform stored in the public processed data.

For Experiment 63, source-only hyperparameter selection instead leaves out one
entire volume/animal at a time and gives the three source animals equal weight.
It never treats the six eyes as six independent cross-validation units.

The established raw pixel pitches are 0.9882 µm (`20231107`) and 0.9884 µm
(`20240530`). The supplied `20240701` mesh is already in physical coordinates,
but its raw acquisition pitch is left unresolved: the public deposition names
a `20240710` acquisition at 1.0065 µm, and the adapter does not assume those
identifiers denote the same scan.

The public position inputs are frozen at `eyemap_T4` commit
`99d2a43123db636cedb55af9ff31a59657e7d17e`:

| RData input | Bytes | SHA-256 |
|---|---:|---|
| `20231107.RData` | 296,662 | `e8cf395625b5fa2a415206614bd4e1a4d74d5372820038135bf77f312349a9ba` |
| `20240530.RData` | 293,647 | `6921aa284338b476cd7db14aea1e0fef19cc45bb141ea19d4baaf03f8d6ca684` |
| `20240701.RData` | 310,361 | `8408447f8c5798fbc6d751de79ef4983d08c342a500c9a55b2a3b4b38d92deec` |

The six WRL hashes are listed in [`data/README.md`](../../data/README.md). The
source adapter enforces filenames, byte sizes and SHA-256 identities for all
nine inputs before parsing them; provenance does not merely self-report
whatever paths appear in a manifest.

Generate the frozen Experiment 63 source bundle after committing the analysis
implementation:

```bash
python experiments/maike-modern-ground-truth/prepare_arthur_source_table.py \
  --manifest /path/to/arthur_manifest.json \
  --eyemap-root /path/to/eyemap_T4 \
  --output results-local/experiment63-arthur-source
```

The primary-ready provenance is emitted only from a clean repository and binds
the source table, all six supplied meshes, all three public RData files, the
`eyemap_T4` commit, distal artifacts, frame audits and analysis-code hashes.

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

## Reconstructed follow-up stress tests

Experiments 60--62 replace analyses that were lost with an unpushed scratch
worktree.  Their surviving questions were retained, but unknown historical
constants and results were not invented: each executable and report labels its
replacement specification explicitly.

| Experiment | Question | Main result | Report |
|---|---|---|---|
| 60 | Does same-eye interpolation still work when a missing patch meets the observed eye margin? | Fixed six-neighbour radius-two median errors: 2.64, 1.20 and 1.45 µm; tails are substantially heavier than for guarded interior patches. | [Margin loss](../../reports/EXPERIMENT_60_ARTHUR_MARGIN_LOSS.md) |
| 61 | How sparse can surviving same-eye proximal donors become? | Two percent was the smallest tested potential-donor fraction with all 25 repeats computable in all six eyes and every eye below 10% median normalized error. | [Donor survival](../../reports/EXPERIMENT_61_ARTHUR_SURVIVAL_THRESHOLD.md) |
| 62 | What can be reported when no same-eye proximal donors survive? | A source-only outer model failed under whole-volume holdout; named residual-morphology scenarios expose sensitivity instead of claiming a unique lens. | [Outer-only partial identification](../../reports/EXPERIMENT_62_ARTHUR_OUTER_ONLY_PARTIAL_IDENTIFICATION.md) |

Experiments 60 and 61 deliberately reuse Experiment 59's legacy extraction
and target-QC contract to isolate margin position and donor thinning. Experiment
62 instead consumes the rebuilt Experiment 63 q90 source table and its
`distal_qc AND target_resolvable` cohort.  Their errors are not row-matched
ablations of one common cohort.  All three remain modern-eye method tests; none
identifies a fossil CT boundary or reconstructs fossil optics.

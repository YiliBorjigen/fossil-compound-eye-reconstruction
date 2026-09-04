# Twelve-animal modern ground-truth validation

> **Experiment 63 status: stopped before model-outcome evaluation.** The frozen visual-QC
> gate found a nonpassing distal cap in four of twelve eyes. No passing
> attestations were issued, and the primary backend was not run. Experiment 63
> therefore has no shape-versus-control result. See the
> [stop report](../../reports/EXPERIMENT_63_MAIKE_PRE_OUTCOME_QC_STOP.md) and
> [machine-readable record](results/experiment_63_stop_record.json).
>
> **Experiment 64 status: stopped before attestation and model-outcome
> evaluation.** Its disjoint visual gate found a connected lobe in a final
> robust core, classified as nonpassing in two concordant AI visual
> assessments. These were not independent human or anatomical-expert reviews.
> No sealed outcome was opened and the backend was not run. See the [stop
> report](../../reports/EXPERIMENT_64_ROBUST_CAP_PRE_OUTCOME_QC_STOP.md) and
> [machine-readable record](results/experiment_64_stop_record.json).

Experiment 64 is the separately numbered successor. It replaces the failed
distal-cap observation rule with a fixed geometric-median/q90 robust core,
commits all 384 previously viewed Experiment 63 identities as exclusions, and
requires one new disjoint visual sample in every eye. Its implementation and
protocol were frozen before that sample was viewed. The robust revision still
failed its visual gate, so no Experiment 63 result is repaired or
reinterpreted. The negative result concerns the cap producer, not the unrun
shape-versus-control comparison.

The complete Experiment 64 contract is in the
[frozen protocol](../../protocols/EXPERIMENT_64_ROBUST_CAP_POST_QC_VALIDATION.md).
The principal files are:

| File | Role |
|---|---|
| `experiment_64_robust_distal_core.py` | Shared physical-coordinate robust-core operator |
| `experiment_64_technical_metrics.py` | Shared fit diagnostics and modality-specific gates |
| `experiment_64_prepare_arthur_source_table.py` | Six-eye Arthur source producer with nested animal metadata |
| `experiment_64_extract_lens_surfaces.py` | Twelve-eye Maike producer with sealed targets separated from technical artifacts |
| `render_experiment_64_instance_qc_sample.py` | One disjoint 32-case sample per eye |
| `attest_experiment_64_instance_qc.py` | Immutable all-pass technical attestation |
| `experiment_64_two_pass_backend.py` | Pass-1 technical barrier and one acknowledged outcome run |
| `results/experiment_64_development_exclusions.json` | Canonical 12 × 32 development-exclusion ledger |
| `results/experiment_64_visual_exposures.json` | All 12 × 32 Experiment 64 identities now treated as viewed development cases |
| `results/experiment_64_stop_record.json` | Immutable pre-outcome stop decision and artifact bindings |

Stage 2 predictor/frame/QC calculations read only sealed distal cores. The
producers necessarily ingest complete meshes or binary bodies, and the visual
review shows complete components, so this is artifact/code-path isolation and
not process security or proximal-anatomy blindness.

The remainder of this README documents the shared cohorts and preserves both
stopped workflows for audit. The Experiment 64 commands below record the frozen
execution contract; they are not instructions to redraw, retune or score the
stopped experiment.

The scientific question, cohorts, thresholds and original decision rules for
Experiment 63 remain frozen in the
[Experiment 63 protocol](../../protocols/EXPERIMENT_63_MAIKE_OUTER_INFORMATION_VALIDATION.md).

The narrow question is whether four distal-cap shape descriptors improve
prediction of a hidden, quadratic-smoothed central **thickness cap** beyond a
nested eye-position-and-scale control. It is not a test of a complete raw lens
surface.

The pipeline is conditional on oracle distal-surface localization. Published
ODA centres and axes may be used only in Stage 1 for lens correspondence and
distal-cap localization. Stage 2 reopens only the sealed distal artifacts when
constructing predictors, eye frames and distal quality decisions. Proximal
geometry is attached afterward as a held-out target.

## Cohorts and independent units

The source cohort comprises three distinct female *D. melanogaster* flies,
represented by the whole-head scan volumes `20231107`, `20240530` and
`20240701` supplied by Arthur Zhao. Each volume contributes two bilateral
eyes, which are always nested within the animal/volume; the six eyes are never
treated as six independent units. This interpretation is supported by the
[source paper](https://doi.org/10.1038/s41586-025-09276-5) and the pinned
analysis code that maps the three dates to the main and two additional flies.
Here, independent means different animals, not different laboratories,
studies or acquisition workflows.

The fixed Arthur Stage-1 denominator is 4,942 oracle-localized caps and 4,897
must pass the frozen distal-only QC. Target-resolvable and target-quality
counts are measured outputs under this protocol rather than values forced to
match an earlier experiment.

The external validation cohort comprises twelve cleaned binary lens stacks
supplied by Maike Kittelmann and associated with [Buffry et al.
(2024)](https://doi.org/10.1186/s12915-024-01864-7): one eye from each of twelve
flies. There are three female and three male M3 *D. simulans* flies and three
female and three male RED3 *D. mauritiana* flies. The independent validation
unit is the animal/eye, not an individual lens. The fixed ODA tables contain
11,369 lens rows across the twelve animals.

| Species/line | Sex | Eye/animal | Fixed ODA rows |
|---|---|---|---:|
| M3 *D. simulans* | female | `M3_F_24_01` | 1,001 |
| M3 *D. simulans* | female | `M3_F_28_03` | 1,011 |
| M3 *D. simulans* | female | `M3_F_35_03` | 1,023 |
| M3 *D. simulans* | male | `M3_M_26_01` | 855 |
| M3 *D. simulans* | male | `M3_M_32_01` | 944 |
| M3 *D. simulans* | male | `M3_M_36_01` | 970 |
| RED3 *D. mauritiana* | female | `RED3_25_F_36` | 1,008 |
| RED3 *D. mauritiana* | female | `RED3_25_F_37` | 984 |
| RED3 *D. mauritiana* | female | `RED3_25_F_38` | 1,003 |
| RED3 *D. mauritiana* | male | `RED3_25_M_26` | 882 |
| RED3 *D. mauritiana* | male | `RED3_25_M_27` | 822 |
| RED3 *D. mauritiana* | male | `RED3_25_M_28` | 866 |
| **Total** |  | **12 animals** | **11,369** |

The primary rule pools all twelve animals. Species- and sex-stratified results
are descriptive and cannot establish separate confirmation in either species
or sex.

## Inputs and redistribution

- The twelve directly supplied 0.325-µm binary TIFF ZIPs. Their exact basenames,
  sizes, SHA-256 digests, slice counts and uncropped shapes are frozen in
  `prepare_maike_masks.py`; substituted or self-consistent replacement ZIPs
  are rejected.
- The official `fig3_share.zip` from Buffry et al. (2024), whose complete
  public payload is verified by `map_oda_to_source.py`. The source record is
  [Figshare version 6](https://doi.org/10.6084/m9.figshare.24769677.v6), and the
  ODA replay is fixed to commit
  `55684a97fb32a95f24d17eaf04c49253c98fee27`.
- Arthur's three exact lens meshes, three exact tip meshes and the matching
  position tables from [`reiserlab/eyemap_T4`](https://github.com/reiserlab/eyemap_T4)
  at commit `99d2a43123db636cedb55af9ff31a59657e7d17e`.
- A clean Git checkout at one frozen commit for source preparation, primary
  extraction and the backend. Diagnostic dirty-worktree flags are not
  admissible for the primary result.

The directly supplied Maike ZIPs and Arthur meshes are not redistributed
pending confirmation of their applicable reuse terms. Generated masks and
per-lens bundles are also large external artifacts and are not committed to
Git. Exact identities and provenance are documented in
[`data/README.md`](../../data/README.md), which permits verification without
presenting derived files as primary data.

## Raw representation shift

The source and validation cohorts share the distal-only frame, q90 lateral
domain, robust normalized quadratic fitter, coefficient convention and
81-point score grid. Their raw observation operators are nevertheless
different:

| Cohort | Distal predictor observations | Proximal target observations |
|---|---|---|
| Arthur source | Irregular vertices of the supplied distal surface mesh | Unique oracle-split proximal mesh vertices inside the final q90 disk |
| Maike validation | Boundary points from a 0.325-µm voxelized binary component | One deterministic minimum-axial point per 0.325-µm lateral bin spanning at least 0.650 µm axially |

This shift affects both predictors and targets. A transfer failure cannot by
itself distinguish voxelization/tessellation effects from acquisition, taxon
or other biological differences. A transfer success would support only these
fixed cohorts and observation operators.

## Archived Experiment 64 execution contract (audit only)

Experiment 64 has already stopped at its visual gate. The commands in this
section document what was frozen and executed before the stop, plus the backend
that was deliberately never invoked. Repeating them cannot create another
Experiment 64 result; any revised analysis needs a new experiment number.

Run the complete warning-as-error test suite, commit the protocol,
implementation and canonical exclusion ledger, and require a clean tree. Save
that commit as `<frozen-commit>`. Every producer and the backend must bind that
same commit.

Build the Arthur source bundle:

```bash
python -W error experiments/maike-modern-ground-truth/experiment_64_prepare_arthur_source_table.py \
  --manifest /path/to/arthur_manifest.json \
  --eyemap-root /path/to/eyemap_T4 \
  --output /new/path/experiment64/arthur \
  --repository-root /path/to/fossil-compound-eye-reconstruction
```

Build each Maike eye from the already verified mask/mapping pair:

```bash
python -W error experiments/maike-modern-ground-truth/experiment_64_extract_lens_surfaces.py \
  --mask /path/to/maike_masks_v2/M3_M_36_01.mask.uint8.npy \
  --provenance /path/to/maike_masks_v2/M3_M_36_01.mask.json \
  --seeds /path/to/maike_oda_mappings_v3/M3_M_36_01/seeds.csv \
  --seed-provenance /path/to/maike_oda_mappings_v3/M3_M_36_01/seeds.json \
  --eye-id M3_M_36_01 \
  --species 'Drosophila simulans' \
  --sex M \
  --output /new/path/experiment64/maike/M3_M_36_01 \
  --repository-root /path/to/fossil-compound-eye-reconstruction
```

Repeat for the twelve fixed eye IDs. M3 is *D. simulans* and RED3 is
*D. mauritiana*; `_F_` and `_M_` encode the recorded sex. Each atomically
published bundle separates `technical_inventory.csv`, `instances/` and
`sealed_distal/` from the hash-bound `sealed_outcomes/` directory.

For each eye, render the single disjoint QC draw. Repeat
`--prior-sample-manifest` exactly once for each of the twelve Experiment 63
sample manifests; the command independently reconstructs and verifies the
committed 384-case ledger before opening a new instance:

The abbreviated example below is not pasteable: replace the literal `...`
with the remaining ten named `--prior-sample-manifest` arguments.

```bash
python -W error experiments/maike-modern-ground-truth/render_experiment_64_instance_qc_sample.py \
  --eye-id M3_M_36_01 \
  --bundle-root /new/path/experiment64/maike/M3_M_36_01 \
  --repository-root /path/to/fossil-compound-eye-reconstruction \
  --prior-sample-manifest M3_F_24_01=/path/to/old/M3_F_24_01/sample_manifest.json \
  --prior-sample-manifest M3_F_28_03=/path/to/old/M3_F_28_03/sample_manifest.json \
  ...
```

Review all 32 renders in manifest order. A fail or indeterminate decision
stops the whole experiment; there is no replacement draw. A passing review
uses schema `experiment64.instance-qc-review.v1`, review scope
`disjoint_stratified_sample_only`, and is then attested with the same twelve
prior-manifest arguments:

```bash
python -W error experiments/maike-modern-ground-truth/attest_experiment_64_instance_qc.py \
  --bundle-root /new/path/experiment64/maike/M3_M_36_01 \
  --repository-root /path/to/fossil-compound-eye-reconstruction \
  --prior-sample-manifest M3_F_24_01=/path/to/old/M3_F_24_01/sample_manifest.json \
  --prior-sample-manifest M3_F_28_03=/path/to/old/M3_F_28_03/sample_manifest.json \
  ...
```

Only after all twelve attestations existed could the two-pass backend have been
invoked. That condition failed, so the following command was never run. Its
acknowledgement switch would have been the point at which separately sealed
target values could first be opened for model evaluation:

```bash
python -W error experiments/maike-modern-ground-truth/experiment_64_two_pass_backend.py \
  --repo /path/to/fossil-compound-eye-reconstruction \
  --expected-commit <frozen-commit> \
  --arthur-root /new/path/experiment64/arthur \
  --maike-root /new/path/experiment64/maike \
  --output-directory /new/path/experiment64/sealed_first_run \
  --execute-sealed-first-experiment64
```

After Pass 1 succeeds, the backend rechecks the clean frozen commit and,
immediately before any Pass-2 outcome read, atomically creates the sibling
`experiment64_outcome_attempt_<commit>/attempt.json` record next to the Maike
root. That record is retained if Pass 2 stops or crashes. Its presence—or an
existing result directory—blocks another invocation, so changing the output
path cannot turn the sealed analysis into a trial run. The 80% per-eye
`distal_qc AND target_resolvable` gate is evaluated after target validation and
joining but before any model fit. A successful result is bound to the attempt
record; a failed attempted outcome run is not retried under Experiment 64.

The primary rule is descriptive because Maike predictor morphology informed
the revised preprocessing. It requires strict shape-model wins in at least 10
of 12 named animals, counts ties as non-wins, and has no minimum effect-size
threshold. Secondary analyses cannot rescue that rule.

## Clean-commit workflow

Install the pinned dependency ranges, run the tests, commit the frozen
protocol and implementation, and record that exact clean commit before
building any primary-ready bundle:

```bash
python -m pip install -r requirements.txt
python -W error -m unittest discover \
  -s experiments/maike-modern-ground-truth \
  -p 'test_*.py' -q
python -W error -m compileall -q experiments/maike-modern-ground-truth
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. Record the printed commit as
`<frozen-commit>`; the producers bind it and the backend rejects a dirty tree,
a different commit, changed producer code or changed inputs. Keep all paths
below outside the repository unless the path is source code or documentation.

Before the freeze, `M3_M_36_01` was used only for coordinate, mask and runtime
testing. An incomplete extraction was deliberately stopped and its staging
directory removed after provenance and temporary-file issues were found. As
recorded in the protocol, no target table, prediction, error or model-comparison
outcome from that pilot was opened; the frozen workflow regenerates the eye.

## Archived Experiment 63 workflow (audit only)

The numbered subsections below preserve the stopped Experiment 63 workflow;
they are not Experiment 64 execution instructions.

### 1. Prepare the Maike masks

Prepare the 12 tight, lossless, memory-mapped masks:

```bash
python -W error experiments/maike-modern-ground-truth/prepare_maike_masks.py \
  /path/to/upload/*.zip \
  --output-dir /path/to/maike_masks_v2
```

The `/path/to/upload/` placeholder above must contain only the twelve frozen
directly supplied archives; do not mix `fig3_share.zip` into that directory.

### 2. Map the public ODA centres

Extract the official Figure 3 nested stacks so that `<fig3-public-root>`
contains the 12 `tiffs_*_eye_lenses_binned/` directories, then replay the ODA
transform and map all published lens centres:

```bash
python -W error experiments/maike-modern-ground-truth/map_oda_to_source.py \
  --batch-public-root /path/to/fig3-public-root \
  --batch-mask-dir /path/to/maike_masks_v2 \
  --batch-output-dir /path/to/maike_oda_mappings_v3 \
  --public-archive /path/to/fig3_share.zip
```

The mapper publishes `manifest.json` only after all 12 eyes and all 11,369
published ODA rows pass the frozen coordinate, axis, uniqueness and foreground
gates.

### 3. Prepare the Arthur source bundle

Create an external JSON manifest with exactly the three dated volumes and the
four exact fields illustrated here:

```json
{
  "volumes": [
    {
      "volume": "20231107",
      "lens_mesh": "/path/to/20231107-PTA-surface-lens.wrl",
      "tip_mesh": "/path/to/20231107-PTA-surface-tip.wrl",
      "rdata": "/path/to/eyemap_T4/data/microCT/20231107.RData"
    },
    {
      "volume": "20240530",
      "lens_mesh": "/path/to/20240530-PTA-surface-lens.wrl",
      "tip_mesh": "/path/to/20240530-PTA-surface-tip.wrl",
      "rdata": "/path/to/eyemap_T4/data/microCT/20240530.RData"
    },
    {
      "volume": "20240701",
      "lens_mesh": "/path/to/20240701-PTA-surface-lens.wrl",
      "tip_mesh": "/path/to/20240701-PTA-surface-tip.wrl",
      "rdata": "/path/to/eyemap_T4/data/microCT/20240701.RData"
    }
  ]
}
```

The exact RData location inside an eyemap checkout may differ; use the files
whose basenames, byte sizes and hashes match the frozen identities. Then build
the source bundle from the clean commit:

```bash
python -W error experiments/maike-modern-ground-truth/prepare_arthur_source_table.py \
  --manifest /path/to/arthur_manifest.json \
  --eyemap-root /path/to/eyemap_T4 \
  --output /path/to/experiment63_arthur_source \
  --repository-root /path/to/fossil-compound-eye-reconstruction
```

The primary backend later expects
`experiment63_arthur_source/arthur_source_table.csv` and
`experiment63_arthur_source/arthur_source_provenance.json`.

### 4. Build the twelve Maike eye bundles

Build one eye bundle with the matching mask and seed pair:

```bash
python -W error experiments/maike-modern-ground-truth/extract_lens_surfaces.py \
  --mask /path/to/maike_masks_v2/M3_M_36_01.mask.uint8.npy \
  --provenance /path/to/maike_masks_v2/M3_M_36_01.mask.json \
  --seeds /path/to/maike_oda_mappings_v3/M3_M_36_01/seeds.csv \
  --seed-provenance /path/to/maike_oda_mappings_v3/M3_M_36_01/seeds.json \
  --eye-id M3_M_36_01 \
  --species 'Drosophila simulans' \
  --sex M \
  --output /path/to/maike_eye_bundles/M3_M_36_01 \
  --repository-root /path/to/fossil-compound-eye-reconstruction
```

Repeat that command for each eye. M3 eyes are *D. simulans*; RED3 eyes are
*D. mauritiana*. The `_F_` and `_M_` identifier fields give the recorded sex.

### Bundle contract

Each atomically published bundle contains one row and one explicit artifact
per mapped seed, including empty or distal-QC-excluded rows:

- `instances/`: the full assigned foreground, dominant 26-connected component
  and component-size evidence;
- `sealed_distal/`: the narrow Stage-2-readable distal artifacts;
- `lenses/`: distal geometry plus target observations for later scoring;
- `lens_summary.csv`: complete technical and target inventory;
- `distal_qc_sampling.csv`: outcome-blind allowlisted fields for visual QC;
- `distal_frame_audit.json`: frozen jackknife, subsample and perturbation gates;
- `completion.json` and `provenance.json`: exact counts, input bindings, Git and
  implementation identities, partition evidence and output hashes.

Extraction stops without publishing the bundle if an input or provenance link
does not match, the foreground is not an exact disjoint partition, the
distal-only fixed point does not converge, the frame audit fails, or the
staging directory contains anything outside the exact artifact inventory.

### 5. Verify the distal-frame QC

`extract_lens_surfaces.py` runs the target-blind distal-frame audit internally
and seals its canonical result as `distal_frame_audit.json`; that file must not
be replaced. To reproduce the calculation independently, write the recheck to
a new external path:

```bash
python -W error experiments/maike-modern-ground-truth/audit_distal_frame_stability.py \
  --distal-dir /path/to/maike_eye_bundles/M3_M_36_01/sealed_distal \
  --eye M3_M_36_01 \
  --output /new/path/M3_M_36_01.distal_frame_audit.recheck.json
cmp \
  /path/to/maike_eye_bundles/M3_M_36_01/distal_frame_audit.json \
  /new/path/M3_M_36_01.distal_frame_audit.recheck.json
```

Repeat for all twelve eyes. A nonzero audit exit status or a non-identical
recheck is a stop condition, not a row-level exclusion.

### 6. Render and review the frozen visual-QC sample

Render exactly 32 outcome-blind, distal-QC-stratified instances for one eye:

```bash
python -W error experiments/maike-modern-ground-truth/render_instance_qc_sample.py \
  --eye-id M3_M_36_01 \
  --bundle-root /path/to/maike_eye_bundles/M3_M_36_01
```

The command reads only the exact allowlisted sampling table, full/dominant
component artifacts and sealed distal artifacts. It does not open the fitted
lens target, a prediction, an error or a model. Review all 32 three-view PNGs
against the criteria in the frozen protocol. The reviewer then writes
`instance_qc_review.json` at the bundle root with the following schema. The
example shows one of the required 32 decisions and is not itself a submit-ready
review:

```json
{
  "schema_version": "experiment63.instance-qc-review.v1",
  "eye_id": "M3_M_36_01",
  "review_scope": "stratified_sample_only",
  "review_mode": "human",
  "reviewer_id": "<nonempty reviewer identifier>",
  "reviewed_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "sample_manifest_sha256": "<sha256 of instance_qc_visual_sample/sample_manifest.json>",
  "decisions": [
    {
      "lens_index": 123,
      "seed_id": "<string copied from sample 1>",
      "decision": "pass",
      "notes": "<review note, possibly empty>"
    }
  ]
}
```

The `decisions` array must contain all 32 samples in manifest order, with
integer `lens_index` values. `review_mode` may instead be
`ai_assisted_visual_review_without_model_outputs` when accurate. Do not issue
a passing review if any case is failed or indeterminate: one such case stops
that eye and therefore the primary experiment, without replacement or manual
relabeling. This review validates only the stratified sample, not every lens.
Because the full binary component is visible, it is model/error-blind technical
QC but is not proximal-target-blind.

After a genuine passing review, issue the immutable sidecar:

```bash
python -W error experiments/maike-modern-ground-truth/attest_instance_qc.py \
  --bundle-root /path/to/maike_eye_bundles/M3_M_36_01
```

Repeat rendering, review and attestation for all twelve eyes. The attester
revalidates the full automatic inventory and hashes but does not turn the
sample review into a claim of complete manual validation.

### 7. Execute the frozen backend once

Only after all source/validation bundles and all twelve attestations pass,
invoke the backend with the same clean commit. The output directory must not
already exist:

```bash
python -W error experiments/maike-modern-ground-truth/experiment_63_primary_backend.py \
  --repo /path/to/fossil-compound-eye-reconstruction \
  --expected-commit <frozen-commit> \
  --arthur-table /path/to/experiment63_arthur_source/arthur_source_table.csv \
  --arthur-provenance /path/to/experiment63_arthur_source/arthur_source_provenance.json \
  --maike-root /path/to/maike_eye_bundles \
  --output-dir /new/path/experiment63_sealed_first_complete_run \
  --execute-frozen-primary
```

The acknowledgement flag is deliberate: this command opens held-out targets,
fits models, computes prediction errors and atomically writes the first
complete result. Do not use it as a preflight or trial run. Failed input
preflight may be repaired without examining an outcome; a successful complete
run is retained as `sealed_first_complete_run` and is not replaced under
changed thresholds.

The backend stops the entire analysis if any eye retains less than 80% of its
fixed ODA denominator in the primary cohort (`distal_qc AND
target_resolvable`) or if any primary normalization denominator is nonfinite
or nonpositive. Such rows are never silently dropped.

The backend also runs the prespecified exact-`target_qc` sensitivity and the
nested within-Maike leave-one-animal-out same-operator diagnostic. Those are
secondary analyses and cannot rescue or redefine the Arthur-to-Maike primary
test. The supporting equal-source-volume template is computed internally from
the frozen Arthur cohort; the sealed run accepts no user-supplied prediction
table.

## What the experiment can and cannot establish

A valid primary result supports only a conditional statement about the fixed
modern cohorts, oracle-localized distal caps, target operator and two nested
models. The shape addition is a set—gradient magnitude, two ordered principal
curvatures and normalized fit residual—not “curvature alone.” Report the
per-animal error magnitudes and absolute errors with the win count because the
pass rule has no minimum practical-effect margin. The prespecified pass
requires a strictly lower median normalized 81-point error for the shape model
in at least 10 of 12 animals; exact ties are non-wins.

A primary failure means that this fixed Arthur-trained descriptor model did
not reliably improve transfer to the supplied Maike cohort. It does not prove
that outer geometry contains no information in principle. A primary success
does not establish automatic distal localization or generalization beyond the
fixed operators and animals.

Neither outcome identifies the candidate *Asaphus* CT boundary, reconstructs
a raw or watertight fossil lens, validates surface normals as optical axes,
recovers fossil optical function, or resolves the mesh-versus-voxel,
acquisition, taxon and biological contributors to domain shift.

## Tests

```bash
python -W error -m unittest discover \
  -s experiments/maike-modern-ground-truth \
  -p 'test_*.py' -q
python -W error -m compileall -q experiments/maike-modern-ground-truth
```

These tests exercise contracts and synthetic fixtures; they do not execute the
one-time backend or inspect Experiment 63 errors.

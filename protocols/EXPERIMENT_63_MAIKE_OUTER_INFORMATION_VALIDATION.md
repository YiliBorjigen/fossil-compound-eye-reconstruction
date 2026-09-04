# Experiment 63 — independent modern-eye outer-information validation

**Protocol status:** frozen before any Maike prediction error or model-comparison
outcome was inspected. The freeze date is 2026-09-04. The Git commit containing
this protocol is the analysis identity recorded by every primary-ready bundle
and by the scoring run.

## Question and scope

Conditional on an oracle having already located the distal corneal-lens cap,
do four distal-shape descriptors improve prediction of the hidden proximal
**thickness cap** beyond a nested eye-position-and-scale control?

This is not an end-to-end segmentation test. Stage 1 may use complete lens
geometry, published lens centres and anatomical axes to establish
correspondence and separate the distal cap from the rest of a lens. The Stage-2
geometry functions must reopen only hash-sealed distal-cap artifacts. They
derive every predictor, eye frame, position, distal quality decision and model
input without receiving a proximal point, thickness target, prediction or
error.

The response is an 81-point quadratic-smoothed central thickness cap. It is
not the raw rim, the complete proximal surface, a watertight two-surface lens,
or an automatically localised fossil feature.

## Data and independent units

### Source cohort

Arthur Zhao supplied complete corneal-lens and photoreceptor-tip meshes for
three distinct female *Drosophila melanogaster* flies, represented by the
whole-head scans `20231107`, `20240530` and `20240701`. Each scan contains both
eyes. Thus source hyperparameter selection has three animal/volume units, not
six eye units. Both eyes may contribute lenses, but leaving out a source unit
means leaving out the entire animal/volume. Here, independent means different
animals; it does not mean an independent laboratory, acquisition study or
biological population.

The source Stage-1 denominator is fixed at 4,942 oracle-localised caps. The
distal-only frozen quality rules retain 4,897; those two counts are protected
against silent cohort changes. Target-resolvable and target-quality counts are
measured under this protocol and are not forced to reproduce an earlier
experiment.

### External validation cohort

Maike Kittelmann supplied twelve manually cleaned, binary corneal-lens TIFF
stacks used in Buffry et al. (2024): six M3 *D. simulans* animals and six RED3
*D. mauritiana* animals, with three females and three males per species and one
eye stack per fly. The animal/eye identifiers and fixed ODA lens counts are:

| Species/line | Sex | Eye | Lenses |
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

The twelve animal/eye stacks, not the 11,369 lenses, are the independent
validation units. Here, independent means different flies; it does not imply
twelve acquisition batches, vials, families or studies. The primary pass rule
pools the twelve animals and therefore cannot by itself establish a result
separately in both species or both sexes.

## Input reconstruction and Stage-1 oracle work

The public Figure 3 archive from Buffry et al. is fixed by its size, MD5 and
SHA-256. Its nested `stacks.zip` and, for each eye, every binned TIFF, published
`ommatidial_data.csv` and zero-byte H5 placeholder are checked byte-for-byte.
The ODA centring and rotation are replayed from the public binned TIFFs at ODA
commit `55684a97fb32a95f24d17eaf04c49253c98fee27`.

ODA columns named `(x,y,z)` originated as `(slice,row,column)`, corresponding
to source `(z,y,x)`. The frozen inverse is:

```text
q_binned = (p_oda @ rotation.T + sphere_center) / 1.3
q_source = 4 * q_binned + 1.5
q_mask = q_source - crop_origin
```

Every published centre must inverse-map inside its 0.325-µm source mask, round
to a unique foreground voxel and retain a finite outward unit axis. The binary
foreground is then assigned losslessly and without overlap to the nearest
centre. A deterministic largest 26-connected component is retained for each
lens. Stage 1 uses the complete component and the ODA axis to identify a
candidate distal boundary layer, then seals only that layer.

Arthur Stage 1 likewise uses the complete supplied mesh and matched tip
landmarks to assign and split the two surface layers, then seals only the
distal points. Neither oracle operation may be described as automatic
distal-surface localisation.

## Distal-only Stage 2 and frozen quality rules

The Stage-2 geometry functions receive only exact-schema, hash-sealed distal
artifacts. All predictor coordinates, frames and quality decisions are
recomputed from those artifacts. The producer orchestrator retains complete
Stage-1 geometry only for target construction after the distal fixed point;
this isolation is a reviewed API/dataflow boundary, not a separate-process
security boundary. Eligibility is a monotone drop-only fixed point with no
re-entry and a maximum of 20 iterations.

In both cohorts, common Stage-2 distal-cap eligibility requires all of the
following:

- q90 radial scale between 3 and 13 µm, inclusive;
- at least 25 raw coordinates in the sealed Stage-1 distal cap, counted in that
  cohort's native predictor representation (a q90 subset is computed for the
  quadratic fit, but its count is not the support gate; this threshold does
  not equate source and validation sampling densities);
- distal quadratic RMSE no greater than 2.5 µm;
- quadratic design condition no greater than 1,000,000.

For Maike Stage 1 only, the foreground partition additionally uses one
deterministically selected nearest centre per foreground voxel and retains the
deterministic largest 26-connected component for each lens. These are
segmentation/partition rules, not common distal-cap QC criteria and not Arthur
source rules. Arthur's supplied surface meshes have no analogous component or
visual-instance gate, so Stage-1 selection/QC is not symmetric across source
and validation modalities.

The distal frame must also pass all three frozen perturbation audits:
exhaustive leave-one-origin-out, deterministic 90% subsampling and a 1%
nearest-neighbour Gaussian perturbation. The hard limits are 5° for the p95
noncentral poleward-axis change, 2° for the p95 outward-axis change, 2% for
central-classification changes and zero noncentral numerical fallbacks.

## Target construction

Within the oracle-localised Stage-1 cohort, Stage-2 distal-QC predictor
eligibility is frozen before and independently of every target field. A target
is structurally resolvable only when the distal frame exists, the required
observation support exists, and a finite full-rank proximal quadratic can be
fit. Positive local thickness and target-fit RMSE are not primary selection
rules.

The source and validation data share the final distal-only frame, q90 lateral
domain, robust normalized quadratic fitter, coefficient convention and
81-point scoring grid. Shared computation does not make their raw point
representations identical. The representation shift affects both the distal
predictor inputs and the proximal target observations:

- Arthur's distal predictors are measured from irregular vertices of the
  supplied distal surface mesh. Its source response uses unique oracle-split
  proximal mesh vertices within the final distal q90 disk.
- Maike's distal predictors are measured from a 0.325-µm voxelized boundary
  cap. Its filled binary target must have a dominant component containing at
  least 99% of assigned foreground; complete-component voxels are grouped into
  0.325-µm lateral bins, bins must span at least 0.650 µm axially, and one
  deterministic minimum-axial point is taken per spanning bin. At least 25
  such bins are required.

This mesh-versus-voxel predictor-and-target representation difference is a
prespecified domain shift. Even with common fitting code, tessellation or
voxelization can affect scale, curvature, fit residual, target weighting and
the normalization denominator. A failed Arthur-to-Maike transfer cannot
distinguish those representation effects from taxon, acquisition or other
biological shifts. A successful transfer supports only the fixed supplied
cohort and operators.

At each retained observation, raw thickness is distal fitted height minus
proximal observed height. The target coefficients `c0..c5` describe the robust
quadratic thickness surface on normalized coordinates. `target_depth_um` is
the median raw thickness and is the fixed normalized-error denominator.

`target_qc` is an exact sensitivity flag:

```text
target_resolvable AND raw_thickness_q05 > 0 AND target_RMSE <= 2.5 µm
```

It may not change primary membership. If any structurally resolvable primary
row has a nonfinite or nonpositive `target_depth_um`, the whole experiment
aborts; that row may not be silently removed.

## Visual instance QC

Before scoring, exactly 32 final-distal-QC lenses per eye are selected without
model or error data: two hash-minimal cases from each cell of a stable-rank 4 ×
4 radial-position by distal-scale grid. The reviewer sees only three
projections of the full assigned foreground, dominant component and sealed
distal cap.

A sample passes only if the dominant component is a single plausible lens
body, does not show an obvious neighbouring-lens merge or major truncation,
and the cyan sealed points form one coherent cap on that component. Any failed
or indeterminate sample stops the eye and therefore the primary experiment;
there is no replacement sample, manual relabelling or outcome-dependent lens
exclusion. This is model/error-blind Stage-1 technical QC of a stratified
sample. It is not complete manual validation of all instances and is not
proximal-target-blind because the full binary component is visible.

## Nested models

Both models receive eleven frozen control features:

1. radial eye position;
2. distance to the distal-QC convex-hull boundary;
3. first-, third- and sixth-neighbour distances;
4. q10, q25, q50, q75 and q90 within-eye pairwise distances;
5. distal q90 scale.

These features are invariant to a global two-dimensional rotation or
reflection of an eye. The shape model adds exactly four collective distal-cap
descriptors: gradient magnitude, the two ordered principal curvatures and
normalized quadratic-fit residual. The test is therefore about this descriptor
set, not curvature alone.

Both models are weighted ridge regressions. They fit reflection-even target
coefficients `c0`, `c1`, `c3` and `c5`; predicted `c2` and `c4` are zero, but
all six target terms contribute to the grid score. Central targets and
predictions receive the frozen rotational symmetrization.

Ridge alpha is selected separately for each model from
`logspace(-2, 3, 12)`. Selection uses only Arthur data: leave one complete
volume/animal out, compute its median per-lens normalized grid MAE, take the
equal mean of the three held-out-animal medians, and choose the smallest alpha
at the minimum. Final source fitting gives each of the three animals equal
total weight. No Maike target enters model or alpha fitting for the primary
analysis.

## Primary estimand, test and decision rule

For each target-resolvable lens, the primary error is:

```text
mean absolute error on the 81-point smoothed thickness grid
-----------------------------------------------------------
                 median raw target thickness
```

Each animal contributes the median of its lens errors for each model. The shape
model wins an animal only when its median is strictly lower than the control's;
an exact tie is a non-win. The primary result passes only with at least 10 wins
among the fixed 12 animals.

The recorded reference calculation is a two-sided fixed-denominator binomial
tail; at 10 of 12 wins it is `0.03857421875`. The conventional two-sided
tie-dropping sign-test value is also reported. The fixed-denominator value is
a prespecified reference for the decision rule, not a general treatment of
structural ties under a continuous null. There is no minimum practical-effect
margin, so per-eye effect magnitudes and absolute errors must accompany the win
count. Passing the rule alone must not be described as a material improvement.

Every eye must contribute at least 80% of its fixed ODA denominator to the
`distal_qc AND target_resolvable` primary cohort. If any eye fails that gate,
the complete primary experiment stops.

## Prespecified secondary analyses

Secondary analyses cannot rescue or redefine the primary result:

- repeat model selection, fitting and scoring on the exact `target_qc`
  sensitivity cohort;
- report 81-point absolute errors and raw unsmoothed errors where available;
- report the equal-source-volume template as a supporting comparator;
- report, as source-only tuning diagnostics rather than independent
  validation, each model's three held-out-volume median normalized errors at
  that model's selected alpha;
- attach the frozen species and sex labels to the twelve rows of
  `per_eye_primary.csv`; for each species margin (six animals), each sex margin
  (six animals) and each of the four species-by-sex cells (three animals),
  report the named-eye `control_minus_shape` effects, their median and strict
  win/loss/tie counts. These are descriptive only: there is no subgroup p-value
  or subgroup pass rule;
- perform a same-operator, leave-one-Maike-animal-out diagnostic. For each
  held-out animal, alpha is selected separately for each model by nested
  leave-one-animal-out validation inside the other eleven animals, with equal
  animal mass; the model is then refit on those eleven and applied once to the
  outer animal. This analysis may show whether the relation is learnable when
  the predictor representation, preprocessing family and target-observation
  operator are matched, but it cannot identify the cause of any
  Arthur-to-Maike failure and has no rescue rule.

No species-to-species model is a prespecified confirmatory result. Any such
analysis must be labelled exploratory.

## Stop conditions and one-time execution

The primary backend refuses to run unless:

- the repository is clean at the protocol commit;
- all source and twelve validation artifacts have exact schemas, hashes and
  counts; Arthur identities are unique and complete relative to its declared
  per-eye manifests, while each Maike eye is the contiguous range `0..N-1`;
- all sealed distal artifacts and frame audits revalidate;
- all foreground partitions are lossless, nonoverlapping and one-candidate;
- all twelve stratified visual-QC attestations pass;
- every source animal and validation animal contributes its required cohort;
- every primary metric denominator is finite and positive; and
- the exclusive output directory does not already exist.

The backend writes atomically to a new directory, refuses to overwrite that
directory and calls the first successful complete result
`sealed_first_complete_run`. A failed preflight may be repaired without
examining outcomes. The one-complete-run rule is a recorded analysis policy,
not an absolute filesystem lock: a user could technically choose another empty
directory. Every run manifest binds its inputs, output hashes, Git commit and
creation time so a later run is detectable. A completed primary result is not
replaced or reinterpreted under changed thresholds.

## Pre-freeze technical work and claim boundary

`M3_M_36_01` was used for coordinate/mask and runtime testing before the
protocol commit. An early extraction was deliberately interrupted after a
provenance fail-open and temporary-file inventory problem was found. It had
serialized some target artifacts, but no target table, prediction, error or
model-comparison outcome was opened; the incomplete staging directory was
deleted and is not an analysis input. The final eye is regenerated from the
frozen commit like every other eye.

Experiment 63 can directly support only a statement conditional on
oracle-localised modern distal caps and the tested models. It cannot establish
automatic distal localisation, anatomical identity of the *Asaphus* internal
transition, universal impossibility of outer-only reconstruction, a fossil
lens reconstruction, or fossil optical function.

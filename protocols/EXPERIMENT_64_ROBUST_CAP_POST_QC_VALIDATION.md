# Experiment 64 — robust-cap post-QC validation

> **Frozen before execution.** This protocol, its implementation, numeric
> thresholds, tests and the 384-case exclusion ledger are bound by the clean
> Git commit containing this file. At that freeze point, no Experiment 64
> disjoint visual-QC sample had been rendered and no Maike target table or
> artifact, prediction, error or model comparison had been opened or computed.
> Subsequent access is permitted only through the ordered gates below.

## Why this is a new experiment

Experiment 63 stopped at its prespecified visual-instance gate before any
model outcome was evaluated. The immutable stop record is
`experiments/maike-modern-ground-truth/results/experiment_63_stop_record.json`.
Four sampled sealed distal caps contained attached peripheral tails, spikes or
lobes. The original largest-component rule could not remove them because each
cap remained one connected three-dimensional voxel component.

Experiment 64 does not repair, continue or reinterpret Experiment 63. It
defines a new robust central-cap observation operator. Rule development used
only the predictor geometry and full-component renders already viewed during
Experiment 63; it did not use a held-out target or model outcome.

All 384 Experiment 63 sample identities are development cases, including the
cases that passed. They must be excluded eye by eye from Experiment 64's only
visual-QC sample. This provides a disjoint lens-level technical check within
the same eyes; it does not create new animal-level independence.

## Scientific question and inference boundary

Conditional on oracle distal localization and the fixed robust central-cap
operator below, do the same four distal-shape descriptors improve animal-level
prediction of an amended central thickness cap beyond the same nested
eye-position-and-scale control?

The twelve M3/RED3 eyes remain one eye from each of twelve flies. However,
their input morphology informed preprocessing development. Experiment 64 is
therefore a model/error-sequestered, post-QC evaluation, not untouched external
confirmation of the complete producer. New animals are required for that
stronger claim.

Arthur's source observations are irregular mesh vertices and Maike's
validation observations are voxel-boundary samples. Applying one
physical-coordinate operator to both does not eliminate that representation,
acquisition or taxon shift.

## Model/error and sealed-outcome sequestration

This is artifact and code-path isolation, not process-level security. The
deterministic producers necessarily ingest complete supplied meshes or binary
components in order to create both technical artifacts and, later, sealed
targets. Arthur's adapter may hold source proximal arrays in memory while it
finishes all six technical eye paths; the Maike technical renderer shows the
complete assigned and dominant lens bodies. Neither the producer process nor
the visual review is therefore proximal-observation-blind.

Stage 2 predictor, frame and automatic distal-QC calculations receive only
the hash-sealed robust distal cores. Before model evaluation, the renderer,
attester and backend Pass 1 may read:

- assigned and dominant-component point sets for disclosed visual review;
- raw localized distal points and robust-core sealed distal points;
- technical inventories containing no derived target variables, provenance
  and hashes; and
- Experiment 63 sample manifests, the committed exclusion ledger and the
  committed stop record.

Those technical consumers must not open any separately sealed target
manifest, target table or target NPZ, nor any target coefficient, prediction,
model, score or error. Thus the enforceable claim is model/error and sealed
target-value sequestration, not blanket denial of access to proximal anatomy.

The backend has two ordered passes. Pass 1 validates the clean Git commit,
producer identities, exact robust-core configuration, all Arthur/Maike
technical manifests, all twelve new visual samples, all twelve reviews and all
twelve attestations without opening any separately sealed target/outcome
artifact. If any technical gate fails, execution ends there. Only after Pass 1
succeeds for all twelve eyes may the backend recheck the clean frozen commit,
atomically create a durable outcome-attempt record, and let Pass 2 open the
sealed Arthur and Maike target manifests for the one outcome analysis. The
record is created before the first Pass-2 outcome read, is retained if Pass 2
later stops or crashes, and blocks another invocation for the same frozen
Maike input root and commit.

## Robust distal-core operator

The identical operator is applied in physical micrometre coordinates to the
unique Arthur mesh vertices and to the Maike integer-voxel distal samples
before the final distal artifact is sealed.

For one raw localized distal point set:

1. Require at least 33 unique, finite three-dimensional points.
2. Compute the geometric median by the frozen modified-Weiszfeld algorithm,
   beginning at the arithmetic mean and failing closed if it does not converge
   within 512 iterations at relative tolerance `1e-12`.
3. Form the tie-inclusive Euclidean q90 set around that median.
4. Estimate a tangent-plane projector by PCA of that Euclidean q90 set. Require
   the frozen normal-eigenspace gap ratio of at least `1e-6`; do not substitute
   a coordinate-axis fallback.
5. Project every original point into that plane and form the tie-inclusive
   lateral-radius q90 set around the same geometric median.
6. Retain the intersection of the Euclidean-q90 and lateral-q90 sets. Require
   at least 27 retained points. With tie-inclusive selection, the two q90 sets
   guarantee at least about 80% raw retention and a subsequent q90 fit support
   of at least 25 points at the minimum input size.
7. Preserve Maike's selected integer source coordinates and sort each final
   retained point set lexicographically. Seal the complete robust-core config
   and SHA-256 with every artifact.

The normal sign is irrelevant because only its projector is used. Boundary
ties are included within the frozen relative tolerance rather than broken by a
coordinate-dependent ordering. The operator is rigid-motion equivariant as a
selected point set.

The unchanged distal-only eye geometry is then recomputed from the robust
sealed points. The q90 lateral subset used by the quadratic descriptor is
therefore nested inside this new core. Final q90 fit support, retained fraction,
PCA eigengap and residual-tail diagnostics must be recorded for every lens;
final-q90 26-connectivity is additionally recorded for Maike's voxel caps and
is explicitly not imputed for Arthur's irregular mesh vertices.

### Technical gates and coherence ordering

Predictor-only calibration was run across all 4,942 Arthur and 11,369 Maike
raw sealed caps before any Maike outcome was opened. Under the robust operator,
the existing final scale range `3–13 µm`, quadratic RMSE maximum `2.5 µm` and
q90 fit-support minimum 25 remain valid numerical gates; the new counts are
measured rather than forced to match Experiment 63.

For Maike's voxel-boundary caps, two additional target-free gates apply to the
final Stage-2 q90 fitting points:

- their largest 26-connected component must contain at least 0.99 of the
  fitting points; and
- the p99 absolute quadratic residual divided by robust q90 scale must not
  exceed 0.75.

The thresholds fall in observed predictor-only gaps and use round values. Of
11,349 caps surviving the unchanged distal fixed point in the development
calculation, two fell below the 0.99 component fraction and four exceeded 0.75
normalized p99 residual; the next cases were approximately 0.992 and 0.741.
These counts are technical calibration facts, not model results. The frozen
producer must recompute them from the new artifacts and must not hard-code an
expected pass count.

Arthur's supplied meshes contain irregular vertex sampling and lack a
corresponding filled 26-neighbour lattice. A radius graph based on median
nearest-neighbour distance fragmented valid surfaces and is not used as a
gate. Arthur therefore retains the robust-core support/eigenspace gates and
the common scale, RMSE and q90-support gates, but not Maike's voxel-connectivity
gate or its calibrated residual-tail gate. This modality-specific QC is a
declared remaining representation asymmetry.

For each eligible Maike lens, define the target-free coherence margin as the
minimum of:

```text
(q90_fit_support - 25) / 25
(2.5 - quadratic_rmse_um) / 2.5
(q90_largest_26_component_fraction - 0.99) / 0.01
(0.75 - q90_p99_abs_residual_over_scale) / 0.75
```

Every term is nonnegative after gating; lower is nearer at least one failure
boundary. This scalar orders the “worst-margin” visual case in each stratum and
is never a model feature.

## Targets and models

Target construction, the 81-point disk, reflection handling, feature lists,
ridge-alpha grid, Arthur-only leave-one-volume-out selection, equal source
volume weighting, primary normalized MAE and secondary analyses are unchanged
from the frozen Experiment 63 protocol. They are re-evaluated on the robust
core's final frame and scale, so Experiment 64 is a new central-cap estimand,
not a numerically interchangeable rerun.

The control receives the same ten eye-position features plus robust distal
q90 scale. The shape model adds gradient magnitude, two ordered principal
curvatures and normalized robust quadratic residual. Neither model may receive
an extraction-QC diagnostic or an identity label as a predictor.

Every eye must retain at least 80% of its fixed ODA denominator in the
`distal_qc AND target_resolvable` cohort. There is no row-level manual repair,
manual relabeling, error-dependent exclusion or post-unsealing exclusion
beyond the two prespecified target-availability cohorts
(`target_resolvable` primary and exact `target_qc` sensitivity).

## Disjoint visual-QC sample

The committed canonical ledger at
`experiments/maike-modern-ground-truth/results/experiment_64_development_exclusions.json`
must bind the exact 32 Experiment 63 identities and old sample-manifest
SHA-256 for each eye, the Experiment 63 frozen commit and the committed
stop-record hash. It must independently reconstruct byte-for-byte from the
twelve old manifests before any new sample is rendered. Exclusion is scoped
to `(eye_id, lens_index)`; equal numeric indices in another eye are unrelated.

After robust geometry and automatic distal QC, the selector removes those 384
identities, recomputes stable-rank 4 × 4 radial-position-by-robust-scale cells,
and selects exactly two unseen lenses per cell under the namespace
`experiment64_instance_qc_v1`:

- one lens with the worst frozen coherence-risk margin in that cell; and
- one hash-minimal lens from the remaining cases.

If a cell lacks two eligible unseen cases, the experiment stops. The manifest
must prove zero overlap with the development ledger. There is one sample draw;
no replacement, redrawing or threshold change is allowed after viewing it.

The reviewer sees three projections of the assigned foreground, dominant
component, raw localized distal points and final robust sealed core, with no
derived target overlay, target coefficient, prediction, error or model. A case
passes only when the dominant body is a plausible single lens without a major
merge/truncation affecting the central analysis domain and the final robust
core is one coherent central cap without a residual shelf, prong or satellite.
Any failed or indeterminate case stops the eye and the whole experiment.

## Model decision and reporting

The decision rule is directional: the shape model meets it only with a
strictly lower per-eye median normalized error in at least 10 of 12 named
animals. Ties remain non-wins in the fixed `n = 12` denominator. At the
threshold, the prespecified two-sided fixed-denominator reference is
`P(W >= 10 or W <= 2)` for `W ~ Binomial(12, 0.5)`, equal to
`0.03857421875`. This is not the conventional tie-dropping sign test; the
backend reports that separately. The directional rule—not a small two-sided
value in the opposite direction—determines whether the criterion is met.

There is no minimum practical-effect threshold: ten arbitrarily small strict
improvements can meet the directional rule. A passing rule therefore supports
added predictive value under the fixed metric, but cannot by itself be called
a material or biologically important improvement; named-eye absolute and
normalized effect sizes control that interpretation.

Because preprocessing was developed from these animals' input morphology, the
win rule and binomial value are descriptive decision references, not pristine
external-confirmation inference. Every report must include named-eye effects,
absolute errors, per-eye cohort-retention counts, Arthur and Maike technical
exclusions with per-reason counts, target-resolvability and target-QC
exclusions, and the mesh-versus-voxel observation-shift caveat.

If the rule is met, the supported statement is limited to added predictive
value of this fixed descriptor set for the robustly defined central-cap
representation among the retained `distal_qc AND target_resolvable` lenses in
the supplied animals. Up to 20% target-unresolvable loss is permitted by the
gate and cannot be assumed random. If the rule is not met, the supported
statement is that this amended model missed its fixed criterion. Neither
outcome establishes universal possibility/impossibility, anatomical identity,
causal curvature–thickness coupling, fossil transfer, a fossil lens or fossil
optical function.

## Stop conditions

The complete experiment stops before predictive-model fitting or scoring if
any of the following occurs. All outcome-independent technical failures stop
in Pass 1 before separately sealed outcomes are opened. The target-dependent
80% resolvability gate is evaluated after Pass 2 validates and joins the target
inventories, and before any model fit.

- the repository is dirty or differs from the expected protocol commit;
- an input, producer, robust-core config, schema, hash, count or manifest fails;
- a separately sealed target/outcome artifact is accessed during technical
  Pass 1;
- the disjoint sample overlaps an Experiment 63 identity or a cell lacks two
  eligible unseen cases;
- any new visual decision is failed or indeterminate;
- any of the twelve attestations is absent or nonpassing;
- a frame, connectivity, residual-tail, support or 80%-retention gate fails;
- a primary denominator is nonfinite or nonpositive; or
- the exclusive output directory or durable outcome-attempt record already
  exists.

A stopped Experiment 64 is not rerun with new thresholds or another sample.
After the durable record has been created, it also must not receive a second
Pass-2 attempt, even if the first attempt produces no result directory. Further
development requires another numbered experiment. A successful Pass 2 writes
one immutable result bound to the attempt record, all inputs, outputs,
implementation hashes and the clean Git commit. This durable fail-closed record
is a workflow control, not an operating-system security boundary.

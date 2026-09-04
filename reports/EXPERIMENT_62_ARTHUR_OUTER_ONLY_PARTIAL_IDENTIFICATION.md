# Experiment 62 — outer-only partial identification

## Verdict

With no surviving same-eye proximal donors, the three Arthur volumes do not
support a unique or reliably accurate proximal-surface reconstruction from the
current invariant distal descriptors.  A source-only outer model has held-out
median central-cap errors of **13.79 µm** (20231107), **3.64 µm** (20240530)
and **6.36 µm** (20240701), corresponding to 48.1%, 22.8% and 47.5% normalized
error.

The useful output at zero donors is therefore not a claimed recovered lens. It
is an explicit set of alternative-morphology scenarios showing how strongly a
point estimate depends on an unobserved population or segmentation regime.

## What was recovered and what was not

The historical Experiment 62 existed only in an interrupted, unpushed
worktree.  Its report title and qualitative conclusion were recovered.  Two
orphan numerical constants also survived in the session handoff:

- ridge alpha: `123.28467394420659`;
- support threshold: `3.095098748914439`.

Their historical feature set, score definition and support-distance definition
did not survive.  The replacement analysis does **not** select the orphan
alpha: its transparent leave-one-volume-out procedure selects **0.01**.  It
records both values and the mismatch, and it does not use the orphan support
threshold to exclude any row.  Forcing a plausible-looking method until it
reproduced an isolated scalar would create false provenance.

The selected 0.01 is the lower boundary of the tested grid, not an identified
interior optimum.  The equal-volume score is also nearly flat at that end:
0.3945305 at alpha 0.01 versus 0.3945338 at 0.02848.  The deterministic grid
choice is adequate for this reconstructed comparator, but it is not evidence
that 0.01 is a generally optimal penalty.

## Reconstructed source-only model

The replacement uses the current distal-only, reflection-invariant source
contract:

1. distal q90 scale;
2. distal gradient magnitude;
3. the two ordered distal-curvature eigenvalues;
4. normalized distal quadratic-fit residual.

No eye position or volume label is a predictor feature, and no photoreceptor
tip or same-eye proximal donor enters the predictor.  Volume labels are used
only to define whole-volume folds and equal-volume fitting weights.  The model
fits the reflection-even thickness coefficients `c0`, `c1`, `c3` and `c5`,
sets the reflection-odd `c2` and `c4` predictions to zero, and is scored against
the complete quadratic target on the 81-point Experiment 57 disk.  Ridge alpha
is selected from the usual 12-value grid by leaving out each whole Arthur
volume, taking its median normalized 81-point MAE, then minimizing the equal
mean of the three volume medians.  Final fitting gives every source volume
equal total weight.

| Held-out volume | Lenses | Median MAE | Median normalized MAE | p90 normalized MAE |
|---|---:|---:|---:|---:|
| 20231107 | 1,584 | 13.79 µm | 48.08% | 57.23% |
| 20240530 | 1,592 | 3.64 µm | 22.77% | 45.85% |
| 20240701 | 1,699 | 6.36 µm | 47.51% | 87.49% |

These are whole-volume holdouts, not lens-level random folds.  Their failure
cannot be hidden by the much larger number of within-volume facets.

Experiments 60 and 61 retain Experiment 59's legacy surface extraction and
target-QC contract.  This experiment instead uses the rebuilt Experiment 63
q90 source table and the `distal_qc AND target_resolvable` cohort (1,584,
1,592 and 1,699 lenses by volume).  Its errors are therefore not a row-matched
zero-donor subtraction from Experiments 60/61.

## Alternative-morphology scenarios

For sensitivity analysis only, each held-out volume supplies its median
quadratic residual from the outer-only point model.  Adding each of those
three residual morphologies to the same point prediction creates named 2023,
May-2024 and July-2024 scenarios.  The median central-depth shifts are
**+13.57**, **−3.52** and **−5.83 µm**, and the average width between the most
separated scenario surfaces across the canonical grid is **19.71 µm**.

When the scenario derived from a held-out volume is reapplied to that same
volume, median errors fall to 3.19, 1.24 and 1.93 µm.  This is deliberately
post-hoc and circular; it is evidence that a volume-regime offset matters, not
an externally validated correction.  Applying the wrong scenario can instead
produce 17–20 µm median error.

The scenario family is not a confidence interval, a calibrated prediction set
or proof of mathematical non-identifiability.  It is an honest partial-
identification display: several materially different proximal morphologies
must remain in view when the data contain only a distal cap and do not identify
which source regime applies.

## Relation to Experiments 60 and 61

Experiments 60 and 61 retain some same-eye proximal donors.  Their success is
not evidence for this zero-donor setting.  Experiment 62 removes that source of
information and recovers the cross-volume failure seen in Experiment 58 in a
more explicit form.  A taxon-matched prior or physical model could narrow the
scenario set, but that narrowing would be an assumption requiring independent
validation.

The experiment does not identify the candidate *Asaphus* CT boundary,
reconstruct a fossil lens, validate automatic distal localization, or infer
fossil optics.

## Reproduction

After building the hash-bound Arthur source table, run:

```bash
python experiments/arthur-modern-ground-truth/experiment_62_outer_only_partial_identification.py \
  --source-table /path/to/arthur_source_table.csv
```

The diagnostics record the source table's exact filename, byte size, SHA-256,
total row count and primary row counts by volume, so a committed result cannot
silently be reassigned to a different table.

With a test table, the code can export the replacement model under the explicit
method name `experiment62_reconstructed`.  Experiment 63 does not accept that
name as its unchanged historical comparator.  This prevents a new model from
silently inheriting the missing model's identity.

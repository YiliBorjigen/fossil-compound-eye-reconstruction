# Experiment 61 — how many proximal donors must survive?

## Verdict

The same-eye neighbour method degrades smoothly rather than collapsing when
proximal donors become sparse.  Under this reconstructed random-thinning
protocol, retaining **2% of distal-defined lens identities as potential
donors** was the smallest tested level at which all 25 repeats in all six eyes
still had the required six target-QC donors.  At that level, the fixed method's
median eye-level errors were **2.26 µm** (20231107), **0.74 µm** (20240530) and
**0.94 µm** (20240701).

Two percent is an operational result on these densely sampled modern eyes,
not a biological constant or a fossil acceptance threshold.  Donor spatial
arrangement, preservation bias and target quality still matter.

## Recovery status and reconstructed protocol

The original Experiment 61 was lost with the same interrupted, unpushed
worktree as Experiments 60 and 62.  Its title and question survive, but its
exact thinning schedule does not.  The replacement run explicitly freezes:

- potential-donor fractions of 1%, 2%, 5%, 10%, 20% and 40%;
- 25 deterministic SHA-256 rankings per eye and fraction;
- selection from distal graph identities before any proximal field is read;
- at least six selected, target-QC proximal donors for a prediction;
- prediction of every other target-resolvable lens in that eye;
- the Experiment 59 inverse-square six-neighbour depth plus shared-shape rule.

A selected potential donor that lacks a target-QC proximal surface is not
silently replaced.  The run records the resulting usable count and marks a
repeat unavailable if fewer than six remain.

Like Experiment 60, this stress test deliberately retains Experiment 59's
legacy surface extraction and target-QC contract.  Experiment 62 uses the
rebuilt Experiment 63 q90 source table and its `distal_qc AND
target_resolvable` cohort.  The zero-donor and nonzero-donor results are thus
qualitatively related but not a row-matched ablation of one common cohort.

## Reconstructed result

The table reports the median of the two eye summaries after taking the median
across deterministic repeats.  Values are median central-cap axial MAE.

| Potential identities retained | 20231107 | 20240530 | 20240701 |
|---:|---:|---:|---:|
| 1% | 2.77 µm | 0.78 µm | 1.01 µm |
| 2% | 2.26 µm | 0.74 µm | 0.94 µm |
| 5% | 1.84 µm | 0.61 µm | 0.82 µm |
| 10% | 1.62 µm | 0.53 µm | 0.74 µm |
| 20% | 1.36 µm | 0.47 µm | 0.66 µm |
| 40% | 1.18 µm | 0.41 µm | 0.60 µm |

At 1%, each eye has only 8–9 potential donors.  One of 150 eye-repeat cases
retained only five usable target-QC donors and correctly produced no neighbour
prediction.  From 2% upward, every repeat was computable.  At 2%, the worse of
the two eye summaries in each volume had normalized median MAE of 7.94%, 4.85%
and 7.36%, respectively.

The operational 2% marker requires both complete repeat availability and no
eye above 10% median normalized error.  It is declared descriptive in the
machine-readable diagnostics.  The experiment cannot determine a continuous
threshold below the tested grid.

## What this solves—and what it does not

This result widens Experiment 59's usable regime.  The proximal boundary need
not survive next to every missing lens: a small, deterministically thinned set of same-eye
donors can still constrain the repeated morphology in these modern volumes.
Accuracy improves continuously as more donors survive, and the unusually deep
20231107 volume remains the hardest case.

The result does **not** apply at zero donors.  The algorithm needs six donor
surfaces by construction, and no number is reported for an undefined fit.
Experiment 62 therefore treats the zero-donor case as partial identification
with explicit alternative-morphology scenarios, not as a uniquely recovered
surface.  None of these experiments establishes fossil transfer or the
anatomical identity of the candidate *Asaphus* boundary.

## Reproduction

```bash
python experiments/arthur-modern-ground-truth/experiment_61_survival_threshold.py \
  --manifest /path/to/local_manifest.json
```

The diagnostics preserve every source hash, thinning constant, repeat count,
donor inventory and the operational-threshold definition.  The supplied WRL
meshes remain external to the repository.

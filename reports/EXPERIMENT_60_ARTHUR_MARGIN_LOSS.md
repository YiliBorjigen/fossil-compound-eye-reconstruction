# Experiment 60 — proximal loss at the eye margin

## Verdict

Same-eye donors remain useful when the missing region touches the observable
eye margin, but the reconstruction is substantially less accurate and has a
much heavier error tail than Experiment 59's guarded interior patches.  On the
reconstructed radius-two protocol, the fixed six-neighbour method has median
central-cap axial errors of **2.64 µm** (20231107), **1.20 µm** (20240530) and
**1.45 µm** (20240701).  Its corresponding p90 lens errors are **11.71 µm**,
**5.30 µm** and **4.08 µm**.

This supports margin reconstruction as a qualified, uncertainty-sensitive use
case.  It does not justify treating an eye edge like the well-supported
interior, and it does not test loss of the distal caps themselves.

## Recovery status

The original Experiment 60 existed only in an unpushed worktree that was lost
during scratch-workspace maintenance.  Its title and scientific question were
recovered, but its exact historical mask constants and result files were not.
The executable replacement therefore freezes a transparent new specification:

- derive the graph and boundary from retained distal-cap centroids only;
- choose eight deterministic, distributed boundary seeds per eye;
- require at least six graph hops between seeds;
- hide graph-radius one, two and three neighbourhoods around every seed;
- treat radius two as primary, matching Experiment 59's primary radius;
- train donors only on target-QC proximal surfaces outside the hidden patch.

These are reproduced results from that replacement specification, not claimed
recovery of unknown historical numbers.

This experiment deliberately reuses Experiment 59's legacy surface extraction
and target-QC contract so that only the position of the missing patch changes.
Experiment 62 instead consumes the rebuilt Experiment 63 q90 source table and
uses its `distal_qc AND target_resolvable` cohort.  The resulting lens cohorts
are therefore not row-matched, and numerical differences between Experiments
60/61 and 62 must not be attributed to donor availability alone.

## Primary result

The 16 radius-two boundary patches per volume contain 12–20 distal-defined
lenses each.  All masks are selected before proximal availability, quality,
depth or error is inspected.

| Volume | Fixed six-neighbour | Graph harmonic | Nearest visible surface | Same-eye template |
|---|---:|---:|---:|---:|
| 20231107 | 2.64 µm | **2.51 µm** | 2.60 µm | 5.42 µm |
| 20240530 | 1.20 µm | 1.23 µm | **0.92 µm** | 1.91 µm |
| 20240701 | 1.45 µm | 1.33 µm | **0.96 µm** | 2.02 µm |

The fixed method's median normalized errors are 10.82%, 7.70% and 10.35%.
Both eyes in each volume give similar fixed-method medians, but the relevant
biological replication remains three supplied volumes and six nested eyes,
not the hundreds of lens predictions.

For comparison, Experiment 59's radius-two interior errors were 1.32, 0.36
and 0.52 µm.  The margin result is therefore not a repeat of the favourable
interior benchmark.  It documents the expected extrapolation penalty.

## Target-QC sensitivity

The primary table conditions on target-QC hidden surfaces.  When every finite,
resolvable target is retained, including failed-QC surfaces, the fixed method's
radius-two medians are **3.16**, **1.29** and **1.53 µm**.  The 20231107 shift is
especially important: poor margin targets cannot be made harmless by reporting
only a central median.

## Interpretation

The nearest-surface comparator is best by median in the two shallower volumes,
while graph harmonic is modestly best in 20231107.  There is no uniform winner.
The defensible practical conclusion is that preserved proximal donors can
constrain a missing margin, but edge extrapolation needs explicit uncertainty
and method sensitivity.  The result does not establish fossil transfer,
anatomical identity of the *Asaphus* boundary, or a reconstruction when every
proximal donor is absent.

## Reproduction

Run all three supplied mesh pairs with:

```bash
python experiments/arthur-modern-ground-truth/experiment_60_margin_loss_validation.py \
  --manifest /path/to/local_manifest.json
```

The manifest follows Experiment 58's schema.  The supplied meshes remain
external; the committed diagnostics bind their hashes and record every graph,
seed and mask choice.

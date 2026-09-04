# Experiment 63 — pre-outcome visual-QC stop

## Verdict

**Experiment 63 stopped before model-outcome evaluation.** At frozen commit
`0e1479740d37b7f25d9fcb66b33e952340515400`, four of the twelve eyes contained
at least one nonpassing lens in the predetermined 32-per-eye visual-QC sample.
The frozen protocol therefore forbids passing attestations and execution of
the twelve-animal primary backend.

No prediction, prediction error or model comparison was opened. Experiment 63
has no result on whether the four distal shape descriptors outperform the
nested position-and-scale control. Its result is a technical one: the frozen
distal-cap extraction/QC producer was not reliable enough for scoring.

## What completed before the stop

All twelve archives supplied by Maike Kittelmann were converted and checked at
the frozen commit. The fixed denominator contained 11,369 ODA-matched lenses;
11,350 passed automatic distal QC, and 11,345 were target-resolvable. Every
eye retained more than 99% of its fixed denominator for the potential primary
cohort, well above the frozen 80% gate. All twelve target-blind distal-frame
audits passed.

Exactly 32 lenses per eye were then selected by the frozen 4 × 4
radial-position-by-scale design. Eight eyes completed all 32 reviews without a
nonpass. Four eyes stopped at the first failure, as required:

| Eye | First nonpass | Reason |
|---|---:|---|
| `M3_M_26_01` | sample 2, lens 26 | Narrow cyan satellite/tail inconsistent with one coherent cap |
| `M3_M_32_01` | sample 1, lens 621 | Narrow cyan spikes inconsistent with one coherent cap |
| `M3_M_36_01` | sample 1, lens 499 | Multiple narrow cyan prongs inconsistent with one coherent cap |
| `RED3_25_F_36` | sample 23, lens 776 | Shelf-like cyan lobe consistent with a neighbouring-lens merge |

“Four of twelve” records the eye-level stop outcome. It is not an estimated
population failure rate: the lens sample was deterministic and stratified,
and review stopped after each eye's first nonpass.

The exact sample-manifest hashes, failed-render hashes and review stopping
positions are sealed in
[`experiment_63_stop_record.json`](../experiments/maike-modern-ground-truth/results/experiment_63_stop_record.json).
No `instance_qc_attestation.json` was issued for these bundles.

## Predictor-only diagnostic after the stop

A read-only diagnostic used only the four named sealed distal point clouds and
the frozen distal-only geometry. It opened no target, model, prediction or
error. All four full cyan point clouds form one 26-connected voxel component,
so a largest-component rule cannot detect the observed shelves and prongs.

| Eye/lens | Full sealed points | q90 fit support | q90 component sizes |
|---|---:|---:|---|
| `M3_M_26_01/26` | 3,525 | 3,172 | 3,172 |
| `M3_M_32_01/621` | 3,855 | 3,469 | 3,469 |
| `M3_M_36_01/499` | 3,950 | 3,555 | 3,552, 2, 1 |
| `RED3_25_F_36/776` | 5,536 | 4,982 | 4,982 |

This does not rescue Experiment 63. The frozen code estimates each cap origin
from all sealed points, uses those origins to construct the eye frame, and only
then applies the radial q90 fit subset. A peripheral artifact can therefore
alter the origin, frame and q90 scale even if it is later excluded from the
quadratic fit. Connectivity alone also misses attached prongs.

## Next experiment

Any revision must be a separately numbered, prospectively frozen analysis.
It may develop a global robust cap producer from model/error-blind geometry,
but it must treat all 384 Experiment 63 sample identities as development cases,
draw a new disjoint visual-QC sample, and preserve the same no-replacement stop
rule. Only after all twelve new eye-level gates pass may it open outcomes once.

Because the same twelve animals informed upstream technical-QC development,
that amended run would be a post-QC evaluation rather than untouched external
confirmation of the complete preprocessing pipeline. The strongest later
confirmation would apply the frozen revised producer to newly supplied
animals.

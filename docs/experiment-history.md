# Experiment history

The work was exploratory, and the experiment numbering reflects that. I have
kept the useful development trail here without turning the main README into a
43-row audit table.

## Missing centres in a modern lattice (Experiments 1–12)

The first experiments progressively removed one or more known ODA-3D
ommatidial centres, hid the truth from the detector, and measured how well the
positions were recovered. They established strong interpolation performance
inside an intact lattice and, just as importantly, the failure at a torn edge.

Representative scripts for the single-hole, adjacent-pair, unknown-count and
edge-tear tests are included. The full original sequence remains on the
[`feature/missing-lens-reconstruction` branch of ODA](https://github.com/YiliBorjigen/ODA/tree/feature/missing-lens-reconstruction/research/missing_lens).

Experiment 11 has no preserved final report. I do not assign it a result here.

## From centres to surfaces (Experiments 14–24)

This phase asked the harder question: can an inner surface be inferred from the
outer surface of a segmented lens? Simple spheres, ellipsoids and unconstrained
quadrics were underconstrained. Experiment 16 retained the best controlled
outer-only test with uncertainty, but its 28.2% normalised median error was not
strong enough to treat it as a fossil solution. Later tests examined loss,
segmentation and population priors.

Experiment 13 has no preserved final report.

## Moving to fossil volumes (Experiments 34–39)

Experiment 34 was the first *Archegonus* exploration. Experiments 36–39 then
established the *Asaphus* surface-facet field, selected stable centres across
neighbouring thresholds, measured a candidate internal edge, compared facet
and inter-facet profiles, and tested possible anatomical interpretations.

Runs 25–33 and 35 do not have preserved final reports. They are omitted from
the evidential chain rather than reconstructed from memory.

## Auditing the fossil claim (Experiments 40–43)

Experiment 40 introduced the decisive nonlinear spatial-only control.
Experiment 41 froze the method and tested transfer to *Archegonus*. Both
weakened the original reconstruction claim: spatial smoothness explained the
*Asaphus* prediction, and the internal stage failed to transfer.

Experiment 42 searched predefined outer-surface and shallow-shell feature
families; none contributed a stable signal beyond position. Experiment 43
therefore changed the question. It tested whether repeated internal CT evidence
could reveal a shared boundary and assist with a deliberately masked local
region. That narrower route remains scientifically plausible.

The failed and limiting experiments are part of the result, not discarded
preliminary noise.

## Testing a population prior in visible anatomy (Experiment 44)

The fossil audits shifted the question from prediction by external curvature
to reconstruction from repeated homologous units. Experiment 44 tested that
idea in a modern *Apis mellifera* scan, where held-out cone intensity provides
real ground truth. A population template outperformed a depth-matched
background in all 15 spatial blocks across 147 cones. Because every cone came
from one animal, the result remained a within-specimen proof of principle.


## Independent-specimen transfer (Experiment 45)

Experiment 45 applied the same idea to a public *Pieris napi* micro-CT scan.
The crystalline-cone lattice was visible after local surface unfolding, but the
strict Apis pipeline failed at its absolute surface-intensity threshold. A
pre-scoring imaging adapter allowed the reconstruction test to proceed without
using held-out error. Across 62 cones and 15 spatial blocks, the global
population template was worse than depth-matched background and won only four
blocks. Two post-hoc diagnostics did not reverse the result. The positive Apis
result is therefore specimen-specific under the current method.

## Error decomposition and anatomy-first redesign (Experiment 46)

Experiment 46 revisited the failed *Pieris* regions as method development. It
removed the local depth profile, aligned residuals by hexagonal orientation and
facet spacing, and kept spatial holdouts. The feasible model remained worse
than local background. A diagnostic given the true internal centre was better
for 49 of 62 targets, while feasible error increased with centre error. This
suggests that centre and axis registration is the immediate bottleneck, but the
current targets are only two-dimensional CT peaks. The next experiment must
trace three-dimensional cone centre-lines before another transfer claim is
tested.

## Candidate centre-line and curvature audit (Experiment 47)

Experiment 47 checked whether unequal z sampling or a straight-axis assumption
caused the Pieris registration error. The source manifest reports isotropic
1.08 micrometre voxels. Low-order curvature modestly improved blocked-depth
tracking in two regions, but not the third, and the typical curved-versus-
straight deviation was only about 0.6 micrometres. More importantly, very few
raw intensity tracks remained continuous from the shallow lattice to the old
internal target depths. Because the public label contains no individual cone
masks, the tracked ridges cannot be called anatomical cone axes. A trained or
manually validated 3D cone segmentation is now the next required evidence.

## Manual axes and a functional consequence (Experiments 48–51)

Experiment 48 replaced ambiguous intensity ridges with manually traced paths.
Within one region, following a fitted straight path improved the intensity
diagnostic; allowing curvature added little. Experiment 49 repeated the manual
measurement in two more regions. One fixed eye-wide tilt transferred from
Patch 1 to Patch 2 and then failed completely in Patch 3, showing that cone
direction changes across the eye.

Patch 4 was selected from surface geometry and reserved before annotation. A
minimal distance-weighted field was frozen on the first three regional median
directions before Patch 4 was scored. It reduced median held-out path RMSE from
5.02 to 1.44 voxels and improved all 13 usable traces. This is a prospective
regional success within one specimen. It does not replace author-provided cone
labels or an independent-eye test.

Experiment 51 converted the path result into visual-direction consequences.
In prospective Patch 4, the surface normals differed from the manual axes by a
median 15.26 degrees, while the frozen field differed by 3.12 degrees and won
for all 13 paths. Sparse pairwise angular-geometry error also fell from 4.25 to
3.56 degrees. The whole-eye diagnostic remained negative: region-held-out
prediction improved only 17/34 paths and failed throughout Patches 2–3. Thus
internal-axis correction matters locally, but the current field model cannot
yet reconstruct the complete eye.

## Preserved fossil geometry and deformation sensitivity (Experiments 52–53)

Experiment 52 used the preserved *Asaphus* facet field to calculate a geometric
surface-normal envelope, then separated threshold precision from anatomical
uncertainty. The 116 robust facets produced a 34.64-degree maximum sampled span
and 2.31-degree median adjacent-normal separation. Matched normals varied by a
median 0.22 degrees across neighbouring intensity thresholds. However, an
unknown axis departure bounded at 15 degrees permits a maximum span anywhere
from 4.64 to 64.64 degrees. The measurement describes the cropped fossil as
preserved; it is not a reconstructed living field of view.

Experiment 53 tested the other major uncertainty: normal-estimation choices and
taphonomic distortion. Reasonable threshold, smoothing and half-voxel sampling
changes were generally small, although an undersmoothed high-threshold setting
failed the hemisphere check. Transparent 10% affine scaling/shear scenarios
gave 31.38–37.84-degree spans and up to 5.72-degree p95 normal changes; 20%
scenarios gave 28.05–40.97 degrees and up to 11.42 degrees. These are
sensitivity bounds, not a recovered retrodeformation. A unilateral crop without
strain markers cannot identify the original geometry.

## Returning to the missing-surface problem (Experiment 54)

Experiment 54 hid complete candidate internal surfaces rather than local gaps.
The primary test removed contiguous spatial blocks, so neither fitting,
parameter selection nor uncertainty calibration could use internal points from
the target region. Ellipsoid-like and repeat-template models remained near
12.9 micrometres median facet error. Strictly local outer curvature did not
improve them. A fixed six-neighbour depth prior plus a shared quadratic shape
reached 8.10 micrometres and improved on the quadratic surface in all five
blocks. An exploratory leave-one-facet-out version reached 6.59 micrometres.
This provides a practical within-specimen reconstruction of the candidate CT
boundary when neighbouring homologous surfaces survive; it does not show that
outer curvature alone determines an invisible lens.

## Blinded human boundary pilot (Experiment 55)

Experiment 55 froze one observer's labels before revealing which local CT
patches had passed the original edge quality controls. Nineteen of 23 reviewed
accepted facets were called visible, compared with one of five failed-QC
controls (Fisher exact p = 0.0148). In 16 accepted cases with clicks, the median
human–algorithm absolute difference was 6.01 micrometres and the two
perpendicular human views differed by a median 1.86 micrometres.

The observer placed the transition deeper than the gradient extractor in 15 of
16 cases, with a median offset of 5.66 micrometres. This supports the existence
of a visually repeatable CT transition and the utility of the QC procedure, but
also shows that "boundary" needs a stricter operational definition. Because
the observer developed the project and all confidence values remained at the
default minimum, the run is an internal pilot rather than independent
anatomical validation.

## Boundary-definition sensitivity (Experiment 56)

Experiment 56 asked whether the systematic observer–gradient offset invalidates
the Experiment 54 reconstruction comparison. For each held-out spatial block,
the median translation was estimated from the other four blocks. The correction
reduced median case-level disagreement from 5.66 to 2.30 micrometres and helped
14 of 16 cases, although its p90 residual remained 8.89 micrometres.

The observer-centred translation moves the median operational depth from 55.50
to 61.16 micrometres. Applying the same shift to targets and predictions leaves
absolute reconstruction errors unchanged by definition and was verified across
all nine Experiment 54 methods. Thus a uniform landmark choice does not alter
the model ranking, but it does alter absolute depth. The remaining non-uniform
ambiguity is separate from prediction error and cannot be mapped reliably from
16 cases from one informed observer.

## Verified modern proximal surfaces (Experiments 57–58)

Arthur Zhao supplied complete lens and photoreceptor-tip surface meshes for
three *Drosophila* micro-CT volumes. Experiment 57 used the 20240701 volume to
freeze a strict outer-only mask and predictor boundary. Tip coordinates and
complete geometry labelled the hidden truth, but all prediction frames and
features were recomputed from the retained distal surface. In two whole-eye
holdouts, an opposite-eye thickness template and an outer-curvature ridge both
reached about 0.86–0.89 micrometres median error; paired performance showed no
consistent added value from curvature within that volume. An axisymmetric
ellipsoid reached 8.53 micrometres.

Experiment 58 inverted the PCA transform stored in the public eyemap RData
files to recover exact raw-coordinate landmarks, then froze 20240701 as the
only training volume. On external 20240530, the population template reached
2.72 micrometres median error and the outer-curvature ridge 2.49 micrometres.
On the coarser 20231107 mesh, the corresponding errors were 15.18 and 14.96
micrometres because its median target depth was much larger. The ellipsoid
errors were 11.53 and 26.99 micrometres. Thus a population prior can
transfer to a similar scan, while outer curvature contributes only a small
increment and cannot absorb major between-volume depth shifts. A target-depth
ceiling inherited from development initially hid much of that shift; removing
the outcome-dependent cutoff and a later complete-patch support gate raised
20231107 coverage to 1,528/1,632 lenses.

## Same-eye patch reconstruction (Experiment 59)

Experiment 59 tested the remaining practical route: borrow information from
proximal surfaces that survive elsewhere in the same eye. Eight deterministic
interior seeds per eye defined non-overlapping graph neighbourhoods; nested
7-, 19- and 37-lens regions were hidden in both eyes of all three Arthur Zhao
volumes. Graphs and masks used retained distal centroids without consulting
target availability, quality, depth or error.

For the primary 19-lens regions, a prespecified six-neighbour inverse-square
depth rule plus a shared within-lens shape reached median axial errors of 1.32,
0.36 and 0.52 micrometres against quadratic-smoothed central proximal targets
on 20231107, 20240530 and 20240701. It improved on a
same-eye template in all six eye summaries and remained accurate across the
2023 depth shift. A new graph-harmonic comparator was modestly lower in all six
eye summaries and remains method-development evidence. Nominal 90% point-error
bands attained 88.96% pooled marginal coverage, but patch-specific coverage
ranged from 46% to 100%.

This is a positive modern within-eye interpolation result for well-supported
interior loss. It does not solve edge loss, total absence of proximal surfaces,
cross-specimen transfer or fossil anatomical identity. Three volumes and six
eyes—not hundreds of hidden lenses—set the biological evidence scale.

## Reconstructed margin, donor-survival and zero-donor tests (Experiments 60–62)

The original unpushed Experiment 60–62 worktree was lost. The scientific
questions and two orphan Experiment 62 scalars survived, but the exact
historical specifications and results did not. The replacements below freeze
new, transparent protocols and do not claim byte-for-byte historical recovery.

Experiment 60 moved Experiment 59's hidden regions to the distal-defined eye
margin. For graph-radius-two patches, the fixed six-neighbour method reached
2.64, 1.20 and 1.45 micrometres median central-cap MAE on 20231107, 20240530
and 20240701, with corresponding p90 lens errors of 11.71, 5.30 and 4.08
micrometres. Preserved same-eye surfaces still constrain margin loss, but the
larger error tail rules out treating the edge like a guarded interior patch.

Experiment 61 deterministically thinned potential proximal donors without
consulting proximal availability or quality. Two percent was the smallest
tested fraction for which every one of 25 repeats in all six eyes retained the
required six target-QC donors and every eye stayed below 10% median normalized
MAE. Median eye-level errors at that fraction were 2.26, 0.74 and 0.94
micrometres. This is a descriptive threshold for these densely sampled modern
eyes, not a biological constant, and the method remains undefined at zero
donors.

Experiment 62 addressed zero donors by refusing a unique reconstruction. Its
rebuilt, reflection-even outer model uses five distal descriptors and
whole-volume holdout. Median errors were 13.79, 3.64 and 6.36 micrometres. The
replacement alpha selection chose 0.01, the lower boundary of a nearly flat
low-alpha grid; it did not reproduce the recovered orphan alpha
123.28467394420659, whose historical definition remains unknown. Three named
held-out residual morphologies span an average 19.71 micrometres on the
canonical grid. They are sensitivity scenarios, not confidence or
identification intervals.

Experiments 60 and 61 preserve Experiment 59's legacy extraction and target-QC
contract to isolate their missingness interventions. Experiment 62 instead
uses the rebuilt Experiment 63 q90 source table and its `distal_qc AND
target_resolvable` cohort. Their numerical differences are therefore not a
row-matched donor-count ablation. None of these modern-eye tests establishes
fossil transfer, anatomical identity or optical function.

## Twelve-animal validation stopped at technical QC (Experiment 63)

Experiment 63 prospectively froze an Arthur-trained comparison between four
distal-cap shape descriptors and a nested position-and-scale control, with the
twelve independently supplied M3 and RED3 eye stacks as one-eye-per-animal
validation data. At commit `0e1479740d37b7f25d9fcb66b33e952340515400`, all
twelve builds and distal-frame audits completed. Of 11,369 ODA-matched lenses,
11,350 passed automatic distal QC and 11,345 were target-resolvable.

The predetermined 32-per-eye visual gate then found noncoherent cyan distal
caps in `M3_M_26_01/lens_26`, `M3_M_32_01/lens_621`,
`M3_M_36_01/lens_499` and `RED3_25_F_36/lens_776`. The frozen rule made any
failed or indeterminate sample a whole-experiment stop. No passing attestations
were issued, the primary backend was not executed, and no prediction error or
model comparison was inspected. Experiment 63 therefore reports a pre-outcome
technical producer failure—not a negative result for distal shape information.

At that point, the next producer was required to be frozen as a new experiment,
exclude all 384 viewed sample identities from a fresh lens-level QC draw, and
retain the no-replacement whole-run stop. Experiment 64 below implemented that
requirement. Reusing the same twelve animals made it a post-QC evaluation;
untouched confirmation of a complete revised pipeline requires new animals.

## Robust-cap post-QC validation stopped at visual QC (Experiment 64)

Experiment 64 implements that separately numbered revision without reopening
Experiment 63. Development used only predictor geometry and the already viewed
full-component renders; no Maike target table, prediction, error or model
comparison was opened. A global physical-coordinate operator takes the
intersection of geometric-median Euclidean-q90 and trimmed-PCA lateral-q90
sets, then recomputes every Stage 2 feature and frame from the sealed core.
Maike voxel caps additionally require at least 0.99 final-fit 26-connected
support and p99 quadratic residual no greater than 0.75 of robust scale.
Arthur's irregular mesh vertices cannot receive the voxel-connectivity gate,
which remains an explicit representation asymmetry.

All 384 Experiment 63 sample identities were committed as a canonical
development-exclusion ledger. At frozen commit
`7ad346c8bde08c81d49e9c417441d825719c6de8`, all six Arthur technical paths
and all twelve Maike technical bundles completed. The Maike robust core passed
for all 11,369 fixed rows; 11,343 passed the final automatic distal-QC gates.
The only Experiment 64 draw then selected one worst-coherence and one
hash-minimal remaining case in every 4 × 4 radial-position-by-scale cell: 384
new cases with zero overlap with Experiment 63.

The visual gate failed. `RED3_25_M_26` sample ordinal 8, lens 260, retained a
large cyan lateral lobe joined to the putative central cap by a narrow neck in
two projections. A second AI visual assessment returned the same nonpass
classification under the frozen criterion; this was not an independent human
or anatomical-expert review. A second already-rendered case in the same eye
showed the same pattern. Under the frozen rule, no passing attestation was
issued, no outcome-attempt record was created and neither backend pass ran. No
target table, prediction, error or model comparison was opened.

Experiment 64 therefore adds a technical negative result, not a
shape-versus-control result: the tested robust-core operator plus connectivity
and residual-tail gates did not guarantee the prespecified visual-coherence
criterion. At least one automatically eligible sampled core violated that
criterion, making Experiment 64 ineligible for scoring. Concurrent AI
assessment exposed additional cases before the global stop propagated; all 384
new identities are conservatively committed as viewed development cases, but
the unadjudicated flags are not a failure-rate estimate. Any later revision
needs another experiment number, must exclude all 768 Experiment 63 and 64
identities, and should use new animals for untouched confirmation.

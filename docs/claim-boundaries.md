# Claim boundaries

## Supported by the current evidence

- Missing centres inside a surviving modern facet lattice can be reconstructed
  accurately in controlled deletion tests.
- The same centre-finding method does not work at a torn outer edge.
- The analysed *Asaphus* scan contains a repeatable, facet-associated internal
  CT transition. Its onset/strongest-gradient summaries lie around 48–52
  micrometres and the complete Experiment 54 target has a 55.50 micrometre
  median depth.
- Aligning repeated facets retains that signal after facet bootstrapping and
  after leaving out complete spatial blocks.
- One blinded internal observer saw a transition in 19/23 QC-accepted facets
  versus 1/5 failed-QC controls. A translation learned without the test spatial
  block reduced median observer–extractor depth disagreement from 5.66 to 2.30
  micrometres in 14/16 accepted cases.
- A common observer-centred boundary translation changes absolute depth but
  leaves Experiment 54 reconstruction errors and model ranking unchanged. This
  invariance was checked across all nine methods.
- When part of the same boundary remains visible, a guarded repeat template is
  a plausible aid for filling a local gap within this specimen.
- When a complete candidate boundary is hidden, a fixed six-neighbour depth
  prior plus shared quadratic shape reaches 8.10 micrometres median error for
  held-out spatial regions and improves on the quadratic baseline in all five
  blocks. In an exploratory isolated-facet scenario it reaches 6.59
  micrometres.
- In one modern *Apis mellifera* eye, a population template reconstructed
  held-out cone intensity better than depth-matched background in all 15
  spatial blocks and better than an axisymmetric template in 13 of 15 blocks.
- In post-transfer method development on the examined *Pieris* regions, a
  diagnostic residual supplied with the true internal centre beat local
  background for 49 of 62 targets and 10 of 15 spatial blocks.
- In a prospectively selected fourth *Pieris* region, a spatial direction field
  frozen on three earlier manually traced regions reduced median held-out path
  RMSE from 5.02 to 1.44 voxels and improved all 13 usable traces.
- In the same prospective fourth-region test, using surface normals as visual
  directions produced a median angular error of 15.26 degrees; the frozen
  field reduced it to 3.12 degrees and improved all 13 usable traces.
- The 116 robust *Asaphus* facets define a reproducible outer-surface geometric
  envelope: matched normals change by a median 0.22 degrees across neighbouring
  thresholds, and the cropped surface-normal field spans 34.64 degrees.
- Experiment 53 quantifies preservation-geometry sensitivity. At the nominal
  smoothing, threshold changes move the 95th-percentile matched normal by at
  most 0.64 degrees; half-voxel sampling offsets move it by at most 1.06
  degrees. These are preserved-state measurement results.

## Not established

- The CT boundary has not been anatomically identified as the proximal lens
  surface. A preservational or mineral-replacement front has not been excluded.
- The observer-centred 61.16 micrometre median is an operational landmark, not
  a lens-thickness estimate. Its held-out p90 disagreement remains 8.89
  micrometres, and 16 labels from one informed observer cannot resolve a
  spatially varying definition shift.
- External curvature does not yet predict the internal boundary better than a
  strong spatial smoother.
- Strictly local outer curvature does not reconstruct a wholly absent surface:
  it gives 13.09 micrometres median error versus 12.94 micrometres for the
  quadratic surface. The positive complete-surface result borrows depth from
  neighbouring preserved internal boundaries.
- The internal-boundary stage did not transfer under frozen criteria to the
  tested *Archegonus* volume.
- Seventy-four facets from one *Asaphus* scan are repeated observations, not
  74 independent fossils.
- The 147 modern cones are repeated observations within one bee, not 147
  independent biological samples.
- Independent transfer to *Pieris napi* failed: after a pre-scoring imaging
  adapter, the global population template was worse than depth-matched
  background and won only 4 of 15 spatial blocks.
- The deployable anatomy-aware residual remains worse than local background;
  only the oracle-centred diagnostic improved.
- Current internal targets are two-dimensional CT intensity peaks, not verified
  three-dimensional cone centre-lines or homologous anatomical landmarks.
- Candidate ridge tracking in Pieris does not establish cone identity: only 24,
  8 and 0 tracks in the three regions spanned the old shallow-to-internal depth
  interval, and the public label contains no individual cone masks.
- The positive Patch 4 direction result is still one specimen, one region and
  one annotator. It does not validate the clicked paths as crystalline cones or
  establish transfer to another eye.
- A whole-eye visual field has not been reconstructed. Across all four
  region-held-out tests, the current direction field improved only 17/34 paths
  and failed for every usable path in Patches 2–3.
- Sparse local angular span and nearest traced-neighbour measurements are not
  full-eye field of view or adjacent-facet interommatidial angles.
- The 34.64-degree *Asaphus* span and 2.31-degree adjacent-normal separation are
  surface-normal baselines for a cropped field as preserved, not living-eye
  geometry, a validated optical field of view or an interommatidial angle. With
  a 15-degree axis-departure allowance, the maximum span is only bounded between
  4.64 and 64.64 degrees.
- No specimen-specific retrodeformation has been recovered. Under transparent
  hypothetical 10% affine scaling/shear, the largest 95th-percentile normal
  change is 5.72 degrees; under 20% it is 11.42 degrees. These scenarios expose
  sensitivity and are not estimates of geological strain.
- No fossil ray tracing, calcite birefringence, rhabdom geometry, sensitivity
  or acuity model has been implemented. Surface-normal geometry cannot by
  itself establish what the animal could see.
- Apposition and superposition eyes have not been shown to share one
  transferable cone model and will be analysed separately.

## Results that must travel with their caveats

The earlier 6.90 micrometre held-out error looked substantially better than a
12.76 micrometre radial baseline. A nonlinear coordinate-only smoother achieved
6.79 micrometres, so the apparent gain cannot be used as evidence that external
lens geometry was learned.

The repeat-aligned local gap-filling result is also within one specimen. On the
off-centre confirmation mask it improved over a quadratic fit, but did not
consistently beat a flexible local RBF interpolator. It remains a candidate
method, not an independently validated reconstruction.

Experiment 54 is also within one specimen and its target is the candidate CT
boundary, not verified lens anatomy. The six-neighbour prior improves complete
held-out regions, but its 8.10 micrometre median error is 14.6% of median target
depth and its calibrated 90% half-width is 20.31 micrometres. The 6.59
micrometre isolated-loss result is exploratory and less independent than the
spatial-block test. These numbers support an approximate neighbourhood prior,
not exact lens recovery, anatomical identity or independent-fossil transfer.

Experiments 55–56 establish image-domain visibility and quantify one
observer's boundary-definition offset. The 5.66 micrometre median translation
can recalibrate an absolute depth but cannot identify an anatomical interface.
The improved 2.30 micrometre median after spatially held-out calibration must
travel with the 8.89 micrometre p90 residual, the informed-observer limitation
and the unresolved possibility of non-uniform definition shifts.

The modern population-prior result is a controlled reconstruction of visible
cone intensity, not anatomical validation of the fossil boundary. Its positive
scope is one *Apis* eye. On independent *Pieris*, the strict pipeline failed at
a scanner-specific outer threshold and the adapted reconstruction did not beat
the depth-matched background. It cannot support a transferable population-prior
claim.

Experiments 57–58 use verified modern *Drosophila* lens surfaces and directly
test distal-only reconstruction. The 0.86 micrometre 20240701 result comes from
whole-eye development holdouts, not an independent volume. The 20240530
transfer result supports a useful population prior on one similar independent
volume, but 20231107 exposes a large depth-domain shift. After removing an
invalid target-depth ceiling,
1,508/1,632 lenses pass QC in that scan. Outer curvature improves the population
template by only 0.084 and 0.225
micrometres on the two tests. These are modern computational-validation results,
not evidence that a Drosophila prior transfers to trilobites or that the fossil
candidate boundary is a proximal lens surface. Lens-level bootstrap intervals
do not turn two test volumes into biological replication.

The Experiment 46 oracle-centred result is a mechanistic diagnostic on the same
*Pieris* regions used after the transfer failed. It supports centre/axis
registration as a working hypothesis for the error source; it is not an
independent replication, a deployable method, or evidence about fossil cone
identity.

The Experiment 47 curvature result also remains a candidate-ridge audit. A
quadratic path modestly improved blocked-depth prediction in two regions but not
the third, and its typical deviation from a straight path was about 0.6
micrometres. This supports permitting gentle curvature in a future segmenter;
it does not show that the tracked ridges are crystalline-cone axes or explain
the earlier approximately 3.2 micrometre registration error.

The Experiment 50 field was frozen before Patch 4 scoring, so its improvement
is not an in-region tuning result. Its scope is nevertheless local: Patch 4 is
near the two training regions with similar directions, while training-only
leave-one-region-out prediction was poor for the distant sign-changing region.
The result supports local interpolation of manual directions, not a solved
whole-eye field, automatic segmentation or fossil reconstruction.

Experiment 51 gives that local result a functional scale: surface normals were
wrong by a median 15.26 degrees in prospective Patch 4, reduced to 3.12 degrees
by the frozen field. This is evidence that internal-axis orientation can matter
to a local viewing-direction estimate. It must travel with the negative
whole-eye diagnostic (17/34 improvements, including complete failures in
Patches 2–3) and with the fact that the manual paths lack independent
anatomical validation.

Experiment 52 deliberately does not copy the *Pieris* tilt into *Asaphus*. Its
bounded sensitivity calculation says instead: if the unknown fossil axes may
depart from the measured normals by beta degrees, the maximum axis span may
change by as much as 2 beta. Threshold stability therefore establishes precise
preserved outer geometry, not precise biological sight direction.

Experiment 53 addresses the separate taphonomic objection. The nominal normal
field is reasonably stable to threshold, smoothing and half-voxel sampling,
but hypothetical 10% and 20% affine strain can change normals by more than the
2.31-degree median separation between neighbours. The available unilateral
crop contains no independent strain constraint, so those transforms cannot be
chosen from the outcome or called a reconstruction of the living eye.

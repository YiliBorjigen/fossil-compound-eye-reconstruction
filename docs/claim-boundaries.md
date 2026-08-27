# Claim boundaries

## Supported by the current evidence

- Missing centres inside a surviving modern facet lattice can be reconstructed
  accurately in controlled deletion tests.
- The same centre-finding method does not work at a torn outer edge.
- The analysed *Asaphus* scan contains a repeatable, facet-associated internal
  CT boundary around 48–52 micrometres beneath the surface.
- Aligning repeated facets retains that signal after facet bootstrapping and
  after leaving out complete spatial blocks.
- When part of the same boundary remains visible, a guarded repeat template is
  a plausible aid for filling a local gap within this specimen.
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

## Not established

- The CT boundary has not been anatomically identified as the proximal lens
  surface. A preservational or mineral-replacement front has not been excluded.
- External curvature does not yet predict the internal boundary better than a
  strong spatial smoother.
- The method does not reconstruct a wholly absent inner lens surface.
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
  surface-normal baselines for a cropped field, not validated optical field of
  view or interommatidial angle. With a 15-degree axis-departure allowance, the
  maximum span is only bounded between 4.64 and 64.64 degrees.
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

The modern population-prior result is a controlled reconstruction of visible
cone intensity, not anatomical validation of the fossil boundary. Its positive
scope is one *Apis* eye. On independent *Pieris*, the strict pipeline failed at
a scanner-specific outer threshold and the adapted reconstruction did not beat
the depth-matched background. It cannot support a transferable population-prior
claim.

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
depart from the measured normals by beta degrees, the maximum field span may
change by as much as 2 beta. Threshold stability therefore establishes precise
outer geometry, not precise biological sight direction.

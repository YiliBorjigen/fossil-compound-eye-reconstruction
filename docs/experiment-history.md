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

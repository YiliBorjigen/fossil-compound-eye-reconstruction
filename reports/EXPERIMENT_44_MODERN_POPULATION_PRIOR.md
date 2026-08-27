# Experiment 44: a population prior in a modern compound eye

## Why this experiment was run

The fossil work established an important limit: external curvature alone did
not predict a missing internal surface better than spatial smoothing. The next
question was therefore not whether another regression model could rescue that
claim. It was whether repeated optical units could provide the missing
information directly.

A modern *Apis mellifera* micro-CT volume provides a controlled setting. The
outer facet lattice and the internal cone signal are both visible, so a cone
can be hidden and reconstructed without guessing what the ground truth was.

## Blind test

The corneal surface was locally unfolded in three non-overlapping patches.
Outer facet centres were detected at the surface and matched to internal cone
centres only to establish ground truth. Each patch was divided into five
contiguous spatial blocks. For every held-out block, the mapping from outer
facets to cones and every reconstruction template were learned from the other
four blocks. The held-out cone intensity was not used to position or build its
prediction.

The primary control was a depth-matched background profile. Additional
controls used an axisymmetric version of the population template, the nearest
training cone, and a local six-cone population.

## What happened

Across 147 cones and 15 held-out blocks, the global population template reduced
median normalised MAE from 0.228 to 0.171, approximately 25%. Median correlation
rose from -0.096 for the background control to 0.624. The global template beat
the background in all 15 blocks and the axisymmetric template in 13.

This is the first positive blind reconstruction result in the project that
uses intact CT intensity as ground truth. It supports a population-prior
strategy: use the surviving outer lattice to locate a missing optical unit,
then infer its likely internal signal from homologous units elsewhere in the
same eye.

## What it does not show

All 147 cones come from one bee. They are spatial replicates, not independent
specimens. The published labels mark anatomical layers rather than individual
cones, and the experiment concerns modern tissue with a strong CT signal. It
does not identify the candidate *Asaphus* boundary anatomically, and it does
not prove that a completely absent fossil lens surface can be reconstructed
from its exterior.

The next test should use the independent *Pieris napi* scan without changing
the evaluation rules in response to its outcome. If transfer succeeds, the
population prior becomes a credible cross-specimen method. If it fails, the
current result remains useful as a within-eye reconstruction result and a
clear boundary on the fossil claim.

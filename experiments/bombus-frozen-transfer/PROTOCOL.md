# Frozen transfer to *Bombus terrestris*

## Status

This is a prospective protocol, not a result. The *Bombus* volume must not be
used to choose segmentation settings or success criteria. Numerical gates will
be filled in and committed after the *Pieris* manual-label stage, before the
held-out *Bombus* outcome is examined.

## Question

Can a three-dimensional cone representation developed on *Apis mellifera* and
*Pieris napi* recover individual cone axes in a new apposition eye, and does an
axis-normalised residual improve reconstruction beyond a local depth-only
background?

The experiment tests transfer between animals and species. It does not test
transfer to superposition eyes or identify fossil structures.

## Intended dataset

- taxon: *Bombus terrestris*;
- specimen: `LU:4_16_: BT_F_CE_10` as reported by Tichit et al. (2022);
- source acquisition: 1.6 micrometre isotropic micro-CT;
- analysis grid: isotropic resampling recorded in the manifest; the published
  study used 4 micrometres for InSegtCone segmentation;
- source and licence: record the exact MorphoSource or Dryad media identifier,
  download terms, file hash and voxel metadata before running.

## What must be frozen first

1. Intensity normalisation and isotropic resampling.
2. Corneal-surface registration and local unfolding.
3. Foreground/background feature construction.
4. Individual-cone segmentation threshold and connected-component rules.
5. Centre-line extraction, including the permitted curvature model.
6. Quality gates for length, radius, continuity and surface-to-cone matching.
7. Spatial blocks, metrics and numerical success thresholds.

Only scanner-facing operations such as file orientation, voxel calibration and
the published corneal mask may be specimen-specific. Any such change must be
recorded before scoring and may not depend on reconstruction accuracy.

## Manual reference labels

Mark a spatially separated reference set without seeing algorithmic scores.
Each selected cone should have centre-line control points at proximal, middle
and distal depths, with extra points where the axis bends. A second pass should
flag ambiguous cones rather than forcing a label. If a second annotator is
available, report inter-annotator centre-line distance and angular difference.

The manual set is divided into tuning-free validation blocks. It must never be
used to retrain the model after the Bombus prediction has been inspected.

## Primary comparisons

| Task | Model | Comparator |
|---|---|---|
| Cone segmentation | Frozen 3D segmenter | Published/manual cone mask where available |
| Axis recovery | Curved 3D centre-line | Straight surface normal |
| Reconstruction | Axis-normalised population residual | Local depth-only background |
| Generalisation | All frozen stages | Radial or spatial-only interpolation |

## Metrics

- segmentation Dice and intersection-over-union on held-out manual cones;
- centre-line distance in micrometres, reported along depth and per cone;
- angular error in degrees and continuity/curvature failure rate;
- cone detection precision and recall after one-to-one matching;
- reconstruction median absolute error and normalised MAE;
- improvement over the local depth-only comparator by spatial block.

Report individual blocks and cones as well as pooled summaries. Multiple cones
from one eye are repeated anatomical observations, not independent animals.

## Decision rule

The frozen method transfers only if it passes the precommitted segmentation and
axis gates and improves reconstruction over the local depth-only comparator in
a majority of spatial blocks. Failure is retained as a result. Post-hoc tuning
may be used to diagnose the failure but must be labelled exploratory and cannot
convert this experiment into independent validation.

## Provenance and credit

Preserve source citations, licences, identifiers and hashes. Contributors who
provide unpublished labels, trained dictionaries or substantive anatomical
guidance will be credited according to their actual contribution; no person is
listed before that contribution is received and confirmed.

## Sources

- Tichit P, Zhou T, Lihoreau M, et al. (2022). InSegtCone: interactive
  segmentation of crystalline cones in compound eyes. *BMC Zoology*.
  <https://doi.org/10.1186/s40850-021-00101-w>
- Taylor GJ, Tichit P, Schmidt MD, Bodey AJ, Rau C, Baird E. (2019). Bumblebee
  visual allometry results. Dryad dataset. <https://doi.org/10.5061/dryad.23rj4pm>

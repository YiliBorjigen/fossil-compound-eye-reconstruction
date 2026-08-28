# Boundary annotation and independent-fossil validation

This protocol addresses the two unresolved questions separately:

1. Does the Asaphus algorithm follow a boundary that a human observer can see
   in the CT data?
2. Does the frozen reconstruction procedure work in a different fossil?

Neither test, on its own, proves that the boundary is the proximal lens
surface. Anatomical identity is a distinct interpretation problem.

## Part A — blinded Asaphus boundary annotation

### Cases

Use the pack produced by
[`apps/asaphus-boundary-annotator/prepare_boundary_pack.py`](../apps/asaphus-boundary-annotator/prepare_boundary_pack.py).
Its default set contains 30 anonymous cases:

- five accepted facets from each of the five existing spatial blocks;
- five facets that failed the internal-edge quality gate.

Selection is deterministic and does not use reconstruction error. The pack
contains raw, surface-aligned local CT intensities. Original facet IDs, spatial
location, quality status and algorithm-derived depths are held in a separate
private key.

### Observers

Obtain at least two independent observers. Because the published anatomical
interpretation is disputed, the most informative panel would include one
observer familiar with the crystalline-cone interpretation and one familiar
with the lens-prism or dissolution interpretation. Observers should not discuss
individual cases until their labels have been frozen.

### What each observer records

For every case, the observer should:

- trace any boundary that is actually visible in the two orthogonal CT
  cross-sections;
- choose visible, uncertain or not visible;
- classify the most plausible structure as a proximal lens/prism boundary,
  cone-like structure, dissolution or mineral-replacement front, other or
  uncertain, or no visible boundary;
- give confidence from 1 to 5 and an optional note.

The algorithm prediction must remain hidden. “No visible boundary” is a valid
and important observation.

### Frozen analysis

After all annotation files are received, freeze them and then unblind once.
Run `score_annotations.py` to compare each clicked human depth with the nearest
point on the algorithm-extracted candidate boundary.

Report:

- the number of visible, uncertain and not-visible cases;
- median and 90th-percentile absolute depth difference in micrometres;
- results separately for accepted facets and failed-QC controls;
- agreement between observers on visibility and structural interpretation;
- between-observer depth difference for cases both observers marked visible.

The primary extraction-validity result is the human–algorithm depth difference
among accepted facets. The controls test whether the QC gate enriches for a
human-visible boundary. Structural classifications must be reported as expert
interpretations, not converted into ground truth by majority vote.

## Part B — frozen test in another fossil

### Specimen inclusion, decided before looking at prediction performance

The first decisive transfer specimen should be a different fossil individual,
preferably another asaphid or comparable holochroal trilobite eye, with:

- at least 30 facets in one contiguous field;
- repeated internal cuticular boundaries visible across several facets;
- calibrated isotropic voxels, preferably 2–4 µm;
- raw reconstruction data and enough metadata to identify orientation,
  specimen, scan and voxel spacing;
- no overlap with the Asaphus volume used for method development.

A second crop or file conversion of the same individual is not an independent
biological replicate. The two public Archegonus stacks are useful technical
material, but the already completed frozen test was negative and they should
not be relabelled as successful validation after retuning.

### What may change

Before inspecting internal-boundary prediction accuracy, record the specimen's
orientation, eye crop, voxel calibration, surface isovalue and neighbouring
surface thresholds. These are imaging and localisation parameters. Confirm
that the outer facet lattice is detected without using the internal result.

If voxel spacing is anisotropic, resample once to a declared isotropic spacing
before local 3D measurements. Preserve the original volume and record the
interpolation method.

### What remains frozen

The confirmatory reconstruction uses:

- six nearest preserved facets;
- inverse-square distance weights for depth;
- the shared within-facet quadratic shape from the training data;
- complete held-out facets in contiguous spatial blocks;
- uncertainty calibration from training blocks only.

No internal-depth window, quality threshold, neighbour count or model setting
may be tuned on the held-out result. Any later specimen-specific adaptation is
reported as exploratory and evaluated on a further untouched specimen.

### Comparators and outcomes

Compare the six-neighbour method with a flat-depth baseline, axisymmetric
ellipsoid, general quadratic surface, local outer-curvature model and
position-only smoother. Report median facet MAE, 90th-percentile facet MAE,
error relative to median boundary depth and empirical coverage of the nominal
90% interval.

Success requires both of the following:

1. the boundary passes the prespecified visibility and facet-association checks;
2. the frozen six-neighbour method improves on the geometric baselines without
   depending on specimen-wide spatial interpolation.

If the boundary is absent, anatomically different or fails the quality gate,
the result is a failed transfer. It should remain in the record.

## Interpretation boundary

Passing Parts A and B would support a reproducible method for reconstructing a
repeated CT boundary from neighbouring preserved homologues. It would still not
recover soft tissue, undo fossil deformation, identify the living optical axis
or justify field-of-view, acuity, sensitivity or ray-tracing claims. Those
questions require independent anatomical evidence and a specimen-appropriate
optical model.

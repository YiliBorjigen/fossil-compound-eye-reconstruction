# Frozen model check on the ten remaining eyes — 9 September 2026

Question: does the M26 model's improvement over its frozen training template
extend beyond the earlier M32 test eye?

Use the original frozen model (SHA-256
`be753b7367db4a4aa8f08bdd739245ea6d8e85d4967404a8c99dad8e3e694cb2`).
No fitting, feature rescaling, hyperparameter changes or target-dependent
crop adjustments are allowed. The earlier M32 result informed the decision
to perform this follow-up; this is not a new development dataset.

Test the four unused M3 eyes and all six RED3 eyes listed in
`transfer_remaining.py`. Byte-identical duplicate archives count once.
M3 is D. simulans; RED3 is D. mauritiana. Report same-species and cross-species
results separately, with individual eyes as the units of replication.

## Outer-only crop selection

The existing predictor operates on the downward-facing (+y) envelope, so this
is a test of a central cap per eye, not complete-eye coverage. In a sparse
preview, sample every 16th z slice and every fourth x column. Retain only the
last occupied y coordinate and occupancy, never the first coordinate or lens
thickness. Smooth this outer map with a normalized Gaussian of sigma (2, 8)
in preview pixels. Choose the greatest smoothed y where distance from missing
outer support exceeds 50 native voxels. This chooses a +y-facing cap without
looking at reconstruction errors.

Centre a 450-column x crop there, clamped to the image boundaries. Clamp the
z centre to leave 160 slices on either side; use seeds within centre ±140.
Within centre ±170 z and the x crop, place the y crop's upper bound 33 voxels
beyond the greatest sampled outer y, clamped to the image bounds, then retain
300 y rows. Write this crop to disk before target extraction.

Use the pilot's unchanged outer peak detection, radii, central disk, outer
features, ellipsoid continuation and target-support criteria. Local x seed
bounds remain 35–415. The automatic crop rule is a new deployment step; the
original two eyes used manually located crops. There is no anatomical
registration between eyes, and orientation/region differences may affect
transfer. Report localization failures explicitly; do not retry another crop.

## Evaluation and reporting

For every candidate, compute predictions with all inner/validity fields
removed and check equality to predictions with the full record. Scoring uses
the withheld binary boundary at corresponding grid positions. Use native
isotropic spacing 0.325 µm, already linked to the authors' Fig. 3 code.

For every eye, report total candidates, scorable candidates, median and 90th
percentile patch MAE for each method, paired template-minus-geometry errors
and how many patches favour geometry. Keep all failures in the ten-eye
denominator. Summaries across eyes must not treat facets as independent
animals. Report whether each eye favours geometry; do not select only wins
or tune a replacement model after seeing results.

This check evaluates transfer against supplied mask boundaries. It cannot
validate segmentation accuracy, complete lens reconstruction, fossil anatomy
or optical function. Preserve all previous code and results.

```bash
python experiments/maike-binary-pilot/transfer_remaining.py /path/to/native-zips \
  --output /path/to/new-remaining-eye-results
```

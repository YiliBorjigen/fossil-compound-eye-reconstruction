# Frozen transfer to the remaining eyes — 9 September 2026

The original M26 model improves on its frozen template in **four of six
additional scorable eyes**, with worse median error in two. Of the ten
planned eyes, two further eyes failed the fixed deployment rules and two
current source ZIPs were unreadable. The earlier M32 test is excluded here.
This is a mixed transfer result, not general validation of outer-only fossil
reconstruction. No model was retrained or selected using these targets.

| Eye | Species | Scorable / candidates | Ellipsoid MAE, µm | Template MAE, µm | Geometry MAE, µm | Outcome |
|---|---|---:|---:|---:|---:|---|
| M3_M_36_01 | D. simulans | 40 / 47 | 4.783 | 1.619 | 1.408 | Geometry lower |
| M3_F_24_01 | D. simulans | 33 / 33 | 4.702 | 0.948 | 1.363 | Template lower |
| M3_F_28_03 | D. simulans | — | — | — | — | File is not a zip file |
| M3_F_35_03 | D. simulans | 7 / 33 | 9.714 | 2.589 | 1.378 | Geometry lower |
| RED3_25_M_26 | D. mauritiana | — | — | — | — | File is not a zip file |
| RED3_25_M_27 | D. mauritiana | 45 / 46 | 4.721 | 1.611 | 0.846 | Geometry lower |
| RED3_25_M_28 | D. mauritiana | — | — | — | — | Insufficient outer-defined candidates |
| RED3_25_F_36 | D. mauritiana | 28 / 28 | 7.110 | 1.748 | 1.815 | Template lower |
| RED3_25_F_37 | D. mauritiana | 42 / 42 | 3.395 | 1.362 | 1.269 | Geometry lower |
| RED3_25_F_38 | D. mauritiana | — | — | — | — | Selected candidate lacks retained outer support |

MAE entries are the median of patch mean absolute axial errors. The six
scorable eyes contribute 195 scorable patches from 229 candidates. Those
counts do not include candidates in the two failed deployments, and must not
be used to imply coverage of all eight accessible eyes. M3_F_35_03 has only
7 of 33 candidates scorable; its apparent gain has especially limited coverage.
Facets within an eye are repeated observations, not independent animals.

## What the result establishes

- Same-species transfer: geometry has lower median error in two of the three
  additional scorable M3 eyes; one M3 source ZIP is unreadable.
- Cross-species transfer: geometry has lower median error in two of the three
  scorable RED3 eyes. Two other RED3 deployments fail, and one ZIP is unreadable.
- Every scorable eye favours both learned methods over the specified ellipsoid
  continuation. The added value of local geometry over the template is inconsistent.
- The outcome supports limited transfer of a learned shape prior. It does not
  justify assuming that the geometry model is always the better choice.

## Limits that affect interpretation

These are automatically selected +y-facing central caps. The crop rule was
fixed from outer observations before target extraction and is new relative
to the original manually located M26/M32 crops. There is no anatomical
registration between eyes. Performance differences cannot therefore be assigned
solely to species or sex; orientation, region, segmentation and acquisition
differences may contribute. No alternative crops were tried after failures.
RED3_25_M_28 lacks sufficient outer-defined candidates under the fixed rule.
RED3_25_F_38 has a selected candidate without retained outer support, so that
entire deployment is recorded as failed rather than silently dropping it.

The unconstrained geometry model also predicts nonpositive thickness at some
grid points in M3_F_24_01 and RED3_25_F_36. The largest within-patch fractions
are 1.23% and 4.94%, respectively. Predictions were not clipped or repaired.
This reinforces that it is not yet a reliable complete-lens construction tool.
Binary-mask accuracy against greyscale CT, complete rims, fossil anatomical
identity, deformation and optical function remain unvalidated.

## Reproducibility and provenance

The [protocol](../REMAINING_EYES_PROTOCOL.md) and initial script were committed
before the run at
[`d49ca61`](https://github.com/YiliBorjigen/fossil-compound-eye-reconstruction/commit/d49ca61806d1b22a0bacdef7e51d7cd4fad7f61f).
Two additive execution fixes followed: convert a NumPy count to an ordinary
integer for JSON output, and record unreadable ZIPs without stopping the batch.
These fixes changed no crop rules, features, predictions or scoring. The
already computed M36 and F24 per-facet files match the final run byte for byte.
The final script hash is recorded in [manifest.json](manifest.json).

The original frozen model SHA-256 remains
`be753b7367db4a4aa8f08bdd739245ea6d8e85d4967404a8c99dad8e3e694cb2`.
For every scored candidate, removing inner-surface and target-validity fields
leaves predictions unchanged. Per-eye candidate counts were checked against
the saved per-facet rows. Native calibration remains 0.325 µm isotropic.
The source data were supplied by Maike Kittelmann and are associated with
[Buffry et al. (2024)](https://doi.org/10.1186/s12915-024-01864-7).

[eye_summary.csv](eye_summary.csv) contains per-eye medians and 90th percentiles.
Individual specimen folders contain per-facet errors and the outer-defined crop.
The manifest keeps all ten planned eyes, source hashes and failure reasons.
No input images are redistributed.

## Source files needed to complete the planned checks

The current scratch copies of the following archives are unreadable by the
ZIP parser and do not have a valid ZIP directory. They were not repaired or
recovered. Earlier provenance checks refer to the earlier accessible copies;
the current bytes must not be treated as identical to those original files.

- `tiffs_M3_F_28_03_eye_lenses-20260903T135112Z-1-001(2).zip`
  (current copy: 1,325,568 bytes).
- `tiffs_RED3_25_M_26_eye_lenses-20260903T135055Z-1-001(1).zip`
  (current copy: 5,661,184 bytes).

Re-uploading these two source archives is required to complete their checks.
The four scored improvements, two scored losses and two deployment failures
already establish that the original 46.4% gain is not uniformly transferable.

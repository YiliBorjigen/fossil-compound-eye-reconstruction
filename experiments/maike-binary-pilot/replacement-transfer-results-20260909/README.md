# Replacement checks completed — 9 September 2026

Both replacement archives are readable and have exactly the SHA-256 hashes
recorded in the earlier calibration evidence. The original M26 predictor and
the fixed automatic crop, feature, prediction and scoring rules are unchanged.
Only the two previously blocked eyes were processed; earlier eyes were not rerun.

Both new checks favour geometry over the frozen template on median patch MAE:

| Eye | Scorable / candidates | Ellipsoid MAE, µm | Template MAE, µm | Geometry MAE, µm | Reduction from template |
|---|---:|---:|---:|---:|---:|
| M3_F_28_03 | 15 / 42 | 9.889 | 1.489 | 1.375 | 7.6% |
| RED3_25_M_26 | 21 / 43 | 6.495 | 1.719 | 1.192 | 30.6% |

Entries are medians of per-patch mean absolute axial errors against the
supplied binary-mask boundaries. The geometry model has lower error in 9 of
15 scorable M3 patches and all 21 scorable RED3 patches. Median paired
template-minus-geometry improvements are 0.129 and 0.577 µm, respectively.
No nonpositive geometry-predicted thickness was found among either eye's
candidates. Predictions were not clipped or repaired.

## Completed ten-eye follow-up

The frozen model now improves on the template in **six of eight scorable
additional eyes**, with worse median error in two. Two other eyes still fail
the fixed deployment rules. There are no remaining unreadable source archives
in this ten-eye follow-up. The earlier M32 result is separate from this table.

| Eye | Scorable / candidates | Template MAE, µm | Geometry MAE, µm | Outcome |
|---|---:|---:|---:|---|
| M3_M_36_01 | 40 / 47 | 1.619 | 1.408 | Geometry lower |
| M3_F_24_01 | 33 / 33 | 0.948 | 1.363 | Template lower |
| M3_F_28_03 | 15 / 42 | 1.489 | 1.375 | Geometry lower |
| M3_F_35_03 | 7 / 33 | 2.589 | 1.378 | Geometry lower |
| RED3_25_M_26 | 21 / 43 | 1.719 | 1.192 | Geometry lower |
| RED3_25_M_27 | 45 / 46 | 1.611 | 0.846 | Geometry lower |
| RED3_25_M_28 | — | — | — | Insufficient outer-defined candidates |
| RED3_25_F_36 | 28 / 28 | 1.748 | 1.815 | Template lower |
| RED3_25_F_37 | 42 / 42 | 1.362 | 1.269 | Geometry lower |
| RED3_25_F_38 | — | — | — | Selected candidate lacks retained outer support |

Geometry improves in three of four scorable additional M3 eyes (D. simulans)
and three of four scorable RED3 eyes (D. mauritiana). Both deployment failures
are RED3 eyes. Thus six of the ten planned eyes yield a scored improvement;
the six-of-eight fraction must not hide the two deployment failures.
Both learned methods outperform the specified ellipsoid in all eight scorable
eyes. This strengthens evidence for limited transfer, including between these
species, while preserving the two losses and the failed deployments.

Coverage is important: the new checks score only 15/42 and 21/43 candidates.
Across the eight scorable eyes, 231/314 candidates are scorable; this count
does not include candidates in the two failed deployments. Patches within
one eye are not independent animals. These central-cap results do not validate
complete lens rims, segmentation accuracy against greyscale CT, fossil anatomy,
deformation correction or optical function. The original training eye and
automatic crop/orientation limitations remain unchanged. See the
[original follow-up report](../remaining-eye-results-20260909/README.md).

## Provenance and checks

- [replacement_receipt.json](replacement_receipt.json) records the two supplied
  filenames, sizes and hashes, matching the earlier
  [calibration evidence](../fig3_calibration_evidence.json).
- [manifest.json](manifest.json) records only the two replacement runs.
- [combined_manifest.json](combined_manifest.json) combines their outcomes
  with the earlier eight entries, which were checked to be unchanged. The
  earlier unreadable-input entries remain in the original report and manifest.
- [combined_eye_summary.csv](combined_eye_summary.csv) contains all eight
  scorable eyes, including median and 90th-percentile errors. Each new eye's
  folder holds per-facet errors and its crop fixed from the outer surface.
- Saved medians, 90th percentiles and candidate counts were checked against
  the new per-facet files. Predictions are invariant to removing all inner
  and target-validity fields. The frozen model and original transfer-script
  hashes are unchanged. Native spacing remains 0.325 µm isotropic.

Data were supplied by Maike Kittelmann and are associated with
[Buffry et al. (2024)](https://doi.org/10.1186/s12915-024-01864-7).
No input images are redistributed. The historical truncated copies were left
untouched; explicit local links selected the replacements without recovery.

Reproduce these two checks using the replacement archives:

```bash
python experiments/maike-binary-pilot/complete_replacement_checks.py \
  /path/to/tiffs_M3_F_28_03_eye_lenses.zip /path/to/tiffs_RED3_25_M_26_eye_lenses.zip \
  --output /path/to/new-replacement-results
```

Use the supplied filenames beginning with `tiffs_` and containing the specimen
IDs; the wrapper checks the exact calibrated source hashes before execution.

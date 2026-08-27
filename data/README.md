# Data not stored in Git

The raw CT volumes and large intermediate tables are intentionally excluded.
This keeps the repository readable and avoids presenting derived files as if
they were primary data.

## Fossil volume used for the Asaphus analysis

- expected file: `1652a_0000_1_cropped_CORRECT_3p7um.nrrd`
- array size: `516 × 552 × 524`, unsigned 8-bit
- voxel spacing: 3.7 micrometres isotropic
- SHA-256: `d1ae2067f8439ffd6ac50faa2e01c0837eeed55944ce949eaf7c77bea675b668`

The original TIFF stack and corrected NRRD contain the same scan after
reorientation and a small edge crop; they are not independent specimens.

## Inputs inherited by Experiments 42 and 43

- `reconstruction_samples.csv`
  - 9,840 rows from 74 quality-controlled facets
  - SHA-256: `76a42873098f3f0b850ba07bb7c4762eb1974f91ef30fd0b6d8945d37e95d3ff`
- `robust_facet_centers.csv`
  - 116 threshold-stable facet centres
  - SHA-256: `5bf12c29c1f76410cafde4eef4cf5e66f639ab9439d1a221d5bb24bdb74ac0cc`

These intermediate files are not committed because they are generated analysis
products. Their hashes are recorded so that a restored copy can be checked
before use.

## Modern ODA input

The synthetic-centre scripts expect the ODA-3D output
`tests/d_mauritiana_ct_stack/ommatidial_data.csv`. Obtain or generate it using
the [ODA repository](https://github.com/jpcurrea/ODA); do not interpret it as
fossil anatomical ground truth.

The outer-only modern experiment expects `labeled_lens_points.npz`, a derived
segmented-lens dataset that is not included here. Its absence is stated rather
than silently replacing it with another file.

## Public fossil source

The source deposit is available at
[HU Berlin, DOI 10.18452/20002](https://doi.org/10.18452/20002). Confirm the
specimen identity and voxel calibration before applying any frozen parameters.

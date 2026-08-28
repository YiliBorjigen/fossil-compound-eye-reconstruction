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

## Inputs inherited by Experiments 42–54

- `reconstruction_samples.csv`
  - 9,840 rows from 74 quality-controlled facets
  - current pipeline SHA-256: `75be2fe2f5ac724da96439169a42cfc8fa58351cf1cf7523b5756198f7ad0dc3`
  - legacy Experiment 40 export SHA-256: `76a42873098f3f0b850ba07bb7c4762eb1974f91ef30fd0b6d8945d37e95d3ff`
- `robust_facet_centers.csv`
  - 116 threshold-stable facet centres
  - SHA-256: `5bf12c29c1f76410cafde4eef4cf5e66f639ab9439d1a221d5bb24bdb74ac0cc`

These intermediate files are not committed because they are generated analysis
products. Their hashes are recorded so that a restored copy can be checked
before use. The current exporter adds the explicit spatial-block column; it
reproduces the archived Experiment 43 full-facet metrics exactly.

Experiment 54 consumes `experiment_42_feature_table.csv`, generated from these
two files and the raw NRRD by the documented Experiment 42 script. The table
used for the reported run has 9,840 rows and SHA-256
`e0b673faa0774238c0f1cdb38afd1c9ea143ee81b1060dca71ca375359856373`.

## Modern ODA input

The synthetic-centre scripts expect the ODA-3D output
`tests/d_mauritiana_ct_stack/ommatidial_data.csv`. Obtain or generate it using
the [ODA repository](https://github.com/jpcurrea/ODA); do not interpret it as
fossil anatomical ground truth.

The outer-only modern experiment expects `labeled_lens_points.npz`, a derived
segmented-lens dataset that is not included here. Its absence is stated rather
than silently replacing it with another file.

## Modern micro-CT population-prior input

Experiment 44 uses the public *Apis mellifera* scan from InSegtCone:

- MorphoSource media: `000396182`
- specimen: `LU:3_14:AM_F_5`
- raw archive used: `60185_AM_F.zip`
- published label: `60185_AM_F_manLabel.nii`
- CT dimensions: `1694 × 916 × 551`, unsigned 8-bit
- acquisition voxel size reported by Tichit et al. (2022): 1.6 micrometres

The NIfTI label header stores unit voxel dimensions, so physical calibration is
taken from the paper and primary measurements remain in voxels. The input label marks
the external cornea and full eye volume, not individual cones.
Individual centres and reconstruction targets are derived from the untouched
CT signal during the controlled test.

Experiment 45 used the independent *Pieris napi* media `000397558`, with
published label media `000397561`:

- raw stack: 992 unsigned 16-bit TIFF slices, each `368 × 496`
- reported voxel size: 1.08 micrometres isotropic
- label: `368 × 496 × 366`, values 1 (full eye volume) and 2 (external cornea)
- registered label crop offset: raw slice 550
- raw download archive SHA-256:
  `ca4c371e3e565a274c7b093d71dd4db0de4a109b8a5f96f18a7833206ad4aea8`
- label download archive SHA-256:
  `370ba0b923b34fbb6a2031356196b32af4f9f95bc455ec55812ad3a51f2434da`
- decompressed label SHA-256:
  `3ef96ff8e26fa96663dc814536ba594c47167b65fd95cd4992139c17fe5fd98b`

The raw MorphoSource files are not redistributed.

## Public fossil source

The source deposit is available at
[HU Berlin, DOI 10.18452/20002](https://doi.org/10.18452/20002). Confirm the
specimen identity and voxel calibration before applying any frozen parameters.

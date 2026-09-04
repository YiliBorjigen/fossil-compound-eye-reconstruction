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

## Arthur Zhao modern surface meshes

Experiments 57--63 use three pairs of complete *Drosophila melanogaster*
surface exports supplied by Arthur Zhao. The cohort comprises three distinct
female flies, represented by the dated whole-head scans; each volume contains
both eyes, so the animal/volume is the independent unit and eyes and lenses are
nested observations. This means different animals, not independent labs,
studies or acquisition workflows. The files remain external pending explicit
redistribution permission. The animal and bilateral-eye interpretation is
supported by the [source paper](https://doi.org/10.1038/s41586-025-09276-5)
and its pinned analysis code.

| Volume | Surface | SHA-256 |
|---|---|---|
| `20231107` | corneal lens | `1a9e5b150208d7f71bc4848b798d988ee52d17bd1aa12d063e5e38dbce7c6b20` |
| `20231107` | photoreceptor tip | `3498710995d31ebfbcab7437a8df92069ca27728e2347ac01380a46288fa68b5` |
| `20240530` | corneal lens | `671cebccc1561ed453e23f0cc58180dd29140f00fe80cc1fabed4054e2093c40` |
| `20240530` | photoreceptor tip | `aa7f7da9fbb424c7e15d59e14505b39bf64095bc43555717128cce3cd6acc55c` |
| `20240701` | corneal lens | `d4d7ab34517e744c5ce1a1d56f6b4fba0deaffcb32b4c9dbbfebf87529beecfa` |
| `20240701` | photoreceptor tip | `1948916674147559ed23751016717b9b3cfc205eb37d4e691b70cd59b6253f08` |

The matching public position inputs are also frozen exactly:

| Input | Bytes | SHA-256 |
|---|---:|---|
| `20231107.RData` | 296,662 | `e8cf395625b5fa2a415206614bd4e1a4d74d5372820038135bf77f312349a9ba` |
| `20240530.RData` | 293,647 | `6921aa284338b476cd7db14aea1e0fef19cc45bb141ea19d4baaf03f8d6ca684` |
| `20240701.RData` | 310,361 | `8408447f8c5798fbc6d751de79ef4983d08c342a500c9a55b2a3b4b38d92deec` |

The matching public position tables are read from
[`reiserlab/eyemap_T4`](https://github.com/reiserlab/eyemap_T4) at commit
`99d2a43123db636cedb55af9ff31a59657e7d17e`. The pinned code maps the three
dates to the main and two additional flies and splits each volume into left
and right eyes. Mesh coordinates are already in
physical micrometres. Raw acquisition pitches are established as 0.9882 µm
for `20231107` and 0.9884 µm for `20240530`. The repository does not equate the
supplied `20240701` mesh with a public deposit labelled `20240710`, so its raw
pitch is left unresolved rather than inferred.

## Maike Kittelmann modern binary lens stacks

Maike Kittelmann directly supplied the twelve manually cleaned binary
corneal-lens stacks associated with [Buffry et al.
(2024)](https://doi.org/10.1186/s12915-024-01864-7). They represent twelve
flies, one eye per fly: three females and three males from each of *D.
simulans* M3 and *D. mauritiana* RED3. Source voxels are binary `uint8` at
0.325 µm isotropic spacing. The archives are not committed or redistributed
pending confirmation of the terms applicable to the directly shared copies.

| Eye/animal | Slices | ODA lenses | Foreground voxels | Direct archive SHA-256 |
|---|---:|---:|---:|---|
| `M3_F_24_01` | 864 | 1,001 | 39,855,977 | `a1af8a4b3f04f9b22f4f416854189072b14dd59b13b6708048d4766580e4a638` |
| `M3_F_28_03` | 875 | 1,011 | 38,419,028 | `d18763929208c1fff12ada55124e1f489cd420ec66f101ebb9309656fce38ffa` |
| `M3_F_35_03` | 1,031 | 1,023 | 38,768,841 | `6a205f1d691e88829861fd56d16269b4c6867d1ae7bd3f14ca558279b18ccef0` |
| `M3_M_26_01` | 694 | 855 | 25,219,465 | `69b0206668e185d333f5fcdf8302a85cec048e4bb39ad93cd54001d5ec415f37` |
| `M3_M_32_01` | 714 | 944 | 29,407,973 | `adc4aa40945bcba20956679262b211444ec307391361f775e15ab4555e797d51` |
| `M3_M_36_01` | 684 | 970 | 36,009,405 | `024cb4ba7de8c56a98fa96bf881b67d94cdb49a2d0c43160cfdffe8b9a30f2bc` |
| `RED3_25_F_36` | 1,158 | 1,008 | 42,773,294 | `9208899f9c2481ebbe43b760b3efe6dfb60cfeb8b3b7fe156c8ee7884feed495` |
| `RED3_25_F_37` | 1,172 | 984 | 43,883,313 | `1d230daea5bacacc92f5ab5aec4da13c1d4b10e7a2aedc554ed3baf7cc0e3012` |
| `RED3_25_F_38` | 1,135 | 1,003 | 44,669,640 | `b735f6767029f512ba094a7ffeb4c1d46a3cc04039022890cce1f54b4e8a6b57` |
| `RED3_25_M_26` | 1,040 | 882 | 32,461,386 | `9c1ab44205610e940617e19249df8c5364a0b88115ff736b3c6a3952bd63b3de` |
| `RED3_25_M_27` | 1,252 | 822 | 27,823,622 | `5997775e6366fc4c268eb1cc65e467b101cc07ecaad2c0d8d4363bd16729d4ef` |
| `RED3_25_M_28` | 1,283 | 866 | 32,490,940 | `d3492e58cde9a6182729f0c9803fcc368f3bb7debdeb798d6e633f03febc40ce` |
| **Total** | **11,902** | **11,369** | **431,782,884** |  |

The fixed public ODA input is the Figure 3 archive in [Buffry et al.'s
Figshare record](https://doi.org/10.6084/m9.figshare.24769677.v6):

- outer archive size: 25,767,536 bytes;
- outer archive MD5: `81bab98c1e6c6aa1fc152132cb5d4c66`;
- outer archive SHA-256:
  `c4703afb44e3a67fc1565bf613b06e2948b9a12e1887c72d32137590a998a095`;
- nested member: `share/stacks.zip`;
- nested archive SHA-256:
  `c6904f692aead34d58896ba99eb8bb74254379902cd05926ea50f5cfbfb92cf8`.

The public per-eye H5 files are zero-byte placeholders. Experiment 63 therefore
replays ODA's deterministic TIFF import, sphere fit and rotations from the
public binned stacks at ODA commit
`55684a97fb32a95f24d17eaf04c49253c98fee27`. All 11,369 published centres
inverse-map to foreground in the directly supplied 0.325-µm stacks. This
mapping is an oracle correspondence/localisation input, not an automatically
predicted outer surface.

See the frozen [Experiment 63
protocol](../protocols/EXPERIMENT_63_MAIKE_OUTER_INFORMATION_VALIDATION.md) for
the exact leakage boundary, cohort gates and source-versus-validation
observation difference.

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

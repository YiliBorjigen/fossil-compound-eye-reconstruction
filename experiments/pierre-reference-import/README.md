# Reference data supplied by Pierre Tichit

Both uploaded specimen archives can be read directly in Python. No source file
was edited, recovered, or reconstructed. The inspection reads MATLAB arrays
and selected TIFF slices without expanding the complete stacks.

| Archive | Automatic regions | Selection-vector entries | Entries marked keep |
|---|---:|---:|---:|
| Pieris napi.zip | 9 | 11,921 | 7,722 |
| 77970 Bombus terrestris.zip | 12 | 6,391 | 5,748 |

These are counts in the author's saved selection vector, **not final unique
cone counts**. Both archives also contain TIFF stacks identified as manual
cone annotations, cornea/eye masks, and figures from the source analysis.
No raw greyscale CT volume was identified in this inventory.

## Verified file interface

`Results/trustedConeCenters.mat` contains a binary keep/discard vector. It
does not contain coordinates or full cone masks. Its object ordering and
position in the author's cleaning pipeline remain to be checked before
applying it. In particular, it must not be applied to a newly sorted list of
cones just because its length appears plausible.

`GatheredConeStruct.mat` contains sparse voxel coordinates for each region.
`Ind` holds one-based linear voxel indices, `Sub` holds one-based x/y/z
coordinates, and `MaskInd` identifies a region in the saved ordering. Individual
cone IDs are encoded in the TIFF voxel values, not in `Ind`.

The linear-index formula matches every supplied sparse voxel in both archives.
At one median-depth slice per region, all tested sparse coordinates occupy
nonzero TIFF labels: 9/9 Pieris regions and 12/12 Bombus regions. This verifies
the sampled coordinate correspondence, not anatomical correctness or the
complete TIFF-to-MAT correspondence.

Bombus uses alphabetical folder order: saved indices 1–12 correspond to
`coneLable1`, `coneLable10`, `coneLable11`, `coneLable12`, `coneLable2`,
`coneLable3`, `coneLable4`, `coneLable5`, `coneLable6`, `coneLable7`,
`coneLable8`, `coneLable9`. Treating saved index 2 as `coneLable2` fails the
sample check. The importer records the verified correspondence explicitly.

The Pieris TIFF export is 1,410 rows × 1,054 columns × 871 slices. It must not
be mixed directly with the earlier Zhou export of 368 × 496 × 366. Neither
the old crop offset nor its calibration was applied to this new export.

The Bombus file `CCorientationOnDistanceBasedGrid.mat` has 114,376 rows and
six columns after conversion from MATLAB storage. Columns 3–6 contain no
finite values. It cannot currently supply a complete position/direction table.
The useful inputs are the underlying label stacks and sparse geometry.

## Next input

The supplied Pieris troubleshooting note names **`ConeLabToOpticalModel7.m`**.
That script, together with any configuration it requires, would establish
how the selection vector maps to objects and which cleaning steps follow it.
Neither uploaded archive contains a MATLAB `.m` script. If the remaining
large package contains the code, only those small code/configuration files
are needed now; a third specimen archive can wait.

The new manual annotations may provide a reference for modern cone tests.
Matching these labels to original greyscale images, and matching them to the
earlier user-traced patches, are separate checks. This import does not establish
the anatomical identity of any fossil boundary or report prediction accuracy.

## Reproduce

Dependencies: `numpy`, `scipy`, `h5py`, `Pillow`.

```bash
python experiments/pierre-reference-import/inspect_archives.py \
  '/path/to/Pieris napi.zip' '/path/to/77970 Bombus terrestris.zip' \
  --output experiments/pierre-reference-import/results/manifest.json
```

[The manifest](results/manifest.json) records archive hashes, file names,
selection-vector counts, coordinate checks and the available Amira metadata.
The archive contents are not redistributed in this repository.

Thanks to Pierre Tichit for the processed reference data and to Tunhe Zhou
for the automatic segmentations and guidance. Source study: Tichit, Zhou,
Kjer and colleagues, [InSegtCone: interactive segmentation of crystalline
cones in compound eyes](https://doi.org/10.1186/s40850-021-00101-w), BMC
Zoology 7, 10 (2022).

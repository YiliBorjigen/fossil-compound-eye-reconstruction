# Pieris cone labels supplied by Tunhe Zhou

The author-provided labels address the missing cone reference identified in
[Experiment 47](../../reports/EXPERIMENT_47_PIERIS_CENTERLINE_AUDIT.md).
They can support a check of the manually traced paths in Experiments 48–51
once matched to the original CT. They do not label the inner corneal-lens
surface and do not establish fossil anatomy.

## What the archive contains

The [source archive](https://github.com/zhoutunhe/InSegtCone/blob/da283cac7818e72f62e9a2f645e2ab93fdb29d27/data/autoSegmentedLabelsPnapi.zip)
contains 3,294 16-bit TIFF images: nine regions of 366 slices, each 368 rows
by 496 columns. All images were decoded and checked. The ZIP is 2.41 MB;
the compressed TIFF files inside total 22.38 MB. Fully decoded voxel arrays
would be larger. The source archive is not redistributed here.

The authors' [P4_cones.m](https://github.com/zhoutunhe/InSegtCone/blob/da283cac7818e72f62e9a2f645e2ab93fdb29d27/code/P4_cones.m)
maps labels from unfolded regions back to the original image grid. Matching
dimensions alone do not confirm alignment to our raw scan. Coordinates in
this audit are TIFF row, column and zero-based slice; distances remain in
voxels until registration and calibration are checked.

## Audit result

| Measure | Count |
|---|---:|
| Region-local labelled objects | 9,828 |
| Objects with multiple 26-connected components | 505 |
| Objects passing the fixed morphology screen | 4,273 |
| Cross-region label pairs with any voxel overlap | 2,153 |
| Pairs overlapping at least half of the smaller object | 1,262 |
| Objects involved in those substantial-overlap pairs | 2,006 |
| Screened objects outside those flagged pairs | 2,999 |

**These are object counts, not verified cone counts.** The permissive screen
requires at least 50 voxels, at least 95% in the largest connected component,
a first/second PCA variance ratio of at least 3, and no contact with a volume
border. Passing means only that an object is worth inspecting. Fragmentation
can also arise from mapping labels between grids.

Substantial overlap is a candidate duplicate or segmentation conflict. It
does not by itself establish that two labels represent exactly the same cone.
All labels were preserved; none were automatically merged or cleaned.

Splitting these files by region alone risks putting overlapping representations
of the same structure into both training and test sets. A later benchmark
must resolve overlaps, group representations of the same physical object,
and keep spatially separated test regions. Even the 2,999 candidates outside
the substantial-overlap pairs are not certified independent observations.

## What is needed next

The matching raw *Pieris napi* CT, MorphoSource media **000397558**, is not in
the current workspace. The previously recorded archive name is
`morphosource_media-id-000397558_download-681a4dea.zip`; an extracted original
volume is also suitable. No missing source was recovered or reconstructed.

With that file, first confirm the alignment and inspect representative labels
against the image signal, including objects that fail the screen. To compare
directly with the earlier manual-axis results, the original four annotation
files and saved patch geometry are also needed. The recorded raw-versus-label
slice offset of 550 is a starting hypothesis for registration, not a
registration verified by this audit.

No new reconstruction accuracy, anatomical identity, or visual-field result
is claimed here. The current result establishes what is present in the
supplied archive and prevents an invalid validation split.

## Reproduce

Install `numpy`, `scipy` and `Pillow`, obtain the source ZIP above, and run:

```bash
python experiments/zhou-cone-labels/audit_labels.py /path/to/autoSegmentedLabelsPnapi.zip \
  --output experiments/zhou-cone-labels/results
```

The script checks the source SHA-256, reads the ZIP without expanding it to
thousands of files, and writes per-object measurements, overlap pairs and
the summary. [PROTOCOL.md](PROTOCOL.md) records the checks fixed before the
run. [results/summary.json](results/summary.json) contains provenance and
counts; [results/objects.csv](results/objects.csv) is the full candidate table.

Thanks to Tunhe Zhou for providing access to the uncleaned labels. The data
and segmentation method are from Pierre Tichit, Tunhe Zhou, Hans Martin Kjer
and colleagues, [InSegtCone: interactive segmentation of crystalline cones
in compound eyes](https://doi.org/10.1186/s40850-021-00101-w), BMC Zoology
7, 10 (2022). Their labels remain subject to their source terms.

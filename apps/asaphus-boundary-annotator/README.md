# Blinded Asaphus boundary annotator

This small GUI asks a human observer to trace a visible internal CT boundary
without seeing the algorithm's proposed depth, original facet identity, spatial
block or quality-control status. It tests whether the extracted boundary is
visually reproducible. It does **not** by itself establish that the boundary is
the proximal lens surface.

## Annotating a prepared pack

On macOS, right-click `RUN_BOUNDARY_ANNOTATOR.command` and choose **Open** the
first time. On Windows, double-click `RUN_BOUNDARY_ANNOTATOR.bat`.

1. Click **Open annotation pack** and select its folder.
2. Enter your name or an anonymous annotator code.
3. Click several points along any visible boundary in the U-depth and V-depth
   sections. Leave the sections unmarked if no boundary is visible.
4. Record visibility, the most plausible interpretation, confidence and notes.
   Selecting a visibility option or clicking a boundary marks the case as
   reviewed. You can also use the explicit **I have reviewed this case** box.
5. Click **Save and export**. The app writes both JSON and CSV.

The app warns before saving if any of the 30 cases remain unreviewed.

The tangential preview helps follow repeated structures through depth. It does
not contain an algorithm overlay.

## Preparing the pack

The preparation script requires the raw cropped NRRD plus the frozen pipeline's
centre, sample and QC tables. Example:

```bash
python prepare_boundary_pack.py \
  --nrrd 1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --centers robust_facet_centers.csv \
  --samples reconstruction_samples.csv \
  --edge-table internal_edge_facets.csv \
  --out asaphus_boundary_pack_v1
```

The default set contains five accepted facets from each of five spatial blocks
and five failed-QC controls. Selection and case order are deterministic hashes,
not model-error rankings. The script creates a private key beside the pack.
Keep that key away from annotators until all labels are frozen.

Raw CT data, generated packs, private keys and annotations are deliberately not
stored in this repository.

After every observer has frozen their labels, `score_annotations.py` uses the
private key and `reconstruction_samples.csv` to compare clicked human boundary
depths with the algorithm-extracted CT edge. Do not run it, or reveal its
outputs, until annotation is complete.

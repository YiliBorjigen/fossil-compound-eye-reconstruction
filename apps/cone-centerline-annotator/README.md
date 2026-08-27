# 3D cone centre-line annotator

This is a clickable fallback when a published individual-cone segmentation is
not available. It does not require command-line use.

## Start

- macOS: double-click `RUN_ANNOTATOR.command`.
- Windows: double-click `RUN_ANNOTATOR.bat`.
- On the first run, the launcher installs the required scientific packages in
  its own `.venv` folder if they are not already available. This can take a few
  minutes; later runs open directly.
- Select one folder named `patch_1`, `patch_2` or `patch_3`. The selected folder
  must contain `unfolded_intensity.npy`.

## Mark one region

1. Click **New cone**.
2. Click the centre of that same cone at several depths. Use the slider or the
   arrow keys to move through the volume. The GUI interpolates between clicks.
3. Repeat for at least three well-separated cones.
4. Select **Explicit background** and click areas that are clearly not cones at
   several depths.
5. Click **Save and export**.
6. Click **Train preliminary mask**. Review the saved preview; the output is an
   annotation aid, not validated anatomy.

Repeat this independently for all three regions. Files are written to a
`manual_annotations` folder beside the selected patch; the CT volume is never
overwritten.

## Outputs

- `annotations.json`: human control points, source hash and provenance;
- `dense_centrelines.csv`: interpolated paths;
- `training_scribbles.npy`: explicit foreground and background training labels;
- `preliminary_segmentation/`: probability map, binary proposal and preview.

The preliminary model is deliberately labelled as training assistance. A cone
mask must be reviewed and corrected before it can serve as the ground truth for
centre-line, length, radius or reconstruction measurements.

## What this annotation means

The published input mask describes the eye/cone-layer volume and external
corneal surface. It does not assign a separate identity to every cone. The
points made here are therefore new human annotations of individual cone axes.
Keep the automatically suggested points switched on only as a visual aid; do
not follow them when the CT image disagrees.

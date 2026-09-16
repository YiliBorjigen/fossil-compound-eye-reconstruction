# Author-supplied Pieris cone labels

This check asks whether Tunhe Zhou's uncleaned labels can provide the cone
reference that Experiments 47–51 lacked. It is an input audit, not a new
reconstruction model or anatomical ground truth.

Before running the audit, the checks are fixed as follows:

- Read all nine regions and all 366 slices per region. Preserve source labels.
- Identify objects by `(region, label_id)`, since region-local IDs can repeat.
- Measure voxel counts, 26-connected components, PCA axes and volume-border contact.
- Report a permissive morphology screen: at least 50 voxels, at least 95% in
  the largest component, first/second PCA variance ratio at least 3, no volume
  border contact. These are diagnostic choices, not biological acceptance rules.
- Check cross-region voxel overlap. Flag a pair if at least half of the
  smaller object overlaps; report all overlaps without automatically merging.
- Do not tune these thresholds to increase the number of retained objects.
- Do not report anatomical reconstruction accuracy without checking labels
  against the matching original CT and resolving the coordinate mapping.

The matching raw Pieris CT and original manual annotation/patch geometry files
are absent from the current workspace. They will be requested from the user;
they will not be recovered or reconstructed from old outputs.

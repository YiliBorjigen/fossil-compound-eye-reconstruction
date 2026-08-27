# Frozen Patch 4 protocol

Patch 4 is reserved as the next held-out regional test of the *Pieris napi*
cone-orientation field.

## Selection before annotation

- Source: the same registered public *Pieris* scan used for Experiments 45–49.
- Candidate regions: nine KMeans medoids selected from the labelled corneal
  surface only (`random_state=45`), with the same safe boundary margins as the
  original three-patch selection.
- Rule: among the six unused medoids, select the candidate with the largest
  minimum Euclidean distance from Patches 1–3.
- Frozen seed: `(x, y, z) = (234, 277, 913)` voxels.
- Patch: 141 × 141 × 61 isotropic voxels at 1.08 µm.
- Surface fit RMSE: 0.750 voxels.

No cone direction, reconstruction error or internal-centre correspondence was
used to select this region.

## Frozen evaluation

1. Preserve the raw annotation archive and its hashes.
2. Audit path continuity before model scoring; document, but do not silently
   repair, any cone switch.
3. Fit the spatial orientation field using Patches 1–3 only.
4. Anchor each Patch 4 prediction at the first point of its held-out manual
   path. Do not use later Patch 4 points to fit direction.
5. Primary outcome: 3D centre-line RMSE versus the surface-normal path.
6. Report per-track results and the number of wins, not only a pooled median.
7. Keep straight and gently curved paths as separate secondary comparisons.

This remains a same-specimen, same-annotator validation. Author-provided cone
segmentations or another specimen are still required for anatomical and
biological replication.

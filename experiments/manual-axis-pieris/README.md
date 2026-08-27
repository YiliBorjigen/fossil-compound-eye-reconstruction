# Manual-axis Pieris pilot

Experiment 48 uses Li Yi's manually traced cone centre-lines from the first
pre-selected *Pieris napi* region. It asks whether a leave-one-cone-out residual
template improves when it is placed on the traced 3D path rather than held on a
static surface normal. Straight and quadratic fits are evaluated separately
from the raw clicked path so that click-to-click jitter is not mistaken for
biological curvature.

This is an oracle-axis diagnostic within previously examined data. It is not an
automatic segmentation result, a biological replication or an independent
validation. The raw annotation archive and CT volume are not stored in Git.

The script records a single auditable correction: the depth-14 and depth-15
points of cone 2 are removed because the next point jumps 79.14 voxels to a
different cone; all other within-track steps are below 1.6 voxels. The original
archive is preserved unchanged.

## Result

Five tracks overlap from depths 34–50. Their median displacement is 4.87 voxels
(5.26 µm). A leave-one-cone-out template placed on an oracle-fitted straight
axis reduces median normalised MAE from 0.399 for local background to 0.282 and
wins for four of five cones. A quadratic axis also gives 0.282 and beats the
straight axis for only one cone. The current evidence therefore supports a
regional tilted axis, not necessary cone-specific curvature.

The uploaded preliminary random-forest mask is not used: three explicit
background clicks were insufficient, and the model labelled 53.3% of the
volume as cone.

Run with:

```bash
python run_manual_axis_pilot.py \
  --volume /path/to/patch_1/unfolded_intensity.npy \
  --annotations /path/to/manual_annotations/annotations.json \
  --output-dir results
```

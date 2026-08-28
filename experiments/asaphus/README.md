# Asaphus surface and candidate internal boundary

`36_37_asaphus_fossil_reconstruction.py` reads the corrected 3.7 micrometre
NRRD, extracts a subvoxel outer surface, finds facet-relief maxima across
neighbouring thresholds, and measures candidate inward intensity edges.

Place `1652a_0000_1_cropped_CORRECT_3p7um.nrrd` beside the script and run:

```bash
python 36_37_asaphus_fossil_reconstruction.py
```

The script was written during exploratory development and writes CSV/Markdown
outputs beside the input. The internal feature is deliberately called a
candidate interface: its anatomical identity has not been independently
validated.

`asaphus_reconstruction_pipeline.py` is the complete command-line version used
for the later audits. Unlike the shorter historical script, it exports the
9,840-point `reconstruction_samples.csv` table and the 116 robust facet centres
needed by Experiments 42–54:

```bash
python asaphus_reconstruction_pipeline.py \
  --input /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --out results/reconstruction_inputs
```

With the corrected scan and frozen defaults it should report 116 robust facets,
74 high-quality candidate surfaces and a 51.8 micrometre median candidate-edge
depth.

`run_visual_field_sensitivity.py` measures the angular geometry of the
preserved outer surface. It computes a surface-normal envelope for the 116
robust facets, checks normal stability across neighbouring thresholds, and
reports analytic bounds for an unknown internal-axis departure. The normals
describe the fossil as preserved; they are not verified optical axes or a
reconstruction of the living field of view. Run:

```bash
python run_visual_field_sensitivity.py \
  --input /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --output-dir results/visual_field
```

`run_preservation_geometry_audit.py` tests normal estimation across thresholds,
surface smoothing and half-voxel sampling offsets. It also applies transparent
10% and 20% affine scaling/shear scenarios. These scenarios quantify
sensitivity; they do not estimate the specimen's actual geological strain or
recover its original shape. Run:

```bash
python run_preservation_geometry_audit.py \
  --input /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --baseline-csv results/visual_field/facet_normal_baseline.csv \
  --output-dir results/preservation_geometry
```

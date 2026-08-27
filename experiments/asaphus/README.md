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

`run_visual_field_sensitivity.py` returns to the preserved outer surface and
asks a narrower functional question. It computes a surface-normal viewing
envelope for the 116 robust facets, checks normal stability across neighbouring
thresholds, and reports analytic bounds for an unknown internal-axis departure.
The normals are a geometric baseline, not verified optical axes. Run:

```bash
python run_visual_field_sensitivity.py \
  --input /path/to/1652a_0000_1_cropped_CORRECT_3p7um.nrrd \
  --output-dir results/visual_field
```

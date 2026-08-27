# Anatomy first

The available scans do not represent one interchangeable class of eyes.

The InSegtCone study used three differently shaped **apposition** eyes. The
honeybee is oval and has strongly skewed ommatidia; *Pieris napi* is almost
hemispherical and required nine local surface regions. Their pipeline does not
describe a cone as a translated image block. It unfolds the cornea locally,
segments a three-dimensional cone, and measures its centre, elongation axis,
length, radius and neighbour spacing.

The moth scans shown in the MorphoSource search add a different optical design.
*Manduca sexta* and *Macroglossum stellatarum* have superposition eyes. These
can be useful stress tests, but they should not be pooled with apposition eyes
as if one mean cone were biologically transferable. The ODA-3D study also shows
that moth eyes are closer to spherical, whereas the honeybee eye needs internal
axis measurements because surface normals are a poor approximation.

## Revised representation

The next model should work with a small set of anatomical variables before it
ever predicts intensity:

1. corneal facet centre and local surface normal;
2. three-dimensional cone centre-line;
3. cone length and proximal/distal radius;
4. tilt relative to the surface normal;
5. local facet spacing and hexagonal orientation;
6. only then, CT residual intensity around the normalised centre-line.

This turns a vague image-generation problem into two measurable tasks. First,
predict the missing centre-line in micrometres and degrees. Second, ask whether
a normalised residual improves over the local depth profile once that axis is
correct.

## Validation order

- Develop the centre-line representation on the already examined Apis and
  Pieris scans.
- Freeze segmentation, matching, scale and evaluation rules.
- Test another apposition eye, preferably Bombus, as the first untouched
  biological replication.
- Test a spherical superposition moth separately as an architecture stress
  test.
- Treat fossil facets as a separate anatomical problem. Modern insect cones
  validate the algorithm; they are not direct templates for trilobite lenses.

Primary sources: [Tichit et al. 2022](https://doi.org/10.1186/s40850-021-00101-w),
[Currea et al. 2023](https://doi.org/10.1038/s42003-023-04575-x), and
[Warrant et al. 1999](https://pubmed.ncbi.nlm.nih.gov/9929453/).

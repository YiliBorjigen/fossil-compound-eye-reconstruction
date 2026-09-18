# Acknowledgements

This page records external contributions to the project, including shared data,
technical help and scientific discussion. Each entry describes the specific
contribution as precisely as possible. Acknowledgement does not imply
co-authorship, responsibility for the analysis, or endorsement of the
conclusions.

## Data access and technical contributions

- **Arthur Zhao** (Janelia Research Campus) exported and shared surface meshes
  for all three *Drosophila* µCT volumes used by Zhao et al., including the
  corneal-lens and photoreceptor-tip segmentations. These files support
  registration and method validation. They are not redistributed in this
  repository pending explicit confirmation of their redistribution terms.
- **Michael Reiser** (Janelia Research Campus) responded to the initial data
  request, connected Li Yi with Arthur Zhao, and directed the project towards
  the relevant *Drosophila* eyemap resources.

## Public datasets and software

The corresponding processed lens–tip positions and analysis code are available
from the [Reiser Lab eyemap repository](https://github.com/reiserlab/eyemap_T4).
The raw imaging data are archived in the
[Janelia Data Repository](https://doi.org/10.25378/janelia.29111339.v1).

Additional dataset and software provenance is maintained in
[`NOTICE.md`](NOTICE.md) and [`data/README.md`](data/README.md).

## Future contributions

Further contributors will be added here with their concrete contribution—for
example data provision, manual annotation, anatomical interpretation, imaging
advice, software or validation—and with the applicable citation and data-use
conditions.

## Maike Kittelmann — corneal-lens data

**Maike Kittelmann** shared binary corneal-lens TIFF stacks for the M3 strain
of *Drosophila simulans* and RED3 strain of *Drosophila mauritiana* with Li Yi.
Eleven archives were accessible in the 5 September 2026 analysis; `M3_M_26_01`
and `M3_M_32_01` support the first binary-lens reconstruction pilot and frozen
cross-file test. The remaining accessible archives received header inspection
only. The supplied stacks are not redistributed here.

Related publication: [Buffry et al. (2024), Evolution of compound eye morphology
underlies differences in vision between closely related Drosophila species](https://doi.org/10.1186/s12915-024-01864-7).

## Pierre Tichit and Tunhe Zhou — crystalline-cone reference data

**Pierre Tichit** shared processed insect-eye datasets from the InSegtCone
work, including automatic cone segmentations, manual annotations, eye and
cornea masks, saved curation selections and comparison files. The supplied
archives include *Pieris napi* and *Bombus terrestris* reference data. He also
explained that the saved cleaning results were retained in MATLAB format and
pointed to `Results/trustedConeCenters.mat`. These contributions support
checking cone geometry and segmentation quality. The original source archives
are not redistributed in this repository.

**Tunhe Zhou** provided access to the [uncleaned automatic *Pieris napi* cone
labels](https://github.com/zhoutunhe/InSegtCone/blob/master/data/autoSegmentedLabelsPnapi.zip),
explained that overlapping segmentations require cleaning, and directed the
request for cleaned results to Pierre Tichit.

Related publication: [Tichit, Zhou, Kjer et al. (2022), InSegtCone: interactive
segmentation of crystalline cones in compound eyes](https://doi.org/10.1186/s40850-021-00101-w).

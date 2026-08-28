# Blinded boundary pilot

This analysis scores a frozen observer file from the Asaphus boundary
annotator. It tests two narrow questions:

1. did the frozen quality-control procedure enrich for CT structures that an
   observer could see without the algorithm overlay; and
2. when the observer clicked a transition, how closely did its depth agree
   with the frozen edge extractor?

It does not test whether the transition is a proximal lens surface, a cone, or
a preservational boundary.

The raw observer file and private unblinding key are intentionally not stored
in the repository. Run locally with:

```bash
python analyse_pilot.py \
  --annotations yili_annotations.json \
  --private-key Asaphus_boundary_annotation_PRIVATE_KEY_v1.csv \
  --samples reconstruction_samples.csv \
  --out results-private
```

Only the aggregate summary and figure from the frozen run are committed.

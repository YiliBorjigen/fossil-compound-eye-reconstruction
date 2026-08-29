# Boundary-definition sensitivity

Experiment 56 asks whether the systematic depth offset in the blinded observer
pilot changes the missing-surface reconstruction result.

For each held-out spatial block, the median human-minus-gradient translation is
estimated from the other four blocks. The test block is not used to choose its
correction. A separate numerical check applies the same translation to every
Experiment 54 target and prediction and verifies that reconstruction errors are
unchanged.

The raw observer labels, case table and unblinding key remain private. Run with:

```bash
python analyse_boundary_definition.py \
  --annotations yili_annotations.json \
  --private-key Asaphus_boundary_annotation_PRIVATE_KEY_v1.csv \
  --samples reconstruction_samples.csv \
  --experiment-54-predictions experiment_54_predictions.csv \
  --experiment-54-summary experiment_54_model_summary.csv \
  --out results-private
```

Only the aggregate JSON and figure from the frozen analysis are committed.

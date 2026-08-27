# Controlled missing-centre tests

These scripts remove known ommatidial centres from modern ODA-3D output and
keep the hidden truth separate until scoring. They test whether the surviving
lattice can locate a missing centre; they do not reconstruct an anatomical
inner lens surface.

- `04_strict_blind_benchmark.py`: one internal hole, 97.7% top-1 detection.
- `06_joint_adjacent_pair_benchmark.py`: 94.0% of held-out adjacent pairs
  recovered within 10% of local spacing.
- `10_geometric_model_selection_3to5.py`: unknown counts of three to five;
  count inference remained model-dependent.
- `12_edge_torn_stress_test.py`: only 2.5% top-1 at torn boundaries, showing
  that the interior interpolation method should not be used for edge loss.

The scripts expect `tests/d_mauritiana_ct_stack/ommatidial_data.csv` beneath
the repository root, matching the original ODA working layout.

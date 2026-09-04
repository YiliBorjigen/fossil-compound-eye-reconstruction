from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import experiment_60_margin_loss_validation as exp60
import experiment_61_survival_threshold as exp61
import experiment_62_outer_only_partial_identification as exp62


class MarginProtocolTests(unittest.TestCase):
    def test_margin_seeds_are_boundary_only_and_deterministic(self) -> None:
        count = 96
        angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        points = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(count)])
        adjacency = tuple(
            np.asarray([(index - 1) % count, (index + 1) % count], dtype=int)
            for index in range(count)
        )
        graph = SimpleNamespace(
            points=points,
            adjacency=adjacency,
            boundary=np.ones(count, dtype=bool),
        )
        protocol = SimpleNamespace(graph=graph)
        first = exp60.select_margin_seeds(protocol)
        second = exp60.select_margin_seeds(protocol)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), exp60.SEEDS_PER_EYE)
        self.assertTrue(np.all(graph.boundary[first]))


class SurvivalRankingTests(unittest.TestCase):
    def test_rank_is_permutation_and_target_blind(self) -> None:
        records = [
            {"landmark_id": index, "target_qc_pass": bool(index % 2)}
            for index in range(20)
        ]
        protocol = SimpleNamespace(volume="v", eye=0, records=records)
        before = exp61.stable_order(protocol, 3)
        for record in records:
            record["target_qc_pass"] = not record["target_qc_pass"]
        after = exp61.stable_order(protocol, 3)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(np.sort(before), np.arange(len(records)))


class PartialIdentificationTests(unittest.TestCase):
    def test_equal_volume_weights_sum_equally(self) -> None:
        groups = np.asarray(["a", "a", "a", "b"])
        weights = exp62.equal_group_weights(groups)
        self.assertAlmostEqual(float(weights[groups == "a"].sum()), 2.0)
        self.assertAlmostEqual(float(weights[groups == "b"].sum()), 2.0)

    def test_outer_model_outputs_reflection_even_coefficients(self) -> None:
        rng = np.random.default_rng(3)
        features = rng.normal(size=(60, len(exp62.OUTER_FEATURE_NAMES)))
        targets = rng.normal(size=(60, 6))
        groups = np.repeat(exp62.VOLUMES, 20)
        model = exp62.fit_ridge(features, targets, groups, alpha=1.0)
        prediction = model.predict(features[:5], np.zeros(5, dtype=bool))
        self.assertEqual(prediction.shape, (5, 6))
        np.testing.assert_array_equal(prediction[:, [2, 4]], 0.0)

    def test_table_validation_rejects_missing_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "required columns"):
            exp62.validate_table(pd.DataFrame({"eye_id": ["x"]}), require_volume=False)

    def test_source_file_hash_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"a,b\n1,2\n")
            self.assertEqual(
                exp62.sha256_file(path),
                "492d5ea496056f1a6a6592241032fab764c321596317930b4fa0e1e8bc3b7470",
            )


if __name__ == "__main__":
    unittest.main()

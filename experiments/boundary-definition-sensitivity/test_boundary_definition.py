from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).with_name("analyse_boundary_definition.py")
SPEC = importlib.util.spec_from_file_location("boundary_definition", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BoundaryDefinitionTests(unittest.TestCase):
    def test_cross_block_calibration_excludes_test_block(self) -> None:
        cases = pd.DataFrame({
            "case_id": ["a", "b", "c", "d"],
            "cv_block": [0, 0, 1, 1],
            "human_minus_algorithm_um": [100.0, 100.0, 4.0, 6.0],
        })
        result = MODULE.cross_block_calibration(cases)
        block_zero = result.loc[result["cv_block"] == 0]
        block_one = result.loc[result["cv_block"] == 1]
        self.assertTrue((block_zero["training_shift_um"] == 5.0).all())
        self.assertTrue((block_one["training_shift_um"] == 100.0).all())

    def test_common_translation_preserves_absolute_error(self) -> None:
        frame = pd.DataFrame({
            "pitch": [10.0, 20.0],
            "target_norm": [1.0, 1.5],
            "first_prediction_norm": [1.2, 1.1],
            "second_prediction_norm": [0.8, 1.7],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv") as stream:
            frame.to_csv(stream.name, index=False)
            change, methods = MODULE.verify_translation_invariance(
                Path(stream.name), spacing_um=3.7, shift_um=5.66
            )
        self.assertEqual(methods, 2)
        self.assertLess(change, 1e-12)


if __name__ == "__main__":
    unittest.main()

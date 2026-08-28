"""Small dependency-light tests for annotation pack I/O."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from annotation_core import (
    PACK_VERSION,
    empty_annotation,
    load_pack,
    point_from_canvas,
    save_annotations,
)


class AnnotationCoreTests(unittest.TestCase):
    def test_pack_round_trip_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cases").mkdir()
            np.savez_compressed(
                root / "cases/F001.npz",
                intensity=np.zeros((3, 3, 4), dtype=np.float32),
                u_vox=np.arange(3),
                v_vox=np.arange(3),
                depth_vox=np.arange(4),
            )
            manifest = {
                "pack_version": PACK_VERSION,
                "pack_id": "test",
                "source_dataset_sha256": "abc",
                "voxel_spacing_um": 3.7,
                "cases": [{"case_id": "F001", "file": "cases/F001.npz"}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            loaded, cases = load_pack(root)
            self.assertEqual([case.case_id for case in cases], ["F001"])
            record = empty_annotation("F001")
            self.assertFalse(record["reviewed"])
            record["reviewed"] = True
            record["u_depth_points"] = [
                point_from_canvas(50, 50, 100, 100, np.arange(3), np.arange(4))
            ]
            save_annotations(root / "labels.json", loaded, "tester", {"F001": record})
            self.assertTrue((root / "labels.json").exists())
            self.assertTrue((root / "labels.csv").exists())


if __name__ == "__main__":
    unittest.main()

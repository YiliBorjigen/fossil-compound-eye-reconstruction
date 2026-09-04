from __future__ import annotations

import ast
import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import render_instance_qc_sample as renderer


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_instance(path: Path, index: int, *, empty: bool = False) -> None:
    full = np.empty((0, 3), dtype=np.int32) if empty else np.array(
        [[index, 0, 0], [index, 1, 0]], dtype=np.int32
    )
    main = np.empty((0, 3), dtype=np.int32) if empty else full[:1].copy()
    np.savez(
        path,
        schema_version=np.array("experiment63.instance.v2"),
        lens_index=np.array(index, dtype=np.int64),
        full_assigned_points_zyx=full,
        main_component_points_zyx=main,
        component_sizes_descending=np.array([] if empty else [len(main), len(full) - len(main)], dtype=np.int64),
        spacing_um=np.array([0.325, 0.325, 0.325], dtype=np.float64),
        seed_source_zyx=np.array([index, 0.0, 0.0], dtype=np.float64),
        oda_axis_source_zyx=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        config_json=np.array("{}"),
        config_sha256=np.array(hashlib.sha256(b"{}").hexdigest()),
    )


def write_sealed(path: Path, index: int, *, empty: bool = False) -> None:
    points = np.empty((0, 3), dtype=np.int32) if empty else np.array(
        [[index, 0, 0]], dtype=np.int32
    )
    np.savez(
        path,
        schema_version=np.array("experiment63.sealed-distal.v2"),
        lens_index=np.array(index, dtype=np.int64),
        points_zyx=points,
        spacing_um=np.array([0.325, 0.325, 0.325], dtype=np.float64),
        config_json=np.array("{}"),
        config_sha256=np.array(hashlib.sha256(b"{}").hexdigest()),
    )


def make_minimal_bundle(root: Path, *, eye_id: str = "TEST_EYE", n: int = 32) -> Path:
    bundle = root / "bundle"
    for directory in ("instances", "sealed_distal", "lenses"):
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        instance_relpath = f"instances/lens_{index:06d}.npz"
        distal_relpath = f"sealed_distal/lens_{index:06d}.npz"
        write_instance(bundle / instance_relpath, index)
        write_sealed(bundle / distal_relpath, index)
        # This is intentionally not a valid NPZ.  A passing renderer proves it
        # did not try to open the fitted/proximal lens artifact.
        (bundle / "lenses" / f"lens_{index:06d}.npz").write_bytes(
            f"PROXIMAL_TARGET_CANARY_{index}".encode()
        )
        rows.append(
            {
                "eye_id": eye_id,
                "lens_index": str(index),
                "seed_id": f"seed-{index:04d}",
                "distal_eligible": "true",
                "position_u_um": str(float(index + 1)),
                "position_v_um": "0.0",
                "distal_scale_um": str(float(index % 8 + 1)),
                "instance_relpath": instance_relpath,
                "sealed_distal_relpath": distal_relpath,
            }
        )
    with (bundle / "distal_qc_sampling.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=renderer.SAMPLING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return bundle


def fake_render(output_path: Path, **kwargs: object) -> None:
    selected = kwargs["selected"]
    output_path.write_bytes(f"PNG-{selected.row.lens_index}".encode())


class FrozenSelectionTests(unittest.TestCase):
    def test_exact_32_and_two_per_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = make_minimal_bundle(Path(temporary))
            rows, count = renderer.read_sampling_table(
                bundle / "distal_qc_sampling.csv", eye_id="TEST_EYE"
            )
            selected = renderer.select_frozen_sample(rows)
        self.assertEqual(count, 32)
        self.assertEqual(len(selected), 32)
        counts = {
            (radial, scale): sum(
                item.radial_stratum == radial and item.scale_stratum == scale
                for item in selected
            )
            for radial in range(4)
            for scale in range(4)
        }
        self.assertEqual(set(counts.values()), {2})
        for item in selected:
            expected = hashlib.sha256(
                f"experiment63_instance_qc_v1|TEST_EYE|{item.row.lens_index}|{item.row.seed_id}".encode()
            ).hexdigest()
            self.assertEqual(item.selection_sha256, expected)

    def test_undersized_cell_fails_instead_of_changing_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = make_minimal_bundle(Path(temporary), n=31)
            rows, _ = renderer.read_sampling_table(
                bundle / "distal_qc_sampling.csv", eye_id="TEST_EYE"
            )
            with self.assertRaises(renderer.QCError):
                renderer.select_frozen_sample(rows)


class OutcomeBlindRendererTests(unittest.TestCase):
    def test_actual_renderer_writes_labeled_three_view_png(self) -> None:
        row = renderer.SamplingRow(
            eye_id="TEST_EYE",
            lens_index=0,
            seed_id="seed-0000",
            position_u_um=1.0,
            position_v_um=0.0,
            distal_scale_um=2.0,
            instance_relpath="instances/lens_000000.npz",
            sealed_distal_relpath="sealed_distal/lens_000000.npz",
        )
        selected = renderer.SelectedRow(row, 0, 0, 0, 0, "0" * 64)
        full = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
        main = full[:2]
        distal = full[:1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "render.png"
            renderer._render_lens(
                output,
                selected=selected,
                full_zyx=full,
                main_zyx=main,
                distal_zyx=distal,
                spacing_zyx=np.array([0.325, 0.325, 0.325]),
            )
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_renderer_opens_npz_only_in_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_minimal_bundle(root)
            output = bundle / renderer.SAMPLE_DIRECTORY_NAME
            opened_npz: list[Path] = []
            original_load = np.load

            def recording_load(path: object, *args: object, **kwargs: object):
                opened_npz.append(Path(path).resolve())
                return original_load(path, *args, **kwargs)

            with mock.patch.object(renderer.np, "load", side_effect=recording_load), mock.patch.object(
                renderer, "_render_lens", side_effect=fake_render
            ):
                manifest_path = renderer.render_eye_sample(
                    eye_id="TEST_EYE",
                    bundle_root=bundle,
                    sampling_table=bundle / "distal_qc_sampling.csv",
                    output_dir=output,
                )
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(opened_npz), 64)
            allowed_roots = {
                (bundle / "instances").resolve(),
                (bundle / "sealed_distal").resolve(),
            }
            for path in opened_npz:
                self.assertIn(path.parent, allowed_roots)
            self.assertFalse(any(path.parent.name == "lenses" for path in opened_npz))

    def test_fitted_lens_canaries_are_not_valid_npz_but_render_still_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_minimal_bundle(root)
            with self.assertRaises(Exception):
                np.load(bundle / "lenses" / "lens_000000.npz", allow_pickle=False)
            with mock.patch.object(renderer, "_render_lens", side_effect=fake_render):
                renderer.render_eye_sample(
                    eye_id="TEST_EYE",
                    bundle_root=bundle,
                    sampling_table=bundle / "distal_qc_sampling.csv",
                    output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
                )

    def test_extra_target_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_minimal_bundle(root)
            table = bundle / "distal_qc_sampling.csv"
            lines = table.read_text(encoding="utf-8").splitlines()
            lines[0] += ",target_depth_um"
            for index in range(1, len(lines)):
                lines[index] += ",999"
            table.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(renderer.QCError, "exact outcome-blind allowlist"):
                renderer.read_sampling_table(table, eye_id="TEST_EYE")

    def test_path_to_fitted_lens_archive_is_rejected_before_np_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_minimal_bundle(root)
            table = bundle / "distal_qc_sampling.csv"
            text = table.read_text(encoding="utf-8")
            text = text.replace(
                "instances/lens_000000.npz", "lenses/lens_000000.npz", 1
            )
            table.write_text(text, encoding="utf-8")
            with mock.patch.object(renderer, "_render_lens", side_effect=fake_render), mock.patch.object(
                renderer.np, "load", wraps=np.load
            ) as load_mock:
                with self.assertRaisesRegex(renderer.QCError, "instances path must be exactly"):
                    renderer.render_eye_sample(
                        eye_id="TEST_EYE",
                        bundle_root=bundle,
                        sampling_table=table,
                        output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
                    )
                load_mock.assert_not_called()

    def test_nonoverwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = make_minimal_bundle(root)
            output = bundle / renderer.SAMPLE_DIRECTORY_NAME
            output.mkdir()
            with self.assertRaisesRegex(renderer.QCError, "Refusing to overwrite"):
                renderer.render_eye_sample(
                    eye_id="TEST_EYE",
                    bundle_root=bundle,
                    sampling_table=bundle / "distal_qc_sampling.csv",
                    output_dir=output,
                )

    def test_static_np_load_calls_exist_only_in_two_allowlisted_loaders(self) -> None:
        tree = ast.parse(Path(renderer.__file__).read_text(encoding="utf-8"))
        parents: list[str] = []
        calls: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                parents.append(node.name)
                self.generic_visit(node)
                parents.pop()

            def visit_Call(self, node: ast.Call) -> None:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                ):
                    calls.append(parents[-1] if parents else "<module>")
                self.generic_visit(node)

        Visitor().visit(tree)
        self.assertEqual(
            calls,
            ["_load_instance_for_render", "_load_distal_for_render"],
        )


if __name__ == "__main__":
    unittest.main()

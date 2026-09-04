from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import experiment_64_robust_distal_core as robust_core
import attest_experiment_64_instance_qc as attester
import render_experiment_64_instance_qc_sample as renderer


EYES = (
    "M3_F_24_01",
    "M3_F_28_03",
    "M3_F_35_03",
    "M3_M_26_01",
    "M3_M_32_01",
    "M3_M_36_01",
    "RED3_25_F_36",
    "RED3_25_F_37",
    "RED3_25_F_38",
    "RED3_25_M_26",
    "RED3_25_M_27",
    "RED3_25_M_28",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def commit_all(root: Path, message: str) -> str:
    run_git(root, "add", ".")
    run_git(
        root,
        "-c",
        "user.name=QC Test",
        "-c",
        "user.email=qc@example.invalid",
        "commit",
        "-m",
        message,
    )
    return run_git(root, "rev-parse", "HEAD")


def make_prior_sample(eye_id: str, path: Path) -> None:
    samples = []
    for ordinal in range(32):
        seed_id = f"seed-{ordinal:04d}"
        token = f"experiment63_instance_qc_v1|{eye_id}|{ordinal}|{seed_id}".encode()
        samples.append(
            {
                "ordinal": ordinal,
                "eye_id": eye_id,
                "lens_index": ordinal,
                "seed_id": seed_id,
                "selection_sha256": hashlib.sha256(token).hexdigest(),
            }
        )
    write_json(
        path,
        {
            "schema_version": "experiment63.instance-qc-sample.v1",
            "eye_id": eye_id,
            "sampling_algorithm": "experiment63_instance_qc_v1",
            "n_selected": 32,
            "samples": samples,
        },
    )


def make_committed_stop_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    run_git(root, "init", "-q")
    (root / "initial.txt").write_text("initial\n", encoding="utf-8")
    frozen_commit = commit_all(root, "initial")

    prior: dict[str, Path] = {}
    stop_eyes = []
    for eye_id in EYES:
        path = root / "prior" / eye_id / "sample_manifest.json"
        make_prior_sample(eye_id, path)
        prior[eye_id] = path
        stop_eyes.append(
            {"eye_id": eye_id, "sample_manifest_sha256": sha256_path(path)}
        )
    stop_path = root / Path(*renderer.STOP_RECORD_RELPATH.parts)
    write_json(
        stop_path,
        {
            "schema_version": "experiment63.pre_outcome_stop.v1",
            "experiment": 63,
            "status": "stopped_before_model_outcome_evaluation",
            "frozen_git_commit": frozen_commit,
            "outcome_access": {
                "targets_opened_during_review": False,
                "predictions_opened": False,
                "errors_opened": False,
                "model_comparisons_opened": False,
                "primary_backend_executed": False,
            },
            "eyes": stop_eyes,
        },
    )
    commit_all(root, "stop record and prior manifests")
    ledger, _ = renderer.build_development_exclusion_ledger(
        repository_root=root,
        stop_record_path=stop_path,
        prior_sample_manifests=prior,
    )
    ledger_path = root / Path(*renderer.EXCLUSION_LEDGER_RELPATH.parts)
    write_json(ledger_path, ledger)
    commit_all(root, "freeze Experiment 64 development exclusions")
    return stop_path, prior


def write_instance(path: Path, lens_index: int) -> None:
    points = np.column_stack(
        [np.arange(50, dtype=np.int32), np.zeros(50, dtype=np.int32), np.zeros(50, dtype=np.int32)]
    )
    np.savez_compressed(
        path,
        schema_version=np.asarray(renderer.INSTANCE_SCHEMA_VERSION),
        lens_index=np.asarray(lens_index, dtype=np.int64),
        full_assigned_points_zyx=points,
        main_component_points_zyx=points[:45],
        raw_distal_points_zyx=points[:33],
        component_sizes_descending=np.asarray([45, 5], dtype=np.int64),
        spacing_um=np.asarray([0.325, 0.325, 0.325], dtype=np.float64),
        seed_source_zyx=np.asarray([0, 0, 0], dtype=np.int64),
        oda_axis_source_zyx=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        config_json=np.asarray("{}"),
        config_sha256=np.asarray(hashlib.sha256(b"{}").hexdigest()),
    )


def write_core(path: Path, lens_index: int) -> None:
    points = np.column_stack(
        [np.arange(27, dtype=np.int32), np.zeros(27, dtype=np.int32), np.zeros(27, dtype=np.int32)]
    )
    config_json = robust_core.canonical_robust_core_config_json()
    np.savez_compressed(
        path,
        schema_version=np.asarray(renderer.SEALED_CORE_SCHEMA_VERSION),
        lens_index=np.asarray(lens_index, dtype=np.int64),
        points_zyx=points,
        spacing_um=np.asarray([0.325, 0.325, 0.325], dtype=np.float64),
        config_json=np.asarray(config_json),
        config_sha256=np.asarray(robust_core.robust_core_config_sha256()),
        raw_distal_support=np.asarray(33, dtype=np.int64),
        robust_core_config_sha256=np.asarray(robust_core.robust_core_config_sha256()),
        robust_core_diagnostics_json=np.asarray("{}"),
    )


def hash_entry(path: Path) -> dict[str, object]:
    return {"sha256": sha256_path(path), "size_bytes": path.stat().st_size}


def coherence_values(lens_index: int) -> tuple[int, float]:
    support = 25 + lens_index % 5
    return support, (support - 25) / 25


def make_bundle(root: Path, *, eye_id: str = EYES[0], n: int = 64) -> Path:
    bundle = root / "bundle"
    (bundle / "instances").mkdir(parents=True)
    (bundle / "sealed_distal").mkdir()
    with (bundle / "distal_qc_sampling.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=renderer.SAMPLING_FIELDS)
        writer.writeheader()
        for lens_index in range(n):
            within_unseen = (lens_index - 32) % 8
            _, coherence_margin = coherence_values(lens_index)
            writer.writerow(
                {
                    "eye_id": eye_id,
                    "lens_index": lens_index,
                    "seed_id": f"seed-{lens_index:04d}",
                    "distal_eligible": "true",
                    "position_u_um": float(lens_index),
                    "position_v_um": 0.0,
                    "distal_scale_um": float(within_unseen + 1),
                    "coherence_margin": coherence_margin,
                    "instance_relpath": f"instances/lens_{lens_index:06d}.npz",
                    "sealed_distal_relpath": f"sealed_distal/lens_{lens_index:06d}.npz",
                }
            )
            write_instance(bundle / "instances" / f"lens_{lens_index:06d}.npz", lens_index)
            write_core(bundle / "sealed_distal" / f"lens_{lens_index:06d}.npz", lens_index)

    inventory_fields = attester.TECHNICAL_INVENTORY_FIELDS
    with (bundle / "technical_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields)
        writer.writeheader()
        for lens_index in range(n):
            support, coherence_margin = coherence_values(lens_index)
            scale = float((lens_index - 32) % 8 + 1)
            writer.writerow(
                {
                    "eye_id": eye_id,
                    "lens_index": lens_index,
                    "seed_id": f"seed-{lens_index:04d}",
                    "species": "test-species",
                    "sex": "unknown",
                    "assignment_status": "ok",
                    "full_assigned_size": 50,
                    "main_component_size": 45,
                    "component_removed_size": 5,
                    "main_component_fraction": 0.9,
                    "component_sizes_json": "[45,5]",
                    "component_fraction_gate_pass": "true",
                    "raw_distal_support": 33,
                    "robust_core_status": "pass",
                    "robust_core_reasons": "",
                    "robust_core_support": 27,
                    "robust_core_retained_fraction": 27 / 33,
                    "robust_core_diagnostics_json": "{}",
                    "distal_qc": "true",
                    "distal_qc_reasons": "",
                    "central": "true",
                    "position_u_um": float(lens_index),
                    "position_v_um": 0.0,
                    "distal_scale_um": scale,
                    "distal_fit_support": support,
                    "distal_fit_rmse_um": 1.0,
                    "distal_abs_residual_p95_um": 0.1,
                    "distal_abs_residual_p99_um": 0.1 * scale,
                    "distal_fit_p99_residual_over_scale": 0.1,
                    "distal_fit_26_component_count": 1,
                    "distal_fit_26_largest_component_support": support,
                    "distal_fit_26_largest_component_fraction": 1.0,
                    "maike_final_fit_gate_pass": "true",
                    "maike_final_fit_gate_reasons": "",
                    "distal_gradient_magnitude": 0.0,
                    "distal_curvature_eigenvalue_1": 0.0,
                    "distal_curvature_eigenvalue_2": 0.0,
                    "distal_normalized_fit_residual": 0.1,
                    "coherence_support_margin": coherence_margin,
                    "coherence_rmse_margin": 0.6,
                    "coherence_lcc_margin": 1.0,
                    "coherence_p99_over_scale_margin": (0.75 - 0.1) / 0.75,
                    "coherence_margin": coherence_margin,
                    "instance_relpath": f"instances/lens_{lens_index:06d}.npz",
                    "sealed_distal_relpath": f"sealed_distal/lens_{lens_index:06d}.npz",
                }
            )
    write_json(
        bundle / "distal_frame_audit.json",
        {"schema_version": "experiment64.distal-frame-audit.v1", "eye_id": eye_id},
    )
    technical_paths = {
        "technical_inventory.csv",
        "distal_qc_sampling.csv",
        "distal_frame_audit.json",
    }
    for lens_index in range(n):
        technical_paths.add(f"instances/lens_{lens_index:06d}.npz")
        technical_paths.add(f"sealed_distal/lens_{lens_index:06d}.npz")
    technical_manifest = {
        relpath: hash_entry(bundle / relpath) for relpath in sorted(technical_paths)
    }
    common = {
        "schema_version": renderer.TECHNICAL_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "experiment": 64,
        "eye_id": eye_id,
        "n_expected": n,
        "n_rows": n,
        "contiguous_indices": True,
        "instance_segmentation_validated": False,
        "robust_core_config": robust_core.ROBUST_CORE_CONFIG,
        "robust_core_config_sha256": robust_core.robust_core_config_sha256(),
        "technical_coherence_config": renderer.EXPECTED_TECHNICAL_COHERENCE_CONFIG,
        "technical_coherence_config_sha256": hashlib.sha256(
            json.dumps(
                renderer.EXPECTED_TECHNICAL_COHERENCE_CONFIG,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "technical_output_manifest": technical_manifest,
        # The file is intentionally absent.  Technical code may validate this
        # binding but must not stat or open the bound outcome artifact.
        "sealed_outcome_manifest_binding": {
            "relative_path": "sealed_outcomes/manifest.json",
            "sha256": "a" * 64,
            "size_bytes": 999,
        },
    }
    write_json(bundle / "technical_completion.json", common)
    write_json(bundle / "technical_provenance.json", common)
    return bundle


def fake_render(output_path: Path, **kwargs: object) -> None:
    selected = kwargs["selected"]
    output_path.write_bytes(f"PNG-{selected.row.lens_index}".encode())


def prepare_fixture(root: Path) -> tuple[Path, Path, dict[str, Path], Path]:
    stop, prior = make_committed_stop_fixture(root)
    bundle = make_bundle(root)
    with mock.patch.object(renderer, "_render_lens", side_effect=fake_render):
        manifest = renderer.render_eye_sample(
            eye_id=EYES[0],
            bundle_root=bundle,
            sampling_table=bundle / "distal_qc_sampling.csv",
            output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
            repository_root=root,
            stop_record_path=stop,
            prior_sample_manifests=prior,
        )
    return bundle, stop, prior, manifest


class Experiment64RendererTests(unittest.TestCase):
    def test_disjoint_fixed_sample_has_one_near_worst_and_one_hash_min_per_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _, _, manifest_path = prepare_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ledger = json.loads(
                (
                    Path(temporary) / Path(*renderer.EXCLUSION_LEDGER_RELPATH.parts)
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["sampling_algorithm"], "experiment64_instance_qc_v1")
        self.assertEqual(manifest["n_selected"], 32)
        self.assertEqual(set(manifest["cell_counts"].values()), {2})
        self.assertEqual(
            manifest["selection_role_counts"],
            {"near_worst_coherence": 16, "hash_minimal_remaining": 16},
        )
        self.assertEqual(manifest["selected_development_overlap_count"], 0)
        self.assertTrue(all(sample["lens_index"] >= 32 for sample in manifest["samples"]))
        self.assertEqual(ledger["n_eye_scoped_exclusions"], 384)
        self.assertEqual(len(ledger["eyes"]), 12)
        self.assertFalse(manifest["visual_disclosure"]["proximal_anatomy_blind"])
        self.assertTrue(manifest["visual_disclosure"]["raw_localized_distal_points_shown"])

    def test_tampered_prior_manifest_stops_before_any_npz_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop, prior = make_committed_stop_fixture(root)
            bundle = make_bundle(root)
            with prior[EYES[-1]].open("ab") as handle:
                handle.write(b"tamper")
            with mock.patch.object(renderer.np, "load", wraps=np.load) as load_mock:
                with self.assertRaisesRegex(renderer.QCError, "SHA-256 mismatch"):
                    renderer.render_eye_sample(
                        eye_id=EYES[0],
                        bundle_root=bundle,
                        sampling_table=bundle / "distal_qc_sampling.csv",
                        output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
                        repository_root=root,
                        stop_record_path=stop,
                        prior_sample_manifests=prior,
                    )
                load_mock.assert_not_called()

    def test_uncommitted_canonical_ledger_change_stops_before_npz_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop, prior = make_committed_stop_fixture(root)
            bundle = make_bundle(root)
            ledger_path = root / Path(*renderer.EXCLUSION_LEDGER_RELPATH.parts)
            ledger_path.write_text(ledger_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with mock.patch.object(renderer.np, "load", wraps=np.load) as load_mock:
                with self.assertRaisesRegex(renderer.QCError, "differs from HEAD"):
                    renderer.render_eye_sample(
                        eye_id=EYES[0],
                        bundle_root=bundle,
                        sampling_table=bundle / "distal_qc_sampling.csv",
                        output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
                        repository_root=root,
                        stop_record_path=stop,
                        prior_sample_manifests=prior,
                    )
                load_mock.assert_not_called()

    def test_core_outside_raw_distal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop, prior = make_committed_stop_fixture(root)
            bundle = make_bundle(root)
            rows, _ = renderer.read_sampling_table(bundle / "distal_qc_sampling.csv", eye_id=EYES[0])
            unseen, _ = renderer.exclude_development_identities(
                rows, {(i, f"seed-{i:04d}") for i in range(32)}, eye_id=EYES[0]
            )
            selected_lens = renderer.select_frozen_sample(unseen)[0].row.lens_index
            core_path = bundle / "sealed_distal" / f"lens_{selected_lens:06d}.npz"
            write_core(core_path, selected_lens)
            with np.load(core_path, allow_pickle=False) as archive:
                values = {name: np.asarray(archive[name]) for name in archive.files}
            bad_points = values["points_zyx"].copy()
            bad_points[-1] = [49, 0, 0]
            values["points_zyx"] = bad_points
            np.savez_compressed(core_path, **values)
            with mock.patch.object(renderer, "_render_lens", side_effect=fake_render):
                with self.assertRaisesRegex(renderer.QCError, "not a subset of raw distal"):
                    renderer.render_eye_sample(
                        eye_id=EYES[0],
                        bundle_root=bundle,
                        sampling_table=bundle / "distal_qc_sampling.csv",
                        output_dir=bundle / renderer.SAMPLE_DIRECTORY_NAME,
                        repository_root=root,
                        stop_record_path=stop,
                        prior_sample_manifests=prior,
                    )

    def test_static_np_load_calls_are_confined_to_two_allowlisted_loaders(self) -> None:
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
        self.assertEqual(calls, ["_load_instance_for_render", "_load_core_for_render"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import experiment_64_extract_lens_surfaces as maike64
import experiment_64_prepare_arthur_source_table as arthur64
import experiment_64_robust_distal_core as robust64
import experiment_64_two_pass_backend as backend
import attest_experiment_64_instance_qc as attester
import render_experiment_64_instance_qc_sample as renderer


def _binding(payload: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_json(path: Path, value: object) -> bytes:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _technical_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "lens_index": 0,
        "distal_qc": True,
        "central": False,
        "position_u_um": 0.0,
        "position_v_um": 1.0,
        "distal_scale_um": 6.0,
        "distal_fit_support": 25,
        "distal_fit_rmse_um": 0.25,
        "distal_abs_residual_p95_um": 0.4,
        "distal_abs_residual_p99_um": 0.5,
        "distal_gradient_magnitude": 0.1,
        "distal_curvature_eigenvalue_1": 0.01,
        "distal_curvature_eigenvalue_2": 0.02,
        "distal_normalized_fit_residual": 0.04,
        "coherence_margin": 0.0,
        "raw_distal_support": 33,
        "robust_core_status": "pass",
        "robust_core_support": 27,
        "robust_core_retained_fraction": 27.0 / 33.0,
        "distal_fit_p99_residual_over_scale": 0.5,
        "distal_fit_26_component_count": 1,
        "distal_fit_26_largest_component_support": 25,
        "distal_fit_26_largest_component_fraction": 1.0,
        "maike_final_fit_gate_pass": True,
        "maike_final_fit_gate_reasons": "",
        "coherence_support_margin": 0.0,
        "coherence_rmse_margin": 0.9,
        "coherence_lcc_margin": 1.0,
        "coherence_p99_over_scale_margin": 1.0 / 3.0,
    }
    row.update(changes)
    return row


def _maike_exclusion_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "assignment_status": "ok",
        "full_assigned_size": 100,
        "component_fraction_gate_pass": True,
        "robust_core_status": "pass",
        "robust_core_reasons": "",
        "distal_qc": True,
        "distal_qc_reasons": "",
        "maike_final_fit_gate_pass": True,
        "maike_final_fit_gate_reasons": "",
    }
    row.update(changes)
    return row


class RobustAndTechnicalContractTests(unittest.TestCase):
    @staticmethod
    def _biological_identity(eye_id: str) -> dict[str, object]:
        return {
            "species": (
                "Drosophila simulans"
                if eye_id.startswith("M3_")
                else "Drosophila mauritiana"
            ),
            "sex": "F" if "_F_" in eye_id else "M",
            "biological_independence": {
                "independent_unit": "animal",
                "animal_id": eye_id,
                "one_eye_per_animal_in_validation": True,
                "source_basis": "one supplied eye stack per uniquely named fly",
            },
        }

    def test_maike_biological_identity_is_bound_to_frozen_cohort(self) -> None:
        for eye_id in backend.EXPECTED_EYES:
            with self.subTest(eye_id=eye_id):
                expected_species, expected_sex = backend._validate_maike_biological_identity(
                    self._biological_identity(eye_id), eye_id
                )
                self.assertEqual(
                    (expected_species, expected_sex),
                    (
                        "Drosophila simulans"
                        if eye_id.startswith("M3_")
                        else "Drosophila mauritiana",
                        "F" if "_F_" in eye_id else "M",
                    ),
                )

        eye_id = "M3_F_24_01"
        for field, value, message in (
            ("species", "Drosophila mauritiana", "species/sex"),
            ("sex", "M", "species/sex"),
            (
                "biological_independence",
                {
                    "independent_unit": "eye",
                    "animal_id": eye_id,
                    "one_eye_per_animal_in_validation": True,
                    "source_basis": "one supplied eye stack per uniquely named fly",
                },
                "biological-independence",
            ),
        ):
            with self.subTest(field=field):
                document = self._biological_identity(eye_id)
                document[field] = value
                with self.assertRaisesRegex(backend.ContractError, message):
                    backend._validate_maike_biological_identity(document, eye_id)

        with self.assertRaisesRegex(backend.ContractError, "not in the frozen Maike cohort"):
            backend._validate_maike_biological_identity(
                self._biological_identity("M3_F_99_99"), "M3_F_99_99"
            )

    def test_csv_float_parser_preserves_exact_producer_round_trip(self) -> None:
        values = np.asarray(
            [
                np.nextafter(0.1, 0.0),
                0.1,
                np.nextafter(0.1, 1.0),
                np.nextafter(12345.6789012345, np.inf),
                np.finfo(np.float64).tiny,
            ],
            dtype=np.float64,
        )
        payload = ("value\n" + "\n".join(repr(float(value)) for value in values) + "\n").encode()
        parsed = backend._csv_table(payload, "synthetic exact-float table")
        np.testing.assert_array_equal(parsed["value"].to_numpy(np.float64), values)

    def test_backend_binds_exact_shared_robust_core(self) -> None:
        document = {
            "robust_core_config": dict(robust64.ROBUST_CORE_CONFIG),
            "robust_core_config_sha256": robust64.robust_core_config_sha256(),
        }
        backend._validate_robust_core_binding(document, "synthetic")
        document["robust_core_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(backend.ContractError, "robust-core hash"):
            backend._validate_robust_core_binding(document, "synthetic")

    def test_post_stage2_support_below_25_fails_closed(self) -> None:
        table = pd.DataFrame([_technical_row(distal_fit_support=24)])
        with self.assertRaisesRegex(backend.ContractError, "fit support below 25"):
            backend._validate_technical_rows(
                table,
                label="synthetic Maike",
                eligible_column="distal_qc",
                expected_rows=1,
            )

    def test_maike_modality_specific_gates_are_exact(self) -> None:
        for change, message in (
            ({"distal_fit_26_largest_component_fraction": 0.989}, "26-LCC"),
            ({"distal_fit_p99_residual_over_scale": 0.751}, "p99/scale"),
            ({"maike_final_fit_gate_pass": False}, "failed the Maike final-fit gate"),
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(backend.ContractError, message):
                    backend._validate_technical_rows(
                        pd.DataFrame([_technical_row(**change)]),
                        label="synthetic Maike",
                        eligible_column="distal_qc",
                        expected_rows=1,
                    )

    def test_backend_imports_producer_field_contracts(self) -> None:
        self.assertEqual(backend.ARTHUR_TECHNICAL_FIELDS, tuple(arthur64.TECHNICAL_FIELDS))
        self.assertEqual(backend._maike_technical_fields(maike64), tuple(maike64.TECHNICAL_FIELDS))
        self.assertEqual(backend._maike_target_fields(maike64), tuple(maike64.TARGET_FIELDS))
        self.assertEqual(backend.MAIKE_SAMPLING_FIELDS, tuple(maike64.DISTAL_QC_SAMPLING_FIELDS))


class TechnicalExclusionAccountingTests(unittest.TestCase):
    def test_assignment_status_is_bound_to_assigned_support(self) -> None:
        table = pd.DataFrame(
            [_maike_exclusion_row(assignment_status="empty_assignment")]
        )
        with self.assertRaisesRegex(backend.ContractError, "assignment_status"):
            backend._maike_technical_exclusion_report(
                table, "synthetic assignment inventory"
            )

    def test_reason_counts_ignore_empty_and_nan_cells(self) -> None:
        table = pd.DataFrame(
            {
                "reasons": [
                    "",
                    None,
                    np.nan,
                    np.float64("nan"),
                    "residual|connectivity",
                    "connectivity",
                ]
            }
        )
        self.assertEqual(
            backend._reason_counts(table, "reasons", "synthetic reasons"),
            {"connectivity": 2, "residual": 1},
        )

    def test_dual_final_fit_reason_is_one_unique_exclusion(self) -> None:
        connectivity = "fit_points_26_lcc_fraction_below_minimum"
        residual = "fit_abs_residual_p99_over_scale_above_maximum"
        table = pd.DataFrame(
            [
                _maike_exclusion_row(),
                _maike_exclusion_row(
                    distal_qc=False,
                    distal_qc_reasons=connectivity,
                    maike_final_fit_gate_pass=False,
                    maike_final_fit_gate_reasons=connectivity,
                ),
                _maike_exclusion_row(
                    distal_qc=False,
                    distal_qc_reasons=residual,
                    maike_final_fit_gate_pass=False,
                    maike_final_fit_gate_reasons=residual,
                ),
                _maike_exclusion_row(
                    distal_qc=False,
                    distal_qc_reasons=f"{connectivity}|{residual}",
                    maike_final_fit_gate_pass=False,
                    maike_final_fit_gate_reasons=f"{connectivity}|{residual}",
                ),
            ]
        )

        report = backend._maike_technical_exclusion_report(
            table, "synthetic Maike inventory"
        )

        self.assertEqual(report["fixed_oda_denominator"], 4)
        self.assertEqual(report["base_distal_qc_retained"], 4)
        self.assertEqual(report["maike_final_fit_connectivity_excluded"], 2)
        self.assertEqual(report["maike_final_fit_residual_excluded"], 2)
        self.assertEqual(report["maike_final_fit_both_excluded"], 1)
        self.assertEqual(report["maike_final_fit_excluded"], 3)
        self.assertEqual(report["distal_qc_retained"], 1)
        self.assertEqual(report["distal_qc_excluded"], 3)
        self.assertEqual(
            report["maike_final_fit_reason_counts"],
            {connectivity: 2, residual: 2},
        )
        self.assertEqual(
            report["maike_final_fit_excluded"],
            report["maike_final_fit_connectivity_excluded"]
            + report["maike_final_fit_residual_excluded"]
            - report["maike_final_fit_both_excluded"],
        )

    def test_contradictory_pass_and_reason_cells_fail_closed(self) -> None:
        connectivity = "fit_points_26_lcc_fraction_below_minimum"
        contradictions = (
            (
                {"robust_core_reasons": "too_few_unique_points"},
                "robust pass/reasons disagree",
            ),
            (
                {"distal_qc_reasons": "scale_out_of_range"},
                "distal-QC pass/reasons disagree",
            ),
            (
                {"maike_final_fit_gate_reasons": connectivity},
                "final-fit pass/reasons disagree",
            ),
        )
        for changes, message in contradictions:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(backend.ContractError, message):
                    backend._maike_technical_exclusion_report(
                        pd.DataFrame([_maike_exclusion_row(**changes)]),
                        "contradictory Maike inventory",
                    )


class LeakageBarrierTests(unittest.TestCase):
    def test_all_predictor_invariant_geometry_is_completed_in_pass1(self) -> None:
        arthur_loader = inspect.getsource(backend._load_arthur_technical_bundle)
        maike_loader = inspect.getsource(backend._load_maike_technical_bundle)
        pass2 = inspect.getsource(backend.run_pass2)

        self.assertIn("add_invariant_features_by_unit", arthur_loader)
        self.assertIn("add_invariant_features_by_unit", maike_loader)
        self.assertNotIn("add_invariant_features_by_unit", pass2)

    def test_cli_reports_reused_experiment63_contract_failure_without_traceback(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                backend,
                "execute_experiment64",
                side_effect=backend.experiment63.ContractError("synthetic reused failure"),
            ),
            mock.patch("sys.stderr", stderr),
        ):
            status = backend.main(
                [
                    "--repo",
                    "/synthetic/repo",
                    "--expected-commit",
                    "a" * 40,
                    "--arthur-root",
                    "/synthetic/arthur",
                    "--maike-root",
                    "/synthetic/maike",
                    "--output-directory",
                    "/synthetic/output",
                    "--execute-sealed-first-experiment64",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("CONTRACT FAILURE: synthetic reused failure", stderr.getvalue())

    def test_guarded_reader_rejects_outcome_path_in_pass1_before_open(self) -> None:
        opened: list[Path] = []

        def tripwire(path: Path) -> bytes:
            opened.append(path)
            raise AssertionError("outcome tripwire opened")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sealed_outcomes/manifest.json"
            path.parent.mkdir()
            path.write_bytes(b"OUTCOME-CANARY")
            reader = backend.GuardedArtifactReader(byte_opener=tripwire)
            with self.assertRaisesRegex(backend.ContractError, "refused outcome path"):
                reader.read(
                    root,
                    "sealed_outcomes/manifest.json",
                    phase="pass1",
                    purpose="forbidden synthetic outcome",
                )
        self.assertEqual(opened, [])

    def test_guarded_reader_rejects_a_symlinked_root_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            real_root = temporary / "real"
            real_root.mkdir()
            (real_root / "technical.json").write_text("{}", encoding="utf-8")
            linked_root = temporary / "linked"
            linked_root.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(backend.ContractError, "Symlinked artifact root"):
                backend.GuardedArtifactReader().read(
                    linked_root,
                    "technical.json",
                    phase="pass1",
                    purpose="synthetic technical document",
                )

    def test_pass1_rejects_symlinked_maike_parent_before_eye_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repository = temporary / "repository"
            arthur = temporary / "arthur"
            real_maike = temporary / "real-maike"
            for path in (repository, arthur, real_maike):
                path.mkdir()
            linked_maike = temporary / "maike"
            linked_maike.symlink_to(real_maike, target_is_directory=True)
            with self.assertRaisesRegex(backend.ContractError, "symlinked Maike bundle root"):
                backend.run_pass1(
                    repository_root=repository,
                    expected_commit="a" * 40,
                    arthur_root=arthur,
                    maike_root=linked_maike,
                    git_validator=lambda _root, expected: expected,
                )

    def test_twelfth_eye_failure_never_opens_any_outcome_or_arthur(self) -> None:
        commit = "a" * 40
        opened: list[Path] = []
        visited: list[str] = []

        def tripwire_opener(path: Path) -> bytes:
            opened.append(path)
            if "sealed_outcomes" in path.parts:
                raise AssertionError(f"outcome canary opened: {path}")
            return path.read_bytes()

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repository = temporary / "repo"
            maike_root = temporary / "maike"
            arthur_root = temporary / "arthur"
            repository.mkdir()
            arthur_root.mkdir()
            for eye_id in backend.EXPECTED_EYES:
                root = maike_root / eye_id
                (root / "sealed_outcomes").mkdir(parents=True)
                (root / "technical-marker").write_bytes(b"technical")
                (root / "sealed_outcomes/manifest.json").write_bytes(
                    b"UNREADABLE-OUTCOME-CANARY"
                )

            reader = backend.GuardedArtifactReader(byte_opener=tripwire_opener)

            def fake_maike_loader(
                guarded: backend.GuardedArtifactReader,
                root: Path,
                eye_id: str,
                expected_rows: int,
                expected_commit: str,
                producer: object,
                repository_root: Path,
            ) -> backend.TechnicalBundle:
                del expected_rows, expected_commit, producer, repository_root
                visited.append(eye_id)
                guarded.read(
                    root,
                    "technical-marker",
                    phase="pass1",
                    purpose=f"{eye_id} synthetic technical marker",
                )
                if eye_id == tuple(backend.EXPECTED_EYES)[-1]:
                    raise backend.ContractError("synthetic twelfth-eye QC failure")
                canary = root / "sealed_outcomes/manifest.json"
                return backend.TechnicalBundle(
                    kind="maike",
                    unit_id=eye_id,
                    root=root,
                    table=pd.DataFrame(),
                    completion_sha256="1" * 64,
                    provenance_sha256="2" * 64,
                    technical_manifest={},
                    outcome_manifest=backend.DeferredArtifact(
                        root=root,
                        relative_path="sealed_outcomes/manifest.json",
                        sha256=hashlib.sha256(canary.read_bytes()).hexdigest(),
                        size_bytes=canary.stat().st_size,
                        owner=eye_id,
                    ),
                )

            with (
                mock.patch.object(backend, "_maike_producer_module", return_value=object()),
                mock.patch.object(backend, "_load_maike_technical_bundle", side_effect=fake_maike_loader),
                mock.patch.object(backend, "_load_arthur_technical_bundle") as arthur_loader,
            ):
                with self.assertRaisesRegex(backend.ContractError, "twelfth-eye QC failure"):
                    backend.run_pass1(
                        repository_root=repository,
                        expected_commit=commit,
                        arthur_root=arthur_root,
                        maike_root=maike_root,
                        reader=reader,
                        git_validator=lambda _root, expected: expected,
                    )
                arthur_loader.assert_not_called()

        self.assertEqual(visited, list(backend.EXPECTED_EYES))
        self.assertEqual(len(opened), 12)
        self.assertTrue(all("sealed_outcomes" not in path.parts for path in opened))
        self.assertTrue(all(event.phase == "pass1" for event in reader.events))
        self.assertFalse(any("sealed_outcomes" in Path(event.path).parts for event in reader.events))

    def test_forged_clearance_cannot_enter_pass2(self) -> None:
        dummy = backend.TechnicalBundle(
            kind="arthur",
            unit_id="arthur_source",
            root=Path("/synthetic"),
            table=pd.DataFrame(),
            completion_sha256="0" * 64,
            provenance_sha256="0" * 64,
            technical_manifest={},
            outcome_manifest=backend.DeferredArtifact(
                root=Path("/synthetic"),
                relative_path="sealed_outcomes/manifest.json",
                sha256="0" * 64,
                size_bytes=0,
                owner="synthetic",
            ),
        )
        forged = backend.Pass1Clearance(
            schema_version=backend.PASS1_SCHEMA,
            analysis_label=backend.ANALYSIS_LABEL,
            expected_commit="a" * 40,
            robust_core_config_sha256=backend.ROBUST_CORE_CONFIG_SHA256,
            maike=(),
            arthur=dummy,
            pass1_reads=(),
            _authority=object(),
        )
        with self.assertRaisesRegex(backend.ContractError, "in-process clearance"):
            backend.run_pass2(forged)

    def test_pass2_resolution_failure_stops_before_any_model_fit(self) -> None:
        expected_commit = "a" * 40
        eye_ids = tuple(backend.EXPECTED_EYES)
        synthetic_counts = {eye_id: 5 for eye_id in eye_ids}

        def deferred(owner: str) -> backend.DeferredArtifact:
            return backend.DeferredArtifact(
                root=Path("/synthetic") / owner,
                relative_path="sealed_outcomes/manifest.json",
                sha256="0" * 64,
                size_bytes=0,
                owner=owner,
            )

        maike_bundles: list[backend.TechnicalBundle] = []
        for eye_id in eye_ids:
            table = pd.DataFrame(
                [
                    {
                        "eye_id": eye_id,
                        "lens_index": lens_index,
                        **_maike_exclusion_row(),
                    }
                    for lens_index in range(5)
                ]
            )
            maike_bundles.append(
                backend.TechnicalBundle(
                    kind="maike",
                    unit_id=eye_id,
                    root=Path("/synthetic") / eye_id,
                    table=table,
                    completion_sha256="1" * 64,
                    provenance_sha256="2" * 64,
                    technical_manifest={},
                    outcome_manifest=deferred(eye_id),
                    qc_binding_sha256="3" * 64,
                    technical_exclusions=backend._maike_technical_exclusion_report(
                        table, f"synthetic {eye_id}"
                    ),
                )
            )

        arthur_table = pd.DataFrame(
            [
                {
                    "volume": "synthetic-volume",
                    "eye_id": 0,
                    "lens_index": 0,
                    "distal_qc": True,
                }
            ]
        )
        arthur_bundle = backend.TechnicalBundle(
            kind="arthur",
            unit_id="arthur_source",
            root=Path("/synthetic/arthur"),
            table=arthur_table,
            completion_sha256="4" * 64,
            provenance_sha256="5" * 64,
            technical_manifest={},
            outcome_manifest=deferred("arthur_source"),
            technical_exclusions={"fixed_source_denominator": 1},
        )
        clearance = backend.Pass1Clearance(
            schema_version=backend.PASS1_SCHEMA,
            analysis_label=backend.ANALYSIS_LABEL,
            expected_commit=expected_commit,
            robust_core_config_sha256=backend.ROBUST_CORE_CONFIG_SHA256,
            maike=tuple(maike_bundles),
            arthur=arthur_bundle,
            pass1_reads=(),
            _authority=backend._CLEARANCE_AUTHORITY,
        )
        with self.assertRaisesRegex(backend.ContractError, "durable outcome attempt"):
            backend.run_pass2(clearance)
        opened: list[str] = []

        def synthetic_outcomes(
            reader: backend.GuardedArtifactReader,
            technical: backend.TechnicalBundle,
            commit: str,
            producer: object,
        ) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, np.ndarray]]]:
            del reader, commit, producer
            opened.append(technical.unit_id)
            if technical.kind == "arthur":
                return (
                    pd.DataFrame(
                        [
                            {
                                "volume": "synthetic-volume",
                                "eye_id": 0,
                                "lens_index": 0,
                                "target_resolvable": True,
                                "target_resolvability_reasons": "",
                                "target_qc": True,
                                "target_qc_reasons": "",
                            }
                        ]
                    ),
                    {},
                )
            resolvable = [True, True, True, False, False]
            return (
                pd.DataFrame(
                    [
                        {
                            "eye_id": technical.unit_id,
                            "lens_index": lens_index,
                            "target_resolvable": is_resolvable,
                            "target_resolvability_reasons": (
                                "" if is_resolvable else "synthetic_unresolvable"
                            ),
                            "target_qc": is_resolvable,
                            "target_qc_reasons": (
                                "" if is_resolvable else "synthetic_unresolvable"
                            ),
                        }
                        for lens_index, is_resolvable in enumerate(resolvable)
                    ]
                ),
                {},
            )

        with (
            mock.patch.object(backend, "EXPECTED_EYES", synthetic_counts),
            mock.patch.object(backend, "EXPECTED_TOTAL", 60),
            mock.patch.object(backend, "EXPECTED_ARTHUR_ROWS", 1),
            mock.patch.object(backend, "_assert_outcome_attempt_authorization"),
            mock.patch.object(backend, "_maike_producer_module", return_value=object()),
            mock.patch.object(
                backend, "_load_outcome_bundle", side_effect=synthetic_outcomes
            ),
            mock.patch.object(
                backend.experiment63,
                "validate_lens_table",
                side_effect=lambda table, **_kwargs: table,
            ),
            mock.patch.object(
                backend.experiment63,
                "add_invariant_features_by_unit",
                side_effect=lambda table, *_args, **_kwargs: table,
            ),
            mock.patch.object(backend.experiment63, "run_primary_models") as primary,
            mock.patch.object(
                backend.experiment63, "run_target_qc_sensitivity_models"
            ) as sensitivity,
            mock.patch.object(
                backend.experiment63, "run_within_maike_nested_loao_secondary"
            ) as secondary,
            mock.patch.object(backend, "_score_with_loaded_raw_targets") as scorer,
        ):
            with self.assertRaisesRegex(
                backend.experiment63.ContractError,
                r"3/5 distal-QC and target-resolvable lenses.*requires 4",
            ):
                backend.run_pass2(clearance)

        self.assertEqual(opened, ["arthur_source", eye_ids[0]])
        primary.assert_not_called()
        sensitivity.assert_not_called()
        secondary.assert_not_called()
        scorer.assert_not_called()


class VisualQCContractTests(unittest.TestCase):
    def _fixture(
        self, root: Path, eye_id: str
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        dict[str, dict[str, object]],
        dict[str, object],
        Path,
    ]:
        sample_root = root / renderer.SAMPLE_DIRECTORY_NAME
        render_root = sample_root / "renders"
        render_root.mkdir(parents=True)
        code_root = root / "experiments/maike-modern-ground-truth"
        code_root.mkdir(parents=True)
        renderer_copy = code_root / "render_experiment_64_instance_qc_sample.py"
        attester_copy = code_root / "attest_experiment_64_instance_qc.py"
        renderer_copy.write_bytes(Path(renderer.__file__).read_bytes())
        attester_copy.write_bytes(Path(attester.__file__).read_bytes())
        robust_config = dict(robust64.ROBUST_CORE_CONFIG)
        robust_hash = robust64.robust_core_config_sha256()
        technical_rows: list[dict[str, object]] = []
        sampling_rows: list[dict[str, object]] = []
        technical_manifest: dict[str, dict[str, object]] = {}
        for lens_index in range(64):
            old = lens_index < 32
            new_index = lens_index - 32
            seed_id = (
                f"old-{eye_id}-{lens_index}" if old else f"seed-{lens_index}"
            )
            position_u = float(100 + lens_index if old else new_index + 1)
            position_v = 0.0
            scale = float(1 + (lens_index % 8))
            coherence = float(lens_index) / 100.0
            instance_relpath = f"instances/lens_{lens_index:06d}.npz"
            core_relpath = f"sealed_distal/lens_{lens_index:06d}.npz"
            technical_rows.append(
                {
                    "lens_index": lens_index,
                    "seed_id": seed_id,
                    "full_assigned_size": 100,
                    "main_component_size": 100,
                    "raw_distal_support": 33,
                    "robust_core_support": 27,
                }
            )
            sampling_rows.append(
                {
                    "eye_id": eye_id,
                    "lens_index": lens_index,
                    "seed_id": seed_id,
                    "distal_eligible": True,
                    "position_u_um": position_u,
                    "position_v_um": position_v,
                    "distal_scale_um": scale,
                    "coherence_margin": coherence,
                    "instance_relpath": instance_relpath,
                    "sealed_distal_relpath": core_relpath,
                }
            )
            technical_manifest[instance_relpath] = _binding(
                f"instance-{lens_index}".encode("ascii")
            )
            technical_manifest[core_relpath] = _binding(
                f"core-{lens_index}".encode("ascii")
            )

        sampling = pd.DataFrame(sampling_rows, columns=backend.MAIKE_SAMPLING_FIELDS)
        replay_rows = [
            renderer.SamplingRow(
                eye_id=str(row["eye_id"]),
                lens_index=int(row["lens_index"]),
                seed_id=str(row["seed_id"]),
                distal_eligible=True,
                position_u_um=float(row["position_u_um"]),
                position_v_um=float(row["position_v_um"]),
                distal_scale_um=float(row["distal_scale_um"]),
                coherence_margin=float(row["coherence_margin"]),
                instance_relpath=str(row["instance_relpath"]),
                sealed_distal_relpath=str(row["sealed_distal_relpath"]),
            )
            for row in sampling_rows
        ]
        old_identities = {
            (index, f"old-{eye_id}-{index}") for index in range(32)
        }
        unseen, eligible_removed = renderer.exclude_development_identities(
            replay_rows, old_identities, eye_id=eye_id
        )
        selected_rows = renderer.select_frozen_sample(unseen)
        samples: list[dict[str, object]] = []
        for ordinal, selected in enumerate(selected_rows):
            lens_index = selected.row.lens_index
            seed_id = selected.row.seed_id
            render_payload = f"render-{ordinal}".encode("ascii")
            render_name = (
                f"sample_{ordinal:02d}_r{selected.radial_stratum}_"
                f"s{selected.scale_stratum}_lens_{lens_index:06d}.png"
            )
            render_path = render_root / render_name
            render_path.write_bytes(render_payload)
            samples.append(
                {
                    "ordinal": ordinal,
                    "eye_id": eye_id,
                    "lens_index": lens_index,
                    "seed_id": seed_id,
                    "radius_um": selected.row.radius_um,
                    "distal_scale_um": selected.row.distal_scale_um,
                    "coherence_margin": selected.row.coherence_margin,
                    "radial_rank_after_exclusion": selected.radial_rank,
                    "radial_stratum": selected.radial_stratum,
                    "scale_rank_within_radial_after_exclusion": selected.scale_rank_within_radial,
                    "scale_stratum": selected.scale_stratum,
                    "selection_role": selected.selection_role,
                    "selection_sha256": selected.selection_sha256,
                    "instance_artifact": {
                        "relative_path": selected.row.instance_relpath,
                        **technical_manifest[selected.row.instance_relpath],
                        "members_present": sorted(renderer.INSTANCE_MEMBERS),
                        "members_accessed": list(renderer.INSTANCE_ACCESSED_MEMBERS),
                    },
                    "sealed_robust_core_artifact": {
                        "relative_path": selected.row.sealed_distal_relpath,
                        **technical_manifest[selected.row.sealed_distal_relpath],
                        "members_present": sorted(renderer.SEALED_CORE_MEMBERS),
                        "members_accessed": list(renderer.SEALED_CORE_ACCESSED_MEMBERS),
                    },
                    "render": {
                        "relative_path": f"renders/{render_path.name}",
                        **_binding(render_payload),
                    },
                    "point_counts": {
                        "full_assigned": 100,
                        "dominant_component": 100,
                        "raw_localized_distal": 33,
                        "final_robust_distal_core": 27,
                    },
                }
            )
        ledger_eyes = []
        for ledger_eye in backend.EXPECTED_EYES:
            ledger_eyes.append(
                {
                    "eye_id": ledger_eye,
                    "n_excluded": 32,
                    "old_sample_manifest_sha256": "d" * 64,
                    "identities": [
                        {
                            "old_ordinal": ordinal,
                            "lens_index": ordinal,
                            "seed_id": f"old-{ledger_eye}-{ordinal}",
                            "old_selection_sha256": "b" * 64,
                        }
                        for ordinal in range(32)
                    ],
                }
            )
        stop_path = root / Path(*renderer.STOP_RECORD_RELPATH.parts)
        stop_bytes = _write_json(stop_path, {"synthetic": "pre-outcome stop"})
        ledger = {
            "schema_version": renderer.EXCLUSION_SCHEMA_VERSION,
            "n_eyes": 12,
            "n_excluded_per_eye": 32,
            "n_eye_scoped_exclusions": 384,
            "eyes": ledger_eyes,
            "experiment63_stop_record": {
                "relative_path": renderer.STOP_RECORD_RELPATH.as_posix(),
                **_binding(stop_bytes),
            },
        }
        ledger_path = root / Path(*renderer.EXCLUSION_LEDGER_RELPATH.parts)
        ledger_bytes = _write_json(ledger_path, ledger)
        expected_commit = "a" * 40
        ledger_binding = {
            "relative_path": renderer.EXCLUSION_LEDGER_RELPATH.as_posix(),
            **_binding(ledger_bytes),
            "tracked_at_head": True,
            "head_commit": expected_commit,
        }
        sample = {
            "schema_version": renderer.SCHEMA_VERSION,
            "experiment": 64,
            "eye_id": eye_id,
            "sampling_algorithm": renderer.SAMPLING_ALGORITHM,
            "review_scope": "disjoint_stratified_sample_only",
            "all_instances_manually_reviewed": False,
            "n_inventory_rows": 64,
            "n_distal_qc_eligible_before_development_exclusion": 64,
            "n_development_identities_for_eye": 32,
            "n_eligible_development_identities_removed": eligible_removed,
            "n_unseen_distal_qc_eligible": len(unseen),
            "n_selected": 32,
            "selected_development_overlap_count": 0,
            "cell_counts": {
                f"r{radial}_s{scale}": 2
                for radial in range(4)
                for scale in range(4)
            },
            "selection_role_counts": {
                "near_worst_coherence": 16,
                "hash_minimal_remaining": 16,
            },
            "robust_core_config": robust_config,
            "robust_core_config_sha256": robust_hash,
            "outcome_sequestration": {
                "sampling_table_exact_field_allowlist": list(backend.MAIKE_SAMPLING_FIELDS),
                "target_proximal_prediction_error_model_or_sealed_outcome_opened": False,
            },
            "development_exclusion_ledger": ledger_binding,
            "renderer_code": {
                "relative_path": "experiments/maike-modern-ground-truth/render_experiment_64_instance_qc_sample.py",
                **_binding(renderer_copy.read_bytes()),
            },
            "prior_sample_manifest_verification": [
                {
                    "eye_id": ledger_eye,
                    "sha256": "d" * 64,
                    "n_verified_identities": 32,
                }
                for ledger_eye in backend.EXPECTED_EYES
            ],
            "samples": samples,
        }
        sample_path = sample_root / "sample_manifest.json"
        sample_bytes = _write_json(sample_path, sample)
        review = {
            "schema_version": attester.REVIEW_SCHEMA_VERSION,
            "eye_id": eye_id,
            "review_scope": "disjoint_stratified_sample_only",
            "review_mode": "ai_assisted_visual_review_without_model_outputs",
            "reviewer_id": "synthetic-reviewer",
            "reviewed_at_utc": "2026-01-01T00:00:00Z",
            "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "decisions": [
                {
                    "lens_index": entry["lens_index"],
                    "seed_id": entry["seed_id"],
                    "decision": "pass",
                    "notes": "",
                }
                for entry in samples
            ],
        }
        review_path = root / attester.REVIEW_FILENAME
        review_bytes = _write_json(review_path, review)
        outcome_binding = {
            "relative_path": "sealed_outcomes/manifest.json",
            "sha256": "c" * 64,
            "size_bytes": 123,
        }
        attestation = {
            "schema_version": attester.ATTESTATION_SCHEMA_VERSION,
            "experiment": 64,
            "status": "passed",
            "eye_id": eye_id,
            "review_scope": "disjoint_stratified_sample_only",
            "stratified_sample_visual_qc_passed": True,
            "whole_run_stop_rule_satisfied_for_this_eye": True,
            "technical_inventory_complete": True,
            "target_free_artifact_hash_verification_passed": True,
            "development_exclusion_verification": {
                "prior_sample_manifests_verified": 12,
                "eye_scoped_identities_verified": 384,
                "identities_for_this_eye": 32,
                "selected_overlap_count": 0,
                "exclusion_before_new_strata": True,
            },
            "reviewed_samples": [
                {
                    "ordinal": ordinal,
                    "lens_index": entry["lens_index"],
                    "seed_id": entry["seed_id"],
                    "decision": "pass",
                }
                for ordinal, entry in enumerate(samples)
            ],
            "outcome_sequestration": {
                "technical_pass_only": True,
                "sealed_outcome_manifest_binding_validated_but_file_not_opened": True,
                "target_proximal_prediction_error_model_or_sealed_outcome_opened": False,
                "model_and_error_blind": True,
            },
            "bindings": {
                "sample_manifest": {
                    "sha256": hashlib.sha256(sample_bytes).hexdigest(),
                },
                "review": {"sha256": hashlib.sha256(review_bytes).hexdigest()},
                "sealed_outcome_manifest": outcome_binding,
                "development_exclusion_ledger": ledger_binding,
            },
            "review": {
                "review_mode": "ai_assisted_visual_review_without_model_outputs",
                "reviewer_id": "synthetic-reviewer",
                "reviewed_at_utc": "2026-01-01T00:00:00Z",
            },
            "attester_code": {
                "relative_path": "attest_experiment_64_instance_qc.py",
                **_binding(attester_copy.read_bytes()),
            },
        }
        _write_json(root / attester.ATTESTATION_FILENAME, attestation)
        return (
            pd.DataFrame(technical_rows),
            sampling,
            technical_manifest,
            outcome_binding,
            code_root,
        )

    def _validate_fixture(
        self,
        *,
        root: Path,
        eye_id: str,
        table: pd.DataFrame,
        sampling: pd.DataFrame,
        technical_manifest: dict[str, dict[str, object]],
        outcome_binding: dict[str, object],
        code_root: Path,
        reader: backend.GuardedArtifactReader | None = None,
    ) -> str:
        with (
            mock.patch.object(renderer, "__file__", str(code_root / "render_experiment_64_instance_qc_sample.py")),
            mock.patch.object(attester, "__file__", str(code_root / "attest_experiment_64_instance_qc.py")),
        ):
            return backend._validate_maike_qc(
                reader=reader or backend.GuardedArtifactReader(),
                bundle_root=root,
                eye_id=eye_id,
                repository_root=root,
                technical_table=table,
                sampling_table=sampling,
                technical_manifest=technical_manifest,
                robust_core_config=robust64.ROBUST_CORE_CONFIG,
                robust_core_hash=robust64.robust_core_config_sha256(),
                outcome_manifest_binding=outcome_binding,
                expected_commit="a" * 40,
            )

    def test_disjoint_qc_bundle_passes_without_outcome_read(self) -> None:
        eye_id = tuple(backend.EXPECTED_EYES)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table, sampling, technical_manifest, outcome_binding, code_root = self._fixture(
                root, eye_id
            )
            reader = backend.GuardedArtifactReader()
            digest = self._validate_fixture(
                root=root,
                eye_id=eye_id,
                table=table,
                sampling=sampling,
                technical_manifest=technical_manifest,
                outcome_binding=outcome_binding,
                code_root=code_root,
                reader=reader,
            )
            self.assertEqual(len(digest), 64)
            self.assertFalse(any("sealed_outcomes" in Path(event.path).parts for event in reader.events))
            self.assertTrue(
                any(event.purpose.endswith("QC renderer frozen implementation") for event in reader.events)
            )
            self.assertTrue(
                any(event.purpose.endswith("QC attester frozen implementation") for event in reader.events)
            )

    def test_old_sample_overlap_stops_eye(self) -> None:
        eye_id = tuple(backend.EXPECTED_EYES)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table, sampling, technical_manifest, outcome_binding, code_root = self._fixture(
                root, eye_id
            )
            sample_path = root / renderer.SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
            sample = json.loads(sample_path.read_text())
            sample["samples"][0]["lens_index"] = 0
            sample["samples"][0]["seed_id"] = "overlap-seed"
            _write_json(sample_path, sample)
            with self.assertRaisesRegex(backend.ContractError, "deterministic reselection"):
                self._validate_fixture(
                    root=root,
                    eye_id=eye_id,
                    table=table,
                    sampling=sampling,
                    technical_manifest=technical_manifest,
                    outcome_binding=outcome_binding,
                    code_root=code_root,
                )

    def test_failed_review_stops_eye(self) -> None:
        eye_id = tuple(backend.EXPECTED_EYES)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table, sampling, technical_manifest, outcome_binding, code_root = self._fixture(
                root, eye_id
            )
            review_path = root / attester.REVIEW_FILENAME
            review = json.loads(review_path.read_text())
            review["decisions"][-1]["decision"] = "indeterminate"
            _write_json(review_path, review)
            with self.assertRaisesRegex(backend.ContractError, "decision 31 is not pass"):
                self._validate_fixture(
                    root=root,
                    eye_id=eye_id,
                    table=table,
                    sampling=sampling,
                    technical_manifest=technical_manifest,
                    outcome_binding=outcome_binding,
                    code_root=code_root,
                )

    def test_hash_consistent_rank_tamper_fails_deterministic_replay(self) -> None:
        eye_id = tuple(backend.EXPECTED_EYES)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table, sampling, technical_manifest, outcome_binding, code_root = self._fixture(
                root, eye_id
            )
            sample_path = root / renderer.SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
            review_path = root / attester.REVIEW_FILENAME
            attestation_path = root / attester.ATTESTATION_FILENAME
            sample = json.loads(sample_path.read_text())
            sample["samples"][0]["radial_rank_after_exclusion"] += 1
            sample_bytes = _write_json(sample_path, sample)
            review = json.loads(review_path.read_text())
            review["sample_manifest_sha256"] = hashlib.sha256(sample_bytes).hexdigest()
            review_bytes = _write_json(review_path, review)
            attestation = json.loads(attestation_path.read_text())
            attestation["bindings"]["sample_manifest"]["sha256"] = hashlib.sha256(
                sample_bytes
            ).hexdigest()
            attestation["bindings"]["review"]["sha256"] = hashlib.sha256(
                review_bytes
            ).hexdigest()
            _write_json(attestation_path, attestation)
            with self.assertRaisesRegex(backend.ContractError, "deterministic reselection"):
                self._validate_fixture(
                    root=root,
                    eye_id=eye_id,
                    table=table,
                    sampling=sampling,
                    technical_manifest=technical_manifest,
                    outcome_binding=outcome_binding,
                    code_root=code_root,
                )

    def test_renderer_and_attester_code_bindings_are_not_trusted(self) -> None:
        eye_id = tuple(backend.EXPECTED_EYES)[0]
        for target, message in (
            ("renderer", "QC renderer"),
            ("attester", "QC attester"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                table, sampling, technical_manifest, outcome_binding, code_root = self._fixture(
                    root, eye_id
                )
                if target == "renderer":
                    path = root / renderer.SAMPLE_DIRECTORY_NAME / "sample_manifest.json"
                    document = json.loads(path.read_text())
                    document["renderer_code"]["sha256"] = "0" * 64
                else:
                    path = root / attester.ATTESTATION_FILENAME
                    document = json.loads(path.read_text())
                    document["attester_code"]["sha256"] = "0" * 64
                _write_json(path, document)
                with self.assertRaisesRegex(backend.ContractError, message):
                    self._validate_fixture(
                        root=root,
                        eye_id=eye_id,
                        table=table,
                        sampling=sampling,
                        technical_manifest=technical_manifest,
                        outcome_binding=outcome_binding,
                        code_root=code_root,
                    )


class AnalysisBoundaryTests(unittest.TestCase):
    def test_target_reason_cells_are_bound_to_recomputed_quality_gates(self) -> None:
        row = pd.Series(
            {
                "target_resolvability_reasons": "",
                "target_qc_reasons": "target_q05_raw_thickness_not_positive",
            }
        )
        backend._validate_target_reason_row(
            row,
            label="synthetic target",
            expected_resolvable=True,
            expected_target_qc=False,
            q05=-0.1,
            rmse=0.5,
        )
        row["target_qc_reasons"] = ""
        with self.assertRaisesRegex(backend.ContractError, "reason emptiness"):
            backend._validate_target_reason_row(
                row,
                label="synthetic target",
                expected_resolvable=True,
                expected_target_qc=False,
                q05=-0.1,
                rmse=0.5,
            )

    def test_unrecognized_target_reason_is_rejected(self) -> None:
        row = pd.Series(
            {
                "target_resolvability_reasons": "invented_reason",
                "target_qc_reasons": "invented_reason",
            }
        )
        with self.assertRaisesRegex(backend.ContractError, "unrecognized"):
            backend._validate_target_reason_row(
                row,
                label="synthetic target",
                expected_resolvable=False,
                expected_target_qc=False,
                q05=np.nan,
                rmse=np.nan,
            )

    def test_maike_no_spanning_bins_reason_is_accepted(self) -> None:
        row = pd.Series(
            {
                "target_resolvability_reasons": "no_spanning_lateral_bins",
                "target_qc_reasons": "no_spanning_lateral_bins",
            }
        )
        backend._validate_target_reason_row(
            row,
            label="synthetic Maike target",
            expected_resolvable=False,
            expected_target_qc=False,
            q05=np.nan,
            rmse=np.nan,
        )

    def test_target_qc_is_recomputed_from_raw_target_payload(self) -> None:
        axis = np.linspace(-0.5, 0.5, 5)
        raw_xy = np.asarray([(x, y) for x in axis for y in axis], dtype=np.float64)
        design = np.column_stack(
            (
                np.ones(len(raw_xy)),
                raw_xy[:, 0],
                raw_xy[:, 1],
                raw_xy[:, 0] ** 2,
                raw_xy[:, 0] * raw_xy[:, 1],
                raw_xy[:, 1] ** 2,
            )
        )
        coefficients = np.asarray([2.0, 0.1, 0.0, 0.05, 0.0, 0.03])
        thickness = design @ coefficients
        technical_hash = "e" * 64
        sealed_relpath = "sealed_distal/lens_000007.npz"
        sealed_sha = "f" * 64
        outcome_config = {
            "technical_config_sha256": technical_hash,
            "target_q05_thickness_min_um_exclusive": 0.0,
            "target_fit_rmse_max_um": 2.5,
        }
        outcome_text = backend._canonical_json(outcome_config)
        outcome_hash = hashlib.sha256(outcome_text.encode()).hexdigest()
        arrays = {
            "schema_version": np.asarray(maike64.TARGET_SCHEMA),
            "eye_id": np.asarray("eye"),
            "lens_index": np.asarray(7, dtype=np.int64),
            "proximal_points_xyz_um": np.column_stack((raw_xy, np.zeros(len(raw_xy)))),
            "canonical_grid_xy": backend.experiment63.CANONICAL_GRID_XY.copy(),
            "target_smoothed_thickness_um": backend.experiment63.CANONICAL_DESIGN @ coefficients,
            "raw_target_xy_normalized": raw_xy,
            "raw_target_thickness_um": thickness,
            "target_coefficients_c0_c5": coefficients,
            "sealed_distal_relpath": np.asarray(sealed_relpath),
            "sealed_distal_sha256": np.asarray(sealed_sha),
            "technical_config_sha256": np.asarray(technical_hash),
            "outcome_config_json": np.asarray(outcome_text),
            "outcome_config_sha256": np.asarray(outcome_hash),
        }
        row = pd.Series(
            {
                "eye_id": "eye",
                "lens_index": 7,
                "target_resolvable": True,
                "target_qc": False,
                "target_support": len(raw_xy),
                "target_depth_um": float(np.median(thickness)),
                "target_q05_raw_thickness_um": float(np.quantile(thickness, 0.05)),
                "target_rmse_um": 0.0,
                **{f"target_c{i}": value for i, value in enumerate(coefficients)},
            }
        )
        technical_row = pd.Series(
            {
                "distal_qc": True,
                "component_fraction_gate_pass": True,
                "sealed_distal_relpath": sealed_relpath,
            }
        )
        dummy = backend.TechnicalBundle(
            kind="maike",
            unit_id="eye",
            root=Path("/synthetic"),
            table=pd.DataFrame(),
            completion_sha256="0" * 64,
            provenance_sha256="0" * 64,
            technical_manifest={sealed_relpath: {"sha256": sealed_sha, "size_bytes": 0}},
            outcome_manifest=backend.DeferredArtifact(
                root=Path("/synthetic"),
                relative_path="sealed_outcomes/manifest.json",
                sha256="0" * 64,
                size_bytes=0,
                owner="eye",
            ),
            technical_config_sha256=technical_hash,
        )
        with self.assertRaisesRegex(backend.ContractError, "target_qc differs"):
            backend._load_target_npz(
                arrays,
                label="synthetic target",
                schema=maike64.TARGET_SCHEMA,
                row=row,
                arthur=False,
                technical_row=technical_row,
                expected_members=set(maike64.TARGET_KEYS),
                technical_bundle=dummy,
                outcome_manifest={
                    "target_config": outcome_config,
                    "target_config_sha256": outcome_hash,
                },
            )

    def test_fixed_rule_is_explicitly_descriptive(self) -> None:
        result = backend.descriptive_10_of_12_result(
            {
                "cohort": "cohort_primary",
                "independent_unit": "animal (one eye per fly)",
                "n_independent_animals": 12,
                "wins": 10,
                "losses": 2,
                "ties_nonwins": 0,
            }
        )
        self.assertTrue(result["meets_descriptive_10_of_12_rule"])
        self.assertFalse(result["pristine_external_confirmation"])
        self.assertEqual(
            result["analysis_label"],
            "post_qc_model_error_sequestered_evaluation",
        )
        self.assertIn("descriptive", result["decision_role"])

    def test_pass2_reuses_all_frozen_secondary_analysis_helpers(self) -> None:
        source = inspect.getsource(backend.run_pass2)
        for helper in (
            "species_sex_descriptive_table",
            "per_eye_method_descriptive_table",
            "summarize_internal_methods",
            "run_target_qc_sensitivity_models",
            "nonconfirmatory_nested_summary",
            "run_within_maike_nested_loao_secondary",
            "aggregate_within_maike_secondary",
        ):
            self.assertIn(f"experiment63.{helper}", source)
        self.assertGreaterEqual(source.count("_score_with_loaded_raw_targets("), 3)

    def test_scoring_delegates_to_experiment63_without_old_artifact_loader(self) -> None:
        table = pd.DataFrame(
            [{"eye_id": "eye", "lens_index": 7, "target_depth_um": 2.0}]
        )
        base_metrics = pd.DataFrame(
            [
                {
                    "eye_id": "eye",
                    "lens_index": 7,
                    "method": "position_scale_control",
                    "raw_unsmoothed_available": False,
                    "raw_unsmoothed_mae_um": np.nan,
                    "raw_unsmoothed_p90_error_um": np.nan,
                    "raw_unsmoothed_normalized_mae": np.nan,
                }
            ]
        )
        predictions = {"position_scale_control": np.zeros((1, 6), dtype=np.float64)}
        raw = {
            ("eye", 7): {
                "raw_target_xy_normalized": np.zeros((3, 2), dtype=np.float64),
                "raw_target_thickness_um": np.ones(3, dtype=np.float64),
            }
        }
        with mock.patch.object(
            backend.experiment63,
            "score_predictions",
            return_value=base_metrics.copy(),
        ) as scorer:
            result = backend._score_with_loaded_raw_targets(table, predictions, raw)
        scorer.assert_called_once_with(table, predictions, artifact_roots=None)
        self.assertTrue(bool(result.loc[0, "raw_unsmoothed_available"]))
        self.assertEqual(float(result.loc[0, "raw_unsmoothed_mae_um"]), 1.0)


class PublicationContractTests(unittest.TestCase):
    def test_execute_publishes_hash_bound_exclusive_output(self) -> None:
        expected_commit = "a" * 40
        descriptive = backend.descriptive_10_of_12_result(
            {
                "cohort": "cohort_primary",
                "independent_unit": "animal (one eye per fly)",
                "n_independent_animals": 12,
                "wins": 10,
                "losses": 2,
                "ties_nonwins": 0,
            }
        )
        exclusion_report = {
            "counting_note": "synthetic non-exclusive reason accounting",
            "arthur_source_predictor_qc": {
                "fixed_source_denominator": 1,
                "distal_qc_retained": 1,
                "distal_qc_excluded": 0,
            },
            "maike_validation_predictor_qc": {
                "aggregate": {
                    "fixed_oda_denominator": 12,
                    "distal_qc_retained": 12,
                    "distal_qc_excluded": 0,
                }
            },
        }
        data = pd.DataFrame(
            [{"eye_id": "synthetic-eye", "lens_index": 0, "value": 1.0}]
        )
        alpha_audit = pd.DataFrame(
            [{"fold": "synthetic-fold", "selected_alpha": 1.0}]
        )
        model = SimpleNamespace(
            feature_mean=np.asarray([1.0], dtype=np.float64),
            feature_scale=np.asarray([2.0], dtype=np.float64),
            target_mean=np.zeros(6, dtype=np.float64),
            coefficients=np.ones((1, 6), dtype=np.float64),
            alpha=1.0,
        )
        analysis = {
            "metrics": data.copy(),
            "sensitivity_metrics": data.copy(),
            "maike_secondary_metrics": data.copy(),
            "per_eye": data.copy(),
            "per_eye_methods": data.copy(),
            "species_sex_descriptive": data.copy(),
            "per_eye_sensitivity": data.copy(),
            "per_eye_maike_secondary": data.copy(),
            "alpha_audit": alpha_audit.copy(),
            "sensitivity_alpha_audit": alpha_audit.copy(),
            "maike_secondary_alpha_audit": alpha_audit.copy(),
            "descriptive_result": descriptive,
            "sensitivity_result": {"status": "synthetic"},
            "maike_secondary_result": {"status": "synthetic"},
            "internal_method_summary": {"status": "synthetic"},
            "metric_validity_gate": {"passed": True},
            "maike_secondary_metric_validity_gate": {"passed": True},
            "retention": {"synthetic-eye": {"fixed_oda_denominator": 1}},
            "technical_exclusion_counts": exclusion_report,
            "selected_alphas": {"position_scale_control": 1.0},
            "sensitivity_selected_alphas": {"position_scale_control": 1.0},
            "maike_secondary_selected_alphas": {
                "synthetic-eye": {"position_scale_control": 1.0}
            },
            "models": {"synthetic": model},
            "equal_volume_source_template": np.arange(6, dtype=np.float64),
            "sensitivity_models": {"synthetic": model},
            "maike_secondary_models": {"synthetic_eye": {"synthetic": model}},
        }

        def deferred(owner: str, digit: str) -> backend.DeferredArtifact:
            return backend.DeferredArtifact(
                root=Path("/synthetic") / owner,
                relative_path="sealed_outcomes/manifest.json",
                sha256=digit * 64,
                size_bytes=int(digit),
                owner=owner,
            )

        maike_bundles = tuple(
            backend.TechnicalBundle(
                kind="maike",
                unit_id=eye_id,
                root=Path("/synthetic") / eye_id,
                table=pd.DataFrame(),
                completion_sha256="1" * 64,
                provenance_sha256="2" * 64,
                technical_manifest={},
                outcome_manifest=deferred(eye_id, "4"),
                qc_binding_sha256="3" * 64,
            )
            for eye_id in backend.EXPECTED_EYES
        )
        arthur = backend.TechnicalBundle(
            kind="arthur",
            unit_id="arthur_source",
            root=Path("/synthetic/arthur"),
            table=pd.DataFrame(),
            completion_sha256="5" * 64,
            provenance_sha256="6" * 64,
            technical_manifest={},
            outcome_manifest=deferred("arthur_source", "7"),
        )
        clearance = backend.Pass1Clearance(
            schema_version=backend.PASS1_SCHEMA,
            analysis_label=backend.ANALYSIS_LABEL,
            expected_commit=expected_commit,
            robust_core_config_sha256=backend.ROBUST_CORE_CONFIG_SHA256,
            maike=maike_bundles,
            arthur=arthur,
            pass1_reads=(),
            _authority=backend._CLEARANCE_AUTHORITY,
        )

        expected_payloads = {
            "per_lens_primary_metrics.csv",
            "per_lens_target_qc_sensitivity_metrics.csv",
            "per_lens_within_maike_nested_loao_secondary_metrics.csv",
            "per_eye_descriptive_10_of_12.csv",
            "per_eye_internal_method_descriptive.csv",
            "species_sex_descriptive.csv",
            "per_eye_target_qc_sensitivity.csv",
            "per_eye_within_maike_nested_loao_secondary.csv",
            "source_only_alpha_selection.csv",
            "within_maike_nested_loao_alpha_selection.csv",
            "primary_result.json",
            "frozen_model_parameters.npz",
        }

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repository = temporary / "repository"
            repository.mkdir()
            output = temporary / "sealed-output"
            with (
                mock.patch.object(
                    backend, "run_pass1", return_value=clearance
                ) as pass1,
                mock.patch.object(
                    backend, "run_pass2", return_value=analysis
                ) as pass2,
                mock.patch.object(
                    backend.experiment63, "require_frozen_git"
                ) as final_git_check,
            ):
                result = backend.execute_experiment64(
                    repository_root=repository,
                    expected_commit=expected_commit,
                    arthur_root=temporary / "arthur",
                    maike_root=temporary / "maike",
                    output_directory=output,
                )

                self.assertIs(
                    pass1.call_args.kwargs["reader"],
                    pass2.call_args.kwargs["reader"],
                )
                self.assertEqual(final_git_check.call_count, 2)
                final_git_check.assert_has_calls(
                    [mock.call(repository, expected_commit)] * 2
                )
                self.assertEqual(result["output_directory"], str(output))
                self.assertEqual(result["wins"], 10)

                primary_result = json.loads(
                    (output / "primary_result.json").read_text(encoding="utf-8")
                )
                sealed_path = output / "sealed_run_manifest.json"
                sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
                attempt_directory = (
                    temporary / f"experiment64_outcome_attempt_{expected_commit}"
                )
                attempt_path = attempt_directory / "attempt.json"
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    attempt["status"],
                    "pass2_attempt_committed_before_outcome_access",
                )
                self.assertFalse(attempt["outcome_artifacts_opened_at_record_creation"])
                self.assertFalse(attempt["rerun_permitted"])
                self.assertEqual(
                    sealed["outcome_attempt"]["record"],
                    {
                        "relative_path": "attempt.json",
                        **_binding(attempt_path.read_bytes()),
                    },
                )
                self.assertEqual(
                    primary_result["outcome_attempt"], sealed["outcome_attempt"]
                )
                self.assertEqual(
                    primary_result["technical_exclusion_counts"], exclusion_report
                )
                self.assertEqual(
                    sealed["technical_exclusion_counts"], exclusion_report
                )
                self.assertEqual(sealed["decision_rule"], descriptive)
                self.assertEqual(sealed["git"], {"commit": expected_commit, "dirty": False})
                self.assertEqual(
                    sealed["robust_core_config_sha256"],
                    backend.ROBUST_CORE_CONFIG_SHA256,
                )
                self.assertEqual(set(sealed["output_manifest"]), expected_payloads)
                for filename, binding in sealed["output_manifest"].items():
                    self.assertEqual(binding, _binding((output / filename).read_bytes()))

                seal_hash = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
                self.assertEqual(
                    (output / "SEALED.sha256").read_text(encoding="ascii"),
                    f"{seal_hash}  sealed_run_manifest.json\n",
                )
                self.assertEqual(
                    sealed["outcome_manifest_bindings"]["arthur"],
                    {
                        "relative_path": "sealed_outcomes/manifest.json",
                        "sha256": "7" * 64,
                        "size_bytes": 7,
                    },
                )
                with np.load(output / "frozen_model_parameters.npz") as arrays:
                    self.assertIn("synthetic_coefficients", arrays.files)
                    self.assertIn(
                        "target_qc_sensitivity_synthetic_coefficients", arrays.files
                    )
                    self.assertIn(
                        "within_maike_loao_synthetic_eye_synthetic_coefficients",
                        arrays.files,
                    )
                    np.testing.assert_array_equal(
                        arrays["synthetic_coefficients"], model.coefficients
                    )
                    np.testing.assert_array_equal(
                        arrays["equal_volume_source_template_coefficients_c0_c5"],
                        analysis["equal_volume_source_template"],
                    )
                self.assertEqual(
                    sealed["method_contract"]["trained_target_indices"],
                    backend.experiment63.EVEN_TARGET_INDICES.tolist(),
                )
                self.assertEqual(primary_result["backend"], sealed["backend"])

                staging_prefix = f".{output.name}.staging-"
                self.assertFalse(
                    any(
                        path.name.startswith(staging_prefix)
                        for path in temporary.iterdir()
                    )
                )
                pass1_calls = pass1.call_count
                with self.assertRaisesRegex(
                    backend.ContractError, "Exclusive Experiment 64 output already exists"
                ):
                    backend.execute_experiment64(
                        repository_root=repository,
                        expected_commit=expected_commit,
                        arthur_root=temporary / "arthur",
                        maike_root=temporary / "maike",
                        output_directory=output,
                    )
                self.assertEqual(pass1.call_count, pass1_calls)

    def test_failed_pass2_leaves_attempt_marker_and_blocks_retry(self) -> None:
        expected_commit = "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repository = temporary / "repository"
            maike_root = temporary / "maike"
            repository.mkdir()
            maike_root.mkdir()

            def deferred(owner: str) -> backend.DeferredArtifact:
                return backend.DeferredArtifact(
                    root=temporary / owner,
                    relative_path="sealed_outcomes/manifest.json",
                    sha256="1" * 64,
                    size_bytes=1,
                    owner=owner,
                )

            maike = tuple(
                backend.TechnicalBundle(
                    kind="maike",
                    unit_id=eye_id,
                    root=temporary / eye_id,
                    table=pd.DataFrame(),
                    completion_sha256="2" * 64,
                    provenance_sha256="3" * 64,
                    technical_manifest={},
                    outcome_manifest=deferred(eye_id),
                    qc_binding_sha256="4" * 64,
                )
                for eye_id in backend.EXPECTED_EYES
            )
            arthur = backend.TechnicalBundle(
                kind="arthur",
                unit_id="arthur_source",
                root=temporary / "arthur",
                table=pd.DataFrame(),
                completion_sha256="5" * 64,
                provenance_sha256="6" * 64,
                technical_manifest={},
                outcome_manifest=deferred("arthur"),
            )
            clearance = backend.Pass1Clearance(
                schema_version=backend.PASS1_SCHEMA,
                analysis_label=backend.ANALYSIS_LABEL,
                expected_commit=expected_commit,
                robust_core_config_sha256=backend.ROBUST_CORE_CONFIG_SHA256,
                maike=maike,
                arthur=arthur,
                pass1_reads=(),
                _authority=backend._CLEARANCE_AUTHORITY,
            )
            output = temporary / "failed-output"
            with (
                mock.patch.object(backend, "run_pass1", return_value=clearance) as pass1,
                mock.patch.object(
                    backend, "run_pass2", side_effect=RuntimeError("synthetic crash after authorization")
                ) as pass2,
                mock.patch.object(backend.experiment63, "require_frozen_git"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                    backend.execute_experiment64(
                        repository_root=repository,
                        expected_commit=expected_commit,
                        arthur_root=temporary / "arthur",
                        maike_root=maike_root,
                        output_directory=output,
                    )
                pass2.assert_called_once()
                attempt_directory = (
                    temporary / f"experiment64_outcome_attempt_{expected_commit}"
                )
                self.assertTrue((attempt_directory / "attempt.json").is_file())
                self.assertFalse(output.exists())
                pass1_calls = pass1.call_count
                with self.assertRaisesRegex(
                    backend.ContractError, "outcome attempt already exists"
                ):
                    backend.execute_experiment64(
                        repository_root=repository,
                        expected_commit=expected_commit,
                        arthur_root=temporary / "arthur",
                        maike_root=maike_root,
                        output_directory=output,
                    )
                self.assertEqual(pass1.call_count, pass1_calls)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attest_experiment_64_instance_qc as attester
import render_experiment_64_instance_qc_sample as renderer
from test_render_experiment_64_instance_qc_sample import (
    EYES,
    prepare_fixture,
    hash_entry,
    sha256_path,
    write_json,
)


def write_passing_review(bundle: Path, sample_manifest: Path) -> Path:
    sample = json.loads(sample_manifest.read_text(encoding="utf-8"))
    review = {
        "schema_version": attester.REVIEW_SCHEMA_VERSION,
        "eye_id": sample["eye_id"],
        "review_scope": "disjoint_stratified_sample_only",
        "review_mode": "ai_assisted_visual_review_without_model_outputs",
        "reviewer_id": "test-reviewer",
        "reviewed_at_utc": "2026-09-04T12:00:00Z",
        "sample_manifest_sha256": sha256_path(sample_manifest),
        "decisions": [
            {
                "lens_index": entry["lens_index"],
                "seed_id": entry["seed_id"],
                "decision": "pass",
                "notes": "",
            }
            for entry in sample["samples"]
        ],
    }
    path = bundle / attester.REVIEW_FILENAME
    write_json(path, review)
    return path


class Experiment64AttesterTests(unittest.TestCase):
    def test_pass_attestation_replays_disjoint_selection_without_outcome_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            self.assertFalse((bundle / "sealed_outcomes" / "manifest.json").exists())
            output = bundle / attester.ATTESTATION_FILENAME
            result = attester.attest_eye(
                bundle_root=bundle,
                sample_manifest_path=sample,
                review_path=review,
                output_path=output,
                repository_root=root,
                stop_record_path=stop,
                prior_sample_manifests=prior,
            )
            document = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], attester.ATTESTATION_SCHEMA_VERSION)
        self.assertEqual(document["status"], "passed")
        self.assertTrue(document["stratified_sample_visual_qc_passed"])
        self.assertEqual(
            document["development_exclusion_verification"]["eye_scoped_identities_verified"],
            384,
        )
        self.assertEqual(document["development_exclusion_verification"]["selected_overlap_count"], 0)
        self.assertEqual(set(document["sample_coverage"].values()), {2})
        self.assertEqual(len(document["reviewed_samples"]), 32)
        self.assertTrue(
            document["outcome_sequestration"]
            ["sealed_outcome_manifest_binding_validated_but_file_not_opened"]
        )
        self.assertFalse(document["outcome_sequestration"]["proximal_anatomy_blind"])

    def test_fail_or_indeterminate_stops_whole_run_and_writes_nothing(self) -> None:
        for verdict in ("fail", "indeterminate"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle, stop, prior, sample = prepare_fixture(root)
                review_path = write_passing_review(bundle, sample)
                review = json.loads(review_path.read_text(encoding="utf-8"))
                review["decisions"][0]["decision"] = verdict
                write_json(review_path, review)
                output = bundle / attester.ATTESTATION_FILENAME
                with self.assertRaisesRegex(attester.QCError, "whole Experiment 64 run must stop"):
                    attester.attest_eye(
                        bundle_root=bundle,
                        sample_manifest_path=sample,
                        review_path=review_path,
                        output_path=output,
                        repository_root=root,
                        stop_record_path=stop,
                        prior_sample_manifests=prior,
                    )
                self.assertFalse(output.exists())

    def test_tampered_target_free_artifact_blocks_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            with (bundle / "instances" / "lens_000063.npz").open("ab") as handle:
                handle.write(b"tamper")
            output = bundle / attester.ATTESTATION_FILENAME
            with self.assertRaisesRegex(attester.QCError, "size mismatch"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample,
                    review_path=review,
                    output_path=output,
                    repository_root=root,
                    stop_record_path=stop,
                    prior_sample_manifests=prior,
                )
            self.assertFalse(output.exists())

    def test_coherence_margin_is_recomputed_from_all_four_frozen_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            inventory_path = bundle / attester.TECHNICAL_INVENTORY_FILENAME
            with inventory_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["coherence_margin"] = "0.999"
            with inventory_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=attester.TECHNICAL_INVENTORY_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            updated_binding = hash_entry(inventory_path)
            for filename in ("technical_completion.json", "technical_provenance.json"):
                path = bundle / filename
                document = json.loads(path.read_text(encoding="utf-8"))
                document["technical_output_manifest"]["technical_inventory.csv"] = updated_binding
                write_json(path, document)
            with self.assertRaisesRegex(attester.QCError, "wrong coherence_margin"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample,
                    review_path=review,
                    output_path=bundle / attester.ATTESTATION_FILENAME,
                    repository_root=root,
                    stop_record_path=stop,
                    prior_sample_manifests=prior,
                )

    def test_target_bearing_sampling_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            sampling = bundle / "distal_qc_sampling.csv"
            lines = sampling.read_text(encoding="utf-8").splitlines()
            lines[0] += ",target_depth_um"
            for index in range(1, len(lines)):
                lines[index] += ",999"
            sampling.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(attester.QCError):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample,
                    review_path=review,
                    output_path=bundle / attester.ATTESTATION_FILENAME,
                    repository_root=root,
                    stop_record_path=stop,
                    prior_sample_manifests=prior,
                )

    def test_prior_manifest_hash_is_reverified_at_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            with prior[EYES[5]].open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(attester.QCError, "SHA-256 mismatch"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample,
                    review_path=review,
                    output_path=bundle / attester.ATTESTATION_FILENAME,
                    repository_root=root,
                    stop_record_path=stop,
                    prior_sample_manifests=prior,
                )

    def test_existing_attestation_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, stop, prior, sample = prepare_fixture(root)
            review = write_passing_review(bundle, sample)
            output = bundle / attester.ATTESTATION_FILENAME
            output.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(attester.QCError, "Refusing to overwrite"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample,
                    review_path=review,
                    output_path=output,
                    repository_root=root,
                    stop_record_path=stop,
                    prior_sample_manifests=prior,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()

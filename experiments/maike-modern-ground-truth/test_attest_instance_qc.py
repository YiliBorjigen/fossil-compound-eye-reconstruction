from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attest_instance_qc as attester
import render_instance_qc_sample as renderer
from test_render_instance_qc_sample import fake_render, make_minimal_bundle, write_instance, write_sealed


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_entry(path: Path) -> dict[str, object]:
    return {"sha256": sha256_path(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_attestation_fixture(root: Path) -> tuple[Path, Path, Path]:
    eye_id = "TEST_EYE"
    n_expected = 33
    bundle = make_minimal_bundle(root, eye_id=eye_id, n=n_expected)

    # Retain 32 eligible rows for exact sample coverage and make the final row
    # an explicit, permitted empty assignment in the complete inventory.
    sampling_path = bundle / "distal_qc_sampling.csv"
    with sampling_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["distal_eligible"] = "false"
    with sampling_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=renderer.SAMPLING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_instance(bundle / "instances" / "lens_000032.npz", 32, empty=True)
    write_sealed(bundle / "sealed_distal" / "lens_000032.npz", 32, empty=True)

    summary_fields = (
        "lens_index",
        "seed_id",
        "assignment_status",
        "full_assigned_size",
        "main_component_size",
        "component_removed_size",
        "main_component_fraction",
        "distal_eligible",
    )
    with (bundle / "lens_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for index in range(n_expected):
            empty = index == n_expected - 1
            writer.writerow(
                {
                    "lens_index": index,
                    "seed_id": f"seed-{index:04d}",
                    "assignment_status": "empty_assignment" if empty else "ok",
                    "full_assigned_size": 0 if empty else 2,
                    "main_component_size": 0 if empty else 1,
                    "component_removed_size": 0 if empty else 1,
                    "main_component_fraction": "0.0" if empty else "0.5",
                    "distal_eligible": "false" if empty else "true",
                }
            )

    write_json(
        bundle / "distal_frame_audit.json",
        {"schema_version": "experiment63.distal-frame-audit.v1", "eye_id": eye_id},
    )
    data_paths = {"lens_summary.csv", "distal_qc_sampling.csv", "distal_frame_audit.json"}
    for index in range(n_expected):
        name = f"lens_{index:06d}.npz"
        data_paths.update(
            {f"instances/{name}", f"sealed_distal/{name}", f"lenses/{name}"}
        )
    output_manifest = {
        relpath: hash_entry(bundle / relpath) for relpath in sorted(data_paths)
    }
    partition = {
        "source_foreground_voxel_count": 64,
        "assigned_voxel_count": 64,
        "assigned_unique_voxel_count": 64,
        "unassigned_foreground_voxel_count": 0,
        "multiply_assigned_voxel_count": 0,
        "exact_partition": True,
        "candidate_seeds_per_voxel": 1,
    }
    provenance = {
        "schema_version": "experiment63-eye-bundle-v2",
        "status": "complete",
        "eye_id": eye_id,
        "n_expected": n_expected,
        "n_rows": n_expected,
        "contiguous_indices": True,
        "instance_segmentation_validated": False,
        "partition_evidence": partition,
        "output_manifest": output_manifest,
    }
    write_json(bundle / "provenance.json", provenance)
    completion = {
        "schema_version": "experiment63-eye-bundle-v2",
        "status": "complete",
        "eye_id": eye_id,
        "n_expected": n_expected,
        "n_rows": n_expected,
        "contiguous_indices": True,
        "instance_segmentation_validated": False,
        "partition_evidence": partition,
        "output_manifest": output_manifest,
    }
    write_json(bundle / "completion.json", completion)

    sample_dir = bundle / renderer.SAMPLE_DIRECTORY_NAME
    with mock.patch.object(renderer, "_render_lens", side_effect=fake_render):
        sample_manifest = renderer.render_eye_sample(
            eye_id=eye_id,
            bundle_root=bundle,
            sampling_table=sampling_path,
            output_dir=sample_dir,
        )
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8"))
    review = {
        "schema_version": "experiment63.instance-qc-review.v1",
        "eye_id": eye_id,
        "review_scope": "stratified_sample_only",
        "review_mode": "ai_assisted_visual_review_without_model_outputs",
        "reviewer_id": "test-reviewer",
        "reviewed_at_utc": "2026-09-04T10:00:00Z",
        "sample_manifest_sha256": sha256_path(sample_manifest),
        "decisions": [
            {
                "lens_index": sample["lens_index"],
                "seed_id": sample["seed_id"],
                "decision": "pass",
                "notes": "",
            }
            for sample in manifest["samples"]
        ],
    }
    review_path = bundle / attester.REVIEW_FILENAME
    write_json(review_path, review)
    return bundle, sample_manifest, review_path


class AttestationTests(unittest.TestCase):
    def test_attestation_is_scoped_hash_bound_and_allows_explicit_empty_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review = prepare_attestation_fixture(root)
            output = bundle / "instance_qc_attestation.json"
            result = attester.attest_eye(
                bundle_root=bundle,
                sample_manifest_path=sample_manifest,
                review_path=review,
                output_path=output,
            )
            document = json.loads(result.read_text(encoding="utf-8"))
            completion = json.loads((bundle / "completion.json").read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], attester.ATTESTATION_SCHEMA_VERSION)
        self.assertEqual(document["review_scope"], "stratified_sample_only")
        self.assertFalse(document["all_instances_manually_reviewed"])
        self.assertTrue(document["stratified_sample_visual_qc_passed"])
        self.assertTrue(document["technical_inventory_complete"])
        self.assertTrue(document["artifact_hash_verification_passed"])
        self.assertEqual(document["technical_inventory"]["n_empty_assignment_rows"], 1)
        self.assertEqual(set(document["sample_coverage"].values()), {2})
        self.assertEqual(len(document["reviewed_samples"]), 32)
        self.assertFalse(completion["instance_segmentation_validated"])

    def test_attester_never_np_loads_fitted_lens_or_sealed_target_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review = prepare_attestation_fixture(root)
            opened: list[Path] = []
            original_load = np.load

            def recording_load(path: object, *args: object, **kwargs: object):
                opened.append(Path(path).resolve())
                return original_load(path, *args, **kwargs)

            with mock.patch.object(attester.np, "load", side_effect=recording_load):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample_manifest,
                    review_path=review,
                    output_path=bundle / "instance_qc_attestation.json",
                )
            self.assertEqual(len(opened), 33)
            self.assertTrue(all(path.parent == (bundle / "instances").resolve() for path in opened))
            self.assertFalse(any(path.parent.name == "lenses" for path in opened))

    def test_single_failed_visual_decision_blocks_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review_path = prepare_attestation_fixture(root)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["decisions"][0]["decision"] = "fail"
            write_json(review_path, review)
            output = bundle / "instance_qc_attestation.json"
            with self.assertRaisesRegex(attester.QCError, "did not pass"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample_manifest,
                    review_path=review_path,
                    output_path=output,
                )
            self.assertFalse(output.exists())

    def test_tampered_artifact_blocks_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review_path = prepare_attestation_fixture(root)
            with (bundle / "lenses" / "lens_000001.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(attester.QCError, "size mismatch"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample_manifest,
                    review_path=review_path,
                    output_path=bundle / "instance_qc_attestation.json",
                )

    def test_review_schema_rejects_extra_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review_path = prepare_attestation_fixture(root)
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["all_instances_manually_reviewed"] = True
            write_json(review_path, review)
            with self.assertRaisesRegex(attester.QCError, "fields are not exact"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample_manifest,
                    review_path=review_path,
                    output_path=bundle / "instance_qc_attestation.json",
                )

    def test_existing_attestation_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, sample_manifest, review_path = prepare_attestation_fixture(root)
            output = bundle / "instance_qc_attestation.json"
            output.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(attester.QCError, "Refusing to overwrite"):
                attester.attest_eye(
                    bundle_root=bundle,
                    sample_manifest_path=sample_manifest,
                    review_path=review_path,
                    output_path=output,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()

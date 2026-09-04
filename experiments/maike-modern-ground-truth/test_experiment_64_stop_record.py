from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results"
STOP_PATH = RESULTS / "experiment_64_stop_record.json"
EXPOSURE_PATH = RESULTS / "experiment_64_visual_exposures.json"
OLD_EXCLUSION_PATH = RESULTS / "experiment_64_development_exclusions.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_EYES = (
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


def _load(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Experiment64StopRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stop = _load(STOP_PATH)
        cls.exposure = _load(EXPOSURE_PATH)
        cls.old_exclusions = _load(OLD_EXCLUSION_PATH)

    def test_stop_is_pre_outcome_and_fail_closed(self) -> None:
        self.assertEqual(self.stop["schema_version"], "experiment64.pre_outcome_stop.v1")
        self.assertEqual(self.stop["experiment"], 64)
        self.assertEqual(
            self.stop["status"],
            "stopped_before_attestation_and_model_outcome_evaluation",
        )
        self.assertEqual(
            self.stop["review_mode"],
            "ai_agent_visual_review_without_model_outputs",
        )
        self.assertTrue(all(value is False for value in self.stop["outcome_access"].values()))
        self.assertEqual(
            self.stop["decision"],
            {
                "visual_gate_passed": False,
                "passing_attestations_may_be_issued": False,
                "experiment_64_may_open_sealed_outcomes": False,
                "experiment_64_may_be_scored": False,
                "scientific_model_comparison_available": False,
            },
        )
        self.assertEqual(
            self.stop["decisive_nonpass"]["concordant_ai_visual_assessments"],
            2,
        )
        self.assertEqual(
            self.stop["corroborating_nonpass_seen_during_concurrent_review"]
            ["concordant_ai_visual_assessments"],
            2,
        )

    def test_exposure_ledger_is_exact_and_bound(self) -> None:
        binding = self.stop["visual_exposure_ledger"]
        self.assertEqual(binding["sha256"], _sha256(EXPOSURE_PATH))
        self.assertEqual(binding["size_bytes"], EXPOSURE_PATH.stat().st_size)
        self.assertEqual(binding["total_identities"], 384)
        self.assertEqual(self.exposure["total_eyes"], 12)
        self.assertEqual(self.exposure["identities_per_eye"], 32)
        self.assertEqual(self.exposure["total_identities"], 384)
        self.assertEqual(
            tuple(eye["eye_id"] for eye in self.exposure["eyes"]),
            EXPECTED_EYES,
        )

        identities: set[tuple[str, int, str]] = set()
        for eye in self.exposure["eyes"]:
            self.assertTrue(SHA256_RE.fullmatch(eye["sample_manifest_sha256"]))
            self.assertEqual(eye["n_identities"], 32)
            self.assertEqual(len(eye["identities"]), 32)
            self.assertEqual(
                [identity["ordinal"] for identity in eye["identities"]],
                list(range(32)),
            )
            cell_roles: dict[tuple[int, int], set[str]] = {}
            for identity in eye["identities"]:
                self.assertTrue(SHA256_RE.fullmatch(identity["render_sha256"]))
                cell = (identity["radial_stratum"], identity["scale_stratum"])
                cell_roles.setdefault(cell, set()).add(identity["selection_role"])
                expected_path = (
                    f"renders/sample_{identity['ordinal']:02d}_"
                    f"r{identity['radial_stratum']}_s{identity['scale_stratum']}_"
                    f"lens_{identity['lens_index']:06d}.png"
                )
                self.assertEqual(identity["render_relative_path"], expected_path)
                key = (eye["eye_id"], identity["lens_index"], identity["seed_id"])
                self.assertNotIn(key, identities)
                identities.add(key)
            self.assertEqual(
                cell_roles,
                {
                    (radial, scale): {
                        "near_worst_coherence",
                        "hash_minimal_remaining",
                    }
                    for radial in range(4)
                    for scale in range(4)
                },
            )
        self.assertEqual(len(identities), 384)

    def test_new_exposures_are_disjoint_from_experiment63(self) -> None:
        old = {
            (eye["eye_id"], identity["lens_index"])
            for eye in self.old_exclusions["eyes"]
            for identity in eye["identities"]
        }
        new = {
            (eye["eye_id"], identity["lens_index"])
            for eye in self.exposure["eyes"]
            for identity in eye["identities"]
        }
        self.assertEqual(len(old), 384)
        self.assertEqual(len(new), 384)
        self.assertFalse(old & new)

    def test_decisive_and_corroborating_renders_are_exact(self) -> None:
        expected = (
            (
                "decisive_nonpass",
                RESULTS / "experiment_64_decisive_nonpass_RED3_25_M_26_lens_260.png",
            ),
            (
                "corroborating_nonpass_seen_during_concurrent_review",
                RESULTS / "experiment_64_corroborating_nonpass_RED3_25_M_26_lens_255.png",
            ),
        )
        eye = next(
            eye for eye in self.exposure["eyes"] if eye["eye_id"] == "RED3_25_M_26"
        )
        self.assertEqual(
            self.stop["decisive_nonpass"]["sample_manifest_sha256"],
            eye["sample_manifest_sha256"],
        )
        for record_key, image_path in expected:
            with self.subTest(record_key=record_key):
                record = self.stop[record_key]
                identity = eye["identities"][record["sample_ordinal_zero_based"]]
                self.assertEqual(identity["lens_index"], record["lens_index"])
                self.assertEqual(identity["seed_id"], record["seed_id"])
                self.assertEqual(identity["render_sha256"], record["render_sha256"])
                self.assertEqual(_sha256(image_path), record["render_sha256"])


if __name__ == "__main__":
    unittest.main()

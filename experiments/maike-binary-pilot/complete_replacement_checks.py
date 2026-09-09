#!/usr/bin/env python3
"""Complete only the two blocked transfers using the unchanged recorded rules."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import transfer_remaining as transfer

EXPECTED_SCRIPT_SHA = "dc61b2b02350f1a2602904f92fb9ce5c6f449b9257f882e863dddf1c62b7d417"
EXPECTED_INPUTS = {
    "M3_F_28_03": "d18763929208c1fff12ada55124e1f489cd420ec66f101ebb9309656fce38ffa",
    "RED3_25_M_26": "9c1ab44205610e940617e19249df8c5364a0b88115ff736b3c6a3952bd63b3de",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m3_archive", type=Path)
    parser.add_argument("red3_archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if sha(base / "transfer_remaining.py") != EXPECTED_SCRIPT_SHA:
        raise ValueError("The fixed transfer script has changed")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    inputs = []
    for specimen, source in zip(EXPECTED_INPUTS, [args.m3_archive, args.red3_archive]):
        if specimen not in source.name or sha(source) != EXPECTED_INPUTS[specimen]:
            raise ValueError("Replacement does not match the calibrated source: " + specimen)
        inputs.append({"specimen": specimen, "archive": source.name,
                       "bytes": source.stat().st_size, "sha256": sha(source)})
    # Explicit input links avoid selecting older, truncated copies. Keep them
    # local; all supplied originals and earlier output directories stay intact.
    staging = args.output.with_name(args.output.name + "-input-links")
    staging.mkdir(parents=True, exist_ok=False)
    for source in [args.m3_archive, args.red3_archive]:
        (staging / source.name).symlink_to(source.resolve())
    (staging / "receipt.json").write_text(json.dumps(inputs, indent=2) + "\n")
    transfer.SPECIMENS = list(EXPECTED_INPUTS)
    sys.argv = ["transfer_remaining.py", str(staging), "--output", str(args.output)]
    transfer.main()
    original_path = base / "remaining-eye-results-20260909/manifest.json"
    original = json.loads(original_path.read_text())
    replacements = json.loads((args.output / "manifest.json").read_text())
    updated = {entry["specimen"]: entry for entry in replacements["results"]}
    combined = copy.deepcopy(original)
    combined["results"] = [updated.get(entry["specimen"], entry) for entry in original["results"]]
    combined["completion"] = {
        "original_manifest_sha256": sha(original_path),
        "replacement_manifest_sha256": sha(args.output / "manifest.json"),
        "completion_script_sha256": sha(Path(__file__)),
        "replacement_receipts": inputs,
        "unchanged_earlier_eyes": [entry["specimen"] for entry in original["results"] if entry["specimen"] not in updated],
        "scope": "Ten planned eyes; replace only two unreadable-input entries with their fixed-rule outcomes. No earlier eye rerun.",
    }
    (args.output / "combined_manifest.json").write_text(json.dumps(combined, indent=2) + "\n")
    rows = [metric for entry in combined["results"] for metric in entry.get("metrics", [])]
    transfer.pilot.write_csv(args.output / "combined_eye_summary.csv", rows)
    (args.output / "replacement_receipt.json").write_text(json.dumps(inputs, indent=2) + "\n")


if __name__ == "__main__":
    main()

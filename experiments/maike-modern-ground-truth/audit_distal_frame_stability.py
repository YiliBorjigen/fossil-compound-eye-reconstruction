#!/usr/bin/env python3
"""Frozen, target-blind stability audit for Experiment 63 distal frames."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from distal_only_geometry import (
    DistalGeometryError,
    canonical_config_json,
    derive_eye_frames_from_origins,
    load_sealed_distal,
    run_monotone_fixed_point,
)


AUDIT_SCHEMA_VERSION = "experiment63.distal-frame-audit.v1"
ANALYSIS_SCOPE = "conditional_on_oracle_distal_surface_localization"
SUBSAMPLE_FRACTION = 0.90
PERTURBATION_NN_FRACTION = 0.01
RANDOM_SEED = 6301
U_ANGLE_P95_MAX_DEG = 5.0
OUTWARD_ANGLE_P95_MAX_DEG = 2.0
CENTRAL_CHANGE_RATE_MAX = 0.02

PERTURBATION_KEYS = (
    "exhaustive_leave_one_origin_out",
    "ninety_percent_subsample",
    "one_percent_nearest_neighbor_gaussian",
)


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = np.sum(first * second, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _compare_frames(
    baseline: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_central = np.asarray(baseline["central"], dtype=bool)
    variant_central = np.asarray(variant["central"], dtype=bool)
    noncentral = ~(baseline_central | variant_central)
    outward_angles = _angle_degrees(
        np.asarray(baseline["outward"]), np.asarray(variant["outward"])
    )
    if np.any(noncentral):
        u_angles = _angle_degrees(
            np.asarray(baseline["u"])[noncentral], np.asarray(variant["u"])[noncentral]
        )
    else:
        u_angles = np.empty(0, dtype=np.float64)
    fallback = np.asarray(variant["numerical_fallback"], dtype=bool)
    return {
        "outward_angles": outward_angles,
        "u_angles": u_angles,
        "central_changes": baseline_central != variant_central,
        "fallback_outside_central": fallback & ~variant_central,
    }


def _safe_p95(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.quantile(values, 0.95))


def _summarise_comparisons(comparisons: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    if not comparisons:
        raise DistalGeometryError("stability audit has no comparisons")
    outward = np.concatenate([item["outward_angles"] for item in comparisons])
    u_parts = [item["u_angles"] for item in comparisons if len(item["u_angles"])]
    u_angles = np.concatenate(u_parts) if u_parts else np.empty(0, dtype=np.float64)
    central_changes = np.concatenate([item["central_changes"] for item in comparisons])
    fallback = np.concatenate([item["fallback_outside_central"] for item in comparisons])
    u_p95 = _safe_p95(u_angles)
    outward_p95 = _safe_p95(outward)
    change_rate = float(np.mean(central_changes))
    fallback_count = int(np.count_nonzero(fallback))
    metrics = {
        "noncentral_poleward_u_angle_p95_deg": u_p95,
        "outward_axis_angle_p95_deg": outward_p95,
        "central_classification_change_rate": change_rate,
        "n_numerical_fallback_outside_central": fallback_count,
        "n_outward_axis_comparisons": int(len(outward)),
        "n_noncentral_poleward_u_comparisons": int(len(u_angles)),
        "n_central_classification_comparisons": int(len(central_changes)),
    }
    gates = {
        "noncentral_poleward_u_angle_p95_le_5_deg": bool(
            math.isfinite(u_p95) and u_p95 <= U_ANGLE_P95_MAX_DEG
        ),
        "outward_axis_angle_p95_le_2_deg": bool(
            math.isfinite(outward_p95) and outward_p95 <= OUTWARD_ANGLE_P95_MAX_DEG
        ),
        "central_classification_change_rate_le_0_02": bool(
            math.isfinite(change_rate) and change_rate <= CENTRAL_CHANGE_RATE_MAX
        ),
        "zero_numerical_fallback_outside_central": fallback_count == 0,
    }
    return {
        **metrics,
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _origin_array(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    origins = []
    for record in records:
        points = np.asarray(record["points_xyz_um"], dtype=np.float64)
        origins.append(np.mean(points, axis=0))
    return np.asarray(origins, dtype=np.float64)


def audit_frame_stability(
    sealed_records: Sequence[Mapping[str, Any]],
    eye_id: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all pre-specified origin perturbations on sealed distal artifacts."""

    if not eye_id:
        raise DistalGeometryError("eye_id must be non-empty")
    if len(sealed_records) < 7:
        raise DistalGeometryError("frame stability audit requires at least seven distal artifacts")
    if any(record.get("sealed_distal_artifact") is not True for record in sealed_records):
        raise DistalGeometryError("frame audit accepts sealed distal artifacts only")
    sealed_records = sorted(sealed_records, key=lambda record: int(record["lens_index"]))
    lens_indices = [record["lens_index"] for record in sealed_records]
    if len(set(lens_indices)) != len(lens_indices):
        raise DistalGeometryError("sealed distal lens_index values must be unique")
    artifact_config_hashes = {str(record["config_sha256"]) for record in sealed_records}
    if len(artifact_config_hashes) != 1:
        raise DistalGeometryError("sealed distal artifacts do not share one configuration hash")
    artifact_config_jsons = {str(record["config_json"]) for record in sealed_records}
    if len(artifact_config_jsons) != 1:
        raise DistalGeometryError("sealed distal artifacts do not share one configuration JSON")
    embedded_config = sealed_records[0]["config"]
    selected_config = embedded_config if config is None else config
    selected_config_json = canonical_config_json(selected_config)
    if selected_config_json != canonical_config_json(embedded_config):
        raise DistalGeometryError("requested geometry config differs from sealed threshold config")

    fixed_point = run_monotone_fixed_point(sealed_records, config=selected_config)
    if not fixed_point["converged"]:
        raise DistalGeometryError("distal-QC fixed point did not converge")
    eligible_records = list(fixed_point["eligible_records"])
    if len(eligible_records) < 7:
        raise DistalGeometryError("fewer than seven distal-QC artifacts remain for stability audit")
    origins = _origin_array(eligible_records)
    n_lenses = len(origins)
    baseline = derive_eye_frames_from_origins(origins, config=selected_config)

    leave_one_out_comparisons = []
    all_indices = np.arange(n_lenses, dtype=np.int64)
    for omitted in range(n_lenses):
        references = np.delete(all_indices, omitted)
        variant = derive_eye_frames_from_origins(
            origins, reference_indices=references, config=selected_config
        )
        leave_one_out_comparisons.append(_compare_frames(baseline, variant))
    leave_one_out = _summarise_comparisons(leave_one_out_comparisons)
    leave_one_out["n_reference_sets"] = n_lenses

    rng = np.random.default_rng(RANDOM_SEED)
    subsample_size = max(4, int(math.floor(SUBSAMPLE_FRACTION * n_lenses)))
    if subsample_size >= n_lenses:
        subsample_size = n_lenses - 1
    subsample_indices = np.sort(rng.choice(n_lenses, size=subsample_size, replace=False))
    subsample_variant = derive_eye_frames_from_origins(
        origins, reference_indices=subsample_indices, config=selected_config
    )
    subsample = _summarise_comparisons([_compare_frames(baseline, subsample_variant)])
    subsample.update(
        {
            "fraction": SUBSAMPLE_FRACTION,
            "random_seed": RANDOM_SEED,
            "reference_count": int(subsample_size),
            "reference_lens_indices": [
                eligible_records[index]["lens_index"] for index in subsample_indices
            ],
        }
    )

    median_nn = float(baseline["median_nearest_neighbour_um"])
    sigma_um = PERTURBATION_NN_FRACTION * median_nn
    perturbed_origins = origins + rng.normal(0.0, sigma_um, size=origins.shape)
    perturbation_variant = derive_eye_frames_from_origins(
        perturbed_origins, config=selected_config
    )
    gaussian = _summarise_comparisons([_compare_frames(baseline, perturbation_variant)])
    gaussian.update(
        {
            "nearest_neighbour_fraction": PERTURBATION_NN_FRACTION,
            "median_nearest_neighbour_um": median_nn,
            "gaussian_sigma_um_per_coordinate": sigma_um,
            "random_seed": RANDOM_SEED,
        }
    )

    perturbations = {
        PERTURBATION_KEYS[0]: leave_one_out,
        PERTURBATION_KEYS[1]: subsample,
        PERTURBATION_KEYS[2]: gaussian,
    }
    all_comparisons = leave_one_out_comparisons + [
        _compare_frames(baseline, subsample_variant),
        _compare_frames(baseline, perturbation_variant),
    ]
    overall = _summarise_comparisons(all_comparisons)
    baseline_fallback_outside_central = int(
        np.count_nonzero(
            np.asarray(baseline["numerical_fallback"], dtype=bool)
            & ~np.asarray(baseline["central"], dtype=bool)
        )
    )
    total_fallback = (
        baseline_fallback_outside_central
        + sum(
            int(item["metrics"]["n_numerical_fallback_outside_central"])
            for item in perturbations.values()
        )
    )
    overall["metrics"]["n_numerical_fallback_outside_central"] = total_fallback
    overall["gates"]["zero_numerical_fallback_outside_central"] = total_fallback == 0
    overall["pass"] = all(overall["gates"].values())

    perturbation_gate_pass = all(item["pass"] for item in perturbations.values())
    # Empty or poor Stage-1 caps are legitimate fixed-point exclusions.  The
    # stability gate concerns the stable eligible cohort; it must never demand
    # that every mapped seed survive distal QC.
    stable_eligible_cohort_valid = all(
        row.get("eligible") is True
        for row in fixed_point["per_lens"]
        if row["lens_index"] in set(fixed_point["eligible_indices"])
    )
    gate_pass = bool(overall["pass"] and perturbation_gate_pass and stable_eligible_cohort_valid)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "analysis_scope": ANALYSIS_SCOPE,
        "eye_id": eye_id,
        "target_blind": True,
        "input_type": "sealed_distal_artifacts_only",
        "config_json": selected_config_json,
        "config_sha256": next(iter(artifact_config_hashes)),
        "sealed_config_json": next(iter(artifact_config_jsons)),
        "n_input_artifacts": len(sealed_records),
        "n_distal_qc_eligible": len(eligible_records),
        "eligible_lens_indices": [record["lens_index"] for record in eligible_records],
        "input_artifacts": [
            {
                "lens_index": record["lens_index"],
                "relative_path": f"sealed_distal/lens_{int(record['lens_index']):06d}.npz",
                "sha256": record["artifact_sha256"],
                **(
                    {"size_bytes": int(Path(record["artifact_path"]).stat().st_size)}
                    if Path(record["artifact_path"]).is_file()
                    else {}
                ),
            }
            for record in sealed_records
        ],
        "fixed_point": {
            "converged": fixed_point["converged"],
            "max_iterations": fixed_point["max_iterations"],
            "iterations": fixed_point["iterations"],
            "eligible_counts": fixed_point["eligible_counts"],
            "readded_count": fixed_point["readded_count"],
        },
        "baseline": {
            "sphere_radius_um": float(baseline["sphere_radius"]),
            "median_nearest_neighbour_um": float(baseline["median_nearest_neighbour_um"]),
            "central_count": int(np.count_nonzero(baseline["central"])),
            "n_numerical_fallback_outside_central": baseline_fallback_outside_central,
        },
        "thresholds": {
            "noncentral_poleward_u_angle_p95_max_deg": U_ANGLE_P95_MAX_DEG,
            "outward_axis_angle_p95_max_deg": OUTWARD_ANGLE_P95_MAX_DEG,
            "central_classification_change_rate_max": CENTRAL_CHANGE_RATE_MAX,
            "numerical_fallback_outside_central_max": 0,
        },
        "perturbations": perturbations,
        "metrics": overall["metrics"],
        "gates": {
            **overall["gates"],
            "every_perturbation_passes_all_gates": perturbation_gate_pass,
            "stable_eligible_cohort_valid": stable_eligible_cohort_valid,
        },
        "gate_passed": gate_pass,
        "pass": gate_pass,
        "status": "pass" if gate_pass else "fail",
    }


def run_frame_stability_audit(
    records_or_paths: Sequence[Mapping[str, Any] | str | Path],
    eye_id: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical extractor-facing entry point.

    Inputs may be already validated sealed-record dictionaries or paths to
    sealed artifacts.  No unsealed in-memory point record is accepted.
    """

    if not records_or_paths:
        raise DistalGeometryError("frame stability audit requires sealed distal artifacts")
    records: list[Mapping[str, Any]] = []
    expected_hash: str | None = None
    for item in records_or_paths:
        if isinstance(item, (str, Path)):
            loaded = load_sealed_distal(item, expected_config_sha256=expected_hash)
            if expected_hash is None:
                expected_hash = str(loaded["config_sha256"])
            records.append(loaded)
        elif isinstance(item, Mapping) and item.get("sealed_distal_artifact") is True:
            item_hash = str(item.get("config_sha256", ""))
            if expected_hash is None:
                expected_hash = item_hash
            elif item_hash != expected_hash:
                raise DistalGeometryError("sealed distal artifacts use different config hashes")
            records.append(item)
        else:
            raise DistalGeometryError("frame audit accepts sealed distal artifacts or paths only")
    return audit_frame_stability(records, eye_id=eye_id, config=config)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distal-dir", type=Path, help="directory containing sealed distal .npz files")
    parser.add_argument(
        "--artifact", type=Path, action="append", default=[], help="one sealed distal .npz (repeatable)"
    )
    parser.add_argument("--eye", required=True, help="eye identifier recorded in the audit")
    parser.add_argument("--output", required=True, type=Path, help="immutable audit JSON output")
    parser.add_argument("--expected-config-sha256", help="optional pre-frozen config hash")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = list(args.artifact)
    if args.distal_dir is not None:
        paths.extend(sorted(args.distal_dir.glob("*.npz")))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise SystemExit("no sealed distal artifacts supplied")
    records = [
        load_sealed_distal(path, expected_config_sha256=args.expected_config_sha256)
        for path in paths
    ]
    result = run_frame_stability_audit(records, eye_id=args.eye)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {args.output}")
    _write_json_atomic(args.output, result)
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

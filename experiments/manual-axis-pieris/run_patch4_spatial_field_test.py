#!/usr/bin/env python3
"""Score the frozen Patch 4 spatial orientation field.

Patch 4 manual paths are audited for continuity before model scoring.  Every
prediction is anchored at the first manual node; all later nodes are held out
from the predicted direction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from run_cross_region_axis_transfer import eye_axes, load_tracks, path_rmse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_annotations(path: Path, minimum_nodes: int = 6) -> list[dict[str, object]]:
    project = json.loads(path.read_text())
    records = []
    for cone in project.get("cones", []):
        cone_id = int(cone["id"])
        nodes = sorted(cone.get("nodes", []), key=lambda node: node["depth"])
        depths = np.asarray([node["depth"] for node in nodes], float)
        uv = np.asarray([[node["row"], node["column"]] for node in nodes], float)
        duplicate_depth = bool(len(depths) > 1 and np.any(np.diff(depths) <= 0))
        if len(nodes) > 1:
            depth_gap = np.diff(depths)
            uv_step = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            threshold = np.maximum(5.0, 2.5 * np.abs(depth_gap))
            discontinuities = np.flatnonzero(uv_step > threshold)
            maximum_uv_step = float(np.max(uv_step))
        else:
            discontinuities = np.asarray([], dtype=int)
            maximum_uv_step = 0.0
        reasons = []
        if len(nodes) < minimum_nodes:
            reasons.append(f"fewer than {minimum_nodes} nodes")
        if duplicate_depth:
            reasons.append("duplicate or non-increasing depth")
        if len(discontinuities):
            reasons.append(
                "possible cone switch after node(s) "
                + ",".join(str(int(index + 1)) for index in discontinuities)
            )
        records.append(
            {
                "cone_id": cone_id,
                "nodes": len(nodes),
                "depth_start": float(depths[0]) if len(depths) else "",
                "depth_stop": float(depths[-1]) if len(depths) else "",
                "maximum_uv_step_voxels": maximum_uv_step,
                "continuity_pass": not reasons,
                "audit_note": "; ".join(reasons),
            }
        )
    return records


def prediction(track, skew: np.ndarray) -> np.ndarray:
    relative_depth = (track.depths - track.depths[0])[:, None]
    direction = track.normal + skew[0] * track.tangent_1 + skew[1] * track.tangent_2
    return track.xyz[[0]] + relative_depth * direction


def quadratic_fit(track) -> np.ndarray:
    relative_depth = track.depths - track.depths[0]
    design = np.column_stack(
        (np.ones(len(relative_depth)), relative_depth, relative_depth**2)
    )
    coefficients, *_ = np.linalg.lstsq(design, track.xyz, rcond=None)
    return design @ coefficients


def held_out_rmse(truth: np.ndarray, predicted: np.ndarray) -> float:
    if len(truth) < 2:
        return float("nan")
    return path_rmse(truth[1:], predicted[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-label", type=Path, required=True)
    parser.add_argument("--patch-4-dir", type=Path, required=True)
    parser.add_argument("--patch-4-annotations", type=Path, required=True)
    parser.add_argument("--patch-4-archive", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_model.read_text())
    if frozen.get("status") != "frozen before reading or scoring Patch 4 annotations":
        raise RuntimeError("The supplied model is not marked as pre-score frozen")

    audit = audit_annotations(args.patch_4_annotations)
    usable_ids = {int(row["cone_id"]) for row in audit if row["continuity_pass"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "patch_4_path_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    axes = eye_axes(args.registered_label)
    tracks = load_tracks(
        "patch_4",
        args.patch_4_dir,
        args.patch_4_annotations,
        axes,
        selected_ids=usable_ids,
        minimum_nodes=6,
    )
    if not tracks:
        raise RuntimeError("No Patch 4 tracks passed the continuity audit")

    primary_skew = np.asarray(frozen["primary_model"]["predicted_skew"], float)
    affine_skew = np.asarray(
        frozen["secondary_affine_model"]["predicted_skew"], float
    )
    constant_skew = np.asarray(
        frozen["secondary_constant_model"]["predicted_skew"], float
    )

    records = []
    for track in tracks:
        normal_path = prediction(track, np.zeros(2))
        primary_path = prediction(track, primary_skew)
        affine_path = prediction(track, affine_skew)
        constant_path = prediction(track, constant_skew)
        quadratic_path = quadratic_fit(track)
        records.append(
            {
                "cone_id": track.cone_id,
                "nodes": len(track.depths),
                "depth_start": float(track.depths[0]),
                "depth_stop": float(track.depths[-1]),
                "observed_skew_1": float(track.skew[0]),
                "observed_skew_2": float(track.skew[1]),
                "normal_rmse_voxels": held_out_rmse(track.xyz, normal_path),
                "primary_idw_rmse_voxels": held_out_rmse(track.xyz, primary_path),
                "secondary_affine_rmse_voxels": held_out_rmse(track.xyz, affine_path),
                "secondary_constant_rmse_voxels": held_out_rmse(track.xyz, constant_path),
                "oracle_straight_rmse_voxels": held_out_rmse(
                    track.xyz, track.straight_fit
                ),
                "oracle_quadratic_rmse_voxels": held_out_rmse(
                    track.xyz, quadratic_path
                ),
                "normal_endpoint_error_voxels": float(
                    np.linalg.norm(track.xyz[-1] - normal_path[-1])
                ),
                "primary_idw_endpoint_error_voxels": float(
                    np.linalg.norm(track.xyz[-1] - primary_path[-1])
                ),
            }
        )

    with (args.output_dir / "patch_4_per_track.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def values(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in records], float)

    normal = values("normal_rmse_voxels")
    primary = values("primary_idw_rmse_voxels")
    affine = values("secondary_affine_rmse_voxels")
    constant = values("secondary_constant_rmse_voxels")
    straight = values("oracle_straight_rmse_voxels")
    quadratic = values("oracle_quadratic_rmse_voxels")
    wins = int(np.sum(primary < normal))
    ties = int(np.sum(np.isclose(primary, normal)))
    non_ties = len(records) - ties
    sign_p = float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else 1.0
    voxel_size = float(frozen["voxel_size_micrometres"])

    annotation_project = json.loads(args.patch_4_annotations.read_text())
    source_hash = annotation_project.get("source_sha256")
    actual_source_hash = sha256(args.patch_4_dir / "unfolded_intensity.npy")
    summary = {
        "status": "frozen Patch 4 held-out regional test",
        "frozen_model_sha256": sha256(args.frozen_model),
        "patch_4_annotation_archive_sha256": sha256(args.patch_4_archive),
        "patch_4_annotations_json_sha256": sha256(args.patch_4_annotations),
        "annotation_source_sha256_recorded": source_hash,
        "annotation_source_sha256_actual": actual_source_hash,
        "annotation_source_match": source_hash in (None, actual_source_hash),
        "tracks_in_archive": len(audit),
        "tracks_passing_continuity_audit": len(records),
        "tracks_excluded_by_audit": [
            row for row in audit if not row["continuity_pass"]
        ],
        "primary": {
            "model": frozen["primary_model"],
            "median_normal_rmse_voxels": float(np.median(normal)),
            "median_primary_rmse_voxels": float(np.median(primary)),
            "median_normal_rmse_micrometres": float(np.median(normal) * voxel_size),
            "median_primary_rmse_micrometres": float(np.median(primary) * voxel_size),
            "median_paired_change_voxels": float(np.median(primary - normal)),
            "median_fractional_change_vs_normal": float(
                np.median((primary - normal) / normal)
            ),
            "wins_vs_normal": wins,
            "ties_vs_normal": ties,
            "tracks": len(records),
            "two_sided_exact_sign_test_p": sign_p,
        },
        "secondary": {
            "median_affine_field_rmse_voxels": float(np.median(affine)),
            "affine_wins_vs_normal": int(np.sum(affine < normal)),
            "median_constant_field_rmse_voxels": float(np.median(constant)),
            "constant_wins_vs_normal": int(np.sum(constant < normal)),
            "median_oracle_straight_rmse_voxels": float(np.median(straight)),
            "median_oracle_quadratic_rmse_voxels": float(np.median(quadratic)),
            "quadratic_better_than_straight_tracks": int(np.sum(quadratic < straight)),
        },
        "claim_limit": (
            "Same specimen and same annotator. A positive result would validate regional "
            "interpolation of manual axes, not automatic cone discovery or fossil transfer."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, plot_axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=180)
    colours = ["#0072B2", "#D55E00", "#009E73"]
    training_coordinates = np.asarray(
        frozen["secondary_affine_model"]["training_coordinates"], float
    )
    patch_4_coordinate = np.asarray(
        frozen["secondary_affine_model"]["patch_4_coordinate"], float
    )
    for index, coordinate in enumerate(training_coordinates):
        plot_axes[0].scatter(*coordinate, s=55, color=colours[index])
        plot_axes[0].annotate(f"Patch {index + 1}", coordinate, xytext=(5, 5), textcoords="offset points")
    plot_axes[0].scatter(*patch_4_coordinate, s=70, marker="*", color="#CC79A7")
    plot_axes[0].annotate("Patch 4", patch_4_coordinate, xytext=(5, 5), textcoords="offset points")
    plot_axes[0].set_xlabel("Whole-eye PCA coordinate 1")
    plot_axes[0].set_ylabel("Whole-eye PCA coordinate 2")
    plot_axes[0].set_title("A  Frozen regional test")

    for left, right in zip(normal, primary):
        plot_axes[1].plot((0, 1), (left, right), color="0.72", linewidth=0.9)
        plot_axes[1].scatter((0, 1), (left, right), s=24, color=("#666666", "#CC79A7"))
    plot_axes[1].set_xticks((0, 1), ("Surface normal", "Frozen spatial field"))
    plot_axes[1].set_ylabel("Held-out 3D path RMSE (voxels)")
    plot_axes[1].set_title("B  Per-track prediction error")

    observed = np.asarray([[row["observed_skew_1"], row["observed_skew_2"]] for row in records])
    plot_axes[2].scatter(observed[:, 0], observed[:, 1], color="#56B4E9", label="Patch 4 tracks")
    plot_axes[2].scatter(*primary_skew, marker="*", s=120, color="#CC79A7", label="Frozen prediction")
    plot_axes[2].axhline(0, color="0.8", linewidth=0.8)
    plot_axes[2].axvline(0, color="0.8", linewidth=0.8)
    plot_axes[2].set_xlabel("Eye-frame skew component 1")
    plot_axes[2].set_ylabel("Eye-frame skew component 2")
    plot_axes[2].set_title("C  Observed and predicted axes")
    plot_axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_50_patch4_spatial_field.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

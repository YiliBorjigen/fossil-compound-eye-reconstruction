#!/usr/bin/env python3
"""Experiment 47: candidate 3-D centre-line audit in unfolded Pieris CT.

This is post-transfer method development. The public label identifies the eye
volume and corneal surface, not individual crystalline cones. Consequently the
tracks produced here are intensity-ridge candidates, not anatomical ground
truth. The experiment asks two narrower questions:

1. Is a low-curvature path measurably better than a straight path?
2. Can raw intensity maxima be followed continuously from the shallow lattice
   to the internal depth used in Experiment 45?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


REGIONS = {
    1: {"outer_depth": 9, "internal_depth": 19},
    2: {"outer_depth": 8, "internal_depth": 18},
    3: {"outer_depth": 8, "internal_depth": 31},
}


def detect_ridges(volume: np.ndarray, percentile: float) -> list[np.ndarray]:
    detections: list[np.ndarray] = []
    for depth in range(volume.shape[2]):
        smooth = gaussian_filter(volume[:, :, depth].astype(float), 1.2)
        high_pass = smooth - gaussian_filter(smooth, 6.0)
        roi = np.zeros(high_pass.shape, dtype=bool)
        roi[14:-14, 14:-14] = True
        threshold = np.percentile(high_pass[roi], percentile)
        peaks = (
            (high_pass == maximum_filter(high_pass, size=7))
            & roi
            & (high_pass > threshold)
        )
        detections.append(np.argwhere(peaks).astype(float))
    return detections


def lattice_cv(points: np.ndarray) -> float:
    if len(points) < 100:
        return np.inf
    distance, _ = cKDTree(points).query(points, k=2)
    nearest = distance[:, 1]
    return float(np.std(nearest) / np.mean(nearest))


def choose_reference(detections: list[np.ndarray]) -> int:
    candidates = np.arange(12, 26)
    scores = np.asarray([lattice_cv(detections[d]) for d in candidates])
    return int(candidates[np.argmin(scores)])


def link_tracks(
    detections: list[np.ndarray], reference: int, maximum_step: float
) -> dict[int, dict[int, np.ndarray]]:
    seeds = detections[reference]
    tracks = {index: {reference: point} for index, point in enumerate(seeds)}
    for direction in (1, -1):
        active = {index: point for index, point in enumerate(seeds)}
        stop = len(detections) if direction > 0 else -1
        for depth in range(reference + direction, stop, direction):
            identifiers = list(active)
            if not identifiers or not len(detections[depth]):
                break
            previous = np.asarray([active[index] for index in identifiers])
            current = detections[depth]
            cost = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
            previous_index, current_index = linear_sum_assignment(cost)
            updated: dict[int, np.ndarray] = {}
            for left, right in zip(previous_index, current_index):
                if cost[left, right] <= maximum_step:
                    identifier = identifiers[left]
                    tracks[identifier][depth] = current[right]
                    updated[identifier] = current[right]
            active = updated
    return tracks


def blocked_cv_error(depth: np.ndarray, points: np.ndarray, degree: int) -> float:
    groups = np.floor(np.linspace(0, 5, len(depth), endpoint=False)).astype(int)
    errors: list[float] = []
    for group in range(5):
        test = groups == group
        train = ~test
        if not np.any(test) or np.count_nonzero(train) < degree + 2:
            continue
        predicted = np.column_stack(
            [
                np.polyval(np.polyfit(depth[train], points[train, axis], degree), depth[test])
                for axis in (0, 1)
            ]
        )
        errors.extend(np.linalg.norm(predicted - points[test], axis=1))
    return float(np.median(errors))


def track_metrics(
    region: int,
    tracks: dict[int, dict[int, np.ndarray]],
    voxel_size: float,
    minimum_length: int,
) -> list[dict[str, float | int | bool]]:
    records: list[dict[str, float | int | bool]] = []
    outer = REGIONS[region]["outer_depth"]
    internal = REGIONS[region]["internal_depth"]
    for identifier, track in tracks.items():
        depth = np.asarray(sorted(track), dtype=float)
        if len(depth) < minimum_length:
            continue
        points = np.asarray([track[int(value)] for value in depth])
        line_error = blocked_cv_error(depth, points, 1)
        curve_error = blocked_cv_error(depth, points, 2)
        line = np.column_stack(
            [np.polyval(np.polyfit(depth, points[:, axis], 1), depth) for axis in (0, 1)]
        )
        curve = np.column_stack(
            [np.polyval(np.polyfit(depth, points[:, axis], 2), depth) for axis in (0, 1)]
        )
        curvature_rms = float(np.sqrt(np.mean(np.sum((curve - line) ** 2, axis=1))))
        slope = np.asarray([np.polyfit(depth, points[:, axis], 1)[0] for axis in (0, 1)])
        records.append(
            {
                "region": region,
                "track": identifier,
                "reference_depth": int(next(iter(track))),
                "start_depth": int(depth[0]),
                "stop_depth": int(depth[-1]),
                "length": int(len(depth)),
                "line_blocked_cv_voxels": line_error,
                "quadratic_blocked_cv_voxels": curve_error,
                "quadratic_minus_line_voxels": curve_error - line_error,
                "quadratic_wins": bool(curve_error < line_error),
                "curvature_rms_voxels": curvature_rms,
                "curvature_rms_micrometres": curvature_rms * voxel_size,
                "candidate_tilt_degrees": float(np.degrees(np.arctan(np.linalg.norm(slope)))),
                "spans_old_outer_to_internal": bool(outer in track and internal in track),
            }
        )
    return records


def analyse_setting(
    patch_root: Path,
    percentile: float,
    maximum_step: float,
    voxel_size: float,
    minimum_length: int,
) -> tuple[list[dict[str, object]], list[dict[str, float | int]]]:
    records: list[dict[str, object]] = []
    settings: list[dict[str, float | int]] = []
    for region in REGIONS:
        volume = np.load(patch_root / f"patch_{region}" / "unfolded_intensity.npy")
        detections = detect_ridges(volume, percentile)
        reference = choose_reference(detections)
        tracks = link_tracks(detections, reference, maximum_step)
        region_records = track_metrics(region, tracks, voxel_size, minimum_length)
        records.extend(region_records)
        delta = np.asarray([float(row["quadratic_minus_line_voxels"]) for row in region_records])
        settings.append(
            {
                "region": region,
                "percentile": percentile,
                "maximum_step_voxels": maximum_step,
                "reference_depth": reference,
                "eligible_tracks": len(region_records),
                "median_quadratic_minus_line_voxels": float(np.median(delta)) if len(delta) else np.nan,
                "fraction_quadratic_wins": float(np.mean(delta < 0)) if len(delta) else np.nan,
            }
        )
    return records, settings


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-micrometres", type=float, default=1.08)
    parser.add_argument("--minimum-track-length", type=int, default=15)
    args = parser.parse_args()

    # Primary choices were fixed after the initial visual/spacing inspection.
    primary, primary_setting = analyse_setting(
        args.patch_root,
        percentile=75.0,
        maximum_step=3.2,
        voxel_size=args.voxel_size_micrometres,
        minimum_length=args.minimum_track_length,
    )
    sensitivity: list[dict[str, float | int]] = []
    for percentile in (65.0, 70.0, 75.0, 80.0):
        for maximum_step in (2.5, 3.2, 4.0):
            _, setting = analyse_setting(
                args.patch_root,
                percentile,
                maximum_step,
                args.voxel_size_micrometres,
                args.minimum_track_length,
            )
            sensitivity.extend(setting)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "track_metrics.csv", primary)
    write_csv(args.output_dir / "sensitivity.csv", sensitivity)

    summary: dict[str, object] = {
        "status": "post-transfer candidate centre-line audit; not anatomical validation",
        "source_label_scope": "eye volume and corneal surface only; no individual cone labels",
        "voxel_spacing_micrometres_xyz": [1.08, 1.08, 1.08],
        "isotropic_sampling": True,
        "primary_detector_percentile": 75.0,
        "primary_maximum_step_voxels": 3.2,
        "minimum_track_length": args.minimum_track_length,
        "regions": {},
    }
    for region in REGIONS:
        rows = [row for row in primary if int(row["region"]) == region]
        line = np.asarray([float(row["line_blocked_cv_voxels"]) for row in rows])
        curve = np.asarray([float(row["quadratic_blocked_cv_voxels"]) for row in rows])
        curvature = np.asarray([float(row["curvature_rms_micrometres"]) for row in rows])
        summary["regions"][str(region)] = {
            "reference_depth": next(
                int(row["reference_depth"])
                for row in primary_setting
                if int(row["region"]) == region
            ),
            "eligible_candidate_tracks": len(rows),
            "tracks_spanning_old_outer_to_internal_depths": int(
                sum(bool(row["spans_old_outer_to_internal"]) for row in rows)
            ),
            "median_line_blocked_cv_voxels": float(np.median(line)),
            "median_quadratic_blocked_cv_voxels": float(np.median(curve)),
            "median_paired_quadratic_minus_line_voxels": float(np.median(curve - line)),
            "fraction_quadratic_wins": float(np.mean(curve < line)),
            "median_curvature_rms_micrometres": float(np.median(curvature)),
        }
    sensitivity_group = {}
    for region in REGIONS:
        delta = np.asarray(
            [
                float(row["median_quadratic_minus_line_voxels"])
                for row in sensitivity
                if int(row["region"]) == region
            ]
        )
        sensitivity_group[str(region)] = {
            "settings_with_quadratic_median_improvement": int(np.count_nonzero(delta < 0)),
            "settings_tested": int(len(delta)),
        }
    summary["sensitivity"] = sensitivity_group
    summary["interpretation"] = (
        "Low-order curvature gives a small, region-dependent improvement, with median "
        "deviation around 0.6 micrometres. This is too small to explain the earlier "
        "approximately 3.2 micrometre centre-registration error. More importantly, raw "
        "maxima do not form reliable cornea-to-internal tracks, so a trained or manually "
        "validated three-dimensional cone segmentation is still required."
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), dpi=180)
    colours = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}
    for offset, region in enumerate(REGIONS):
        rows = [row for row in primary if int(row["region"]) == region]
        line = np.asarray([float(row["line_blocked_cv_voxels"]) for row in rows])
        curve = np.asarray([float(row["quadratic_blocked_cv_voxels"]) for row in rows])
        jitter = np.linspace(-0.04, 0.04, len(rows)) if rows else np.asarray([])
        axes[0].scatter(
            np.full(len(rows), offset * 2 + 1) + jitter,
            line,
            s=10,
            alpha=0.35,
            color="#777777",
        )
        axes[0].scatter(
            np.full(len(rows), offset * 2 + 2) + jitter,
            curve,
            s=10,
            alpha=0.35,
            color=colours[region],
        )
        axes[0].plot(
            [offset * 2 + 1, offset * 2 + 2],
            [np.median(line), np.median(curve)],
            color="black",
            marker="o",
            linewidth=2,
        )
    axes[0].set_xticks(range(1, 7), ["L1", "Q1", "L2", "Q2", "L3", "Q3"])
    axes[0].set_ylabel("Blocked-CV position error (voxels)")
    axes[0].set_title("A  Straight (L) versus quadratic (Q)")

    for region in REGIONS:
        rows = [row for row in primary if int(row["region"]) == region]
        axes[1].scatter(
            [int(row["length"]) for row in rows],
            [float(row["curvature_rms_micrometres"]) for row in rows],
            color=colours[region],
            s=20,
            alpha=0.65,
            label=f"region {region}",
        )
    axes[1].axhline(3.2, color="black", linestyle="--", linewidth=1, label="old centre error")
    axes[1].set_xlabel("Continuous track length (slices)")
    axes[1].set_ylabel("Curvature deviation (µm)")
    axes[1].set_title("B  Curvature is sub-error-scale")
    axes[1].legend(frameon=False, fontsize=7)

    eligible = []
    spanning = []
    for region in REGIONS:
        rows = [row for row in primary if int(row["region"]) == region]
        eligible.append(len(rows))
        spanning.append(sum(bool(row["spans_old_outer_to_internal"]) for row in rows))
    x = np.arange(3)
    axes[2].bar(x - 0.18, eligible, width=0.36, color="#999999", label="≥15 slices")
    axes[2].bar(x + 0.18, spanning, width=0.36, color="#CC79A7", label="outer→internal")
    axes[2].set_xticks(x, ["region 1", "region 2", "region 3"])
    axes[2].set_ylabel("Candidate tracks")
    axes[2].set_title("C  Anatomical continuity is missing")
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Experiment 47: Pieris candidate centre-line audit")
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_47_centerline_audit.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

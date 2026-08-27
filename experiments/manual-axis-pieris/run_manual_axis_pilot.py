#!/usr/bin/env python3
"""Experiment 48: diagnostic value of manually traced Pieris cone axes.

This is a within-region, oracle-axis pilot. It tests whether placing a
leave-one-cone-out residual template on a human-traced path improves the hidden
cone core. It is not an independent validation and does not train or score an
automatic segmenter.
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
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import shift
from scipy.ndimage import map_coordinates


METHODS = (
    "local_depth_background",
    "surface_normal_static",
    "oracle_fitted_straight_axis",
    "oracle_fitted_quadratic_axis",
    "raw_manual_path",
    "permuted_quadratic_axis",
)


def dense_nodes(nodes: list[dict[str, float]], depths: np.ndarray) -> np.ndarray:
    ordered = sorted(nodes, key=lambda item: item["depth"])
    source_depth = np.asarray([item["depth"] for item in ordered], float)
    source_row = np.asarray([item["row"] for item in ordered], float)
    source_column = np.asarray([item["column"] for item in ordered], float)
    if len(source_depth) >= 3:
        row = PchipInterpolator(source_depth, source_row)(depths)
        column = PchipInterpolator(source_depth, source_column)(depths)
    else:
        row = np.interp(depths, source_depth, source_row)
        column = np.interp(depths, source_depth, source_column)
    return np.column_stack((row, column))


def extract_path_patch(
    volume: np.ndarray, path: np.ndarray, depths: np.ndarray, radius: int
) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=float)
    row_offset, column_offset = np.meshgrid(offsets, offsets, indexing="ij")
    result = np.empty((len(offsets), len(offsets), len(depths)), float)
    for index, (depth, centre) in enumerate(zip(depths, path)):
        coordinates = np.asarray(
            (row_offset + centre[0], column_offset + centre[1])
        )
        result[:, :, index] = map_coordinates(
            volume[:, :, int(depth)].astype(float),
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        )
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.ravel() - np.mean(left)
    y = right.ravel() - np.mean(right)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else 0.0


def fit_straight_path(path: np.ndarray, depths: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(depths)), depths - depths[0]))
    coefficient, *_ = np.linalg.lstsq(design, path, rcond=None)
    return design @ coefficient


def fit_quadratic_path(path: np.ndarray, depths: np.ndarray) -> np.ndarray:
    relative_depth = depths - np.mean(depths)
    design = np.column_stack(
        (np.ones(len(depths)), relative_depth, relative_depth**2)
    )
    coefficient, *_ = np.linalg.lstsq(design, path, rcond=None)
    return design @ coefficient


def shifted_template(template: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    output = np.empty_like(template)
    for index in range(template.shape[2]):
        displacement = predicted[index] - truth[index]
        output[:, :, index] = shift(
            template[:, :, index],
            shift=tuple(displacement),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-cones", default="1,2,3,4,5")
    parser.add_argument("--depth-start", type=int, default=34)
    parser.add_argument("--depth-stop", type=int, default=50)
    parser.add_argument(
        "--drop-nodes",
        action="append",
        default=[],
        metavar="CONE_ID:DEPTH,DEPTH",
        help="Documented continuity-audit correction; may be supplied more than once",
    )
    args = parser.parse_args()

    volume = np.load(args.volume)
    project = json.loads(args.annotations.read_text())
    original_project = json.loads(json.dumps(project))

    corrections = []
    requested_drops: dict[int, set[int]] = {}
    for specification in args.drop_nodes:
        cone_text, depth_text = specification.split(":", maxsplit=1)
        requested_drops.setdefault(int(cone_text), set()).update(
            int(value) for value in depth_text.split(",")
        )
    for cone in project["cones"]:
        cone_id = int(cone["id"])
        depths_to_drop = requested_drops.get(cone_id, set())
        if depths_to_drop:
            removed = [
                node for node in cone["nodes"] if int(node["depth"]) in depths_to_drop
            ]
            cone["nodes"] = [
                node for node in cone["nodes"] if int(node["depth"]) not in depths_to_drop
            ]
            corrections.append(
                {
                    "cone_id": cone_id,
                    "action": "removed nodes after continuity audit",
                    "depths": sorted(depths_to_drop),
                    "reason": "explicit command-line correction; original archive preserved",
                    "removed_nodes": removed,
                }
            )

    project["source_volume"] = f"{args.volume.parent.name}/{args.volume.name}"
    project["audit_corrections"] = corrections
    project["original_annotation_file_sha256"] = hashlib.sha256(
        args.annotations.read_bytes()
    ).hexdigest()

    selected_ids = tuple(int(value) for value in args.selected_cones.split(","))
    depths = np.arange(args.depth_start, args.depth_stop + 1, dtype=int)
    paths = {}
    for cone in project["cones"]:
        cone_id = int(cone["id"])
        node_depths = [int(item["depth"]) for item in cone["nodes"]]
        if cone_id in selected_ids and min(node_depths) <= depths[0] and max(node_depths) >= depths[-1]:
            paths[cone_id] = dense_nodes(cone["nodes"], depths)
    if tuple(sorted(paths)) != selected_ids:
        raise RuntimeError(f"Required overlapping tracks not found: {sorted(paths)}")

    radius = 6
    rows, columns = np.indices((2 * radius + 1, 2 * radius + 1))
    radial_distance = np.hypot(rows - radius, columns - radius)
    core = radial_distance <= 2.5
    annulus = (radial_distance >= 4.5) & (radial_distance <= 6.0)
    core_3d = np.repeat(core[:, :, None], len(depths), axis=2)
    patches = {
        cone_id: extract_path_patch(volume, path, depths, radius)
        for cone_id, path in paths.items()
    }

    records: list[dict[str, float | int | str]] = []
    for test_id in selected_ids:
        training_ids = [cone_id for cone_id in selected_ids if cone_id != test_id]
        residuals = []
        for cone_id in training_ids:
            training = patches[cone_id]
            profile = np.median(training[annulus, :], axis=0)
            contrast = max(
                float(np.percentile(training[annulus, :], 95) - np.percentile(training[annulus, :], 5)),
                1.0,
            )
            residuals.append((training - profile[None, None, :]) / contrast)
        template = np.median(np.stack(residuals), axis=0)

        observed = patches[test_id]
        test_profile = np.median(observed[annulus, :], axis=0)
        test_contrast = max(
            float(np.percentile(observed[annulus, :], 95) - np.percentile(observed[annulus, :], 5)),
            1.0,
        )
        background = np.broadcast_to(test_profile, observed.shape).copy()
        truth_path = paths[test_id]
        static_path = np.repeat(truth_path[[0]], len(depths), axis=0)
        straight_path = fit_straight_path(truth_path, depths)
        quadratic_path = fit_quadratic_path(truth_path, depths)
        next_id = selected_ids[(selected_ids.index(test_id) + 1) % len(selected_ids)]
        other = paths[next_id]
        other_quadratic = fit_quadratic_path(other, depths)
        permuted_path = truth_path[[0]] + (other_quadratic - other_quadratic[[0]])
        method_images = {
            "local_depth_background": background,
            "surface_normal_static": background
            + test_contrast * shifted_template(template, static_path, truth_path),
            "oracle_fitted_straight_axis": background
            + test_contrast * shifted_template(template, straight_path, truth_path),
            "oracle_fitted_quadratic_axis": background
            + test_contrast * shifted_template(template, quadratic_path, truth_path),
            "raw_manual_path": background + test_contrast * template,
            "permuted_quadratic_axis": background
            + test_contrast * shifted_template(template, permuted_path, truth_path),
        }
        dynamic_range = max(
            float(np.percentile(observed[core_3d], 95) - np.percentile(observed[core_3d], 5)),
            1.0,
        )
        for method, prediction in method_images.items():
            error = np.abs(observed[core_3d] - prediction[core_3d])
            records.append(
                {
                    "cone_id": test_id,
                    "method": method,
                    "normalised_mae": float(np.mean(error) / dynamic_range),
                    "correlation": correlation(observed[core_3d], prediction[core_3d]),
                    "path_shift_voxels": float(np.linalg.norm(truth_path[-1] - truth_path[0])),
                    "straight_fit_rmse_voxels": float(
                        np.sqrt(np.mean(np.sum((truth_path - straight_path) ** 2, axis=1)))
                    ),
                    "quadratic_fit_rmse_voxels": float(
                        np.sqrt(np.mean(np.sum((truth_path - quadratic_path) ** 2, axis=1)))
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "corrected_annotations.json").write_text(
        json.dumps(project, indent=2) + "\n"
    )
    with (args.output_dir / "per_cone_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary: dict[str, object] = {
        "status": "within-region oracle-axis diagnostic; not independent validation",
        "source_annotation_cones": len(original_project["cones"]),
        "source_background_points": len(original_project.get("background_points", [])),
        "documented_corrections": corrections,
        "evaluated_cones": list(selected_ids),
        "evaluation_depths": [int(depths[0]), int(depths[-1])],
        "evaluation_voxel_size_micrometres": float(project["voxel_size_micrometres"]),
        "methods": {},
    }
    for method in METHODS:
        subset = [record for record in records if record["method"] == method]
        summary["methods"][method] = {
            "median_normalised_mae": float(np.median([record["normalised_mae"] for record in subset])),
            "median_correlation": float(np.median([record["correlation"] for record in subset])),
            "wins_vs_background": int(
                sum(
                    record["normalised_mae"]
                    < next(
                        item["normalised_mae"]
                        for item in records
                        if item["cone_id"] == record["cone_id"]
                        and item["method"] == "local_depth_background"
                    )
                    for record in subset
                )
            ),
        }
    summary["median_path_shift_voxels"] = float(
        np.median(
            [
                record["path_shift_voxels"]
                for record in records
                if record["method"] == "local_depth_background"
            ]
        )
    )
    summary["median_straight_fit_rmse_voxels"] = float(
        np.median(
            [
                record["straight_fit_rmse_voxels"]
                for record in records
                if record["method"] == "local_depth_background"
            ]
        )
    )
    summary["median_quadratic_fit_rmse_voxels"] = float(
        np.median(
            [
                record["quadratic_fit_rmse_voxels"]
                for record in records
                if record["method"] == "local_depth_background"
            ]
        )
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), dpi=180)
    for cone_id, path in paths.items():
        relative = path - path[[0]]
        axes[0].plot(relative[:, 1], relative[:, 0], marker="o", markersize=2, label=f"cone {cone_id}")
    axes[0].axhline(0, color="0.75", linewidth=0.7)
    axes[0].axvline(0, color="0.75", linewidth=0.7)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("Column shift (voxels)")
    axes[0].set_ylabel("Row shift (voxels)")
    axes[0].set_title(
        f"A  Human-traced paths, depths {depths[0]}–{depths[-1]}"
    )
    axes[0].legend(fontsize=7, frameon=False)

    labels = [
        "Background",
        "Static normal",
        "Straight axis",
        "Quadratic axis",
        "Raw manual path",
        "Permuted quadratic",
    ]
    values = [
        [record["normalised_mae"] for record in records if record["method"] == method]
        for method in METHODS
    ]
    axes[1].boxplot(values, tick_labels=labels, showfliers=False)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("Normalised MAE in hidden core")
    axes[1].set_title("B  Leave-one-cone-out diagnostic")

    background = {
        int(record["cone_id"]): float(record["normalised_mae"])
        for record in records
        if record["method"] == "local_depth_background"
    }
    for method, label in zip(METHODS[1:], labels[1:]):
        differences = [
            float(record["normalised_mae"]) - background[int(record["cone_id"])]
            for record in records
            if record["method"] == method
        ]
        axes[2].scatter(range(1, len(differences) + 1), differences, label=label, s=24)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xticks(range(1, len(selected_ids) + 1), [str(value) for value in selected_ids])
    axes[2].set_xlabel("Held-out manually traced cone")
    axes[2].set_ylabel("Method − background NMAE")
    axes[2].set_title("C  Negative values improve on background")
    axes[2].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_48_manual_axis_pilot.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

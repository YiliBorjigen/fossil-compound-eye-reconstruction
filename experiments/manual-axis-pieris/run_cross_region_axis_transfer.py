#!/usr/bin/env python3
"""Experiment 49: transfer a simple cone-skew model between Pieris regions.

Manual centre-lines are converted from each unfolded patch back into the common
scanner coordinate system.  Cone skew is represented in a deterministic local
tangent frame derived from the whole corneal surface.  Patch 1 trains a
constant-skew model that is tested on Patch 2.  Patches 1 and 2 then train the
same model before a held-out test on Patch 3.

This tests one deliberately simple predictor.  It does not claim that the
manual paths are anatomical ground truth or that a constant-skew model is the
only possible spatial model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class Track:
    region: str
    cone_id: int
    depths: np.ndarray
    xyz: np.ndarray
    normal: np.ndarray
    tangent_1: np.ndarray
    tangent_2: np.ndarray
    skew: np.ndarray
    straight_fit: np.ndarray


def canonical_axis(vector: np.ndarray) -> np.ndarray:
    """Give a PCA axis a reproducible sign."""
    vector = np.asarray(vector, float).copy()
    largest = int(np.argmax(np.abs(vector)))
    if vector[largest] < 0:
        vector *= -1
    return vector / np.linalg.norm(vector)


def eye_axes(label_path: Path, surface_value: int = 7) -> np.ndarray:
    label = np.load(label_path, mmap_mode="r")
    points = []
    for start in range(0, label.shape[2], 64):
        slab = np.asarray(label[:, :, start : start + 64])
        selected = np.argwhere(slab == surface_value)
        if len(selected):
            selected[:, 2] += start
            points.append(selected)
    if not points:
        raise RuntimeError(f"No surface value {surface_value} in {label_path}")
    surface = np.concatenate(points).astype(float)
    stride = max(1, len(surface) // 100_000)
    sample = surface[::stride]
    _, _, axes = np.linalg.svd(sample - np.mean(sample, axis=0), full_matrices=False)
    return np.asarray([canonical_axis(axis) for axis in axes])


def tangent_frame(normal: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent_1 = axes[0] - np.dot(axes[0], normal) * normal
    if np.linalg.norm(tangent_1) < 0.2:
        tangent_1 = axes[1] - np.dot(axes[1], normal) * normal
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


def sample_vector_field(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    coordinates = np.asarray((uv[:, 0], uv[:, 1]))
    return np.column_stack(
        [
            map_coordinates(
                field[:, :, component],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for component in range(3)
        ]
    )


def load_tracks(
    region: str,
    patch_dir: Path,
    annotation_path: Path,
    axes: np.ndarray,
    selected_ids: set[int] | None = None,
    depth_window: tuple[int, int] | None = None,
    minimum_nodes: int = 6,
) -> list[Track]:
    surface = np.load(patch_dir / "surface_xyz.npy")
    normals = np.load(patch_dir / "inward_normals.npy")
    project = json.loads(annotation_path.read_text())
    tracks = []
    for cone in project["cones"]:
        cone_id = int(cone["id"])
        if selected_ids is not None and cone_id not in selected_ids:
            continue
        nodes = sorted(cone["nodes"], key=lambda node: node["depth"])
        if depth_window is not None:
            nodes = [
                node
                for node in nodes
                if depth_window[0] <= int(node["depth"]) <= depth_window[1]
            ]
        if len(nodes) < minimum_nodes:
            continue
        depths = np.asarray([node["depth"] for node in nodes], float)
        uv = np.asarray([[node["row"], node["column"]] for node in nodes], float)
        sampled_surface = sample_vector_field(surface, uv)
        sampled_normals = sample_vector_field(normals, uv)
        sampled_normals /= np.linalg.norm(sampled_normals, axis=1, keepdims=True)
        xyz = sampled_surface + depths[:, None] * sampled_normals

        relative_depth = depths - depths[0]
        design = np.column_stack((np.ones(len(depths)), relative_depth))
        coefficients, *_ = np.linalg.lstsq(design, xyz, rcond=None)
        straight_fit = design @ coefficients
        direction = coefficients[1]
        normal = sampled_normals[0]
        normal_component = float(np.dot(direction, normal))
        if normal_component <= 0:
            raise RuntimeError(f"Non-inward track: {region} cone {cone_id}")
        direction /= normal_component
        tangent_1, tangent_2 = tangent_frame(normal, axes)
        skew = np.asarray(
            (float(np.dot(direction, tangent_1)), float(np.dot(direction, tangent_2)))
        )
        tracks.append(
            Track(
                region=region,
                cone_id=cone_id,
                depths=depths,
                xyz=xyz,
                normal=normal,
                tangent_1=tangent_1,
                tangent_2=tangent_2,
                skew=skew,
                straight_fit=straight_fit,
            )
        )
    return tracks


def fit_constant_skew(tracks: list[Track]) -> np.ndarray:
    return np.median(np.asarray([track.skew for track in tracks]), axis=0)


def path_rmse(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((truth - predicted) ** 2, axis=1))))


def evaluate(
    tracks: list[Track], skew: np.ndarray, training_regions: str, test_name: str
) -> list[dict[str, float | int | str]]:
    records = []
    for track in tracks:
        relative_depth = (track.depths - track.depths[0])[:, None]
        anchor = track.xyz[[0]]
        normal_prediction = anchor + relative_depth * track.normal
        transferred_direction = (
            track.normal
            + skew[0] * track.tangent_1
            + skew[1] * track.tangent_2
        )
        transferred_prediction = anchor + relative_depth * transferred_direction
        records.append(
            {
                "test": test_name,
                "training_regions": training_regions,
                "region": track.region,
                "cone_id": track.cone_id,
                "depth_start": float(track.depths[0]),
                "depth_stop": float(track.depths[-1]),
                "nodes": len(track.depths),
                "observed_skew_1": float(track.skew[0]),
                "observed_skew_2": float(track.skew[1]),
                "predicted_skew_1": float(skew[0]),
                "predicted_skew_2": float(skew[1]),
                "normal_path_rmse_voxels": path_rmse(track.xyz, normal_prediction),
                "transferred_path_rmse_voxels": path_rmse(
                    track.xyz, transferred_prediction
                ),
                "oracle_straight_path_rmse_voxels": path_rmse(
                    track.xyz, track.straight_fit
                ),
                "normal_endpoint_error_voxels": float(
                    np.linalg.norm(track.xyz[-1] - normal_prediction[-1])
                ),
                "transferred_endpoint_error_voxels": float(
                    np.linalg.norm(track.xyz[-1] - transferred_prediction[-1])
                ),
            }
        )
    return records


def test_summary(records: list[dict[str, float | int | str]]) -> dict[str, object]:
    normal = np.asarray([record["normal_path_rmse_voxels"] for record in records], float)
    transferred = np.asarray(
        [record["transferred_path_rmse_voxels"] for record in records], float
    )
    oracle = np.asarray(
        [record["oracle_straight_path_rmse_voxels"] for record in records], float
    )
    return {
        "tracks": len(records),
        "median_normal_path_rmse_voxels": float(np.median(normal)),
        "median_transferred_path_rmse_voxels": float(np.median(transferred)),
        "median_oracle_straight_path_rmse_voxels": float(np.median(oracle)),
        "transferred_wins_vs_normal": int(np.sum(transferred < normal)),
        "median_fractional_change_vs_normal": float(
            np.median((transferred - normal) / normal)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered-label", type=Path, required=True)
    parser.add_argument("--patch-1-dir", type=Path, required=True)
    parser.add_argument("--patch-1-annotations", type=Path, required=True)
    parser.add_argument("--patch-2-dir", type=Path, required=True)
    parser.add_argument("--patch-2-annotations", type=Path, required=True)
    parser.add_argument("--patch-3-dir", type=Path, required=True)
    parser.add_argument("--patch-3-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    axes = eye_axes(args.registered_label)
    patch_1 = load_tracks(
        "patch_1",
        args.patch_1_dir,
        args.patch_1_annotations,
        axes,
        selected_ids={1, 2, 3, 4, 5},
        depth_window=(34, 50),
    )
    patch_2 = load_tracks(
        "patch_2",
        args.patch_2_dir,
        args.patch_2_annotations,
        axes,
        selected_ids={1, 2, 3, 4, 5},
        depth_window=(30, 60),
    )
    patch_3 = load_tracks(
        "patch_3",
        args.patch_3_dir,
        args.patch_3_annotations,
        axes,
        selected_ids=set(range(2, 13)),
        minimum_nodes=6,
    )
    if (len(patch_1), len(patch_2), len(patch_3)) != (5, 5, 11):
        raise RuntimeError(
            f"Unexpected track counts: {len(patch_1)}, {len(patch_2)}, {len(patch_3)}"
        )

    patch_1_model = fit_constant_skew(patch_1)
    patch_1_to_2 = evaluate(patch_2, patch_1_model, "patch_1", "patch_2")
    patch_1_2_model = fit_constant_skew(patch_1 + patch_2)
    patch_1_2_to_3 = evaluate(
        patch_3, patch_1_2_model, "patch_1+patch_2", "patch_3"
    )
    records = patch_1_to_2 + patch_1_2_to_3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_track_transfer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "held-out regional transfer diagnostic; same specimen and annotator",
        "coordinate_system": (
            "whole-eye PCA tangent frame; constant median skew coefficients"
        ),
        "patch_1_model": patch_1_model.tolist(),
        "patch_1_plus_2_model": patch_1_2_model.tolist(),
        "patch_1_to_patch_2": test_summary(patch_1_to_2),
        "patch_1_plus_2_to_patch_3": test_summary(patch_1_2_to_3),
        "interpretation": (
            "Manual cone skew is reproducible within regions, but a single "
            "eye-wide constant skew does not transfer to Patch 3. The next "
            "model must predict a spatially varying orientation field."
        ),
        "claim_limits": [
            "Manual centre-lines are not independent anatomical segmentations.",
            "All regions come from one Pieris specimen and one annotator.",
            "Patch 3 tests a constant-skew model, not every possible spatial model.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes_plot = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=180)
    colours = {"patch_1": "#0072B2", "patch_2": "#D55E00", "patch_3": "#009E73"}
    for region_tracks in (patch_1, patch_2, patch_3):
        region = region_tracks[0].region
        values = np.asarray([track.skew for track in region_tracks])
        axes_plot[0].scatter(
            values[:, 0], values[:, 1], label=region.replace("_", " "),
            color=colours[region], s=30, alpha=0.85
        )
    axes_plot[0].axhline(0, color="0.75", linewidth=0.8)
    axes_plot[0].axvline(0, color="0.75", linewidth=0.8)
    axes_plot[0].set_xlabel("Eye-frame skew component 1")
    axes_plot[0].set_ylabel("Eye-frame skew component 2")
    axes_plot[0].set_title("A  Manual axis directions differ by region")
    axes_plot[0].legend(frameon=False, fontsize=8)

    def paired_plot(axis: plt.Axes, subset: list[dict[str, object]], title: str) -> None:
        normal = np.asarray([row["normal_path_rmse_voxels"] for row in subset], float)
        transfer = np.asarray([row["transferred_path_rmse_voxels"] for row in subset], float)
        for index, (left, right) in enumerate(zip(normal, transfer)):
            axis.plot((0, 1), (left, right), color="0.7", linewidth=0.8)
            axis.scatter((0, 1), (left, right), s=22, color=("#666666", "#CC79A7"))
        axis.set_xticks((0, 1), ("Surface normal", "Transferred skew"))
        axis.set_ylabel("3D path RMSE (voxels)")
        axis.set_title(title)

    paired_plot(axes_plot[1], patch_1_to_2, "B  Patch 1 model tested on Patch 2")
    paired_plot(axes_plot[2], patch_1_2_to_3, "C  Patches 1–2 model tested on Patch 3")
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_49_cross_region_axis_transfer.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 60: same-eye reconstruction when proximal loss meets the margin.

This file was reconstructed after the original unpushed worktree was lost.
The scientific question and report title were recovered, but the historical
mask constants were not.  The constants below therefore define a new,
auditable specification; they are not represented as a byte-for-byte recovery
of the missing run.

The benchmark reuses Experiment 59's distal-only graph and predictors.  Eight
seeds per eye are chosen from the graph-derived eye boundary, without target
information, and graph-radius 1--3 proximal regions are hidden around them.
The radius-two result is primary.  This deliberately replaces Experiment 59's
five-hop interior guard with a true margin stress test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_58_cross_volume_confirmation import load_manifest, prepare_all
from experiment_59_neighbor_block_validation import (
    METHOD_ORDER,
    PATCH_RADII,
    PRIMARY_RADIUS,
    build_protocols,
    coordinate_rank,
    graph_distances,
    predict_methods,
)


SEEDS_PER_EYE = 8
MIN_SEED_SEPARATION_HOPS = 6
MAIN_METHOD = "six_neighbor_depth_shared_shape"
SPECIFICATION_STATUS = "reconstructed_after_unpushed_worktree_loss"


def select_margin_seeds(protocol) -> np.ndarray:
    """Choose deterministic, distributed seeds from the distal-only boundary."""
    candidates = np.flatnonzero(protocol.graph.boundary)
    if len(candidates) < SEEDS_PER_EYE:
        raise ValueError("Fewer boundary nodes than required margin seeds")
    rank = coordinate_rank(protocol.graph.points)
    selected = [int(candidates[np.argmin(rank[candidates])])]
    distance_to_selected = graph_distances(protocol.graph.adjacency, selected[0])
    while len(selected) < SEEDS_PER_EYE:
        pool = candidates[
            distance_to_selected[candidates] >= MIN_SEED_SEPARATION_HOPS
        ]
        if not len(pool):
            break
        farthest = np.max(distance_to_selected[pool])
        tied = pool[distance_to_selected[pool] == farthest]
        chosen = int(tied[np.argmin(rank[tied])])
        selected.append(chosen)
        distance_to_selected = np.minimum(
            distance_to_selected,
            graph_distances(protocol.graph.adjacency, chosen),
        )
    if len(selected) != SEEDS_PER_EYE:
        raise ValueError(
            f"Only {len(selected)} margin seeds satisfy the reconstructed "
            f"{MIN_SEED_SEPARATION_HOPS}-hop separation rule"
        )
    return np.asarray(selected, dtype=int)


def margin_masks(protocol, seeds: np.ndarray) -> dict[int, dict[int, np.ndarray]]:
    return {
        int(seed): {
            radius: np.flatnonzero(
                graph_distances(protocol.graph.adjacency, int(seed)) <= radius
            )
            for radius in PATCH_RADII
        }
        for seed in seeds
    }


def run_margin_validation(protocols) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    mask_rows: list[dict] = []
    diagnostics: list[dict] = []
    for protocol in protocols:
        seeds = select_margin_seeds(protocol)
        masks = margin_masks(protocol, seeds)
        qc = np.asarray(
            [record["target_qc_pass"] for record in protocol.records], dtype=bool
        )
        minimum_pair_distance = min(
            graph_distances(protocol.graph.adjacency, int(first))[int(second)]
            for position, first in enumerate(seeds)
            for second in seeds[position + 1 :]
        )
        diagnostics.append(
            {
                "volume": protocol.volume,
                "eye": protocol.eye,
                "n_distal_graph_nodes": len(protocol.records),
                "n_boundary_nodes": int(np.sum(protocol.graph.boundary)),
                "seed_landmark_ids": [
                    int(protocol.records[int(seed)]["landmark_id"]) for seed in seeds
                ],
                "all_seeds_on_distal_only_boundary": bool(
                    np.all(protocol.graph.boundary[seeds])
                ),
                "minimum_seed_separation_hops": int(minimum_pair_distance),
                "mask_sizes_by_radius": {
                    str(radius): [
                        int(len(masks[int(seed)][radius])) for seed in seeds
                    ]
                    for radius in PATCH_RADII
                },
                "mask_landmark_ids_by_radius": {
                    str(radius): [
                        [
                            int(protocol.records[int(node)]["landmark_id"])
                            for node in masks[int(seed)][radius]
                        ]
                        for seed in seeds
                    ]
                    for radius in PATCH_RADII
                },
            }
        )
        for seed_rank, seed in enumerate(seeds):
            for radius in PATCH_RADII:
                test_nodes = masks[int(seed)][radius]
                hidden = np.zeros(len(protocol.records), dtype=bool)
                hidden[test_nodes] = True
                train_nodes = np.flatnonzero(qc & ~hidden)
                predictions, neighbour_distances = predict_methods(
                    protocol, train_nodes, test_nodes
                )
                test_records = [protocol.records[int(node)] for node in test_nodes]
                mask_rows.append(
                    {
                        "volume": protocol.volume,
                        "eye": protocol.eye,
                        "seed_rank": seed_rank,
                        "seed_landmark_id": int(
                            protocol.records[int(seed)]["landmark_id"]
                        ),
                        "radius_hops": radius,
                        "hidden_distal_defined_nodes": len(test_nodes),
                        "target_resolvable_nodes": int(
                            sum(r["target_resolvable"] for r in test_records)
                        ),
                        "target_qc_nodes": int(
                            sum(r["target_qc_pass"] for r in test_records)
                        ),
                        "visible_target_qc_donors": len(train_nodes),
                    }
                )
                for position, (node, record) in enumerate(
                    zip(test_nodes, test_records)
                ):
                    if not record["target_resolvable"]:
                        continue
                    truth = np.asarray(record["thickness"], dtype=float)
                    depth = float(np.median(truth))
                    for method in METHOD_ORDER:
                        error = np.abs(predictions[method][position] - truth)
                        rows.append(
                            {
                                "volume": protocol.volume,
                                "eye": protocol.eye,
                                "seed_rank": seed_rank,
                                "radius_hops": radius,
                                "landmark_id": int(record["landmark_id"]),
                                "method": method,
                                "target_qc": bool(record["target_qc_pass"]),
                                "mae_um": float(np.mean(error)),
                                "normalized_mae": float(np.mean(error) / depth),
                                "p90_error_um": float(np.quantile(error, 0.90)),
                                "target_depth_um": depth,
                                "nearest_visible_graph_distance_um": float(
                                    neighbour_distances[position, 0]
                                ),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(mask_rows), diagnostics


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for cohort, data in (
        ("primary_target_qc", metrics[metrics["target_qc"]]),
        ("all_target_resolvable", metrics),
    ):
        summary = (
            data.groupby(["volume", "radius_hops", "method"], sort=True)
            .agg(
                n_lens_predictions=("mae_um", "size"),
                median_lens_mae_um=("mae_um", "median"),
                p90_lens_mae_um=("mae_um", lambda x: float(np.quantile(x, 0.9))),
                median_normalized_mae=("normalized_mae", "median"),
                median_target_depth_um=("target_depth_um", "median"),
            )
            .reset_index()
        )
        summary.insert(0, "cohort", cohort)
        parts.append(summary)
    combined = pd.concat(parts, ignore_index=True)
    primary = metrics[
        metrics["target_qc"] & (metrics["radius_hops"] == PRIMARY_RADIUS)
    ]
    eye = (
        primary.groupby(["volume", "eye", "method"], sort=True)
        .agg(
            n_lens_predictions=("mae_um", "size"),
            median_lens_mae_um=("mae_um", "median"),
            median_normalized_mae=("normalized_mae", "median"),
        )
        .reset_index()
    )
    return combined, eye


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    data = summary[
        (summary["cohort"] == "primary_target_qc")
        & (summary["radius_hops"] == PRIMARY_RADIUS)
        & summary["method"].isin(
            [MAIN_METHOD, "same_eye_template", "graph_harmonic_depth_shared_shape"]
        )
    ]
    methods = [MAIN_METHOD, "graph_harmonic_depth_shared_shape", "same_eye_template"]
    labels = ["Six-neighbour", "Graph harmonic", "Same-eye template"]
    volumes = sorted(data["volume"].unique())
    x = np.arange(len(volumes), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    for offset, method, label in zip((-width, 0.0, width), methods, labels):
        values = [
            float(data[(data["volume"] == volume) & (data["method"] == method)]["median_lens_mae_um"].iloc[0])
            for volume in volumes
        ]
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x, volumes)
    ax.set_ylabel("Median proximal-surface MAE (µm)")
    ax.set_title("Experiment 60: graph-radius-two loss at the eye margin")
    ax.legend(frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, source_diagnostics = prepare_all(
        load_manifest(args.manifest), include_invalid_targets=True
    )
    protocols, graph_diagnostics = build_protocols(records)
    metrics, masks, margin_diagnostics = run_margin_validation(protocols)
    summary, eye = summarize(metrics)
    metrics.to_csv(args.output / "experiment_60_per_lens_metrics.csv", index=False)
    masks.to_csv(args.output / "experiment_60_mask_counts.csv", index=False)
    summary.to_csv(args.output / "experiment_60_summary.csv", index=False)
    eye.to_csv(args.output / "experiment_60_eye_summary.csv", index=False)
    diagnostics = {
        "experiment": 60,
        "specification_status": SPECIFICATION_STATUS,
        "target_blind_mask_selection": True,
        "primary_radius_hops": PRIMARY_RADIUS,
        "patch_radii_hops": list(PATCH_RADII),
        "seeds_per_eye": SEEDS_PER_EYE,
        "minimum_seed_separation_hops": MIN_SEED_SEPARATION_HOPS,
        "source_diagnostics": source_diagnostics,
        "graph_diagnostics": graph_diagnostics,
        "margin_protocols": margin_diagnostics,
    }
    (args.output / "experiment_60_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    plot_summary(summary, args.output / "experiment_60_margin_comparison.png")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

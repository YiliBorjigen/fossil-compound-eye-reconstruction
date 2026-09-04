#!/usr/bin/env python3
"""Experiment 61: donor-survival stress test for same-eye reconstruction.

The original unpushed implementation was lost.  Its scientific question and
report title survive, but its exact thinning grid does not.  This replacement
therefore freezes and records a conservative reconstructed protocol rather
than pretending to recover unknown historical numbers.

For each eye, deterministic SHA-256 ranks retain 1, 2, 5, 10, 20 or 40 percent
of distal-defined lens identities as *potential* proximal donors in 25 repeats.
Only a selected lens with a target-QC proximal surface can actually donate.
Selection never reads proximal availability, depth, quality or error.  All
other resolvable lenses are pseudo-hidden and predicted from the surviving
same-eye donors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_58_cross_volume_confirmation import load_manifest, prepare_all
from experiment_59_neighbor_block_validation import (
    build_protocols,
    neighbour_weights,
    training_shape,
)


SURVIVAL_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
N_REPEATS = 25
N_NEIGHBOURS = 6
MAIN_METHOD = "six_neighbor_depth_shared_shape"
METHODS = (
    "donor_template",
    "nearest_visible_surface",
    "six_neighbor_surface",
    MAIN_METHOD,
)
SPECIFICATION_STATUS = "reconstructed_after_unpushed_worktree_loss"
RANK_NAMESPACE = "experiment61_survival_v1"


def stable_order(protocol, repeat: int) -> np.ndarray:
    """Return a target-blind pseudorandom permutation of distal graph nodes."""
    keys = []
    for node, record in enumerate(protocol.records):
        payload = (
            f"{RANK_NAMESPACE}|{protocol.volume}|{protocol.eye}|{repeat}|"
            f"{int(record['landmark_id'])}"
        ).encode("utf-8")
        keys.append((hashlib.sha256(payload).digest(), node))
    return np.asarray([node for _, node in sorted(keys)], dtype=int)


def predict_sparse(protocol, donors: np.ndarray, tests: np.ndarray) -> dict[str, np.ndarray]:
    donor_thickness = np.vstack(
        [protocol.records[int(node)]["thickness"] for node in donors]
    )
    donor_depth, shared_shape = training_shape(donor_thickness)
    selected, _, weights = neighbour_weights(protocol, donors, tests)
    row_for_node = {int(node): row for row, node in enumerate(donors)}
    selected_rows = np.asarray(
        [[row_for_node[int(node)] for node in row] for row in selected], dtype=int
    )
    local_depth = np.sum(weights * donor_depth[selected_rows], axis=1)
    return {
        "donor_template": np.tile(
            np.median(donor_thickness, axis=0), (len(tests), 1)
        ),
        "nearest_visible_surface": donor_thickness[selected_rows[:, 0]],
        "six_neighbor_surface": np.einsum(
            "ij,ijk->ik", weights, donor_thickness[selected_rows]
        ),
        MAIN_METHOD: local_depth[:, None] + shared_shape[None, :],
    }


def run_survival(protocols) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    inventory_rows: list[dict] = []
    for protocol in protocols:
        target_qc = np.asarray(
            [record["target_qc_pass"] for record in protocol.records], dtype=bool
        )
        resolvable = np.asarray(
            [record["target_resolvable"] for record in protocol.records], dtype=bool
        )
        for repeat in range(N_REPEATS):
            order = stable_order(protocol, repeat)
            for fraction in SURVIVAL_FRACTIONS:
                n_potential = int(np.ceil(fraction * len(order)))
                selected = np.zeros(len(order), dtype=bool)
                selected[order[:n_potential]] = True
                donors = np.flatnonzero(selected & target_qc)
                tests = np.flatnonzero(~selected & resolvable)
                inventory_rows.append(
                    {
                        "volume": protocol.volume,
                        "eye": protocol.eye,
                        "repeat": repeat,
                        "survival_fraction": fraction,
                        "n_distal_graph_nodes": len(order),
                        "n_potential_donors": n_potential,
                        "n_usable_target_qc_donors": len(donors),
                        "n_resolvable_test_lenses": len(tests),
                        "n_target_qc_test_lenses": int(np.sum(target_qc[tests])),
                        "prediction_available": len(donors) >= N_NEIGHBOURS,
                    }
                )
                if len(donors) < N_NEIGHBOURS or not len(tests):
                    continue
                predictions = predict_sparse(protocol, donors, tests)
                for position, node in enumerate(tests):
                    record = protocol.records[int(node)]
                    truth = np.asarray(record["thickness"], dtype=float)
                    depth = float(np.median(truth))
                    for method in METHODS:
                        error = np.abs(predictions[method][position] - truth)
                        metric_rows.append(
                            {
                                "volume": protocol.volume,
                                "eye": protocol.eye,
                                "repeat": repeat,
                                "survival_fraction": fraction,
                                "landmark_id": int(record["landmark_id"]),
                                "method": method,
                                "target_qc": bool(record["target_qc_pass"]),
                                "mae_um": float(np.mean(error)),
                                "normalized_mae": float(np.mean(error) / depth),
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(inventory_rows)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = metrics[metrics["target_qc"]].copy()
    repeat = (
        primary.groupby(
            ["volume", "eye", "repeat", "survival_fraction", "method"], sort=True
        )
        .agg(
            n_test_lenses=("mae_um", "size"),
            median_lens_mae_um=("mae_um", "median"),
            median_normalized_mae=("normalized_mae", "median"),
        )
        .reset_index()
    )
    eye = (
        repeat.groupby(["volume", "eye", "survival_fraction", "method"], sort=True)
        .agg(
            n_repeats=("repeat", "nunique"),
            median_repeat_mae_um=("median_lens_mae_um", "median"),
            p90_repeat_mae_um=(
                "median_lens_mae_um",
                lambda x: float(np.quantile(x, 0.90)),
            ),
            median_repeat_normalized_mae=("median_normalized_mae", "median"),
        )
        .reset_index()
    )
    summary = (
        eye.groupby(["volume", "survival_fraction", "method"], sort=True)
        .agg(
            n_eyes=("eye", "nunique"),
            median_eye_mae_um=("median_repeat_mae_um", "median"),
            worst_eye_mae_um=("median_repeat_mae_um", "max"),
            median_eye_normalized_mae=("median_repeat_normalized_mae", "median"),
            worst_eye_normalized_mae=("median_repeat_normalized_mae", "max"),
        )
        .reset_index()
    )
    return repeat, eye, summary


def operational_threshold(eye: pd.DataFrame, inventory: pd.DataFrame) -> dict:
    """Descriptive 10%-normalized-error threshold; not a universal cutoff."""
    main = eye[eye["method"] == MAIN_METHOD]
    passing = (
        main.groupby("survival_fraction")["median_repeat_normalized_mae"]
        .max()
        .sort_index()
    )
    availability = (
        inventory.groupby("survival_fraction")["prediction_available"].all().sort_index()
    )
    acceptable = passing[(passing <= 0.10) & availability.reindex(passing.index)]
    return {
        "definition": (
            "smallest tested potential-donor fraction for which every one of "
            "the six eye summaries has median normalized MAE <= 0.10 and all "
            "25 repeats have at least six usable donors"
        ),
        "descriptive_only": True,
        "threshold_fraction": (
            None if acceptable.empty else float(acceptable.index.min())
        ),
        "worst_eye_normalized_mae_by_fraction": {
            str(float(index)): float(value) for index, value in passing.items()
        },
        "all_repeats_predictable_by_fraction": {
            str(float(index)): bool(value) for index, value in availability.items()
        },
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=False, constrained_layout=True)
    for axis, volume in zip(axes, sorted(summary["volume"].unique())):
        subset = summary[(summary["volume"] == volume) & (summary["method"] == MAIN_METHOD)]
        axis.plot(
            100.0 * subset["survival_fraction"],
            subset["median_eye_mae_um"],
            marker="o",
            label="Median of eye summaries",
        )
        axis.plot(
            100.0 * subset["survival_fraction"],
            subset["worst_eye_mae_um"],
            marker="s",
            linestyle="--",
            label="Worse eye",
        )
        axis.set_xscale("log")
        axis.set_title(volume)
        axis.set_xlabel("Potential proximal donors retained (%)")
        axis.set_ylabel("Median lens MAE (µm)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 61: same-eye donor survival")
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
    metrics, inventory = run_survival(protocols)
    repeat, eye, summary = summarize(metrics)
    inventory.to_csv(args.output / "experiment_61_donor_inventory.csv", index=False)
    repeat.to_csv(args.output / "experiment_61_repeat_summary.csv", index=False)
    eye.to_csv(args.output / "experiment_61_eye_summary.csv", index=False)
    summary.to_csv(args.output / "experiment_61_summary.csv", index=False)
    diagnostics = {
        "experiment": 61,
        "specification_status": SPECIFICATION_STATUS,
        "rank_namespace": RANK_NAMESPACE,
        "target_blind_potential_donor_selection": True,
        "survival_fractions": list(SURVIVAL_FRACTIONS),
        "n_repeats": N_REPEATS,
        "minimum_usable_donors": N_NEIGHBOURS,
        "zero_donor_point_prediction": "undefined; addressed by Experiment 62 scenarios",
        "operational_threshold": operational_threshold(eye, inventory),
        "source_diagnostics": source_diagnostics,
        "graph_diagnostics": graph_diagnostics,
    }
    (args.output / "experiment_61_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    plot_summary(summary, args.output / "experiment_61_survival_curve.png")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(diagnostics["operational_threshold"], indent=2), flush=True)


if __name__ == "__main__":
    main()

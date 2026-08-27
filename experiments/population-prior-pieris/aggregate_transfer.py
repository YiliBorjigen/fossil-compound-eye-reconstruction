#!/usr/bin/env python3
"""Aggregate the three pre-selected Pieris transfer regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "background",
    "axisymmetric_population",
    "global_population",
    "nearest_cone",
    "local_population",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--apis-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    patch_summaries = []
    for patch in (1, 2, 3):
        frame = pd.read_csv(args.results_dir / f"patch_{patch}" / "per_cone_results.csv")
        frame["patch"] = patch
        frames.append(frame)
        patch_summaries.append(
            json.loads((args.results_dir / f"patch_{patch}" / "summary.json").read_text())
        )
    data = pd.concat(frames, ignore_index=True)
    fold = (
        data.groupby(["patch", "fold", "method"], as_index=False)[
            ["normalised_mae", "correlation"]
        ]
        .median()
    )
    fold_wide = fold.pivot(
        index=["patch", "fold"], columns="method", values="normalised_mae"
    )

    pooled = {}
    for method in METHODS:
        subset = data[data.method == method]
        pooled[method] = {
            "median_normalised_mae": float(subset.normalised_mae.median()),
            "median_correlation": float(subset.correlation.median()),
        }
    cone_wide = data.pivot_table(
        index=["patch", "sample", "fold"], columns="method", values="normalised_mae"
    )
    summary = {
        "status": "negative independent-specimen transfer after pre-scoring imaging adapter",
        "strict_frozen_status": "failed before reconstruction because the absolute surface threshold detected zero outer centres",
        "matched_cones": int(len(cone_wide)),
        "matched_cones_by_patch": {
            str(index + 1): int(item["matched_cones"])
            for index, item in enumerate(patch_summaries)
        },
        "folds": int(len(fold_wide)),
        "global_beats_background_folds": int(
            (fold_wide.global_population < fold_wide.background).sum()
        ),
        "axisymmetric_beats_background_folds": int(
            (fold_wide.axisymmetric_population < fold_wide.background).sum()
        ),
        "global_beats_background_cones": int(
            (cone_wide.global_population < cone_wide.background).sum()
        ),
        "median_relative_global_change_vs_background": float(
            np.median(
                (cone_wide.global_population - cone_wide.background)
                / cone_wide.background
            )
        ),
        "methods": pooled,
        "posthoc_training_selected_mapping": "did not reverse the result",
        "posthoc_four_voxel_core": "did not reverse the result",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fold.to_csv(args.output_dir / "fold_summary.csv", index=False)

    apis = json.loads(args.apis_summary.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=180)
    x = np.arange(len(fold_wide))
    axes[0].plot(x, fold_wide.background, "o", color="#777777", label="Background")
    axes[0].plot(x, fold_wide.global_population, "o", color="#0072B2", label="Population")
    for index, (_, row) in enumerate(fold_wide.iterrows()):
        colour = "#009E73" if row.global_population < row.background else "#D55E00"
        axes[0].plot([index, index], [row.background, row.global_population], color=colour, alpha=0.55)
    axes[0].set_xlabel("Held-out spatial block")
    axes[0].set_ylabel("Median normalised MAE")
    axes[0].set_title("A  Pieris transfer: population wins 4/15 blocks")
    axes[0].legend(frameon=False)

    datasets = ["Apis\nExperiment 44", "Pieris\nExperiment 45"]
    background = [
        apis["methods"]["background"]["median_normalised_mae"],
        pooled["background"]["median_normalised_mae"],
    ]
    population = [
        apis["methods"]["global_population"]["median_normalised_mae"],
        pooled["global_population"]["median_normalised_mae"],
    ]
    positions = np.arange(2)
    width = 0.34
    axes[1].bar(positions - width / 2, background, width, color="#999999", label="Background")
    axes[1].bar(positions + width / 2, population, width, color="#0072B2", label="Population")
    axes[1].set_xticks(positions, datasets)
    axes[1].set_ylabel("Median normalised MAE")
    axes[1].set_title("B  The Apis gain does not transfer")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, max(background + population) + 0.035)
    axes[1].text(0, max(background[0], population[0]) + 0.011, "population better", ha="center", fontsize=9)
    axes[1].text(1, max(background[1], population[1]) + 0.011, "population worse", ha="center", fontsize=9)
    figure.suptitle("Experiment 45: independent Pieris population-prior transfer")
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_45_transfer.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


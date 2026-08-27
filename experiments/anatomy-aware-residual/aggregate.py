#!/usr/bin/env python3
"""Aggregate the three Pieris regions used for Experiment 46."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


METHODS = (
    "local_depth_background",
    "raw_population",
    "axisymmetric_residual",
    "anatomy_residual",
    "oracle_centred_residual",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for patch in (1, 2, 3):
        frame = pd.read_csv(args.results_dir / f"patch_{patch}" / "per_cone_results.csv")
        frame["patch"] = patch
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    cone = data.pivot_table(
        index=["patch", "sample", "fold"], columns="method", values="normalised_mae"
    )
    fold = data.groupby(
        ["patch", "fold", "method"], as_index=False
    ).normalised_mae.median()
    fold_wide = fold.pivot(
        index=["patch", "fold"], columns="method", values="normalised_mae"
    )
    centre = data[data.method == METHODS[0]].set_index(["patch", "sample", "fold"])[
        "centre_error_voxels"
    ]

    rng = np.random.default_rng(46)
    methods = {}
    for method in METHODS:
        methods[method] = {
            "median_normalised_mae": float(cone[method].median()),
            "median_fold_normalised_mae": float(fold_wide[method].median()),
            "beats_background_cones": int(
                (cone[method] < cone.local_depth_background).sum()
            ) if method != METHODS[0] else None,
            "beats_background_folds": int(
                (fold_wide[method] < fold_wide.local_depth_background).sum()
            ) if method != METHODS[0] else None,
        }

    comparisons = {}
    for method in ("anatomy_residual", "oracle_centred_residual"):
        delta = (fold_wide[method] - fold_wide.local_depth_background).to_numpy()
        bootstrap = np.median(
            delta[rng.integers(0, len(delta), size=(10000, len(delta)))], axis=1
        )
        comparisons[method] = {
            "median_fold_delta_vs_background": float(np.median(delta)),
            "bootstrap_95_interval": [float(value) for value in np.percentile(bootstrap, (2.5, 97.5))],
            "paired_wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue),
        }

    anatomy_delta = cone.anatomy_residual - cone.local_depth_background
    correlation = spearmanr(centre.loc[cone.index], anatomy_delta)
    summary = {
        "status": "mechanistic method development on previously examined Pieris data",
        "claim_boundary": "not independent validation and not fossil anatomical reconstruction",
        "matched_cones": int(len(cone)),
        "spatial_blocks": int(len(fold_wide)),
        "methods": methods,
        "comparisons": comparisons,
        "centre_error_vs_anatomy_delta_spearman_r": float(correlation.statistic),
        "centre_error_vs_anatomy_delta_p": float(correlation.pvalue),
        "interpretation": (
            "Depth-only local background remains the best feasible method. "
            "The diagnostic oracle-centred residual is substantially better for most cones, "
            "which is consistent with centre/axis definition and registration being the dominant "
            "remaining bottleneck; the present two-dimensional peak targets still require "
            "three-dimensional anatomical validation."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fold.to_csv(args.output_dir / "fold_summary.csv", index=False)

    labels = ["Local depth\nbackground", "Raw\npopulation", "Axisymmetric\nresidual", "Anatomy\nresidual", "Oracle-centred\nresidual"]
    colours = ["#777777", "#CC79A7", "#E69F00", "#0072B2", "#009E73"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    values = [cone[method].to_numpy() for method in METHODS]
    boxes = axes[0].boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for box, colour in zip(boxes["boxes"], colours):
        box.set_facecolor(colour)
        box.set_alpha(0.75)
    axes[0].set_ylabel("Normalised MAE in hidden core")
    axes[0].set_title("A  Correcting intensity is not enough")
    axes[0].tick_params(axis="x", rotation=18)

    delta = cone.anatomy_residual - cone.local_depth_background
    patch_colours = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}
    for patch in (1, 2, 3):
        keep = cone.index.get_level_values("patch") == patch
        axes[1].scatter(
            centre.loc[cone.index][keep],
            delta[keep],
            color=patch_colours[patch],
            label=f"region {patch}",
            s=28,
            alpha=0.8,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Predicted centre error (voxels)")
    axes[1].set_ylabel("Anatomy residual − background MAE")
    axes[1].set_title("B  Centre error erases repeatable shape")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Experiment 46: decomposing the Pieris reconstruction error")
    figure.tight_layout()
    figure.savefig(args.output_dir / "experiment_46_error_decomposition.png", bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

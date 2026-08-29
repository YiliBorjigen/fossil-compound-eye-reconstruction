#!/usr/bin/env python3
"""Experiment 56: audit sensitivity to the operational boundary definition.

The gradient extractor and one blinded observer selected systematically
different depths within the same broad Asaphus CT transition in Experiment 55.
This script estimates that translation without using a held-out spatial block,
then checks which Experiment 54 conclusions can and cannot change under it.

Raw annotations and the unblinding key remain private.  Only aggregate outputs
and a diagnostic figure are intended for the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPACING_UM = 3.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--experiment-54-predictions", required=True, type=Path)
    parser.add_argument("--experiment-54-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--spacing-um", type=float, default=SPACING_UM)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_case_offsets(
    annotations: Path,
    private_key: Path,
    samples_path: Path,
    spacing_um: float,
) -> pd.DataFrame:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    key = pd.read_csv(private_key)
    samples = pd.read_csv(samples_path)
    rows: list[dict] = []

    for record in payload["annotations"]:
        for view, field in (("u-depth", "u_depth_points"),
                            ("v-depth", "v_depth_points")):
            for point in record.get(field, []):
                rows.append({
                    "case_id": record["case_id"],
                    "view": view,
                    "lateral_vox": float(point["lateral_vox"]),
                    "human_depth_vox": float(point["depth_vox"]),
                })
    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("No boundary clicks were found")

    points = points.merge(
        key[["case_id", "facet_id", "source_group", "cv_block"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    algorithm_depths = []
    for row in points.itertuples(index=False):
        facet = samples.loc[samples["facet_id"] == row.facet_id]
        if facet.empty:
            algorithm_depths.append(np.nan)
            continue
        pitch = float(facet["pitch"].iloc[0])
        lateral = row.lateral_vox / max(pitch / 2.0, 1e-6)
        if row.view == "u-depth":
            distance = np.hypot(
                facet["un"].to_numpy(float) - lateral,
                facet["vn"].to_numpy(float),
            )
        else:
            distance = np.hypot(
                facet["un"].to_numpy(float),
                facet["vn"].to_numpy(float) - lateral,
            )
        algorithm_depths.append(
            float(facet.iloc[int(np.argmin(distance))]["target_vox"])
        )
    points["algorithm_depth_vox"] = algorithm_depths

    accepted = points.loc[
        (points["source_group"] == "accepted")
        & points["algorithm_depth_vox"].notna()
    ].copy()
    cases = accepted.groupby("case_id", as_index=False).agg(
        facet_id=("facet_id", "first"),
        cv_block=("cv_block", "first"),
        clicked_points=("human_depth_vox", "size"),
        human_depth_vox=("human_depth_vox", "median"),
        algorithm_depth_vox=("algorithm_depth_vox", "median"),
    )
    cases["human_minus_algorithm_um"] = (
        cases["human_depth_vox"] - cases["algorithm_depth_vox"]
    ) * spacing_um
    return cases


def cross_block_calibration(cases: pd.DataFrame) -> pd.DataFrame:
    """Estimate the observer shift without using the test spatial block."""
    rows = []
    for heldout, test in cases.groupby("cv_block"):
        train = cases.loc[cases["cv_block"] != heldout]
        train_shift = float(train["human_minus_algorithm_um"].median())
        for row in test.itertuples(index=False):
            rows.append({
                "case_id": row.case_id,
                "cv_block": int(heldout),
                "observed_shift_um": row.human_minus_algorithm_um,
                "training_shift_um": train_shift,
                "uncorrected_absolute_difference_um": abs(
                    row.human_minus_algorithm_um
                ),
                "corrected_absolute_difference_um": abs(
                    row.human_minus_algorithm_um - train_shift
                ),
            })
    return pd.DataFrame(rows)


def verify_translation_invariance(
    predictions_path: Path,
    spacing_um: float,
    shift_um: float,
) -> tuple[float, int]:
    """Numerically verify that translating target and prediction preserves error."""
    frame = pd.read_csv(predictions_path)
    columns = [
        column for column in frame.columns
        if column.endswith("_prediction_norm")
    ]
    shift_norm = shift_um / (frame["pitch"].to_numpy(float) * spacing_um)
    target = frame["target_norm"].to_numpy(float)
    maximum_change = 0.0
    for column in columns:
        prediction = frame[column].to_numpy(float)
        original = np.abs(prediction - target) * (
            frame["pitch"].to_numpy(float) * spacing_um
        )
        translated = np.abs(
            (prediction + shift_norm) - (target + shift_norm)
        ) * (frame["pitch"].to_numpy(float) * spacing_um)
        maximum_change = max(
            maximum_change,
            float(np.max(np.abs(original - translated))),
        )
    return maximum_change, len(columns)


def analyse(args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    cases = load_case_offsets(
        args.annotations,
        args.private_key,
        args.samples,
        args.spacing_um,
    )
    calibrated = cross_block_calibration(cases)
    shifts = cases["human_minus_algorithm_um"].to_numpy(float)
    q10, q25, median, q75, q90 = np.percentile(shifts, [10, 25, 50, 75, 90])

    predictions = pd.read_csv(
        args.experiment_54_predictions,
        usecols=["target_vox"],
    )
    target_median_um = float(
        predictions["target_vox"].median() * args.spacing_um
    )
    maximum_change, method_count = verify_translation_invariance(
        args.experiment_54_predictions,
        args.spacing_um,
        float(median),
    )
    model_summary = pd.read_csv(args.experiment_54_summary)
    six_neighbour = model_summary.loc[
        model_summary["method"] == "six_neighbor_prior"
    ].iloc[0]

    improved = (
        calibrated["corrected_absolute_difference_um"]
        < calibrated["uncorrected_absolute_difference_um"]
    )
    summary = {
        "experiment": 56,
        "annotation_sha256": sha256(args.annotations),
        "accepted_cases_with_depth_comparison": int(len(cases)),
        "spatial_blocks": int(cases["cv_block"].nunique()),
        "median_human_minus_algorithm_um": float(median),
        "mad_human_minus_algorithm_um": float(
            np.median(np.abs(shifts - median))
        ),
        "p10_human_minus_algorithm_um": float(q10),
        "p25_human_minus_algorithm_um": float(q25),
        "p75_human_minus_algorithm_um": float(q75),
        "p90_human_minus_algorithm_um": float(q90),
        "algorithm_target_median_depth_um": target_median_um,
        "observer_calibrated_median_depth_um": float(target_median_um + median),
        "observer_translation_p10_depth_um": float(target_median_um + q10),
        "observer_translation_p90_depth_um": float(target_median_um + q90),
        "uncorrected_case_median_absolute_difference_um": float(
            calibrated["uncorrected_absolute_difference_um"].median()
        ),
        "cross_block_corrected_median_absolute_difference_um": float(
            calibrated["corrected_absolute_difference_um"].median()
        ),
        "cross_block_corrected_p90_absolute_difference_um": float(
            np.percentile(
                calibrated["corrected_absolute_difference_um"], 90
            )
        ),
        "cases_improved_by_cross_block_translation": int(improved.sum()),
        "translated_experiment_54_methods_checked": method_count,
        "translation_invariance_max_absolute_change_um": maximum_change,
        "six_neighbor_median_MAE_um_unchanged": float(
            six_neighbour["median_facet_MAE_um"]
        ),
        "six_neighbor_p90_MAE_um_unchanged": float(
            six_neighbour["p90_facet_MAE_um"]
        ),
        "scope": (
            "Operational boundary-definition sensitivity within one Asaphus "
            "scan and one non-independent observer; not anatomical identity."
        ),
    }
    return summary, calibrated.merge(
        cases[["case_id", "cv_block", "human_minus_algorithm_um"]],
        on=["case_id", "cv_block"],
        validate="one_to_one",
    )


def make_figure(summary: dict, cases: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    blue = "#3366a8"
    red = "#b04a4a"

    rng = np.random.default_rng(20260829)
    for block, group in cases.groupby("cv_block"):
        jitter = rng.uniform(-0.07, 0.07, size=len(group))
        axes[0].scatter(
            np.full(len(group), block) + jitter,
            group["human_minus_algorithm_um"],
            color=blue,
            s=34,
            alpha=0.9,
        )
    axes[0].axhline(
        summary["median_human_minus_algorithm_um"],
        color=red,
        ls="--",
        lw=1.2,
        label="overall median",
    )
    axes[0].set(
        xlabel="Held-out spatial block",
        ylabel="Human − gradient depth (µm)",
        xticks=range(summary["spatial_blocks"]),
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    before = cases["uncorrected_absolute_difference_um"].to_numpy(float)
    after = cases["corrected_absolute_difference_um"].to_numpy(float)
    for first, second in zip(before, after):
        axes[1].plot([0, 1], [first, second], color="#a9adb4", lw=0.8)
    axes[1].scatter(np.zeros(len(before)), before, color=red, s=28, zorder=3)
    axes[1].scatter(np.ones(len(after)), after, color=blue, s=28, zorder=3)
    axes[1].set(
        ylabel="Absolute difference from human click (µm)",
        xticks=[0, 1],
        xticklabels=["Uncorrected", "Training-block\ntranslation"],
    )
    axes[1].text(
        0.32,
        0.96,
        "median %.2f → %.2f µm"
        % (
            summary["uncorrected_case_median_absolute_difference_um"],
            summary["cross_block_corrected_median_absolute_difference_um"],
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )

    original = summary["algorithm_target_median_depth_um"]
    corrected = summary["observer_calibrated_median_depth_um"]
    lower = summary["observer_translation_p10_depth_um"]
    upper = summary["observer_translation_p90_depth_um"]
    axes[2].bar(
        [0, 1],
        [original, corrected],
        color=[red, blue],
        width=0.62,
    )
    axes[2].errorbar(
        [1],
        [corrected],
        yerr=[[corrected - lower], [upper - corrected]],
        fmt="none",
        ecolor="#333333",
        capsize=5,
        lw=1.3,
    )
    axes[2].set(
        ylabel="Median operational depth (µm)",
        xticks=[0, 1],
        xticklabels=["Gradient\nlandmark", "Human-centred\nlandmark"],
        ylim=(0, max(upper * 1.15, 75)),
    )
    axes[2].text(0, original + 2, f"{original:.1f}", ha="center")
    axes[2].text(1, corrected + 2, f"{corrected:.1f}", ha="center")
    axes[2].text(
        0.03,
        0.97,
        "Error ranking unchanged\nunder common translation",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8.5,
    )

    for axis, letter in zip(axes, "ABC"):
        axis.text(
            -0.16, 1.07, letter,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, cases = analyse(args)
    (args.out / "experiment_56_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    cases.to_csv(args.out / "private_case_sensitivity.csv", index=False)
    make_figure(
        summary,
        cases,
        args.out / "experiment_56_boundary_definition_sensitivity.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

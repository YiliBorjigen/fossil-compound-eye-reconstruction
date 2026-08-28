#!/usr/bin/env python3
"""Analyse a frozen blinded Asaphus boundary-annotation pilot.

Raw annotations and the unblinding key remain private.  The script writes an
aggregate summary, a private case table, and a publication-size diagnostic
figure.  Anatomical identity is deliberately outside the scored claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--spacing-um", type=float, default=3.7)
    return parser.parse_args()


def load_annotations(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = pd.DataFrame(payload["annotations"])
    points = []
    for record in payload["annotations"]:
        for view, field in (("u-depth", "u_depth_points"),
                            ("v-depth", "v_depth_points")):
            for point in record.get(field, []):
                points.append({
                    "case_id": record["case_id"],
                    "view": view,
                    "lateral_vox": float(point["lateral_vox"]),
                    "human_depth_vox": float(point["depth_vox"]),
                })
    return payload, records, pd.DataFrame(points)


def attach_algorithm_depths(
    points: pd.DataFrame,
    key: pd.DataFrame,
    samples: pd.DataFrame,
    spacing_um: float,
) -> pd.DataFrame:
    points = points.merge(
        key[["case_id", "facet_id", "source_group"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    predictions = []
    for row in points.itertuples(index=False):
        facet = samples.loc[samples["facet_id"] == row.facet_id]
        if facet.empty:
            predictions.append(np.nan)
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
        predictions.append(
            float(facet.iloc[int(np.argmin(distance))]["target_vox"])
        )
    points["algorithm_depth_vox"] = predictions
    points["signed_algorithm_minus_human_um"] = (
        points["algorithm_depth_vox"] - points["human_depth_vox"]
    ) * spacing_um
    points["absolute_difference_um"] = (
        points["signed_algorithm_minus_human_um"].abs()
    )
    return points


def analyse(args: argparse.Namespace) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    payload, records, points = load_annotations(args.annotations)
    key = pd.read_csv(args.private_key)
    samples = pd.read_csv(args.samples)

    records = records.merge(
        key[["case_id", "source_group"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    records["has_points"] = records.apply(
        lambda row: bool(row["u_depth_points"] or row["v_depth_points"]), axis=1
    )
    reviewed = records.loc[records["reviewed"].astype(bool)].copy()

    visibility = pd.crosstab(reviewed["source_group"], reviewed["visibility"])
    for column in ("visible", "uncertain", "not visible"):
        if column not in visibility:
            visibility[column] = 0
    accepted_visible = int(visibility.loc["accepted", "visible"])
    accepted_other = int(
        visibility.loc["accepted", ["uncertain", "not visible"]].sum()
    )
    control_visible = int(visibility.loc["failed_qc_control", "visible"])
    control_other = int(
        visibility.loc[
            "failed_qc_control", ["uncertain", "not visible"]
        ].sum()
    )
    visible_test = fisher_exact(
        [[accepted_visible, accepted_other],
         [control_visible, control_other]],
        alternative="two-sided",
    )

    points = attach_algorithm_depths(points, key, samples, args.spacing_um)
    accepted_points = points.loc[
        (points["source_group"] == "accepted")
        & points["algorithm_depth_vox"].notna()
    ].copy()
    accepted_cases = accepted_points.groupby("case_id", as_index=False).agg(
        human_median_depth_vox=("human_depth_vox", "median"),
        algorithm_median_depth_vox=("algorithm_depth_vox", "median"),
        median_point_absolute_difference_um=("absolute_difference_um", "median"),
        clicked_points=("human_depth_vox", "size"),
    )
    accepted_cases["signed_algorithm_minus_human_um"] = (
        accepted_cases["algorithm_median_depth_vox"]
        - accepted_cases["human_median_depth_vox"]
    ) * args.spacing_um

    view_depth = points.groupby(["case_id", "view"])["human_depth_vox"].median()
    view_depth = view_depth.unstack()
    view_depth = view_depth.dropna(subset=["u-depth", "v-depth"])
    view_depth["absolute_u_v_difference_um"] = (
        view_depth["u-depth"] - view_depth["v-depth"]
    ).abs() * args.spacing_um

    signed = accepted_cases["signed_algorithm_minus_human_um"]
    negative = int((signed < 0).sum())
    nonzero = int((signed != 0).sum())
    sign_test = binomtest(negative, nonzero, 0.5, alternative="two-sided")

    per_case_abs = accepted_cases["median_point_absolute_difference_um"]
    summary = {
        "experiment": 55,
        "pack_id": payload.get("pack_id"),
        "annotation_file_sha256": __import__("hashlib").sha256(
            args.annotations.read_bytes()
        ).hexdigest(),
        "cases_total": int(len(records)),
        "cases_reviewed": int(len(reviewed)),
        "cases_unreviewed": int(len(records) - len(reviewed)),
        "reviewed_visible": int((reviewed["visibility"] == "visible").sum()),
        "reviewed_uncertain": int((reviewed["visibility"] == "uncertain").sum()),
        "reviewed_not_visible": int(
            (reviewed["visibility"] == "not visible").sum()
        ),
        "cases_with_points_in_both_views": int(
            points.groupby(["case_id", "view"]).size().unstack(fill_value=0)
            .gt(0).all(axis=1).sum()
        ),
        "clicked_points": int(len(points)),
        "accepted_reviewed": accepted_visible + accepted_other,
        "accepted_visible": accepted_visible,
        "control_reviewed": control_visible + control_other,
        "control_visible": control_visible,
        "visibility_odds_ratio": float(visible_test.statistic),
        "visibility_fisher_two_sided_p": float(visible_test.pvalue),
        "accepted_cases_with_algorithm_comparison": int(len(accepted_cases)),
        "median_case_point_absolute_difference_um": float(per_case_abs.median()),
        "p90_case_point_absolute_difference_um": float(
            np.percentile(per_case_abs, 90)
        ),
        "median_algorithm_minus_human_depth_um": float(signed.median()),
        "algorithm_shallower_cases": negative,
        "nonzero_signed_cases": nonzero,
        "signed_depth_two_sided_sign_test_p": float(sign_test.pvalue),
        "median_cross_view_absolute_difference_um": float(
            view_depth["absolute_u_v_difference_um"].median()
        ),
        "p90_cross_view_absolute_difference_um": float(
            np.percentile(view_depth["absolute_u_v_difference_um"], 90)
        ),
        "scope": (
            "Internal blinded observer pilot. Tests visual repeatability and "
            "QC enrichment, not anatomical identity or independent validation."
        ),
    }
    return summary, accepted_cases, view_depth.reset_index()


def make_figure(summary: dict, cases: pd.DataFrame, out: Path, spacing: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    colours = ["#3366a8", "#a9adb4"]

    totals = [summary["accepted_reviewed"], summary["control_reviewed"]]
    visible = [summary["accepted_visible"], summary["control_visible"]]
    percentages = [100.0 * v / n for v, n in zip(visible, totals)]
    axes[0].bar([0, 1], percentages, color=colours, width=0.65)
    axes[0].set_xticks([0, 1], ["QC accepted", "Failed-QC\ncontrol"])
    axes[0].set_ylabel("Cases called visible (%)")
    axes[0].set_ylim(0, 100)
    for index, (percentage, count, total) in enumerate(
        zip(percentages, visible, totals)
    ):
        axes[0].text(index, percentage + 4, f"{count}/{total}", ha="center")
    axes[0].text(
        0.03, 0.97,
        f"Fisher p = {summary['visibility_fisher_two_sided_p']:.3f}",
        transform=axes[0].transAxes, va="top", fontsize=9,
    )

    human_um = cases["human_median_depth_vox"] * spacing
    algorithm_um = cases["algorithm_median_depth_vox"] * spacing
    low = float(min(human_um.min(), algorithm_um.min()) - 3)
    high = float(max(human_um.max(), algorithm_um.max()) + 3)
    axes[1].scatter(human_um, algorithm_um, s=34, color="#3366a8", alpha=0.9)
    axes[1].plot([low, high], [low, high], "--", color="#666666", lw=1)
    axes[1].set(xlabel="Human median depth (µm)",
                ylabel="Algorithm median depth (µm)",
                xlim=(low, high), ylim=(low, high))
    axes[1].set_aspect("equal", adjustable="box")

    signed = cases["signed_algorithm_minus_human_um"].to_numpy()
    axes[2].axhline(0, color="#666666", ls="--", lw=1)
    axes[2].scatter(np.arange(1, len(signed) + 1), np.sort(signed),
                    s=34, color="#b04a4a")
    axes[2].set_xlabel("Accepted cases, sorted")
    axes[2].set_ylabel("Algorithm − human depth (µm)")
    axes[2].text(
        0.03, 0.03,
        f"median = {summary['median_algorithm_minus_human_depth_um']:.1f} µm",
        transform=axes[2].transAxes, fontsize=9,
    )

    for axis, letter in zip(axes, "ABC"):
        axis.text(-0.16, 1.07, letter, transform=axis.transAxes,
                  fontsize=12, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, cases, view_depth = analyse(args)
    (args.out / "experiment_55_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    cases.to_csv(args.out / "private_case_comparison.csv", index=False)
    view_depth.to_csv(args.out / "private_cross_view_comparison.csv", index=False)
    make_figure(
        summary,
        cases,
        args.out / "experiment_55_blinded_boundary_pilot.png",
        args.spacing_um,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

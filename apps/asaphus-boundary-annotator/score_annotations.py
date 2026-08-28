#!/usr/bin/env python3
"""Unblind frozen human labels and compare them with the CT-edge extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path, nargs="+")
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--spacing-um", type=float, default=3.7)
    return parser.parse_args()


def annotation_points(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotator = payload.get("annotator_id", path.stem)
    rows = []
    for record in payload.get("annotations", []):
        for view, field in (("u-depth", "u_depth_points"),
                            ("v-depth", "v_depth_points")):
            for point in record.get(field, []):
                rows.append({
                    "annotator_id": annotator,
                    "case_id": record["case_id"],
                    "visibility": record.get("visibility"),
                    "structure_class": record.get("structure_class"),
                    "confidence": record.get("confidence"),
                    "view": view,
                    "lateral_vox": float(point["lateral_vox"]),
                    "human_depth_vox": float(point["depth_vox"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    key = pd.read_csv(args.private_key)
    samples = pd.read_csv(args.samples)
    frames = [annotation_points(path) for path in args.annotations]
    points = pd.concat([frame for frame in frames if not frame.empty],
                       ignore_index=True)
    if points.empty:
        raise ValueError("No clicked boundary points were found")
    points = points.merge(key[["case_id", "facet_id", "source_group", "cv_block"]],
                          on="case_id", how="left", validate="many_to_one")

    predictions = []
    for row in points.itertuples(index=False):
        facet = samples.loc[samples["facet_id"] == row.facet_id]
        if facet.empty:
            predictions.append(np.nan)
            continue
        pitch = float(facet["pitch"].iloc[0])
        normalized_lateral = row.lateral_vox / max(pitch / 2.0, 1e-6)
        if row.view == "u-depth":
            distance = np.hypot(
                facet["un"].to_numpy(float) - normalized_lateral,
                facet["vn"].to_numpy(float),
            )
        else:
            distance = np.hypot(
                facet["un"].to_numpy(float),
                facet["vn"].to_numpy(float) - normalized_lateral,
            )
        predictions.append(float(facet.iloc[int(np.argmin(distance))]["target_vox"]))
    points["algorithm_depth_vox"] = predictions
    points["signed_difference_um"] = (
        points["algorithm_depth_vox"] - points["human_depth_vox"]
    ) * args.spacing_um
    points["absolute_difference_um"] = points["signed_difference_um"].abs()
    args.out.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out / "unblinded_point_comparison.csv", index=False)

    per_case = points.groupby(
        ["annotator_id", "case_id", "facet_id", "source_group"], as_index=False
    ).agg(
        human_median_depth_vox=("human_depth_vox", "median"),
        algorithm_median_depth_vox=("algorithm_depth_vox", "median"),
        median_absolute_difference_um=("absolute_difference_um", "median"),
        clicked_points=("human_depth_vox", "size"),
    )
    per_case.to_csv(args.out / "unblinded_case_comparison.csv", index=False)

    accepted = per_case.loc[per_case["source_group"] == "accepted"]
    summary = {
        "annotators": int(per_case["annotator_id"].nunique()),
        "annotated_cases": int(per_case["case_id"].nunique()),
        "accepted_cases_with_points": int(accepted["case_id"].nunique()),
        "accepted_case_median_absolute_difference_um": (
            None if accepted.empty else
            float(accepted["median_absolute_difference_um"].median())
        ),
        "accepted_case_p90_absolute_difference_um": (
            None if accepted.empty else
            float(np.percentile(accepted["median_absolute_difference_um"], 90))
        ),
        "interpretation": (
            "Agreement with a human-visible CT boundary tests extraction "
            "validity only. It does not establish anatomical identity."
        ),
    }
    (args.out / "annotation_score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

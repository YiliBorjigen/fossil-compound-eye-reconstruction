#!/usr/bin/env python3
"""Diagnostic decomposition for Experiment 42.

Separates prediction of between-facet mean depth from prediction of the
within-facet boundary shape.  The within-facet target is centred using its true
facet mean and is therefore diagnostic only; it is not a deployable
reconstruction by itself.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SPACING_UM = 3.7
WITHIN = ["un", "vn", "rn", "rn2", "un2", "vn2", "uv"]
ABSOLUTE = ["surf_x", "y", "z"]
CURRENT = ["relief", "fy", "fz", "lap", "pitch"]
FACET_SURFACE = [
    "facet_sag_amp", "facet_sag_std", "facet_core_rim",
    "normal_y", "normal_z", "nn_mean", "nn_cv", "nn_range",
    "lattice_anisotropy", "angle_gap_rmse",
]
POINT_SURFACE = [
    "surface_sag",
    *[f"{kind}_s{scale}" for scale in (2, 4, 8, 16)
      for kind in ("relief", "slope", "lap", "det", "aniso")],
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def facet_errors(facet: np.ndarray, block: np.ndarray,
                 target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({
        "facet_id": facet, "cv_block": block,
        "error_um": np.abs(target - prediction) * SPACING_UM,
    })
    return d.groupby(["facet_id", "cv_block"], as_index=False)[
        "error_um"
    ].mean().rename(columns={"error_um": "MAE_um"})


def depth_spatial_model(degree: int, alpha: float):
    ct = ColumnTransformer([
        ("xyz", PolynomialFeatures(degree=degree, include_bias=False), ABSOLUTE)
    ], remainder="drop")
    return make_pipeline(ct, StandardScaler(), Ridge(alpha=alpha))


def ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def depth_score(frame: pd.DataFrame, pred: np.ndarray) -> float:
    return float(np.median(np.abs(frame["target_mean"].to_numpy() - pred))
                 * SPACING_UM)


def select_depth_spatial(train: pd.DataFrame) -> tuple[float, int, float]:
    groups = train["cv_block"].to_numpy()
    y = train["target_mean"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    for degree, alpha in itertools.product(
        (1, 2, 3), (0.1, 1.0, 10.0, 100.0, 1000.0)
    ):
        pred = np.full(len(train), np.nan)
        for tr, va in cv.split(train, y, groups):
            model = depth_spatial_model(degree, alpha)
            model.fit(train.iloc[tr], y[tr])
            pred[va] = model.predict(train.iloc[va])
        candidates.append((depth_score(train, pred), degree, alpha))
    return min(candidates)


def select_depth_residual(train: pd.DataFrame, columns: list[str],
                          degree: int, spatial_alpha: float) -> tuple[float, float]:
    groups = train["cv_block"].to_numpy()
    y = train["target_mean"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        pred = np.full(len(train), np.nan)
        for tr, va in cv.split(train, y, groups):
            sm = depth_spatial_model(degree, spatial_alpha)
            sm.fit(train.iloc[tr], y[tr])
            base_tr = sm.predict(train.iloc[tr])
            base_va = sm.predict(train.iloc[va])
            rm = ridge(alpha)
            rm.fit(train.iloc[tr][columns], y[tr] - base_tr)
            pred[va] = base_va + rm.predict(train.iloc[va][columns])
        candidates.append((depth_score(train, pred), alpha))
    return min(candidates)


def run_depth(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    groups = frame["cv_block"].to_numpy()
    y = frame["target_mean"].to_numpy()
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    base = np.full(len(frame), np.nan)
    aug = np.full(len(frame), np.nan)
    for tr, te in cv.split(frame, y, groups):
        train = frame.iloc[tr]
        _, degree, sa = select_depth_spatial(train)
        _, ra = select_depth_residual(train, columns, degree, sa)
        sm = depth_spatial_model(degree, sa)
        sm.fit(train, y[tr])
        base_tr = sm.predict(train)
        base[te] = sm.predict(frame.iloc[te])
        rm = ridge(ra)
        rm.fit(train[columns], y[tr] - base_tr)
        aug[te] = base[te] + rm.predict(frame.iloc[te][columns])
    out = frame[["facet_id", "cv_block", "target_mean"]].copy()
    out["baseline_prediction"] = base
    out["augmented_prediction"] = aug
    out["baseline_MAE_um"] = np.abs(y - base) * SPACING_UM
    out["augmented_MAE_um"] = np.abs(y - aug) * SPACING_UM
    out["improvement_um"] = out["baseline_MAE_um"] - out["augmented_MAE_um"]
    return out


def shape_score(frame: pd.DataFrame, pred: np.ndarray) -> float:
    f = facet_errors(
        frame["facet_id"].to_numpy(), frame["cv_block"].to_numpy(),
        frame["target_centered"].to_numpy(), pred,
    )
    return float(f["MAE_um"].median())


def select_shape_base(train: pd.DataFrame) -> tuple[float, float]:
    groups = train["cv_block"].to_numpy()
    y = train["target_centered"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        pred = np.full(len(train), np.nan)
        for tr, va in cv.split(train, y, groups):
            model = ridge(alpha)
            model.fit(train.iloc[tr][WITHIN], y[tr])
            pred[va] = model.predict(train.iloc[va][WITHIN])
        candidates.append((shape_score(train, pred), alpha))
    return min(candidates)


def select_shape_residual(train: pd.DataFrame, columns: list[str],
                          base_alpha: float) -> tuple[float, float]:
    groups = train["cv_block"].to_numpy()
    y = train["target_centered"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        pred = np.full(len(train), np.nan)
        for tr, va in cv.split(train, y, groups):
            bm = ridge(base_alpha)
            bm.fit(train.iloc[tr][WITHIN], y[tr])
            base_tr = bm.predict(train.iloc[tr][WITHIN])
            base_va = bm.predict(train.iloc[va][WITHIN])
            rm = ridge(alpha)
            rm.fit(train.iloc[tr][columns], y[tr] - base_tr)
            pred[va] = base_va + rm.predict(train.iloc[va][columns])
        candidates.append((shape_score(train, pred), alpha))
    return min(candidates)


def run_shape(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    groups = frame["cv_block"].to_numpy()
    y = frame["target_centered"].to_numpy()
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    base = np.full(len(frame), np.nan)
    aug = np.full(len(frame), np.nan)
    for tr, te in cv.split(frame, y, groups):
        train = frame.iloc[tr]
        _, ba = select_shape_base(train)
        _, ra = select_shape_residual(train, columns, ba)
        bm = ridge(ba)
        bm.fit(train[WITHIN], y[tr])
        base_tr = bm.predict(train[WITHIN])
        base[te] = bm.predict(frame.iloc[te][WITHIN])
        rm = ridge(ra)
        rm.fit(train[columns], y[tr] - base_tr)
        aug[te] = base[te] + rm.predict(frame.iloc[te][columns])
    fbase = facet_errors(frame["facet_id"].to_numpy(), groups, y, base).rename(
        columns={"MAE_um": "baseline_MAE_um"}
    )
    faug = facet_errors(frame["facet_id"].to_numpy(), groups, y, aug).rename(
        columns={"MAE_um": "augmented_MAE_um"}
    )
    out = fbase.merge(faug, on=["facet_id", "cv_block"])
    out["improvement_um"] = out["baseline_MAE_um"] - out["augmented_MAE_um"]
    return out


def summarize(task: str, result: pd.DataFrame) -> dict[str, float | int | str]:
    blocks = result.groupby("cv_block")["improvement_um"].median().to_numpy()
    return {
        "task": task,
        "baseline_median_MAE_um": float(result["baseline_MAE_um"].median()),
        "augmented_median_MAE_um": float(result["augmented_MAE_um"].median()),
        "median_paired_improvement_um": float(result["improvement_um"].median()),
        "blocks_improved": int(np.sum(blocks > 0)),
        "block_sign_flip_p": exact_sign_flip_p(blocks),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.features)
    frame["target_centered"] = frame["target_vox"] - frame.groupby(
        "facet_id"
    )["target_vox"].transform("mean")
    facet = frame.groupby("facet_id", as_index=False).agg({
        "cv_block": "first", "target_vox": "mean", "surf_x": "first",
        "y": "first", "z": "first", **{c: "first" for c in CURRENT + FACET_SURFACE}
    }).rename(columns={"target_vox": "target_mean"})

    depth = run_depth(facet, CURRENT + FACET_SURFACE)
    shape = run_shape(frame, POINT_SURFACE)
    depth.to_csv(args.out / "experiment_42_mean_depth_component.csv", index=False)
    shape.to_csv(args.out / "experiment_42_within_facet_shape_component.csv",
                 index=False)
    summary = pd.DataFrame([
        summarize("between_facet_mean_depth", depth),
        summarize("within_facet_shape_diagnostic", shape),
    ])
    summary.to_csv(args.out / "experiment_42_component_summary.csv", index=False)

    d = summary.iloc[0]
    s = summary.iloc[1]
    report = f"""# Experiment 42 component analysis

This diagnostic separates two questions that the complete reconstruction score
combines.

## Between-facet mean depth

- spatial-only MAE: **{d['baseline_median_MAE_um']:.2f} µm**
- spatial plus facet geometry MAE: **{d['augmented_median_MAE_um']:.2f} µm**
- blocks improved: **{int(d['blocks_improved'])}/5**
- exact block sign-flip p: **{d['block_sign_flip_p']:.4f}**

This tests whether whole-facet surface and lattice measurements explain mean
boundary depth after specimen position.

## Within-facet shape

- radial-position MAE: **{s['baseline_median_MAE_um']:.2f} µm**
- radial position plus local surface shape MAE: **{s['augmented_median_MAE_um']:.2f} µm**
- blocks improved: **{int(s['blocks_improved'])}/5**
- exact block sign-flip p: **{s['block_sign_flip_p']:.4f}**

The within-facet target was centred using its observed facet mean.  This is a
diagnostic of curvature correspondence only and cannot reconstruct an entirely
missing boundary unless mean depth is supplied or predicted independently.
"""
    (args.out / "EXPERIMENT_42_COMPONENT_REPORT.md").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()

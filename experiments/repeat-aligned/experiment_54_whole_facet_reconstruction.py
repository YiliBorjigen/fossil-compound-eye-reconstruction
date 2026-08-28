#!/usr/bin/env python3
"""Experiment 54: reconstruct wholly hidden Asaphus candidate surfaces.

The complete internal CT boundary of every facet in one contiguous spatial
block is hidden. Models are trained and calibrated on the other four blocks.
Only quantities available when the internal surface is absent are supplied at
test time: within-facet coordinates, facet spacing, the preserved outer
surface and (for explicit controls) specimen position.

The target is the candidate internal CT boundary extracted by the frozen
Asaphus pipeline. It is not anatomically verified lens ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SPACING_UM = 3.7
SEED = 20260828
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SIGMAS = (0.15, 0.25, 0.40)
DEGREES = (1, 2, 3)
QUADRATIC = ["un", "vn", "un2", "vn2", "uv"]
LOCAL_OUTER = [
    "surface_sag_norm", "facet_sag_amp_norm", "facet_sag_std_norm",
    "facet_core_rim_norm",
]
OUTER = [
    *LOCAL_OUTER, "normal_y", "normal_z", "nn_cv",
    "nn_range_norm", "lattice_anisotropy", "angle_gap_rmse",
]
METHODS = (
    "flat_depth",
    "axisymmetric_ellipsoid",
    "quadratic_surface",
    "local_outer_curvature",
    "outer_curvature",
    "repeat_template",
    "six_neighbor_prior",
    "position_smoother",
    "position_plus_outer",
)
LABELS = {
    "flat_depth": "flat depth",
    "axisymmetric_ellipsoid": "axisymmetric ellipsoid",
    "quadratic_surface": "quadratic surface",
    "local_outer_curvature": "local outer curvature",
    "outer_curvature": "outer geometry + context",
    "repeat_template": "repeat template",
    "six_neighbor_prior": "six-neighbour prior",
    "position_smoother": "position smoother",
    "position_plus_outer": "position + local outer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["surface_sag_norm"] = frame["surface_sag"] / frame["pitch"]
    frame["facet_sag_amp_norm"] = frame["facet_sag_amp"] / frame["pitch"]
    frame["facet_sag_std_norm"] = frame["facet_sag_std"] / frame["pitch"]
    frame["facet_core_rim_norm"] = frame["facet_core_rim"] / frame["pitch"]
    frame["nn_range_norm"] = frame["nn_range"] / frame["pitch"]
    required = {
        "facet_id", "cv_block", "target_norm", "target_vox", "pitch",
        "rn2", "surf_x", "y", "z", *QUADRATIC, *OUTER,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required features contain missing values")
    return frame


def rbf_centres() -> np.ndarray:
    grid = np.linspace(-0.8, 0.8, 7)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    centres = np.c_[u.ravel(), v.ravel()]
    return centres[np.hypot(centres[:, 0], centres[:, 1]) <= 1.05]


RBF_CENTRES = rbf_centres()


def rbf_features(frame: pd.DataFrame, sigma: float) -> np.ndarray:
    points = frame[["un", "vn"]].to_numpy(float)
    distance2 = ((points[:, None] - RBF_CENTRES[None]) ** 2).sum(axis=2)
    return np.exp(-distance2 / (2 * sigma * sigma))


def spatial_features(frame: pd.DataFrame, degree: int) -> np.ndarray:
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(
        frame[["surf_x", "y", "z"]].to_numpy(float)
    )


def design(frame: pd.DataFrame, method: str, params: dict) -> np.ndarray:
    if method == "axisymmetric_ellipsoid":
        return frame[["rn2"]].to_numpy(float)
    if method == "quadratic_surface":
        return frame[QUADRATIC].to_numpy(float)
    if method == "local_outer_curvature":
        return frame[QUADRATIC + LOCAL_OUTER].to_numpy(float)
    if method == "outer_curvature":
        return frame[QUADRATIC + OUTER].to_numpy(float)
    if method == "repeat_template":
        return np.c_[frame[QUADRATIC].to_numpy(float),
                     rbf_features(frame, params["sigma"])]
    if method == "position_smoother":
        return np.c_[frame[QUADRATIC].to_numpy(float),
                     spatial_features(frame, params["degree"])]
    if method == "position_plus_outer":
        return np.c_[
            frame[QUADRATIC + LOCAL_OUTER].to_numpy(float),
            spatial_features(frame, params["degree"]),
        ]
    raise ValueError(method)


def candidates(method: str) -> list[dict]:
    if method in {"flat_depth", "six_neighbor_prior"}:
        return [{}]
    if method in {"axisymmetric_ellipsoid", "quadratic_surface",
                  "local_outer_curvature", "outer_curvature"}:
        return [{"alpha": alpha} for alpha in ALPHAS]
    if method == "repeat_template":
        return [
            {"sigma": sigma, "alpha": alpha}
            for sigma, alpha in itertools.product(SIGMAS, ALPHAS)
        ]
    if method == "position_smoother":
        return [
            {"degree": degree, "alpha": alpha}
            for degree, alpha in itertools.product(DEGREES, ALPHAS)
        ]
    if method == "position_plus_outer":
        return [
            {"degree": degree, "alpha": alpha}
            for degree, alpha in itertools.product(DEGREES, ALPHAS)
        ]
    raise ValueError(method)


def facet_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("facet_id")["facet_id"].transform("size")
    weights = 1.0 / counts.to_numpy(float)
    return weights * len(weights) / weights.sum()


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
    params: dict,
) -> np.ndarray:
    if method == "flat_depth":
        facet_depth = train.groupby("facet_id")["target_norm"].median()
        return np.full(len(test), float(facet_depth.median()))
    if method == "six_neighbor_prior":
        train = train.copy()
        train["target_centered"] = train["target_norm"] - train.groupby(
            "facet_id"
        )["target_norm"].transform("median")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[QUADRATIC].to_numpy(float))
        shape = Ridge(alpha=20.0)
        shape.fit(
            x_train,
            train["target_centered"].to_numpy(float),
            sample_weight=facet_balanced_weights(train),
        )
        train_facets = train.groupby("facet_id", as_index=False).agg(
            y=("y", "first"), z=("z", "first"),
            target_median=("target_norm", "median"),
        )
        prediction = np.full(len(test), np.nan)
        for _, locations in test.groupby("facet_id").groups.items():
            locations = np.asarray(list(locations), dtype=int)
            local = test.loc[locations]
            y0 = float(local["y"].iloc[0])
            z0 = float(local["z"].iloc[0])
            distance = np.hypot(
                train_facets["y"].to_numpy(float) - y0,
                train_facets["z"].to_numpy(float) - z0,
            )
            nearest = np.argsort(distance)[:6]
            weights = 1.0 / np.maximum(distance[nearest], 1e-6) ** 2
            depth = float(np.sum(
                weights * train_facets.iloc[nearest]["target_median"].to_numpy()
            ) / weights.sum())
            shape_prediction = shape.predict(
                scaler.transform(local[QUADRATIC].to_numpy(float))
            )
            prediction[test.index.get_indexer(locations)] = (
                depth + shape_prediction
            )
        return prediction
    x_train = design(train, method, params)
    x_test = design(test, method, params)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    estimator = Ridge(alpha=params["alpha"])
    estimator.fit(
        x_train,
        train["target_norm"].to_numpy(float),
        sample_weight=facet_balanced_weights(train),
    )
    return estimator.predict(x_test)


def facet_error(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    error_um = np.abs(
        prediction - frame["target_norm"].to_numpy(float)
    ) * frame["pitch"].to_numpy(float) * SPACING_UM
    rows = pd.DataFrame({
        "facet_id": frame["facet_id"].to_numpy(int),
        "cv_block": frame["cv_block"].to_numpy(int),
        "error_um": error_um,
    })
    return rows.groupby(["facet_id", "cv_block"], as_index=False)[
        "error_um"
    ].mean().rename(columns={"error_um": "facet_MAE_um"})


def validation_score(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    return float(facet_error(frame, prediction)["facet_MAE_um"].median())


def grouped_prediction(
    frame: pd.DataFrame,
    method: str,
    params: dict,
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    groups = frame["cv_block"].to_numpy(int)
    for heldout in sorted(np.unique(groups)):
        train = frame[groups != heldout]
        test_index = np.where(groups == heldout)[0]
        prediction[test_index] = fit_predict(
            train, frame.iloc[test_index], method, params
        )
    return prediction


def tune(train: pd.DataFrame, method: str) -> tuple[dict, float]:
    scored = []
    for order, params in enumerate(candidates(method)):
        prediction = grouped_prediction(train, method, params)
        scored.append((validation_score(train, prediction), order, params))
    score, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
    return chosen, float(score)


def calibration_width(
    train: pd.DataFrame,
    method: str,
    params: dict,
) -> float:
    prediction = grouped_prediction(train, method, params)
    error_um = np.abs(
        prediction - train["target_norm"].to_numpy(float)
    ) * train["pitch"].to_numpy(float) * SPACING_UM
    residuals = pd.DataFrame({
        "facet_id": train["facet_id"].to_numpy(int),
        "error_um": error_um,
    })
    facet_p90 = residuals.groupby("facet_id")["error_um"].quantile(0.90)
    return float(np.quantile(facet_p90, 0.90, method="higher"))


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def run(frame: pd.DataFrame):
    groups = frame["cv_block"].to_numpy(int)
    prediction_table = frame[[
        "facet_id", "cv_block", "un", "vn", "pitch",
        "target_norm", "target_vox",
    ]].copy()
    facet_tables = []
    tuning_rows = []
    coverage_rows = []

    for method in METHODS:
        prediction = np.full(len(frame), np.nan)
        for heldout in sorted(np.unique(groups)):
            train_index = np.where(groups != heldout)[0]
            test_index = np.where(groups == heldout)[0]
            train = frame.iloc[train_index]
            test = frame.iloc[test_index]
            params, inner_score = tune(train, method)
            width_um = calibration_width(train, method, params)
            test_prediction = fit_predict(train, test, method, params)
            prediction[test_index] = test_prediction
            test_error_um = np.abs(
                test_prediction - test["target_norm"].to_numpy(float)
            ) * test["pitch"].to_numpy(float) * SPACING_UM
            test_coverage = test.assign(
                covered=test_error_um <= width_um
            ).groupby("facet_id")["covered"].mean()
            coverage_rows.append({
                "method": method,
                "heldout_block": int(heldout),
                "calibrated_half_width_um": width_um,
                "median_facet_point_coverage": float(test_coverage.median()),
                "facets_with_at_least_90pct_coverage": int(
                    np.sum(test_coverage >= 0.90)
                ),
                "heldout_facets": int(len(test_coverage)),
            })
            tuning_rows.append({
                "method": method,
                "heldout_block": int(heldout),
                "inner_median_facet_MAE_um": inner_score,
                "parameters": json.dumps(params, sort_keys=True),
            })
        prediction_table[f"{method}_prediction_norm"] = prediction
        facets = facet_error(frame, prediction)
        facets["method"] = method
        facet_tables.append(facets)

    facets = pd.concat(facet_tables, ignore_index=True)
    return (
        prediction_table,
        facets,
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coverage_rows),
    )


def run_isolated_loss(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory leave-one-facet-out version of the practical use case."""
    rows = []
    for facet_id in sorted(frame["facet_id"].unique()):
        train = frame[frame["facet_id"] != facet_id]
        test = frame[frame["facet_id"] == facet_id]
        predictions = {
            "flat_depth": fit_predict(train, test, "flat_depth", {}),
            "quadratic_surface_fixed": fit_predict(
                train, test, "quadratic_surface", {"alpha": 20.0}
            ),
            "six_neighbor_prior": fit_predict(
                train, test, "six_neighbor_prior", {}
            ),
        }
        for method, prediction in predictions.items():
            error = facet_error(test, prediction).iloc[0]
            rows.append({
                "facet_id": int(facet_id),
                "cv_block": int(error["cv_block"]),
                "method": method,
                "facet_MAE_um": float(error["facet_MAE_um"]),
            })
    facets = pd.DataFrame(rows)
    summary = facets.groupby("method", as_index=False).agg(
        median_facet_MAE_um=("facet_MAE_um", "median"),
        p90_facet_MAE_um=("facet_MAE_um", lambda x: x.quantile(0.90)),
        facets=("facet_id", "size"),
    ).sort_values("median_facet_MAE_um")
    return facets, summary


def export_six_neighbor_surfaces(frame: pd.DataFrame, out: Path) -> None:
    """Export one complete canonical surface for every held-out facet."""
    coordinate = np.linspace(-0.9, 0.9, 61)
    uu, vv = np.meshgrid(coordinate, coordinate, indexing="xy")
    keep = uu * uu + vv * vv <= 0.9 ** 2
    u = uu[keep]
    v = vv[keep]
    facet_ids = np.asarray(sorted(frame["facet_id"].unique()), dtype=int)
    surfaces = np.empty((len(facet_ids), len(u)), dtype=np.float32)
    blocks = np.empty(len(facet_ids), dtype=np.int16)
    widths = np.empty(len(facet_ids), dtype=np.float32)
    width_by_block: dict[int, float] = {}
    for row, facet_id in enumerate(facet_ids):
        metadata = frame[frame["facet_id"] == facet_id].iloc[0]
        block = int(metadata["cv_block"])
        train = frame[frame["cv_block"] != block]
        grid = pd.DataFrame({
            "facet_id": facet_id,
            "un": u,
            "vn": v,
            "un2": u * u,
            "vn2": v * v,
            "uv": u * v,
            "y": float(metadata["y"]),
            "z": float(metadata["z"]),
        })
        prediction = fit_predict(train, grid, "six_neighbor_prior", {})
        surfaces[row] = (
            prediction * float(metadata["pitch"]) * SPACING_UM
        ).astype(np.float32)
        blocks[row] = block
        if block not in width_by_block:
            width_by_block[block] = calibration_width(
                train, "six_neighbor_prior", {}
            )
        widths[row] = width_by_block[block]
    np.savez_compressed(
        out / "experiment_54_six_neighbor_surfaces.npz",
        facet_id=facet_ids,
        cv_block=blocks,
        u_normalized=u.astype(np.float32),
        v_normalized=v.astype(np.float32),
        predicted_depth_um=surfaces,
        calibrated_interval_half_width_um=widths,
    )


def summarize(
    frame: pd.DataFrame,
    facets: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    block = facets.groupby(["method", "cv_block"], as_index=False).agg(
        median_facet_MAE_um=("facet_MAE_um", "median"),
        facets=("facet_id", "size"),
    )
    block_winner = block.loc[
        block.groupby("cv_block")["median_facet_MAE_um"].idxmin(),
        ["cv_block", "method"],
    ]
    target_depth_um = float(frame["target_vox"].median() * SPACING_UM)
    summary_rows = []
    for method, values in facets.groupby("method"):
        q = coverage[coverage["method"] == method]
        summary_rows.append({
            "method": method,
            "median_facet_MAE_um": float(values["facet_MAE_um"].median()),
            "p90_facet_MAE_um": float(values["facet_MAE_um"].quantile(0.90)),
            "normalized_median_error": float(
                values["facet_MAE_um"].median() / target_depth_um
            ),
            "blocks_won": int(np.sum(block_winner["method"] == method)),
            "median_calibrated_half_width_um": float(
                q["calibrated_half_width_um"].median()
            ),
            "median_facet_point_coverage": float(
                q["median_facet_point_coverage"].median()
            ),
        })
    summary = pd.DataFrame(summary_rows).sort_values("median_facet_MAE_um")

    comparison_rows = []

    def add_comparison(method: str, reference_method: str) -> None:
        reference = facets[facets["method"] == reference_method].rename(
            columns={"facet_MAE_um": "reference_MAE_um"}
        )[["facet_id", "cv_block", "reference_MAE_um"]]
        tested = facets[facets["method"] == method].merge(
            reference, on=["facet_id", "cv_block"]
        )
        tested["advantage_um"] = (
            tested["reference_MAE_um"] - tested["facet_MAE_um"]
        )
        block_advantage = tested.groupby("cv_block")["advantage_um"].median()
        comparison_rows.append({
            "method": method,
            "reference": reference_method,
            "median_paired_advantage_um": float(
                tested["advantage_um"].median()
            ),
            "blocks_favouring_method": int(np.sum(block_advantage > 0)),
            "exact_block_sign_flip_p": exact_sign_flip_p(
                block_advantage.to_numpy(float)
            ),
        })
    for method in METHODS:
        if method != "quadratic_surface":
            add_comparison(method, "quadratic_surface")
    add_comparison("position_plus_outer", "position_smoother")
    return summary, block, pd.DataFrame(comparison_rows)


def map_values(facet: pd.DataFrame, values: np.ndarray):
    grid = np.linspace(-0.9, 0.9, 121)
    uu, vv = np.meshgrid(grid, grid, indexing="xy")
    points = facet[["un", "vn"]].to_numpy(float)
    image = griddata(points, values, (uu, vv), method="linear")
    return grid, image


def make_figure(
    predictions: pd.DataFrame,
    facets: pd.DataFrame,
    summary: pd.DataFrame,
    block: pd.DataFrame,
    coverage: pd.DataFrame,
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5))
    order = list(METHODS)
    summary_indexed = summary.set_index("method").loc[order]
    x = np.arange(len(order))
    axes[0, 0].bar(
        x, summary_indexed["median_facet_MAE_um"], color="#2563EB"
    )
    axes[0, 0].set_xticks(x, [LABELS[m] for m in order], rotation=35,
                           ha="right")
    axes[0, 0].set_ylabel("Median facet MAE (µm)")
    axes[0, 0].set_title("Complete held-out surfaces")

    selected = ["axisymmetric_ellipsoid", "local_outer_curvature",
                "outer_curvature", "repeat_template", "six_neighbor_prior",
                "position_smoother", "position_plus_outer"]
    reference = block[block["method"] == "quadratic_surface"].set_index(
        "cv_block"
    )["median_facet_MAE_um"]
    for i, method in enumerate(selected):
        values = block[block["method"] == method].set_index("cv_block")[
            "median_facet_MAE_um"
        ]
        advantage = reference - values
        axes[0, 1].scatter(np.full(len(advantage), i), advantage, s=45)
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set_xticks(np.arange(len(selected)),
                          [LABELS[m] for m in selected], rotation=35,
                          ha="right")
    axes[0, 1].set_ylabel("Quadratic − method MAE (µm)")
    axes[0, 1].set_title("Five held-out spatial blocks")

    coverage_indexed = coverage.groupby("method").agg(
        width=("calibrated_half_width_um", "median"),
        coverage=("median_facet_point_coverage", "median"),
    ).loc[order]
    axes[0, 2].bar(x, coverage_indexed["width"], color="#D97706")
    axes[0, 2].set_xticks(x, [LABELS[m] for m in order], rotation=35,
                          ha="right")
    axes[0, 2].set_ylabel("Calibrated 90% half-width (µm)")
    axes[0, 2].set_title("Training-only uncertainty")

    axes[0, 3].bar(x, coverage_indexed["coverage"], color="#0F766E")
    axes[0, 3].axhline(0.9, color="black", ls="--", lw=1)
    axes[0, 3].set_ylim(0, 1.02)
    axes[0, 3].set_xticks(x, [LABELS[m] for m in order], rotation=35,
                          ha="right")
    axes[0, 3].set_ylabel("Median facet point coverage")
    axes[0, 3].set_title("Held-out interval coverage")

    neighbour_facets = facets[facets["method"] == "six_neighbor_prior"].copy()
    median_error = neighbour_facets["facet_MAE_um"].median()
    example_id = int(neighbour_facets.iloc[
        np.argmin(np.abs(neighbour_facets["facet_MAE_um"] - median_error))
    ]["facet_id"])
    facet = predictions[predictions["facet_id"] == example_id]
    observed_um = facet["target_vox"].to_numpy(float) * SPACING_UM
    panels = [
        (observed_um, "Hidden CT target"),
        (facet["quadratic_surface_prediction_norm"].to_numpy(float)
         * facet["pitch"].to_numpy(float) * SPACING_UM,
         "Quadratic prediction"),
        (facet["local_outer_curvature_prediction_norm"].to_numpy(float)
         * facet["pitch"].to_numpy(float) * SPACING_UM,
         "Local outer-curvature prediction"),
        (facet["six_neighbor_prior_prediction_norm"].to_numpy(float)
         * facet["pitch"].to_numpy(float) * SPACING_UM,
         "Six-neighbour prediction"),
    ]
    finite = observed_um[np.isfinite(observed_um)]
    vmin, vmax = np.percentile(finite, [3, 97])
    for ax, (values, title) in zip(axes[1], panels):
        grid, image = map_values(facet, values)
        artist = ax.imshow(
            image, origin="lower", extent=[grid.min(), grid.max(),
                                           grid.min(), grid.max()],
            vmin=vmin, vmax=vmax, cmap="viridis"
        )
        ax.set_title(title)
        ax.set_xlabel("u / facet radius")
        ax.set_ylabel("v / facet radius")
    fig.suptitle(
        f"Experiment 54 — wholly held-out candidate surface; facet {example_id}"
    )
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.08, top=0.92,
                        wspace=0.34, hspace=0.52)
    colour_axis = fig.add_axes([0.93, 0.12, 0.012, 0.30])
    fig.colorbar(artist, cax=colour_axis, label="Boundary depth (µm)")
    fig.savefig(out / "experiment_54_whole_facet_reconstruction.png", dpi=220)
    fig.savefig(out / "experiment_54_whole_facet_reconstruction.pdf")
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    isolated_summary: pd.DataFrame,
    out: Path,
) -> None:
    table = summary.set_index("method")
    quadratic_compare = comparisons[
        comparisons["reference"] == "quadratic_surface"
    ].set_index("method")
    target_depth = float(frame["target_vox"].median() * SPACING_UM)
    local_outer = quadratic_compare.loc["local_outer_curvature"]
    outer = quadratic_compare.loc["outer_curvature"]
    repeat = quadratic_compare.loc["repeat_template"]
    neighbour = quadratic_compare.loc["six_neighbor_prior"]
    spatial = quadratic_compare.loc["position_smoother"]
    matched = comparisons[
        (comparisons["method"] == "position_plus_outer")
        & (comparisons["reference"] == "position_smoother")
    ].iloc[0]
    isolated = isolated_summary.set_index("method")
    if (
        local_outer["median_paired_advantage_um"] > 0
        and local_outer["blocks_favouring_method"] >= 4
        and matched["median_paired_advantage_um"] > 0
        and matched["blocks_favouring_method"] >= 4
    ):
        verdict = (
            "Surviving local outer curvature contributes reproducible signal "
            "for a wholly missing internal CT boundary within this specimen. "
            "The reconstruction remains approximate and still requires "
            "anatomical and independent-specimen validation."
        )
    elif (
        neighbour["median_paired_advantage_um"] > 0
        and neighbour["blocks_favouring_method"] >= 4
        and table.loc["six_neighbor_prior", "normalized_median_error"] < 0.15
    ):
        verdict = (
            "A wholly missing candidate boundary can be approximated within "
            "this specimen by combining neighbouring preserved internal "
            "surfaces with a shared quadratic shape. The predictive "
            "information comes from anatomical repetition and spatial "
            "continuity, not from local outer curvature alone."
        )
    else:
        verdict = (
            "The wholly missing internal surface is not recovered reliably. "
            "Neither measured outer curvature nor the repeat template gives "
            "a material, spatially consistent gain over a quadratic surface."
        )
    report = f"""# Experiment 54 — wholly held-out internal surfaces

## Verdict

{verdict}

## The question

This experiment directly tests the case described by Lauren Sumner-Rooney:
the outer lens curvature survives, but the complete internal lens surface is
missing. The target facet contributes no internal boundary point to model
fitting, parameter selection or uncertainty calibration. Five contiguous
spatial blocks are held out in turn.

The scored target is the frozen candidate internal CT boundary in this
*Asaphus* scan. It is not verified anatomical lens ground truth.

Scoring uses the locations where the frozen edge extractor produced a valid
target sample. That support mask is used only for evaluation and the example
figure; it is not supplied to the predictor. A separate NPZ output contains a
complete fixed canonical grid for every held-out facet.

## Main result

- median target depth: **{target_depth:.2f} µm**
- axisymmetric ellipsoid MAE: **{table.loc['axisymmetric_ellipsoid', 'median_facet_MAE_um']:.2f} µm**
- general quadratic MAE: **{table.loc['quadratic_surface', 'median_facet_MAE_um']:.2f} µm**
- strictly local outer-curvature MAE: **{table.loc['local_outer_curvature', 'median_facet_MAE_um']:.2f} µm**
- measured outer-curvature MAE: **{table.loc['outer_curvature', 'median_facet_MAE_um']:.2f} µm**
- repeat-template MAE: **{table.loc['repeat_template', 'median_facet_MAE_um']:.2f} µm**
- fixed six-neighbour prior MAE: **{table.loc['six_neighbor_prior', 'median_facet_MAE_um']:.2f} µm**
- position-only smoother MAE: **{table.loc['position_smoother', 'median_facet_MAE_um']:.2f} µm**
- position plus local-outer MAE: **{table.loc['position_plus_outer', 'median_facet_MAE_um']:.2f} µm**

Relative to the general quadratic surface:

- strictly local outer curvature: median paired advantage **{local_outer['median_paired_advantage_um']:.2f} µm**, **{int(local_outer['blocks_favouring_method'])}/5** blocks, p **{local_outer['exact_block_sign_flip_p']:.4f}**
- outer curvature: median paired advantage **{outer['median_paired_advantage_um']:.2f} µm**, **{int(outer['blocks_favouring_method'])}/5** blocks, exact block sign-flip p **{outer['exact_block_sign_flip_p']:.4f}**
- repeat template: median paired advantage **{repeat['median_paired_advantage_um']:.2f} µm**, **{int(repeat['blocks_favouring_method'])}/5** blocks, p **{repeat['exact_block_sign_flip_p']:.4f}**
- six-neighbour prior: median paired advantage **{neighbour['median_paired_advantage_um']:.2f} µm**, **{int(neighbour['blocks_favouring_method'])}/5** blocks, p **{neighbour['exact_block_sign_flip_p']:.4f}**
- position smoother: median paired advantage **{spatial['median_paired_advantage_um']:.2f} µm**, **{int(spatial['blocks_favouring_method'])}/5** blocks, p **{spatial['exact_block_sign_flip_p']:.4f}**
- local outer curvature beyond the matched position smoother: median paired advantage **{matched['median_paired_advantage_um']:.2f} µm**, **{int(matched['blocks_favouring_method'])}/5** blocks, p **{matched['exact_block_sign_flip_p']:.4f}**

With five blocks, the smallest possible two-sided sign-flip p-value is 0.0625.
The block result is therefore used to assess consistency, not to manufacture a
conventional significance claim.

## Practical isolated-loss scenario

The primary analysis deliberately removes a complete spatial region. A second,
exploratory analysis removes one facet at a time, matching the common practical
case in which neighbouring internal surfaces remain visible. Its rules are
fixed: the six nearest surviving facets set depth by inverse-square distance,
and a shared quadratic residual supplies within-facet shape.

- flat training depth: **{isolated.loc['flat_depth', 'median_facet_MAE_um']:.2f} µm**
- fixed global quadratic: **{isolated.loc['quadratic_surface_fixed', 'median_facet_MAE_um']:.2f} µm**
- six-neighbour depth plus shared shape: **{isolated.loc['six_neighbor_prior', 'median_facet_MAE_um']:.2f} µm**
- six-neighbour p90 facet MAE: **{isolated.loc['six_neighbor_prior', 'p90_facet_MAE_um']:.2f} µm**

This scenario is closer to deployment but less independent than the spatial-
block stress test. It is reported as method development, not confirmation.

## Uncertainty

Every outer fold receives an error interval calibrated from leave-one-block-out
predictions made only inside its four training blocks. The summary reports the
median calibrated 90% half-width and held-out point coverage. These intervals
describe prediction error against the candidate CT boundary; they do not cover
uncertainty in anatomical identity or fossil deformation.

## Interpretation

The axisymmetric and general quadratic models are ellipsoid-like baselines.
The strictly local outer-curvature model uses only the sag of the surviving
surface and per-facet curvature amplitudes; it excludes absolute position,
surface orientation and lattice context. The broader outer model adds those
context variables. The repeat template uses homologous facets but no
test-facet internal points. The position smoother is a confound control: good
performance from it indicates interpolation of a specimen-level depth field.
The fixed six-neighbour prior is the directly deployable version of that idea:
it borrows depth only from the nearest preserved facets and combines it with a
shared quadratic within-facet shape.
The matched position-plus-local-outer comparison asks whether local curvature
adds anything after that spatial trend has already been supplied.

This experiment can establish an image-domain reconstruction benchmark within
one specimen. It cannot prove that the CT edge is the proximal lens surface,
recover a living undeformed eye, or establish transfer to another fossil.

## Provenance

- feature table: `{args.features}`
- feature table SHA-256: `{sha256(args.features)}`
- rows: {len(frame)}
- facets: {frame['facet_id'].nunique()}
- spatial blocks: {frame['cv_block'].nunique()}
- voxel spacing: {SPACING_UM} µm isotropic
- random seed declared for provenance: {SEED}
"""
    (out / "EXPERIMENT_54_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = prepare(pd.read_csv(args.features))
    predictions, facets, tuning, coverage = run(frame)
    isolated_facets, isolated_summary = run_isolated_loss(frame)
    export_six_neighbor_surfaces(frame, args.out)
    summary, block, comparisons = summarize(frame, facets, coverage)
    predictions.to_csv(args.out / "experiment_54_predictions.csv", index=False)
    facets.to_csv(args.out / "experiment_54_facet_MAE.csv", index=False)
    tuning.to_csv(args.out / "experiment_54_nested_tuning.csv", index=False)
    coverage.to_csv(args.out / "experiment_54_interval_coverage.csv", index=False)
    summary.to_csv(args.out / "experiment_54_model_summary.csv", index=False)
    block.to_csv(args.out / "experiment_54_block_MAE.csv", index=False)
    comparisons.to_csv(args.out / "experiment_54_comparisons.csv", index=False)
    isolated_facets.to_csv(
        args.out / "experiment_54_isolated_facet_MAE.csv", index=False
    )
    isolated_summary.to_csv(
        args.out / "experiment_54_isolated_summary.csv", index=False
    )
    make_figure(predictions, facets, summary, block, coverage, args.out)
    write_report(
        args, frame, summary, comparisons, isolated_summary, args.out
    )
    print(summary.to_string(index=False))
    print("\nComparisons with the quadratic surface:")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()

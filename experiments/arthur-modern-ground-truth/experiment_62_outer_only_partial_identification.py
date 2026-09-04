#!/usr/bin/env python3
"""Experiment 62: outer-only point comparator and morphology scenarios.

The historical unpushed file was lost.  The experiment title, conclusion and
two orphan scalars described as a ridge alpha and support threshold were
recovered, but the definitions that produced those scalars were not.  This
replacement preserves that distinction and makes every reconstructed choice
explicit.  It does not claim that an outer-only point estimate identifies the
hidden proximal surface.

The replacement point model uses distal scale plus four reflection-invariant
distal shape descriptors, never same-eye proximal donors or eye position.
Alpha is selected by leave-one-Arthur-volume-out validation with equal volume
weights.  Because it does not recover the historical definition, its optional
test predictions are explicitly labelled ``experiment62_reconstructed`` and
cannot silently enter Experiment 63 as the unchanged comparator.  Separately,
held-out-volume median residuals define three named alternative-morphology
scenarios.  These expose sensitivity to an unobserved population/segmentation
regime; they are not calibrated confidence or identification intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VOLUMES = ("20231107", "20240530", "20240701")
RIDGE_ALPHAS = np.logspace(-2, 3, 12)
RECOVERED_SELECTED_ALPHA = 123.28467394420659
RECOVERED_SUPPORT_THRESHOLD = 3.095098748914439
SPECIFICATION_STATUS = "reconstructed_after_unpushed_worktree_loss"
OUTER_FEATURE_NAMES = (
    "distal_scale_um",
    "distal_gradient_magnitude",
    "distal_curvature_eigenvalue_1",
    "distal_curvature_eigenvalue_2",
    "distal_normalized_fit_residual",
)
TARGET_NAMES = tuple(f"target_c{index}" for index in range(6))
EVEN_INDICES = np.array([0, 1, 3, 5], dtype=int)


def canonical_design() -> np.ndarray:
    axis = np.arange(-0.65, 0.65 + 0.13 / 2.0, 0.13)
    xx, yy = np.meshgrid(axis, axis)
    keep = xx * xx + yy * yy <= 0.65**2 + 1e-12
    x, y = xx[keep], yy[keep]
    design = np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])
    if design.shape != (81, 6):
        raise AssertionError(f"Unexpected canonical design shape {design.shape}")
    return design


CANONICAL_DESIGN = canonical_design()


def strict_bool(values: pd.Series) -> np.ndarray:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "1": True,
        "0": False,
    }
    converted = values.map(mapping)
    if converted.isna().any():
        raise ValueError(f"Column {values.name} is not strictly boolean")
    return converted.to_numpy(bool)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_table(table: pd.DataFrame, require_volume: bool) -> pd.DataFrame:
    required = {
        "eye_id",
        "lens_index",
        "distal_qc",
        "target_resolvable",
        "central",
        "target_depth_um",
        *OUTER_FEATURE_NAMES,
        *TARGET_NAMES,
    }
    if require_volume:
        required.add("volume")
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Input table lacks required columns: {missing}")
    result = table.copy()
    for name in ("distal_qc", "target_resolvable", "central"):
        result[name] = strict_bool(result[name])
    numeric = ["lens_index", "target_depth_um", *OUTER_FEATURE_NAMES, *TARGET_NAMES]
    for name in numeric:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    primary = result["distal_qc"] & result["target_resolvable"]
    if not primary.any():
        raise ValueError("No distal-QC, target-resolvable rows")
    values = result.loc[primary, ["target_depth_um", *OUTER_FEATURE_NAMES, *TARGET_NAMES]].to_numpy(float)
    if not np.all(np.isfinite(values)) or np.any(result.loc[primary, "target_depth_um"] <= 0):
        raise ValueError("Primary rows contain invalid features or targets")
    if require_volume and tuple(sorted(result.loc[primary, "volume"].astype(str).unique())) != tuple(sorted(VOLUMES)):
        raise ValueError(f"Source table must contain exactly {VOLUMES}")
    return result


def symmetrized_targets(table: pd.DataFrame) -> np.ndarray:
    targets = table.loc[:, TARGET_NAMES].to_numpy(float, copy=True)
    central = table["central"].to_numpy(bool)
    targets[central, 1] = 0.0
    targets[central, 2] = 0.0
    targets[central, 4] = 0.0
    curvature = 0.5 * (targets[central, 3] + targets[central, 5])
    targets[central, 3] = curvature
    targets[central, 5] = curvature
    return targets


def equal_group_weights(groups: Sequence[str]) -> np.ndarray:
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    count = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray(
        [len(groups) / (len(unique) * count[value]) for value in groups], dtype=float
    )


@dataclass(frozen=True)
class OuterOnlyRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray, central: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_scale
        even = self.target_mean + standardized @ self.coefficients
        full = np.zeros((len(standardized), 6), dtype=float)
        full[:, EVEN_INDICES] = even
        central = np.asarray(central, dtype=bool)
        full[central, 1] = 0.0
        average = 0.5 * (full[central, 3] + full[central, 5])
        full[central, 3] = average
        full[central, 5] = average
        return full

    def support_radius(self, features: np.ndarray) -> np.ndarray:
        """Reconstructed descriptive score; it never gates a prediction."""
        standardized = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_scale
        return np.linalg.norm(standardized, axis=1)


def fit_ridge(features: np.ndarray, targets: np.ndarray, groups, alpha: float) -> OuterOnlyRidge:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)[:, EVEN_INDICES]
    weights = equal_group_weights(groups)
    total = float(np.sum(weights))
    feature_mean = np.sum(x * weights[:, None], axis=0) / total
    variance = np.sum((x - feature_mean) ** 2 * weights[:, None], axis=0) / total
    if np.any(variance <= 1e-24):
        raise ValueError("An outer-only feature has zero weighted variance")
    feature_scale = np.sqrt(variance)
    xs = (x - feature_mean) / feature_scale
    target_mean = np.sum(y * weights[:, None], axis=0) / total
    yc = y - target_mean
    xw = xs * np.sqrt(weights)[:, None]
    yw = yc * np.sqrt(weights)[:, None]
    coefficients = np.linalg.solve(
        xw.T @ xw + float(alpha) * np.eye(x.shape[1]), xw.T @ yw
    )
    return OuterOnlyRidge(
        feature_mean, feature_scale, target_mean, coefficients, float(alpha)
    )


def normalized_mae(prediction: np.ndarray, truth: np.ndarray, depth: np.ndarray) -> np.ndarray:
    predicted_grid = prediction @ CANONICAL_DESIGN.T
    truth_grid = truth @ CANONICAL_DESIGN.T
    return np.mean(np.abs(predicted_grid - truth_grid), axis=1) / depth


def leave_one_volume_out(source: pd.DataFrame) -> tuple[float, pd.DataFrame, np.ndarray]:
    features = source.loc[:, OUTER_FEATURE_NAMES].to_numpy(float)
    truth = symmetrized_targets(source)
    groups = source["volume"].astype(str).to_numpy()
    depth = source["target_depth_um"].to_numpy(float)
    rows: list[dict] = []
    predictions_by_alpha: dict[float, np.ndarray] = {}
    for alpha in RIDGE_ALPHAS:
        prediction = np.full_like(truth, np.nan)
        heldout_medians = []
        for heldout in VOLUMES:
            train = groups != heldout
            test = ~train
            model = fit_ridge(features[train], truth[train], groups[train], float(alpha))
            prediction[test] = model.predict(features[test], source.loc[test, "central"].to_numpy(bool))
            median = float(np.median(normalized_mae(prediction[test], truth[test], depth[test])))
            heldout_medians.append(median)
            rows.append(
                {
                    "alpha": float(alpha),
                    "heldout_volume": heldout,
                    "heldout_median_81pt_normalized_mae": median,
                }
            )
        mean = float(np.mean(heldout_medians))
        rows.append(
            {
                "alpha": float(alpha),
                "heldout_volume": "equal_volume_mean",
                "heldout_median_81pt_normalized_mae": mean,
            }
        )
        predictions_by_alpha[float(alpha)] = prediction
    audit = pd.DataFrame(rows)
    means = audit[audit["heldout_volume"] == "equal_volume_mean"]
    best = float(means["heldout_median_81pt_normalized_mae"].min())
    selected = float(means.loc[means["heldout_median_81pt_normalized_mae"] == best, "alpha"].min())
    return selected, audit, predictions_by_alpha[selected]


def scenario_shifts(source: pd.DataFrame, loo_prediction: np.ndarray) -> pd.DataFrame:
    truth = symmetrized_targets(source)
    residual = truth - loo_prediction
    rows = []
    for volume in VOLUMES:
        select = source["volume"].astype(str).to_numpy() == volume
        median = np.median(residual[select], axis=0)
        median[[2, 4]] = 0.0
        row = {"scenario": f"heldout_residual_{volume}", "source_volume": volume}
        row.update({f"shift_c{index}": float(value) for index, value in enumerate(median)})
        rows.append(row)
    return pd.DataFrame(rows)


def loo_summary(source: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    truth = symmetrized_targets(source)
    error = normalized_mae(prediction, truth, source["target_depth_um"].to_numpy(float))
    rows = []
    for volume in VOLUMES:
        select = source["volume"].astype(str).to_numpy() == volume
        absolute_grid_error = np.mean(
            np.abs(
                prediction[select] @ CANONICAL_DESIGN.T
                - truth[select] @ CANONICAL_DESIGN.T
            ),
            axis=1,
        )
        rows.append(
            {
                "heldout_volume": volume,
                "n_lenses": int(np.sum(select)),
                "median_81pt_mae_um": float(np.median(absolute_grid_error)),
                "median_81pt_normalized_mae": float(np.median(error[select])),
                "p90_81pt_normalized_mae": float(np.quantile(error[select], 0.90)),
            }
        )
    return pd.DataFrame(rows)


def scenario_validation(
    source: pd.DataFrame, prediction: np.ndarray, shifts: pd.DataFrame
) -> pd.DataFrame:
    """Describe post-hoc scenario sensitivity; this is not model validation."""
    truth = symmetrized_targets(source)
    depth = source["target_depth_um"].to_numpy(float)
    groups = source["volume"].astype(str).to_numpy()
    candidates = [("unshifted_point_model", np.zeros(6, dtype=float))]
    for row in shifts.itertuples(index=False):
        candidates.append(
            (
                str(row.scenario),
                np.asarray([getattr(row, f"shift_c{i}") for i in range(6)]),
            )
        )
    rows = []
    for target_volume in VOLUMES:
        select = groups == target_volume
        for scenario, shift in candidates:
            shifted = prediction[select] + shift[None, :]
            central = source.loc[select, "central"].to_numpy(bool)
            shifted[central, 1] = shifted[central, 2] = shifted[central, 4] = 0.0
            average = 0.5 * (shifted[central, 3] + shifted[central, 5])
            shifted[central, 3] = shifted[central, 5] = average
            grid_error = np.mean(
                np.abs(
                    shifted @ CANONICAL_DESIGN.T
                    - truth[select] @ CANONICAL_DESIGN.T
                ),
                axis=1,
            )
            rows.append(
                {
                    "target_heldout_volume": target_volume,
                    "scenario": scenario,
                    "n_lenses": int(np.sum(select)),
                    "median_81pt_mae_um": float(np.median(grid_error)),
                    "median_81pt_normalized_mae": float(
                        np.median(grid_error / depth[select])
                    ),
                }
            )
    return pd.DataFrame(rows)


def export_test_predictions(
    test: pd.DataFrame,
    model: OuterOnlyRidge,
    shifts: pd.DataFrame,
    output: Path,
    scenario_output: Path,
    support_output: Path,
    reconstructed_support_threshold: float,
) -> None:
    primary = test[test["distal_qc"] & test["target_resolvable"]].copy()
    features = primary.loc[:, OUTER_FEATURE_NAMES].to_numpy(float)
    prediction = model.predict(features, primary["central"].to_numpy(bool))
    rows = primary.loc[:, ["eye_id", "lens_index"]].copy()
    rows["method"] = "experiment62_reconstructed"
    for index in range(6):
        rows[f"prediction_c{index}"] = prediction[:, index]
    rows.to_csv(output, index=False)

    scenario_rows = []
    for scenario in shifts.itertuples(index=False):
        shift = np.asarray([getattr(scenario, f"shift_c{i}") for i in range(6)])
        values = prediction + shift[None, :]
        central = primary["central"].to_numpy(bool)
        values[central, 1] = 0.0
        values[central, 2] = 0.0
        values[central, 4] = 0.0
        average = 0.5 * (values[central, 3] + values[central, 5])
        values[central, 3] = average
        values[central, 5] = average
        for position, identity in enumerate(primary[["eye_id", "lens_index"]].itertuples(index=False)):
            row = {
                "eye_id": identity.eye_id,
                "lens_index": int(identity.lens_index),
                "scenario": scenario.scenario,
            }
            row.update({f"prediction_c{i}": float(values[position, i]) for i in range(6)})
            scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(scenario_output, index=False)

    score = model.support_radius(features)
    pd.DataFrame(
        {
            "eye_id": primary["eye_id"].astype(str),
            "lens_index": primary["lens_index"].astype(int),
            "reconstructed_standardized_radius": score,
            "reconstructed_source_q95_threshold": reconstructed_support_threshold,
            "inside_reconstructed_support_flag": score <= reconstructed_support_threshold,
        }
    ).to_csv(support_output, index=False)


def plot_audit(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    selected_alpha: float,
    output: Path,
) -> None:
    means = audit[audit["heldout_volume"] == "equal_volume_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].plot(means["alpha"], means["heldout_median_81pt_normalized_mae"], marker="o")
    selected_score = float(
        means.loc[
            means["alpha"] == selected_alpha,
            "heldout_median_81pt_normalized_mae",
        ].iloc[0]
    )
    axes[0].scatter(
        [selected_alpha],
        [selected_score],
        marker="*",
        s=130,
        color="tab:orange",
        zorder=4,
        label=f"Selected grid alpha ({selected_alpha:g})",
    )
    axes[0].axvline(
        RECOVERED_SELECTED_ALPHA,
        color="black",
        linestyle="--",
        label=f"Recovered orphan alpha ({RECOVERED_SELECTED_ALPHA:.2f}; unused)",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Ridge alpha")
    axes[0].set_ylabel("Equal-volume mean held-out median normalized MAE")
    axes[0].set_title("Source-only tuning")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(summary["heldout_volume"], summary["median_81pt_mae_um"])
    axes[1].set_ylabel("Held-out median MAE (µm)")
    axes[1].set_title("Outer-only transfer by volume")
    fig.suptitle("Experiment 62: point comparator, not identification")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--test-table", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_all = validate_table(pd.read_csv(args.source_table), require_volume=True)
    source = source_all[source_all["distal_qc"] & source_all["target_resolvable"]].copy()
    selected, audit, loo_prediction = leave_one_volume_out(source)
    recovered_alpha_reproduced = selected == RECOVERED_SELECTED_ALPHA
    summary = loo_summary(source, loo_prediction)
    shifts = scenario_shifts(source, loo_prediction)
    scenario_audit = scenario_validation(source, loo_prediction, shifts)
    model = fit_ridge(
        source.loc[:, OUTER_FEATURE_NAMES].to_numpy(float),
        symmetrized_targets(source),
        source["volume"].astype(str).to_numpy(),
        selected,
    )
    support_radius = model.support_radius(source.loc[:, OUTER_FEATURE_NAMES].to_numpy(float))
    reconstructed_support_threshold = float(np.quantile(support_radius, 0.95))

    audit.to_csv(args.output / "experiment_62_alpha_selection.csv", index=False)
    summary.to_csv(args.output / "experiment_62_loo_summary.csv", index=False)
    shifts.to_csv(args.output / "experiment_62_alternative_morphology_shifts.csv", index=False)
    scenario_audit.to_csv(
        args.output / "experiment_62_scenario_sensitivity.csv", index=False
    )
    source_binding = {
        "filename": args.source_table.name,
        "bytes": int(args.source_table.stat().st_size),
        "sha256": sha256_file(args.source_table),
        "n_rows_total": int(len(source_all)),
        "n_primary_distal_qc_target_resolvable_rows": int(len(source)),
        "n_primary_rows_by_volume": {
            str(volume): int(count)
            for volume, count in source.groupby("volume", sort=True).size().items()
        },
    }
    diagnostics = {
        "experiment": 62,
        "specification_status": SPECIFICATION_STATUS,
        "scope": "outer-only source model with no same-eye proximal donors",
        "source_table_binding": source_binding,
        "outer_features": list(OUTER_FEATURE_NAMES),
        "selected_alpha": selected,
        "selected_alpha_is_lower_grid_boundary": bool(selected == float(RIDGE_ALPHAS[0])),
        "recovered_historical_selected_alpha": RECOVERED_SELECTED_ALPHA,
        "recovered_alpha_reproduced": recovered_alpha_reproduced,
        "selected_alpha_status": (
            "exact_scalar_recovered_from_lost_run_and_reproduced"
            if recovered_alpha_reproduced
            else "unresolved: reconstructed protocol does not reproduce the recovered scalar; historical feature/score definition was lost"
        ),
        "recovered_support_threshold": RECOVERED_SUPPORT_THRESHOLD,
        "support_threshold_status": (
            "orphan scalar only: historical score derivation unresolved and the "
            "scalar is not applied to the reconstructed score"
        ),
        "reconstructed_support_threshold": reconstructed_support_threshold,
        "reconstructed_support_threshold_definition": (
            "unweighted q95 source standardized Euclidean feature radius; "
            "descriptive only and never removes a row"
        ),
        "reconstructed_source_support_radius_q50_q90_q95_q99": [
            float(value) for value in np.quantile(support_radius, [0.50, 0.90, 0.95, 0.99])
        ],
        "alternative_morphology_scenarios": shifts["scenario"].tolist(),
        "scenario_status": (
            "explicit held-out-volume median-residual sensitivity scenarios; "
            "not confidence intervals and not unique reconstructions"
        ),
        "scenario_grid_envelope_mean_width_um": float(
            np.mean(
                np.ptp(
                    shifts[[f"shift_c{i}" for i in range(6)]].to_numpy(float)
                    @ CANONICAL_DESIGN.T,
                    axis=0,
                )
            )
        ),
    }
    (args.output / "experiment_62_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    plot_audit(
        audit,
        summary,
        selected,
        args.output / "experiment_62_outer_only_audit.png",
    )

    if args.test_table is not None:
        test = validate_table(pd.read_csv(args.test_table), require_volume=False)
        export_test_predictions(
            test,
            model,
            shifts,
            args.output / "experiment_62_reconstructed_predictions.csv",
            args.output / "experiment_62_test_scenarios.csv",
            args.output / "experiment_62_test_support_flags.csv",
            reconstructed_support_threshold,
        )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 58: frozen-model cross-volume validation on three Drosophila scans."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_57_outer_only_validation import (
    RIDGE_ALPHAS,
    SEED,
    bootstrap_ci,
    parse_wrl_surfaces,
    prepare_records,
    sha256,
)


def raw_positions_from_rdata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Invert the saved prcomp transform to recover raw mesh coordinates."""
    try:
        import rdata
    except ImportError as exc:
        raise RuntimeError("Install the Python package 'rdata' for Experiment 58") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = rdata.conversion.convert(rdata.parser.parse_file(path))
    n_lenses = len(data["lens"])
    pca = data["eye_pca"]
    scores = np.asarray(pca["x"], dtype=float)
    rotation = np.asarray(pca["rotation"], dtype=float)
    centre = np.asarray(pca["center"], dtype=float)
    raw = scores @ rotation.T + centre
    if len(raw) != 2 * n_lenses:
        raise ValueError(f"Unexpected PCA score count in {path}")
    return raw[:n_lenses], raw[n_lenses:]


def load_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    rows = manifest.get("volumes", manifest)
    required = {"volume", "lens_mesh", "tip_mesh", "rdata"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest row is missing {sorted(missing)}")
    return rows


def prepare_all(
    rows: list[dict], include_invalid_targets: bool = False
) -> tuple[list[dict], list[dict]]:
    all_records: list[dict] = []
    all_diagnostics: list[dict] = []
    for row in rows:
        volume = str(row["volume"])
        lens_mesh = Path(row["lens_mesh"])
        tip_mesh = Path(row["tip_mesh"])
        rdata_path = Path(row["rdata"])
        _, lens_surfaces = parse_wrl_surfaces(lens_mesh)
        _, tip_surfaces = parse_wrl_surfaces(tip_mesh)
        lens_positions, tip_positions = raw_positions_from_rdata(rdata_path)

        records, diagnostics = prepare_records(
            lens_surfaces,
            lens_positions,
            tip_positions,
            include_invalid_targets=include_invalid_targets,
        )
        for record in records:
            record["volume"] = volume

        tip_distances = np.column_stack(
            [cKDTree(surface).query(tip_positions, workers=-1)[0] for surface in tip_surfaces]
        )
        diagnostics.update(
            {
                "volume": volume,
                "median_tip_landmark_mesh_distance_um": float(
                    np.median(np.min(tip_distances, axis=1))
                ),
                "lens_mesh_sha256": sha256(lens_mesh),
                "tip_mesh_sha256": sha256(tip_mesh),
                "rdata_sha256": sha256(rdata_path),
            }
        )
        all_records.extend(records)
        all_diagnostics.append(diagnostics)
        print(json.dumps(diagnostics, indent=2), flush=True)
    return all_records, all_diagnostics


def run_frozen_transfer(
    records: list[dict], training_volume: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if any(
        not record.get("target_valid", True)
        or not np.all(np.isfinite(record["thickness"]))
        for record in records
    ):
        raise ValueError("run_frozen_transfer requires target-QC records only")
    train = [record for record in records if record["volume"] == training_volume]
    tests = [record for record in records if record["volume"] != training_volume]
    if not train or not tests:
        raise ValueError("Manifest needs the training volume and at least one test volume")

    x_train = np.vstack([record["outer_features"] for record in train])
    y_train = np.vstack([record["thickness"] for record in train])
    model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_absolute_error")
    )
    model.fit(x_train, y_train)
    template = np.median(y_train, axis=0)

    rows = []
    for record in tests:
        ridge_thickness = model.predict(record["outer_features"][None, :])[0]
        predictions = {
            "axisymmetric_ellipsoid": record["ellipsoid_inner"],
            "training_volume_template": record["outer_grid"] - template,
            "outer_curvature_ridge": record["outer_grid"] - ridge_thickness,
        }
        for method, prediction in predictions.items():
            error = np.abs(prediction - record["inner_grid"])
            target_depth = float(np.median(record["thickness"]))
            rows.append(
                {
                    "training_volume": training_volume,
                    "test_volume": record["volume"],
                    "test_eye": record["eye"],
                    "landmark_id": record["landmark_id"],
                    "method": method,
                    "mae_um": float(np.mean(error)),
                    "p90_error_um": float(np.quantile(error, 0.90)),
                    "target_depth_um": target_depth,
                    "normalized_mae": float(np.mean(error) / target_depth),
                }
            )
    metrics = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for (volume, method), group in metrics.groupby(["test_volume", "method"], sort=True):
        low, high = bootstrap_ci(group["mae_um"].to_numpy(), rng)
        summary_rows.append(
            {
                "test_volume": volume,
                "method": method,
                "n_lenses": len(group),
                "median_lens_mae_um": group["mae_um"].median(),
                "bootstrap_ci_low_um": low,
                "bootstrap_ci_high_um": high,
                "median_normalized_mae": group["normalized_mae"].median(),
                "median_target_depth_um": group["target_depth_um"].median(),
            }
        )
    for method, group in metrics.groupby("method", sort=True):
        low, high = bootstrap_ci(group["mae_um"].to_numpy(), rng)
        summary_rows.append(
            {
                "test_volume": "pooled_test_volumes",
                "method": method,
                "n_lenses": len(group),
                "median_lens_mae_um": group["mae_um"].median(),
                "bootstrap_ci_low_um": low,
                "bootstrap_ci_high_um": high,
                "median_normalized_mae": group["normalized_mae"].median(),
                "median_target_depth_um": group["target_depth_um"].median(),
            }
        )

    pivot = metrics.pivot(
        index=["test_volume", "test_eye", "landmark_id"],
        columns="method",
        values="mae_um",
    )
    comparison_rows = []
    for volume in [*sorted(metrics["test_volume"].unique()), "pooled_test_volumes"]:
        subset = pivot if volume == "pooled_test_volumes" else pivot.loc[[volume]]
        advantage = (
            subset["training_volume_template"] - subset["outer_curvature_ridge"]
        ).to_numpy()
        low, high = bootstrap_ci(advantage, rng)
        comparison_rows.append(
            {
                "test_volume": volume,
                "n_lenses": len(advantage),
                "median_template_minus_outer_ridge_um": float(np.median(advantage)),
                "bootstrap_ci_low_um": low,
                "bootstrap_ci_high_um": high,
                "outer_ridge_wins": int(np.sum(advantage > 0)),
                "outer_ridge_win_fraction": float(np.mean(advantage > 0)),
            }
        )
    return metrics, pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def plot_summary(metrics: pd.DataFrame, output: Path) -> None:
    order = ["axisymmetric_ellipsoid", "training_volume_template", "outer_curvature_ridge"]
    labels = ["Outer-only\nellipsoid", "Training-volume\ntemplate", "Outer-curvature\nridge"]
    colours = ["#C44E52", "#4C72B0", "#55A868"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    test_volumes = sorted(metrics["test_volume"].unique())
    for x, method in enumerate(order):
        values = metrics.loc[metrics["method"] == method, "mae_um"].to_numpy()
        axes[0].boxplot(values, positions=[x], widths=0.48, showfliers=False)
        for offset, volume, marker in zip([-0.08, 0.08], test_volumes, ["o", "s"]):
            subset = metrics[(metrics["method"] == method) & (metrics["test_volume"] == volume)]
            axes[0].scatter(
                x + offset,
                subset["mae_um"].median(),
                s=55,
                color=colours[x],
                edgecolor="black",
                linewidth=0.5,
                marker=marker,
                label=volume if x == 0 else None,
            )
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Per-lens proximal-surface MAE (µm)")
    axes[0].set_title("Frozen transfer from 20240701")
    axes[0].legend(frameon=False, fontsize=8)

    pivot = metrics.pivot(
        index=["test_volume", "test_eye", "landmark_id"], columns="method", values="mae_um"
    )
    advantage = pivot["training_volume_template"] - pivot["outer_curvature_ridge"]
    axes[1].hist(advantage, bins=50, color="#55A868", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axvline(np.median(advantage), color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Template MAE − outer-curvature MAE (µm)")
    axes[1].set_ylabel("Lenses")
    axes[1].set_title("Added information from outer curvature")
    figure.suptitle("Experiment 58 — independent-volume validation", fontsize=12)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-volume", default="20240701")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records, diagnostics = prepare_all(load_manifest(args.manifest))
    metrics, summary, comparison = run_frozen_transfer(records, args.training_volume)
    metrics.to_csv(args.output / "experiment_58_per_lens_metrics.csv", index=False)
    summary.to_csv(args.output / "experiment_58_summary.csv", index=False)
    comparison.to_csv(args.output / "experiment_58_outer_added_value.csv", index=False)
    (args.output / "experiment_58_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    plot_summary(metrics, args.output / "experiment_58_comparison.png")
    print(summary.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

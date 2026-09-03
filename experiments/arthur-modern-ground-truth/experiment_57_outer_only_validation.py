#!/usr/bin/env python3
"""Experiment 57: strict outer-only validation against Arthur Zhao's meshes.

The complete lens mesh and matched photoreceptor-tip landmarks are used only
to construct the synthetic mask and the hidden proximal-surface target.  Once
that mask is made, every predictor is built from the retained distal surface.

This first implementation targets the 20240701 volume because its raw lens and
tip landmark CSV files are public in reiserlab/eyemap_T4 and remain in the same
coordinate system as the supplied WRL meshes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree, distance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260902
GRID_LIMIT = 0.65
GRID_STEP = 0.13
# The three supplied meshes have different tessellation densities. Cap support
# is checked after the oracle layer split so proximal tessellation cannot decide
# whether a retained distal cap enters an outer-only prediction universe.
MIN_CAP_VERTICES = 25
RIDGE_ALPHAS = np.logspace(-2, 3, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_wrl_surfaces(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Read a triangulated VRML97 file with one shared coordinate table."""
    text = path.read_text(errors="ignore")
    point_match = re.search(r"point\s*\[", text)
    if point_match is None:
        raise ValueError(f"No Coordinate point block in {path}")

    start = point_match.end()
    stop = text.find("]", start)
    values = np.fromstring(
        re.sub(r"[,\n\r\t]+", " ", text[start:stop]), sep=" ", dtype=float
    )
    if values.size % 3:
        raise ValueError(f"Malformed point block in {path}")
    points = values.reshape(-1, 3)

    surfaces: list[np.ndarray] = []
    for match in re.finditer(r"coordIndex\s*\[", text):
        start = match.end()
        stop = text.find("]", start)
        indices = np.fromstring(
            re.sub(r"[,\n\r\t]+", " ", text[start:stop]), sep=" ", dtype=np.int64
        )
        used = np.unique(indices[indices >= 0])
        surfaces.append(points[used])

    if len(surfaces) != 2:
        raise ValueError(f"Expected two eye surfaces in {path}; found {len(surfaces)}")
    return points, surfaces


def read_positions(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    preferred = ["Position.X", "Position.Y", "Position.Z"]
    if all(column in table for column in preferred):
        return table[preferred].to_numpy(float)
    numeric = table.select_dtypes(include=[np.number])
    if numeric.shape[1] < 3:
        raise ValueError(f"Could not find three coordinate columns in {path}")
    return numeric.iloc[:, :3].to_numpy(float)


def sphere_center(points: np.ndarray) -> np.ndarray:
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.einsum("ij,ij->i", points, points)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return solution[:3]


def tangent_frame(outward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = outward / np.linalg.norm(outward)
    reference = np.array([0.0, 0.0, 1.0])
    u = reference - np.dot(reference, w) * w
    if np.linalg.norm(u) < 0.15:
        reference = np.array([1.0, 0.0, 0.0])
        u = reference - np.dot(reference, w) * w
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    v /= np.linalg.norm(v)
    return u, v, w


def split_two_layers(values: np.ndarray) -> np.ndarray:
    """Deterministic one-dimensional two-means split."""
    centres = np.quantile(values, [0.2, 0.8])
    for _ in range(30):
        assignment = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        updated = np.array(
            [values[assignment == group].mean() for group in range(2)]
        )
        if np.allclose(updated, centres):
            break
        centres = updated
    return assignment == np.argmax(centres)


def polynomial_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])


def robust_quadratic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    design = polynomial_design(x, y)
    weights = np.ones(len(z))
    beta = np.zeros(design.shape[1])
    for _ in range(8):
        root_weight = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(
            design * root_weight[:, None], z * root_weight, rcond=None
        )
        residual = z - design @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        standardized = np.abs(residual) / (1.345 * scale)
        weights = np.ones_like(standardized)
        tail = standardized > 1.0
        weights[tail] = 1.0 / standardized[tail]
    rmse = float(np.sqrt(np.mean((z - design @ beta) ** 2)))
    return beta, rmse


def canonical_grid() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(-GRID_LIMIT, GRID_LIMIT + GRID_STEP / 2.0, GRID_STEP)
    xx, yy = np.meshgrid(axis, axis)
    keep = xx * xx + yy * yy <= GRID_LIMIT * GRID_LIMIT + 1e-12
    return xx[keep], yy[keep]


def axisymmetric_ellipsoid_prediction(
    outer_local: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Continue an axisymmetric ellipsoid from its visible to hidden branch.

    The lateral semi-axis and distal sag are fitted independently.  This is
    the operational version of the ellipsoid proposal in Lauren's email; a
    spherical continuation would force equal axes and is catastrophically
    ill-conditioned for a shallow corneal cap.
    """
    x, y, z = outer_local.T
    rho = np.sqrt(x * x + y * y)
    central = rho <= scale
    rho, z = rho[central], z[central]
    rim = z[rho >= np.quantile(rho, 0.75)]
    apex = z[rho <= np.quantile(rho, 0.20)]
    initial_sag = float(np.clip(np.median(apex) - np.median(rim), 0.1, 15.0))
    initial_rim = float(np.median(rim))

    def residual(parameters: np.ndarray) -> np.ndarray:
        lateral_axis, vertical_axis, rim_z = parameters
        fraction = np.maximum(1.0 - (rho / lateral_axis) ** 2, 1e-9)
        predicted = rim_z + vertical_axis * np.sqrt(fraction)
        return predicted - z

    fit = least_squares(
        residual,
        x0=np.array([scale * 1.05, initial_sag, initial_rim]),
        bounds=(
            np.array([scale * 1.001, 0.05, z.min() - 10.0]),
            np.array([scale * 2.0, 30.0, z.max() + 5.0]),
        ),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=300,
    )
    lateral_axis, vertical_axis, rim_z = fit.x
    target_rho = np.sqrt((grid_x * scale) ** 2 + (grid_y * scale) ** 2)
    fraction = np.maximum(1.0 - (target_rho / lateral_axis) ** 2, 0.0)
    return rim_z - vertical_axis * np.sqrt(fraction)


def prepare_records(
    lens_surfaces: list[np.ndarray],
    lens_positions: np.ndarray,
    tip_positions: np.ndarray,
    include_invalid_targets: bool = False,
) -> tuple[list[dict], dict]:
    if len(lens_positions) != len(tip_positions):
        raise ValueError("Lens and tip landmark tables must have equal lengths")

    tip_rows, lens_columns = linear_sum_assignment(distance.cdist(tip_positions, lens_positions))
    tip_for_lens = np.empty_like(lens_positions)
    tip_for_lens[lens_columns] = tip_positions[tip_rows]
    pair_distance = np.linalg.norm(lens_positions - tip_for_lens, axis=1)

    surface_distance = np.column_stack(
        [cKDTree(surface).query(lens_positions, workers=-1)[0] for surface in lens_surfaces]
    )
    eye = np.argmin(surface_distance, axis=1)

    raw_records: list[dict] = []
    failure_counts = {
        "patch_support": 0,
        "layer_support": 0,
        "outer_scale": 0,
        "central_support": 0,
        "nonfinite_target": 0,
        "target_nonpositive": 0,
        "outer_fit_rmse": 0,
        "inner_fit_rmse": 0,
    }
    for eye_id, surface in enumerate(lens_surfaces):
        landmark_ids = np.flatnonzero(eye == eye_id)
        _, nearest = cKDTree(lens_positions[landmark_ids]).query(surface, workers=-1)
        order = np.argsort(nearest)
        nearest_sorted = nearest[order]
        surface_sorted = surface[order]
        starts = np.searchsorted(nearest_sorted, np.arange(len(landmark_ids)), side="left")
        stops = np.searchsorted(nearest_sorted, np.arange(len(landmark_ids)), side="right")

        for local_id, landmark_id in enumerate(landmark_ids):
            patch = surface_sorted[starts[local_id] : stops[local_id]]
            # Two vertices are the mathematical minimum for the deterministic
            # oracle split. Eligibility after that split depends on retained
            # distal support, never on total (distal + hidden proximal) count.
            if len(patch) < 2:
                failure_counts["patch_support"] += 1
                continue

            # Oracle information is confined to this masking operation.
            oracle_outward = lens_positions[landmark_id] - tip_for_lens[landmark_id]
            oracle_outward /= np.linalg.norm(oracle_outward)
            oracle_depth = (patch - lens_positions[landmark_id]) @ oracle_outward
            is_outer = split_two_layers(oracle_depth)
            outer = patch[is_outer]
            inner = patch[~is_outer]
            if len(outer) < MIN_CAP_VERTICES:
                failure_counts["layer_support"] += 1
                continue
            target_support = len(inner) >= MIN_CAP_VERTICES
            if not target_support:
                failure_counts["layer_support"] += 1
            raw_records.append(
                {
                    "eye": eye_id,
                    "landmark_id": int(landmark_id),
                    "outer": outer,
                    "inner": inner,
                    "target_support": target_support,
                    "patch_vertices": len(patch),
                }
            )

    # This operational centre is fitted only after all hidden inner vertices
    # and all tip landmarks have been discarded from the predictor input.
    outer_centres = np.vstack([record["outer"].mean(axis=0) for record in raw_records])
    eye_centres = {
        eye_id: sphere_center(outer_centres[[r["eye"] == eye_id for r in raw_records]])
        for eye_id in sorted({r["eye"] for r in raw_records})
    }

    grid_x, grid_y = canonical_grid()
    records: list[dict] = []
    candidate_scale = []
    candidate_depth = []
    candidate_outer_rmse = []
    candidate_inner_rmse = []
    n_outer_qc = 0
    n_target_resolvable = 0
    n_target_qc = 0
    for record in raw_records:
        outer = record["outer"]
        inner = record["inner"]
        outer_origin = outer.mean(axis=0)
        u, v, w = tangent_frame(outer_origin - eye_centres[record["eye"]])

        def local(points: np.ndarray) -> np.ndarray:
            delta = points - outer_origin
            return np.column_stack([delta @ u, delta @ v, delta @ w])

        outer_local = local(outer)
        inner_local = local(inner)
        outer_rho = np.linalg.norm(outer_local[:, :2], axis=1)
        scale = float(np.quantile(outer_rho, 0.90))
        candidate_scale.append(scale)
        if not 3.0 <= scale <= 13.0:
            failure_counts["outer_scale"] += 1
            continue

        outer_keep = outer_rho <= scale
        inner_rho = np.linalg.norm(inner_local[:, :2], axis=1)
        inner_keep = inner_rho <= scale
        if outer_keep.sum() < MIN_CAP_VERTICES:
            failure_counts["central_support"] += 1
            continue

        outer_beta, outer_rmse = robust_quadratic(
            outer_local[outer_keep, 0] / scale,
            outer_local[outer_keep, 1] / scale,
            outer_local[outer_keep, 2],
        )
        design_grid = polynomial_design(grid_x, grid_y)
        outer_grid = design_grid @ outer_beta
        candidate_outer_rmse.append(outer_rmse)
        if outer_rmse > 2.5:
            failure_counts["outer_fit_rmse"] += 1
            continue

        ellipsoid_inner = axisymmetric_ellipsoid_prediction(
            outer_local[outer_keep], grid_x, grid_y, scale
        )
        # Predictor features: retained outer surface only.  The constant term
        # is removed because the local origin is arbitrary.
        outer_features = np.r_[scale, outer_beta[1:], outer_rmse]

        # Target availability and target quality are deliberately separate.
        # Experiment 59 assigns masks to every outer-QC record without looking
        # at either flag.  A finite target can still be retained for the
        # prespecified sensitivity analysis even when it fails anatomical QC.
        target_resolvable = False
        target_qc_pass = False
        target_reason = "valid"
        inner_rmse = float("nan")
        inner_grid = np.full_like(outer_grid, np.nan)
        thickness = np.full_like(outer_grid, np.nan)
        if not record["target_support"]:
            target_reason = "layer_support"
        elif inner_keep.sum() < MIN_CAP_VERTICES:
            failure_counts["central_support"] += 1
            target_reason = "central_support"
        else:
            inner_beta, inner_rmse = robust_quadratic(
                inner_local[inner_keep, 0] / scale,
                inner_local[inner_keep, 1] / scale,
                inner_local[inner_keep, 2],
            )
            inner_grid = design_grid @ inner_beta
            thickness = outer_grid - inner_grid
            candidate_inner_rmse.append(inner_rmse)
            if not np.all(np.isfinite(thickness)):
                failure_counts["nonfinite_target"] += 1
                target_reason = "nonfinite_target"
            else:
                target_resolvable = True
                candidate_depth.append(float(np.median(thickness)))
                # Target magnitude is not an outer-data QC variable.  It is
                # nevertheless an anatomical target-quality check: a fitted
                # proximal surface should lie inside the distal surface.
                if np.quantile(thickness, 0.05) <= 0.0:
                    failure_counts["target_nonpositive"] += 1
                    target_reason = "target_nonpositive"
                elif inner_rmse > 2.5:
                    failure_counts["inner_fit_rmse"] += 1
                    target_reason = "inner_fit_rmse"
                else:
                    target_qc_pass = True

        n_outer_qc += 1
        n_target_resolvable += int(target_resolvable)
        n_target_qc += int(target_qc_pass)
        if not target_qc_pass and not include_invalid_targets:
            continue
        records.append(
            {
                **record,
                # Retained for outer-only spatial blocking and neighbour
                # selection in Experiment 59. This centroid is computed after
                # the hidden proximal vertices have been discarded.
                "outer_origin": outer_origin,
                "scale": scale,
                "outer_rmse": outer_rmse,
                "inner_rmse": inner_rmse,
                "outer_features": outer_features,
                "outer_grid": outer_grid,
                "inner_grid": inner_grid,
                "thickness": thickness,
                "ellipsoid_inner": ellipsoid_inner,
                # ``target_valid`` remains as a compatibility alias for the
                # Experiment 57/58 code and their frozen result tables.
                "target_valid": target_qc_pass,
                "target_resolvable": target_resolvable,
                "target_qc_pass": target_qc_pass,
                "target_reason": target_reason,
            }
        )

    def quantiles(values: list[float]) -> list[float]:
        if not values:
            return []
        return [float(value) for value in np.quantile(values, [0.05, 0.5, 0.95])]

    diagnostics = {
        "include_invalid_targets": bool(include_invalid_targets),
        "n_records_returned": int(len(records)),
        "n_landmarks": int(len(lens_positions)),
        "n_eye_0": int(np.sum(eye == 0)),
        "n_eye_1": int(np.sum(eye == 1)),
        "median_lens_tip_distance_um": float(np.median(pair_distance)),
        "median_landmark_mesh_distance_um": float(
            np.median(surface_distance[np.arange(len(lens_positions)), eye])
        ),
        "n_raw_patches": int(len(raw_records)),
        "n_outer_qc_records": int(n_outer_qc),
        "n_target_resolvable": int(n_target_resolvable),
        "n_qc_records": int(n_target_qc),
        "qc_failure_counts": failure_counts,
        "candidate_scale_q05_q50_q95_um": quantiles(candidate_scale),
        "candidate_depth_q05_q50_q95_um": quantiles(candidate_depth),
        "candidate_outer_rmse_q05_q50_q95_um": quantiles(candidate_outer_rmse),
        "candidate_inner_rmse_q05_q50_q95_um": quantiles(candidate_inner_rmse),
    }
    return records, diagnostics


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = rng.choice(values, size=(4000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def run_cross_eye(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if any(
        not record.get("target_valid", True)
        or not np.all(np.isfinite(record["thickness"]))
        for record in records
    ):
        raise ValueError("run_cross_eye requires target-QC records only")
    rows = []
    for test_eye in [0, 1]:
        train = [record for record in records if record["eye"] != test_eye]
        test = [record for record in records if record["eye"] == test_eye]
        x_train = np.vstack([record["outer_features"] for record in train])
        y_train = np.vstack([record["thickness"] for record in train])
        x_test = np.vstack([record["outer_features"] for record in test])

        ridge = make_pipeline(
            StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_absolute_error")
        )
        ridge.fit(x_train, y_train)
        ridge_thickness = ridge.predict(x_test)
        template_thickness = np.median(y_train, axis=0)

        for index, record in enumerate(test):
            truth = record["inner_grid"]
            predictions = {
                "axisymmetric_ellipsoid": record["ellipsoid_inner"],
                "train_eye_template": record["outer_grid"] - template_thickness,
                "outer_curvature_ridge": record["outer_grid"] - ridge_thickness[index],
            }
            target_depth = float(np.median(record["thickness"]))
            for method, prediction in predictions.items():
                error = np.abs(prediction - truth)
                rows.append(
                    {
                        "test_eye": test_eye,
                        "landmark_id": record["landmark_id"],
                        "method": method,
                        "mae_um": float(np.mean(error)),
                        "p90_error_um": float(np.quantile(error, 0.90)),
                        "target_depth_um": target_depth,
                        "normalized_mae": float(np.mean(error) / target_depth),
                    }
                )

    per_lens = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summaries = []
    for (eye, method), group in per_lens.groupby(["test_eye", "method"], sort=True):
        low, high = bootstrap_ci(group["mae_um"].to_numpy(), rng)
        summaries.append(
            {
                "test_eye": eye,
                "method": method,
                "n_lenses": len(group),
                "median_lens_mae_um": group["mae_um"].median(),
                "bootstrap_ci_low_um": low,
                "bootstrap_ci_high_um": high,
                "median_normalized_mae": group["normalized_mae"].median(),
                "median_target_depth_um": group["target_depth_um"].median(),
            }
        )
    for method, group in per_lens.groupby("method", sort=True):
        low, high = bootstrap_ci(group["mae_um"].to_numpy(), rng)
        summaries.append(
            {
                "test_eye": "pooled",
                "method": method,
                "n_lenses": len(group),
                "median_lens_mae_um": group["mae_um"].median(),
                "bootstrap_ci_low_um": low,
                "bootstrap_ci_high_um": high,
                "median_normalized_mae": group["normalized_mae"].median(),
                "median_target_depth_um": group["target_depth_um"].median(),
            }
        )
    return per_lens, pd.DataFrame(summaries)


def plot_results(per_lens: pd.DataFrame, output: Path) -> None:
    order = ["axisymmetric_ellipsoid", "train_eye_template", "outer_curvature_ridge"]
    labels = ["Outer-only\nellipsoid", "Opposite-eye\ntemplate", "Outer-curvature\nridge"]
    colours = ["#C44E52", "#4C72B0", "#55A868"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)

    for eye, marker in [(0, "o"), (1, "s")]:
        subset = per_lens[per_lens["test_eye"] == eye]
        for x, method in enumerate(order):
            values = subset.loc[subset["method"] == method, "mae_um"].to_numpy()
            axes[0].scatter(
                x + (-0.07 if eye == 0 else 0.07), np.median(values),
                s=55, marker=marker, color=colours[x], edgecolor="black", linewidth=0.5,
                label=f"held-out eye {eye}" if x == 0 else None,
            )
    pooled = [per_lens.loc[per_lens["method"] == method, "mae_um"].to_numpy() for method in order]
    axes[0].boxplot(pooled, positions=np.arange(3), widths=0.42, showfliers=False)
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Per-lens proximal-surface MAE (µm)")
    axes[0].set_title("Whole-eye transfer")
    axes[0].legend(frameon=False, fontsize=8)

    pivot = per_lens.pivot(index=["test_eye", "landmark_id"], columns="method", values="mae_um")
    advantage = pivot["train_eye_template"] - pivot["outer_curvature_ridge"]
    axes[1].hist(advantage, bins=40, color="#55A868", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axvline(np.median(advantage), color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Template MAE − outer-curvature MAE (µm)")
    axes[1].set_ylabel("Lenses")
    axes[1].set_title("Does outer curvature add information?")
    figure.suptitle("Experiment 57 — 20240701 Drosophila validation", fontsize=12)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-mesh", type=Path, required=True)
    parser.add_argument("--tip-mesh", type=Path, required=True)
    parser.add_argument("--lens-csv", type=Path, required=True)
    parser.add_argument("--tip-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    _, lens_surfaces = parse_wrl_surfaces(args.lens_mesh)
    # Parse the tip mesh to verify the supplied file, but do not expose it to
    # prepare_records or any predictor.  Only the landmark CSV labels truth.
    _, tip_surfaces = parse_wrl_surfaces(args.tip_mesh)
    lens_positions = read_positions(args.lens_csv)
    tip_positions = read_positions(args.tip_csv)
    records, diagnostics = prepare_records(lens_surfaces, lens_positions, tip_positions)
    diagnostics["tip_mesh_eye_surfaces"] = len(tip_surfaces)
    diagnostics["lens_mesh_sha256"] = sha256(args.lens_mesh)
    diagnostics["tip_mesh_sha256"] = sha256(args.tip_mesh)
    diagnostics["lens_csv_sha256"] = sha256(args.lens_csv)
    diagnostics["tip_csv_sha256"] = sha256(args.tip_csv)

    per_lens, summary = run_cross_eye(records)
    # Detailed predictions are local-only by repository policy; committed
    # outputs contain aggregate results and QC counts, not supplied raw data.
    per_lens.to_csv(args.output / "experiment_57_per_lens_metrics.csv", index=False)
    summary.to_csv(args.output / "experiment_57_summary.csv", index=False)
    (args.output / "experiment_57_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    plot_results(per_lens, args.output / "experiment_57_comparison.png")
    print(json.dumps(diagnostics, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

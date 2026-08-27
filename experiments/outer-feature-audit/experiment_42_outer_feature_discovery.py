#!/usr/bin/env python3
"""Experiment 42: search for outer cues beyond specimen position.

The analysis keeps the Experiment 40 targets, facets, and five spatial blocks.
All new predictors are extracted outside the 9--19 voxel internal-edge window.
The primary feature family uses only the reconstructed outer surface.  A
secondary family also uses CT intensities from -1 to 6 voxels relative to the
outer tangent plane and is therefore interpreted as an outer-layer imaging
cue, not as pure geometry.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SPACING_UM = 3.7
WITHIN = ["un", "vn", "rn", "rn2", "un2", "vn2", "uv"]
ABSOLUTE = ["surf_x", "y", "z"]
CURRENT = ["relief", "fy", "fz", "lap", "pitch"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--volume", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--centers", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--surface-threshold", type=float, default=50.0)
    return p.parse_args()


def read_nrrd(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in NRRD header")
            if line in (b"\n", b"\r\n"):
                body_offset = f.tell()
                break
            header_lines.append(line.decode("ascii").rstrip())
    header: dict[str, str] = {}
    for line in header_lines:
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()
    shape = tuple(map(int, header["sizes"].split()))
    with path.open("rb") as f:
        f.seek(body_offset)
        if header.get("encoding", "raw").lower() in ("gzip", "gz"):
            raw = gzip.GzipFile(fileobj=f).read()
        else:
            raw = f.read()
    volume = np.frombuffer(raw, dtype=np.uint8).reshape(shape, order="F")
    return volume, header


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subvoxel_surface(smoothed: np.ndarray, threshold: float) -> np.ndarray:
    above = smoothed >= threshold
    rev = np.argmax(above[::-1], axis=0)
    any_above = above.any(axis=0)
    x0 = (smoothed.shape[0] - 1 - rev).astype(np.int32)
    surface = np.full(any_above.shape, np.nan, dtype=np.float32)
    ys, zs = np.where(any_above)
    xs = x0[ys, zs]
    valid = xs < smoothed.shape[0] - 1
    yv, zv, xv = ys[valid], zs[valid], xs[valid]
    v0 = smoothed[xv, yv, zv]
    v1 = smoothed[xv + 1, yv, zv]
    denom = v0 - v1
    frac = np.divide(v0 - threshold, denom, out=np.zeros_like(v0),
                     where=np.abs(denom) > 1e-6)
    surface[yv, zv] = xv + frac
    surface[ys[~valid], zs[~valid]] = xs[~valid]
    return surface


def map2(array: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return ndi.map_coordinates(array, np.vstack([y, z]), order=1,
                               mode="nearest")


def surface_maps(volume: np.ndarray, threshold: float):
    smoothed = ndi.gaussian_filter1d(volume, sigma=1.0, axis=0,
                                    output=np.float32)
    surface = subvoxel_surface(smoothed, threshold)
    del smoothed
    valid = np.isfinite(surface)
    indices = ndi.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = surface[tuple(indices)].astype(np.float32)
    del indices
    maps: dict[str, np.ndarray] = {"surface": filled}
    for scale in (2, 4, 8, 16):
        s = ndi.gaussian_filter(filled, sigma=scale).astype(np.float32)
        gy, gz = np.gradient(s)
        gyy, gyz = np.gradient(gy)
        _, gzz = np.gradient(gz)
        maps[f"smooth_{scale}"] = s
        maps[f"gy_{scale}"] = gy.astype(np.float32)
        maps[f"gz_{scale}"] = gz.astype(np.float32)
        maps[f"lap_{scale}"] = (gyy + gzz).astype(np.float32)
        maps[f"det_{scale}"] = (gyy * gzz - gyz * gyz).astype(np.float32)
        maps[f"aniso_{scale}"] = np.sqrt(
            np.maximum((gyy - gzz) ** 2 + 4 * gyz * gyz, 0)
        ).astype(np.float32)
    return maps


def neighbour_features(centers: pd.DataFrame) -> pd.DataFrame:
    xyz = centers[["x_vox", "y_vox", "z_vox"]].to_numpy(float)
    yz = centers[["y_vox", "z_vox"]].to_numpy(float)
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=7)
    rows = []
    for i in range(len(centers)):
        d = distances[i, 1:]
        vectors = yz[indices[i, 1:]] - yz[i]
        cov = np.cov(vectors.T)
        eig = np.linalg.eigvalsh(cov)
        angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]),
                                2 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        rows.append({
            "facet_id": int(centers.iloc[i]["facet_id"]),
            "nn_mean": float(np.mean(d)),
            "nn_cv": float(np.std(d) / max(np.mean(d), 1e-6)),
            "nn_range": float(np.max(d) - np.min(d)),
            "lattice_anisotropy": float(eig[-1] / max(eig[0], 1e-6)),
            "angle_gap_rmse": float(np.sqrt(np.mean(
                (gaps - np.pi / 3) ** 2
            ))),
        })
    return pd.DataFrame(rows)


def extract_features(
    volume: np.ndarray,
    maps: dict[str, np.ndarray],
    samples: pd.DataFrame,
    centers: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    center_lookup = centers.set_index("facet_id")
    neighbour = neighbour_features(centers).set_index("facet_id")
    output = samples.copy()
    surface_columns: list[str] = []
    shell_columns: list[str] = []

    for facet_id, idx in output.groupby("facet_id").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        c = center_lookup.loc[int(facet_id)]
        x, y, z = c[["x_vox", "y_vox", "z_vox"]].to_numpy(float)
        pitch = float(output.loc[idx[0], "pitch"])
        gy = float(map2(maps["gy_2"], np.array([y]), np.array([z]))[0])
        gz = float(map2(maps["gz_2"], np.array([y]), np.array([z]))[0])
        normal = np.array([1.0, -gy, -gz], float)
        normal /= np.linalg.norm(normal)
        t1 = np.array([gy, 1.0, 0.0], float)
        t1 -= normal * np.dot(t1, normal)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        t2 /= np.linalg.norm(t2)

        u = output.loc[idx, "un"].to_numpy(float) * pitch / 2
        v = output.loc[idx, "vn"].to_numpy(float) * pitch / 2
        q = (np.array([x, y, z])[:, None]
             + t1[:, None] * u[None]
             + t2[:, None] * v[None])
        qy, qz = q[1], q[2]
        surf_here = map2(maps["surface"], qy, qz)
        sag = (surf_here - q[0]) * normal[0]
        output.loc[idx, "surface_sag"] = sag

        for scale in (2, 4, 8, 16):
            smooth = map2(maps[f"smooth_{scale}"], qy, qz)
            gyv = map2(maps[f"gy_{scale}"], qy, qz)
            gzv = map2(maps[f"gz_{scale}"], qy, qz)
            output.loc[idx, f"relief_s{scale}"] = surf_here - smooth
            output.loc[idx, f"slope_s{scale}"] = np.hypot(gyv, gzv)
            output.loc[idx, f"lap_s{scale}"] = map2(
                maps[f"lap_{scale}"], qy, qz
            )
            output.loc[idx, f"det_s{scale}"] = map2(
                maps[f"det_{scale}"], qy, qz
            )
            output.loc[idx, f"aniso_s{scale}"] = map2(
                maps[f"aniso_{scale}"], qy, qz
            )

        # Per-facet surface shape over a disk approximately bounded by the rim.
        grid = np.linspace(-0.45 * pitch, 0.45 * pitch, 17)
        uu, vv = np.meshgrid(grid, grid, indexing="ij")
        disk = uu * uu + vv * vv <= (0.45 * pitch) ** 2
        qg = (np.array([x, y, z])[:, None]
              + t1[:, None] * uu.ravel()[None]
              + t2[:, None] * vv.ravel()[None])
        sg = map2(maps["surface"], qg[1], qg[2])
        sagg = ((sg - qg[0]) * normal[0]).reshape(uu.shape)
        r = np.hypot(uu, vv)
        core = disk & (r <= 0.15 * pitch)
        rim = disk & (r >= 0.35 * pitch)
        vals = sagg[disk]
        facet_values = {
            "facet_sag_amp": float(np.percentile(vals, 95)
                                    - np.percentile(vals, 5)),
            "facet_sag_std": float(np.std(vals)),
            "facet_core_rim": float(np.mean(sagg[core])
                                    - np.mean(sagg[rim])),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
        }
        facet_values.update(neighbour.loc[int(facet_id)].to_dict())
        for name, value in facet_values.items():
            output.loc[idx, name] = value

        # Secondary preserved-outer-layer intensities.  The deepest point is
        # 6 voxels, leaving a 3-voxel gap before the 9-voxel edge-search window.
        depths = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 6.0])
        profiles = []
        for depth in depths:
            xyz = q - normal[:, None] * depth
            profiles.append(ndi.map_coordinates(
                volume, xyz, order=1, mode="nearest"
            ))
        profiles = np.asarray(profiles)
        for j, depth in enumerate(depths):
            label = str(depth).replace("-", "m").replace(".", "p")
            output.loc[idx, f"shell_i_{label}"] = profiles[j]
        output.loc[idx, "shell_mean"] = profiles[1:].mean(axis=0)
        output.loc[idx, "shell_std"] = profiles[1:].std(axis=0)
        output.loc[idx, "shell_gradient"] = (
            profiles[-1] - profiles[1]
        ) / 6.0
        output.loc[idx, "shell_curvature"] = (
            profiles[4] - 2 * profiles[3] + profiles[2]
        )

    surface_columns = [
        "surface_sag",
        *[f"{kind}_s{scale}" for scale in (2, 4, 8, 16)
          for kind in ("relief", "slope", "lap", "det", "aniso")],
        "facet_sag_amp", "facet_sag_std", "facet_core_rim",
        "normal_y", "normal_z", "nn_mean", "nn_cv", "nn_range",
        "lattice_anisotropy", "angle_gap_rmse",
    ]
    shell_columns = [c for c in output.columns if c.startswith("shell_")]
    return output, surface_columns, shell_columns


def spatial_model(degree: int, alpha: float):
    features = ColumnTransformer([
        ("within", "passthrough", WITHIN),
        ("absolute", PolynomialFeatures(degree=degree, include_bias=False),
         ABSOLUTE),
    ], remainder="drop")
    return make_pipeline(features, StandardScaler(), Ridge(alpha=alpha))


def residual_model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def facet_mae(frame: pd.DataFrame, prediction_vox: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "facet_id": frame["facet_id"].to_numpy(),
        "cv_block": frame["cv_block"].to_numpy(),
        "error_um": np.abs(prediction_vox - frame["target_vox"].to_numpy())
                    * SPACING_UM,
    })
    return tmp.groupby(["facet_id", "cv_block"], as_index=False)[
        "error_um"
    ].mean().rename(columns={"error_um": "facet_MAE_um"})


def score(frame: pd.DataFrame, prediction_norm: np.ndarray) -> float:
    pred_vox = prediction_norm * frame["pitch"].to_numpy()
    return float(facet_mae(frame, pred_vox)["facet_MAE_um"].median())


def select_spatial(train: pd.DataFrame) -> tuple[float, int, float]:
    candidates = []
    groups = train["cv_block"].to_numpy()
    target = train["target_norm"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    for degree, alpha in itertools.product(
        (1, 2, 3), (0.1, 1.0, 10.0, 20.0, 100.0, 1000.0)
    ):
        pred = np.full(len(train), np.nan)
        for tr, va in cv.split(train, target, groups):
            model = spatial_model(degree, alpha)
            model.fit(train.iloc[tr], target[tr])
            pred[va] = model.predict(train.iloc[va])
        candidates.append((score(train, pred), degree, alpha))
    return min(candidates)


def select_residual_alpha(
    train: pd.DataFrame,
    feature_columns: list[str],
    degree: int,
    spatial_alpha: float,
) -> tuple[float, float]:
    groups = train["cv_block"].to_numpy()
    target = train["target_norm"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        prediction = np.full(len(train), np.nan)
        for tr, va in cv.split(train, target, groups):
            sm = spatial_model(degree, spatial_alpha)
            sm.fit(train.iloc[tr], target[tr])
            base_train = sm.predict(train.iloc[tr])
            base_val = sm.predict(train.iloc[va])
            rm = residual_model(alpha)
            rm.fit(train.iloc[tr][feature_columns],
                   target[tr] - base_train)
            prediction[va] = base_val + rm.predict(
                train.iloc[va][feature_columns]
            )
        candidates.append((score(train, prediction), alpha))
    return min(candidates)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def run_family(
    frame: pd.DataFrame,
    feature_columns: list[str],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame["cv_block"].to_numpy()
    target = frame["target_norm"].to_numpy()
    outer = GroupKFold(n_splits=len(np.unique(groups)))
    pred_base = np.full(len(frame), np.nan)
    pred_aug = np.full(len(frame), np.nan)
    tuning_rows = []
    coefficient_rows = []

    for tr, te in outer.split(frame, target, groups):
        train = frame.iloc[tr]
        test = frame.iloc[te]
        spatial_score, degree, spatial_alpha = select_spatial(train)
        residual_score, residual_alpha = select_residual_alpha(
            train, feature_columns, degree, spatial_alpha
        )
        sm = spatial_model(degree, spatial_alpha)
        sm.fit(train, target[tr])
        base_train = sm.predict(train)
        pred_base[te] = sm.predict(test)
        rm = residual_model(residual_alpha)
        rm.fit(train[feature_columns], target[tr] - base_train)
        pred_aug[te] = pred_base[te] + rm.predict(test[feature_columns])
        held = int(np.unique(groups[te])[0])
        tuning_rows.append({
            "family": family, "heldout_block": held,
            "spatial_degree": degree, "spatial_alpha": spatial_alpha,
            "spatial_inner_MAE_um": spatial_score,
            "residual_alpha": residual_alpha,
            "augmented_inner_MAE_um": residual_score,
        })
        coef = rm.named_steps["ridge"].coef_
        for name, value in zip(feature_columns, coef):
            coefficient_rows.append({
                "family": family, "heldout_block": held,
                "feature": name, "standardized_coefficient": float(value),
            })

    base_vox = pred_base * frame["pitch"].to_numpy()
    aug_vox = pred_aug * frame["pitch"].to_numpy()
    predictions = frame[["facet_id", "cv_block", "target_vox"]].copy()
    predictions["family"] = family
    predictions["spatial_prediction_vox"] = base_vox
    predictions["augmented_prediction_vox"] = aug_vox
    fbase = facet_mae(frame, base_vox).rename(
        columns={"facet_MAE_um": "spatial_MAE_um"}
    )
    faug = facet_mae(frame, aug_vox).rename(
        columns={"facet_MAE_um": "augmented_MAE_um"}
    )
    facets = fbase.merge(faug, on=["facet_id", "cv_block"])
    facets["family"] = family
    facets["improvement_um"] = (
        facets["spatial_MAE_um"] - facets["augmented_MAE_um"]
    )
    return predictions, facets, pd.DataFrame(tuning_rows), pd.DataFrame(coefficient_rows)


def make_figure(summary: pd.DataFrame, folds: pd.DataFrame,
                coefficients: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    x = np.arange(len(summary))
    axes[0].bar(x - 0.18, summary["spatial_median_MAE_um"], 0.36,
                label="spatial only", color="#A9B4C2")
    axes[0].bar(x + 0.18, summary["augmented_median_MAE_um"], 0.36,
                label="+ outer features", color="#2563EB")
    axes[0].set_xticks(x, summary["family"], rotation=25, ha="right")
    axes[0].set_ylabel("Median facet MAE (µm)")
    axes[0].set_title("Nested spatial-block validation")
    axes[0].legend(frameon=False)

    families = summary["family"].tolist()
    for i, family in enumerate(families):
        f = folds[folds["family"] == family]
        axes[1].scatter(np.full(len(f), i), f["improvement_um"], s=45)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(np.arange(len(families)), families,
                       rotation=25, ha="right")
    axes[1].set_ylabel("Spatial MAE − augmented MAE (µm)")
    axes[1].set_title("Improvement in each held-out block")

    chosen = summary.sort_values("augmented_median_MAE_um").iloc[0]["family"]
    c = coefficients[coefficients["family"] == chosen]
    top = (c.groupby("feature")["standardized_coefficient"]
           .agg(["median", lambda x: np.median(np.abs(x))])
           .rename(columns={"<lambda_0>": "median_abs"})
           .sort_values("median_abs").tail(10))
    axes[2].barh(top.index, top["median"], color="#D97706")
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_xlabel("Median standardized coefficient")
    axes[2].set_title(f"Exploratory coefficients: {chosen}")
    fig.tight_layout()
    fig.savefig(out / "experiment_42_outer_feature_discovery.png", dpi=220)
    fig.savefig(out / "experiment_42_outer_feature_discovery.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    volume, header = read_nrrd(args.volume)
    samples = pd.read_csv(args.samples)
    centers = pd.read_csv(args.centers)
    maps = surface_maps(volume, args.surface_threshold)
    features, surface_columns, shell_columns = extract_features(
        volume, maps, samples, centers
    )
    if features[surface_columns + shell_columns].isna().any().any():
        missing = features[surface_columns + shell_columns].isna().sum()
        raise RuntimeError(f"Missing extracted features:\n{missing[missing > 0]}")
    features.to_csv(args.out / "experiment_42_feature_table.csv", index=False)

    families = {
        "current_geometry": CURRENT,
        "surface_shape": CURRENT + surface_columns,
        "outer_shell": CURRENT + shell_columns,
        "all_outer": CURRENT + surface_columns + shell_columns,
    }
    predictions_all = []
    facets_all = []
    tuning_all = []
    coefficients_all = []
    for name, columns in families.items():
        pred, facets, tuning, coefficients = run_family(features, columns, name)
        predictions_all.append(pred)
        facets_all.append(facets)
        tuning_all.append(tuning)
        coefficients_all.append(coefficients)
    predictions = pd.concat(predictions_all, ignore_index=True)
    facets = pd.concat(facets_all, ignore_index=True)
    tuning = pd.concat(tuning_all, ignore_index=True)
    coefficients = pd.concat(coefficients_all, ignore_index=True)

    fold_rows = []
    summary_rows = []
    for family, f in facets.groupby("family"):
        block = f.groupby("cv_block").agg(
            spatial_MAE_um=("spatial_MAE_um", "median"),
            augmented_MAE_um=("augmented_MAE_um", "median"),
            improvement_um=("improvement_um", "median"),
            facets=("facet_id", "size"),
        ).reset_index()
        block["family"] = family
        fold_rows.append(block)
        differences = block["improvement_um"].to_numpy()
        summary_rows.append({
            "family": family,
            "spatial_median_MAE_um": float(f["spatial_MAE_um"].median()),
            "augmented_median_MAE_um": float(f["augmented_MAE_um"].median()),
            "median_paired_improvement_um": float(
                f["improvement_um"].median()
            ),
            "blocks_improved": int(np.sum(differences > 0)),
            "block_sign_flip_p": exact_sign_flip_p(differences),
        })
    folds = pd.concat(fold_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        "augmented_median_MAE_um"
    )

    predictions.to_csv(args.out / "experiment_42_predictions.csv", index=False)
    facets.to_csv(args.out / "experiment_42_facet_MAE.csv", index=False)
    folds.to_csv(args.out / "experiment_42_fold_MAE.csv", index=False)
    tuning.to_csv(args.out / "experiment_42_nested_tuning.csv", index=False)
    coefficients.to_csv(args.out / "experiment_42_coefficients.csv", index=False)
    summary.to_csv(args.out / "experiment_42_model_summary.csv", index=False)
    make_figure(summary, folds, coefficients, args.out)

    best = summary.iloc[0]
    surface = summary.loc[summary["family"] == "surface_shape"].iloc[0]
    shell = summary.loc[summary["family"] == "outer_shell"].iloc[0]
    if surface["blocks_improved"] >= 4 and surface["median_paired_improvement_um"] > 0:
        verdict = (
            "The surface-only family produced a candidate geometric signal, "
            "but it remains exploratory within one specimen and requires a "
            "frozen independent test."
        )
    else:
        verdict = (
            "No stable surface-only geometric signal was discovered beyond "
            "the specimen-level spatial field."
        )
    report = f"""# Experiment 42 — outer-feature discovery

## Verdict

{verdict}

## Design

The input, 74 quality-controlled facets, 9,840 internal-edge samples and five
contiguous spatial blocks are unchanged from Experiment 40.  New predictors
were extracted without using voxels in the 9--19 voxel internal-edge search
window.  The spatial model was fit first in every training fold; Ridge models
then attempted to predict only its residuals.  Spatial degree, spatial
regularisation and residual-model regularisation were selected inside the
training data.

The **surface_shape** family uses the outer isosurface only.  The
**outer_shell** family uses intensities no deeper than 6 voxels (22.2 µm) and
must be interpreted as a preserved outer-layer imaging cue rather than pure
geometry.

## Main numbers

- nested spatial-only median facet MAE: **{surface['spatial_median_MAE_um']:.2f} µm**
- spatial + surface-shape MAE: **{surface['augmented_median_MAE_um']:.2f} µm**
- surface-shape median paired improvement: **{surface['median_paired_improvement_um']:.2f} µm**
- surface-shape blocks improved: **{int(surface['blocks_improved'])}/5**
- surface-shape exact block sign-flip p: **{surface['block_sign_flip_p']:.4f}**
- spatial + outer-shell MAE: **{shell['augmented_median_MAE_um']:.2f} µm**
- best exploratory family: **{best['family']}**, **{best['augmented_median_MAE_um']:.2f} µm**

## Interpretation rule

A useful geometric feature family should improve the spatial-only model in at
least four of five held-out blocks and should then be frozen before testing an
independent fossil.  Results from this single Asaphus specimen are feature
hypotheses, not biological replication.  A lower error from outer-shell
intensity does not establish that external shape determines the hidden
boundary.

## Provenance

- volume: `{args.volume}`
- volume SHA-256: `{sha256(args.volume)}`
- NRRD sizes: `{header.get('sizes')}`
- NRRD directions: `{header.get('space directions')}`
- feature families were declared in this script before outcome inspection
"""
    (args.out / "EXPERIMENT_42_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "volume": str(args.volume),
        "volume_sha256": sha256(args.volume),
        "samples": str(args.samples),
        "samples_sha256": sha256(args.samples),
        "centers": str(args.centers),
        "centers_sha256": sha256(args.centers),
        "spacing_um": SPACING_UM,
        "surface_threshold": args.surface_threshold,
        "surface_feature_count": len(surface_columns),
        "outer_shell_feature_count": len(shell_columns),
        "rows": len(features),
        "facets": int(features["facet_id"].nunique()),
    }
    (args.out / "experiment_42_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

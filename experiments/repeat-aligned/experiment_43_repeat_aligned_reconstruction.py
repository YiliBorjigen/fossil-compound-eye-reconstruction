#!/usr/bin/env python3
"""Experiment 43: repeat-aligned reconstruction from weak internal CT signal.

This experiment does not claim that the outer surface predicts an invisible
inner lens.  It tests two narrower questions:

1. Does robust alignment of repeated facets reveal a shared internal CT edge?
2. Can a shared template plus the visible part of an edge fill deliberately
   masked regions better than local interpolation alone?

The five spatial blocks and 74 quality-controlled edge maps are inherited from
Experiment 40.  The intensity template uses all 116 threshold-stable facets.
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
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SPACING_UM = 3.7
WITHIN = ["un", "vn", "rn", "rn2", "un2", "vn2", "uv"]
RBF_SIGMA = 0.25
RBF_ALPHA = 10.0
CALIBRATION_ALPHA = 0.1
BOOTSTRAPS = 500
SEED = 20260827


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--volume", required=True, type=Path)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--centers", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--surface-threshold", type=float, default=50.0)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_nrrd(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    with path.open("rb") as f:
        lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in NRRD header")
            if line in (b"\n", b"\r\n"):
                body_offset = f.tell()
                break
            lines.append(line.decode("ascii").rstrip())
    header: dict[str, str] = {}
    for line in lines:
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


def surface_frame_maps(volume: np.ndarray, threshold: float):
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
    smooth = ndi.gaussian_filter(filled, sigma=2.0).astype(np.float32)
    fy, fz = np.gradient(smooth)
    return filled, fy.astype(np.float32), fz.astype(np.float32)


def local_frame(point: np.ndarray, fy: np.ndarray, fz: np.ndarray):
    x, y, z = point
    yi, zi = int(round(y)), int(round(z))
    normal = np.array([1.0, -fy[yi, zi], -fz[yi, zi]], float)
    normal /= np.linalg.norm(normal)
    t1 = np.array([fy[yi, zi], 1.0, 0.0], float)
    t1 -= normal * np.dot(t1, normal)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)
    return normal, t1, t2


def assign_all_blocks(centers: pd.DataFrame, samples: pd.DataFrame) -> np.ndarray:
    known = samples[["facet_id", "cv_block"]].drop_duplicates()
    merged = known.merge(centers[["facet_id", "y_vox", "z_vox"]],
                         on="facet_id")
    centroids = merged.groupby("cv_block")[["y_vox", "z_vox"]].mean()
    block_ids = centroids.index.to_numpy(int)
    tree = cKDTree(centroids.to_numpy(float))
    _, nearest = tree.query(centers[["y_vox", "z_vox"]].to_numpy(float))
    return block_ids[nearest]


def extract_aligned_intensity(
    volume: np.ndarray,
    centers: pd.DataFrame,
    fy: np.ndarray,
    fz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = np.linspace(-0.8, 0.8, 17)
    dn = np.linspace(-0.2, 1.4, 41)
    un, vn, dd = np.meshgrid(uv, uv, dn, indexing="ij")
    aligned = np.empty((len(centers), *un.shape), dtype=np.float32)
    for row_i, row in centers.iterrows():
        point = row[["x_vox", "y_vox", "z_vox"]].to_numpy(float)
        pitch = float(row["nearest_neighbor_um"] / SPACING_UM)
        normal, t1, t2 = local_frame(point, fy, fz)
        xyz = (
            point[:, None, None, None]
            + t1[:, None, None, None] * (un * pitch / 2)[None]
            + t2[:, None, None, None] * (vn * pitch / 2)[None]
            - normal[:, None, None, None] * (dd * pitch)[None]
        )
        aligned[row_i] = ndi.map_coordinates(
            volume, xyz.reshape(3, -1), order=1, mode="nearest"
        ).reshape(un.shape)
    return aligned, uv, dn


def template_profile(template: np.ndarray, uv: np.ndarray, dn: np.ndarray):
    uu, vv = np.meshgrid(uv, uv, indexing="ij")
    central = np.hypot(uu, vv) <= 0.25
    profile = template[central].mean(axis=0)
    smooth = ndi.gaussian_filter1d(profile, sigma=1.0)
    gradient = np.gradient(smooth, dn)
    search = np.where((dn >= 0.6) & (dn <= 1.3))[0]
    edge_i = search[np.argmin(gradient[search])]
    return profile, gradient, float(dn[edge_i])


def intensity_template_analysis(
    aligned: np.ndarray,
    centers: pd.DataFrame,
    blocks: np.ndarray,
    uv: np.ndarray,
    dn: np.ndarray,
):
    template = np.median(aligned, axis=0)
    profile, gradient, edge_norm = template_profile(template, uv, dn)
    median_pitch_vox = float(np.median(
        centers["nearest_neighbor_um"].to_numpy() / SPACING_UM
    ))
    rng = np.random.default_rng(SEED)
    bootstrap_edges = []
    for _ in range(BOOTSTRAPS):
        ids = rng.integers(0, len(aligned), len(aligned))
        boot = np.median(aligned[ids], axis=0)
        bootstrap_edges.append(template_profile(boot, uv, dn)[2])
    bootstrap_edges = np.asarray(bootstrap_edges)
    block_rows = []
    for block in sorted(np.unique(blocks)):
        train_template = np.median(aligned[blocks != block], axis=0)
        _, _, edge = template_profile(train_template, uv, dn)
        block_rows.append({
            "heldout_block": int(block),
            "training_facets": int(np.sum(blocks != block)),
            "edge_depth_norm_pitch": edge,
            "edge_depth_um_at_global_median_pitch": (
                edge * median_pitch_vox * SPACING_UM
            ),
        })
    metrics = {
        "template_edge_norm_pitch": edge_norm,
        "template_edge_um_at_global_median_pitch": (
            edge_norm * median_pitch_vox * SPACING_UM
        ),
        "bootstrap_edge_norm_ci95": [
            float(np.percentile(bootstrap_edges, 2.5)),
            float(np.percentile(bootstrap_edges, 97.5)),
        ],
        "bootstrap_edge_um_ci95": [
            float(np.percentile(bootstrap_edges, 2.5)
                  * median_pitch_vox * SPACING_UM),
            float(np.percentile(bootstrap_edges, 97.5)
                  * median_pitch_vox * SPACING_UM),
        ],
        "median_pitch_vox_all_facets": median_pitch_vox,
    }
    return template, profile, gradient, bootstrap_edges, pd.DataFrame(block_rows), metrics


def rbf_centers() -> np.ndarray:
    grid = np.linspace(-0.8, 0.8, 7)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    centers = np.c_[u.ravel(), v.ravel()]
    return centers[np.hypot(centers[:, 0], centers[:, 1]) <= 1.05]


RBF_CENTERS = rbf_centers()


def gaussian_rbf(un: np.ndarray, vn: np.ndarray, sigma: float) -> np.ndarray:
    points = np.c_[un, vn]
    d2 = ((points[:, None, :] - RBF_CENTERS[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2 * sigma * sigma))


def radial_x(frame: pd.DataFrame) -> np.ndarray:
    return frame[WITHIN].to_numpy(float)


def canonical_x(frame: pd.DataFrame, sigma: float) -> np.ndarray:
    return np.c_[radial_x(frame), gaussian_rbf(
        frame["un"].to_numpy(float), frame["vn"].to_numpy(float), sigma
    )]


def model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def facet_mae(frame: pd.DataFrame, prediction_norm: np.ndarray) -> pd.DataFrame:
    error_um = np.abs(
        prediction_norm - frame["target_norm"].to_numpy()
    ) * frame["pitch"].to_numpy() * SPACING_UM
    temp = pd.DataFrame({
        "facet_id": frame["facet_id"].to_numpy(),
        "cv_block": frame["cv_block"].to_numpy(),
        "error_um": error_um,
    })
    return temp.groupby(["facet_id", "cv_block"], as_index=False)[
        "error_um"
    ].mean().rename(columns={"error_um": "facet_MAE_um"})


def score(frame: pd.DataFrame, prediction_norm: np.ndarray) -> float:
    return float(facet_mae(frame, prediction_norm)["facet_MAE_um"].median())


def tune_model(train: pd.DataFrame, canonical: bool):
    y = train["target_norm"].to_numpy()
    groups = train["cv_block"].to_numpy()
    cv = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    candidates = []
    sigmas = (0.15, 0.25, 0.40) if canonical else (None,)
    for sigma, alpha in itertools.product(
        sigmas, (0.1, 1.0, 10.0, 100.0, 1000.0)
    ):
        prediction = np.full(len(train), np.nan)
        for tr, va in cv.split(train, y, groups):
            xtr = (canonical_x(train.iloc[tr], sigma) if canonical
                   else radial_x(train.iloc[tr]))
            xva = (canonical_x(train.iloc[va], sigma) if canonical
                   else radial_x(train.iloc[va]))
            m = model(alpha)
            m.fit(xtr, y[tr])
            prediction[va] = m.predict(xva)
        candidates.append((score(train, prediction), sigma, alpha))
    return min(candidates, key=lambda x: x[0])


def full_facet_transfer(samples: pd.DataFrame):
    y = samples["target_norm"].to_numpy()
    groups = samples["cv_block"].to_numpy()
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    radial_prediction = np.full(len(samples), np.nan)
    canonical_prediction = np.full(len(samples), np.nan)
    tuning_rows = []
    for tr, te in cv.split(samples, y, groups):
        train, test = samples.iloc[tr], samples.iloc[te]
        radial_inner, _, radial_alpha = tune_model(train, canonical=False)
        canonical_inner, sigma, canonical_alpha = tune_model(train, canonical=True)
        rm = model(radial_alpha)
        rm.fit(radial_x(train), y[tr])
        radial_prediction[te] = rm.predict(radial_x(test))
        cm = model(canonical_alpha)
        cm.fit(canonical_x(train, sigma), y[tr])
        canonical_prediction[te] = cm.predict(canonical_x(test, sigma))
        tuning_rows.append({
            "heldout_block": int(np.unique(groups[te])[0]),
            "radial_inner_MAE_um": radial_inner,
            "radial_alpha": radial_alpha,
            "canonical_inner_MAE_um": canonical_inner,
            "canonical_sigma": sigma,
            "canonical_alpha": canonical_alpha,
        })
    predictions = samples[[
        "facet_id", "cv_block", "un", "vn", "rn", "pitch", "target_norm"
    ]].copy()
    predictions["radial_prediction_norm"] = radial_prediction
    predictions["canonical_prediction_norm"] = canonical_prediction
    radial_facets = facet_mae(samples, radial_prediction).rename(
        columns={"facet_MAE_um": "radial_MAE_um"}
    )
    canonical_facets = facet_mae(samples, canonical_prediction).rename(
        columns={"facet_MAE_um": "canonical_MAE_um"}
    )
    facets = radial_facets.merge(canonical_facets,
                                 on=["facet_id", "cv_block"])
    facets["canonical_improvement_um"] = (
        facets["radial_MAE_um"] - facets["canonical_MAE_um"]
    )
    return predictions, facets, pd.DataFrame(tuning_rows)


def mask_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "central_disk":
        return frame["rn"].to_numpy() <= 0.35
    if name == "half_plane":
        return frame["un"].to_numpy() >= 0
    if name == "quadrant":
        return ((frame["un"].to_numpy() >= 0)
                & (frame["vn"].to_numpy() >= 0))
    if name == "diagonal_band":
        return np.abs(
            frame["un"].to_numpy() - frame["vn"].to_numpy()
        ) <= 0.20
    if name == "offset_disk_confirmation":
        # Frozen after the guarded rule was defined; this mask was not used to
        # motivate or tune that rule.
        u = frame["un"].to_numpy()
        v = frame["vn"].to_numpy()
        return (u - 0.30) ** 2 + (v + 0.20) ** 2 <= 0.30 ** 2
    raise ValueError(name)


def polynomial_x(frame: pd.DataFrame) -> np.ndarray:
    u = frame["un"].to_numpy(float)
    v = frame["vn"].to_numpy(float)
    return np.c_[u, v, u * u, v * v, u * v]


def gap_fill_validation(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    methods = (
        "canonical_uncalibrated", "local_quadratic", "local_rbf",
        "template_calibrated", "hierarchical_template_residual",
        "hierarchical_guarded",
    )
    masks = (
        "central_disk", "half_plane", "quadrant", "diagonal_band",
        "offset_disk_confirmation",
    )
    for facet_id, facet in predictions.groupby("facet_id"):
        facet = facet.copy().reset_index(drop=True)
        y = facet["target_norm"].to_numpy()
        canonical = facet["canonical_prediction_norm"].to_numpy()
        pitch = float(facet["pitch"].iloc[0])
        block = int(facet["cv_block"].iloc[0])
        for mask_name in masks:
            masked = mask_array(facet, mask_name)
            visible = ~masked
            if masked.sum() < 12 or visible.sum() < 20:
                continue

            # Local quadratic baseline.
            qm = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
            qm.fit(polynomial_x(facet)[visible], y[visible])
            quadratic = qm.predict(polynomial_x(facet))

            # Flexible local interpolation baseline with declared settings.
            local_features = np.c_[
                polynomial_x(facet),
                gaussian_rbf(facet["un"].to_numpy(),
                             facet["vn"].to_numpy(), RBF_SIGMA),
            ]
            lm = model(RBF_ALPHA)
            lm.fit(local_features[visible], y[visible])
            local_rbf = lm.predict(local_features)

            # Calibrate the cross-block template using only visible edge data.
            calibration = Ridge(alpha=CALIBRATION_ALPHA)
            calibration.fit(canonical[visible, None], y[visible])
            calibrated = calibration.predict(canonical[:, None])

            residual = y[visible] - calibrated[visible]
            hm = model(RBF_ALPHA)
            hm.fit(local_features[visible], residual)
            raw_correction = hm.predict(local_features)
            hierarchical = calibrated + raw_correction

            # Post-hoc safety refinement after the unguarded residual model
            # proved unstable for extrapolative masks. Corrections decay with
            # distance from visible data and are bounded using visible
            # residuals only. This variant remains exploratory until tested on
            # independent data.
            visible_uv = facet.loc[visible, ["un", "vn"]].to_numpy(float)
            all_uv = facet[["un", "vn"]].to_numpy(float)
            nearest_visible = cKDTree(visible_uv).query(all_uv, k=1)[0]
            support_weight = np.exp(
                -0.5 * (nearest_visible / RBF_SIGMA) ** 2
            )
            residual_median = float(np.median(residual))
            residual_mad = float(np.median(np.abs(residual - residual_median)))
            radius = max(3.0 * 1.4826 * residual_mad, 0.05)
            bounded_correction = np.clip(
                raw_correction,
                residual_median - radius,
                residual_median + radius,
            )
            guarded = calibrated + support_weight * bounded_correction

            predictions_by_method = {
                "canonical_uncalibrated": canonical,
                "local_quadratic": quadratic,
                "local_rbf": local_rbf,
                "template_calibrated": calibrated,
                "hierarchical_template_residual": hierarchical,
                "hierarchical_guarded": guarded,
            }
            for method in methods:
                error = np.abs(
                    predictions_by_method[method][masked] - y[masked]
                ) * pitch * SPACING_UM
                records.append({
                    "facet_id": int(facet_id), "cv_block": block,
                    "mask": mask_name, "method": method,
                    "masked_points": int(masked.sum()),
                    "visible_points": int(visible.sum()),
                    "masked_MAE_um": float(np.mean(error)),
                })
    return pd.DataFrame(records)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-12))


def summarize_gap(gap: pd.DataFrame):
    summary = gap.groupby(["mask", "method"], as_index=False).agg(
        median_facet_MAE_um=("masked_MAE_um", "median"),
        p90_facet_MAE_um=("masked_MAE_um", lambda x: np.percentile(x, 90)),
        facets=("facet_id", "nunique"),
    )
    comparisons = []
    pivot = gap.pivot_table(index=["facet_id", "cv_block", "mask"],
                            columns="method", values="masked_MAE_um")
    for mask in sorted(gap["mask"].unique()):
        m = pivot.xs(mask, level="mask")
        for baseline in ("local_quadratic", "local_rbf", "template_calibrated"):
            advantage = m[baseline] - m["hierarchical_guarded"]
            blocks = advantage.groupby(level="cv_block").median()
            comparisons.append({
                "mask": mask, "baseline": baseline,
                "median_paired_hierarchical_advantage_um": float(
                    advantage.median()
                ),
                "blocks_favouring_hierarchical": int(np.sum(blocks > 0)),
                "exact_block_sign_flip_p": exact_sign_flip_p(blocks.to_numpy()),
            })
    return summary, pd.DataFrame(comparisons)


def make_intensity_figure(
    template: np.ndarray,
    uv: np.ndarray,
    dn: np.ndarray,
    profile: np.ndarray,
    gradient: np.ndarray,
    bootstrap_edges: np.ndarray,
    block_edges: pd.DataFrame,
    median_pitch_vox: float,
    out: Path,
):
    depth_um = dn * median_pitch_vox * SPACING_UM
    edge_um = np.median(bootstrap_edges) * median_pitch_vox * SPACING_UM
    mid = len(uv) // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    im = axes[0].imshow(
        template[:, mid, :].T, origin="lower", aspect="auto",
        extent=[uv.min(), uv.max(), depth_um.min(), depth_um.max()],
        cmap="gray",
    )
    axes[0].axhline(edge_um, color="#E11D48", lw=1.5)
    axes[0].set_xlabel("Normalised tangential position")
    axes[0].set_ylabel("Depth from surface (µm)")
    axes[0].set_title("Median of 116 aligned facets")
    fig.colorbar(im, ax=axes[0], label="CT intensity")

    ax2 = axes[1].twinx()
    axes[1].plot(depth_um, profile, color="#2563EB", label="intensity")
    ax2.plot(depth_um, -gradient, color="#D97706", label="−gradient")
    axes[1].axvline(edge_um, color="#E11D48", lw=1.5)
    axes[1].set_xlabel("Depth from surface (µm)")
    axes[1].set_ylabel("Mean CT intensity", color="#2563EB")
    ax2.set_ylabel("Negative gradient", color="#D97706")
    axes[1].set_title("Central repeat-aligned profile")

    values = block_edges["edge_depth_um_at_global_median_pitch"].to_numpy()
    axes[2].scatter(np.arange(len(values)), values, s=55, color="#0F766E")
    axes[2].axhspan(
        np.percentile(bootstrap_edges, 2.5) * median_pitch_vox * SPACING_UM,
        np.percentile(bootstrap_edges, 97.5) * median_pitch_vox * SPACING_UM,
        color="#94A3B8", alpha=0.35, label="bootstrap 95% interval",
    )
    axes[2].set_xticks(np.arange(len(values)),
                       block_edges["heldout_block"].astype(str))
    axes[2].set_xlabel("Held-out spatial block")
    axes[2].set_ylabel("Template edge depth (µm)")
    axes[2].set_title("Template stability")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "experiment_43_repeat_aligned_intensity.png", dpi=220)
    fig.savefig(out / "experiment_43_repeat_aligned_intensity.pdf")
    plt.close(fig)


def example_maps(predictions: pd.DataFrame, gap: pd.DataFrame):
    central = gap[gap["mask"] == "central_disk"]
    errors = central.pivot_table(index="facet_id", columns="method",
                                 values="masked_MAE_um")
    advantage = errors["local_quadratic"] - errors["hierarchical_guarded"]
    facet_id = int((advantage - advantage.median()).abs().idxmin())
    facet = predictions[predictions["facet_id"] == facet_id].copy().reset_index(drop=True)
    y = facet["target_norm"].to_numpy()
    canonical = facet["canonical_prediction_norm"].to_numpy()
    masked = mask_array(facet, "central_disk")
    visible = ~masked
    local_features = np.c_[
        polynomial_x(facet),
        gaussian_rbf(facet["un"].to_numpy(), facet["vn"].to_numpy(), RBF_SIGMA),
    ]
    calibration = Ridge(alpha=CALIBRATION_ALPHA).fit(
        canonical[visible, None], y[visible]
    )
    calibrated = calibration.predict(canonical[:, None])
    residual = y[visible] - calibrated[visible]
    hm = model(RBF_ALPHA).fit(local_features[visible], residual)
    raw_correction = hm.predict(local_features)
    visible_uv = facet.loc[visible, ["un", "vn"]].to_numpy(float)
    all_uv = facet[["un", "vn"]].to_numpy(float)
    nearest_visible = cKDTree(visible_uv).query(all_uv, k=1)[0]
    support_weight = np.exp(-0.5 * (nearest_visible / RBF_SIGMA) ** 2)
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    radius = max(3.0 * 1.4826 * residual_mad, 0.05)
    bounded = np.clip(raw_correction, residual_median - radius,
                      residual_median + radius)
    hierarchical = calibrated + support_weight * bounded
    qm = make_pipeline(StandardScaler(), Ridge(alpha=0.1)).fit(
        polynomial_x(facet)[visible], y[visible]
    )
    quadratic = qm.predict(polynomial_x(facet))
    return facet_id, facet, masked, quadratic, hierarchical


def map_on_grid(facet: pd.DataFrame, values: np.ndarray):
    grid = np.linspace(-0.85, 0.85, 100)
    gu, gv = np.meshgrid(grid, grid, indexing="xy")
    z = griddata(
        facet[["un", "vn"]].to_numpy(), values, (gu, gv), method="linear"
    )
    z[np.hypot(gu, gv) > 0.9] = np.nan
    return grid, z


def make_reconstruction_figure(
    predictions: pd.DataFrame,
    facets: pd.DataFrame,
    gap: pd.DataFrame,
    gap_summary: pd.DataFrame,
    out: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    axes[0, 0].bar(
        [0, 1],
        [facets["radial_MAE_um"].median(), facets["canonical_MAE_um"].median()],
        color=["#94A3B8", "#2563EB"],
    )
    axes[0, 0].set_xticks([0, 1], ["radial shape", "repeat template"])
    axes[0, 0].set_ylabel("Median facet MAE (µm)")
    axes[0, 0].set_title("Full-facet transfer")

    methods = ["local_quadratic", "template_calibrated",
               "hierarchical_guarded"]
    masks = ["central_disk", "half_plane", "quadrant", "diagonal_band",
             "offset_disk_confirmation"]
    x = np.arange(len(masks))
    width = 0.25
    colors = ["#94A3B8", "#0EA5E9", "#2563EB"]
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = []
        for mask in masks:
            row = gap_summary[(gap_summary["mask"] == mask)
                              & (gap_summary["method"] == method)].iloc[0]
            values.append(row["median_facet_MAE_um"])
        axes[0, 1].bar(x + (i - 1) * width, values, width,
                       label=method.replace("_", " "), color=color)
    axes[0, 1].set_xticks(x, [m.replace("_", " ") for m in masks],
                          rotation=20, ha="right")
    axes[0, 1].set_ylabel("Masked-region MAE (µm)")
    axes[0, 1].set_title("Deliberate gap filling")
    axes[0, 1].legend(frameon=False, fontsize=8)

    comparison = gap.pivot_table(
        index=["facet_id", "mask"], columns="method", values="masked_MAE_um"
    ).reset_index()
    comparison["advantage"] = (
        comparison["local_quadratic"]
        - comparison["hierarchical_guarded"]
    )
    for i, mask in enumerate(masks):
        vals = comparison.loc[comparison["mask"] == mask, "advantage"]
        axes[0, 2].scatter(np.full(len(vals), i), vals, s=18, alpha=0.65)
    axes[0, 2].axhline(0, color="black", lw=1)
    axes[0, 2].set_xticks(np.arange(len(masks)),
                          [m.replace("_", " ") for m in masks],
                          rotation=20, ha="right")
    axes[0, 2].set_ylabel("Quadratic − hierarchical MAE (µm)")
    axes[0, 2].set_title("Facet-level hierarchical advantage")

    facet_id, facet, masked, quadratic, hierarchical = example_maps(
        predictions, gap
    )
    observed = facet["target_norm"].to_numpy() * facet["pitch"].to_numpy() * SPACING_UM
    q_um = quadratic * facet["pitch"].to_numpy() * SPACING_UM
    h_um = hierarchical * facet["pitch"].to_numpy() * SPACING_UM
    panels = [
        (np.where(masked, np.nan, observed), "Visible data after masking"),
        (np.where(masked, q_um, np.nan), "Quadratic reconstruction"),
        (np.where(masked, h_um, np.nan), "Hierarchical reconstruction"),
    ]
    finite = observed[np.isfinite(observed)]
    vmin, vmax = np.percentile(finite, [5, 95])
    for ax, (values, title) in zip(axes[1], panels):
        grid, image = map_on_grid(facet, values)
        im = ax.imshow(image, origin="lower", extent=[grid.min(), grid.max(),
                       grid.min(), grid.max()], vmin=vmin, vmax=vmax,
                       cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("u / facet radius")
        ax.set_ylabel("v / facet radius")
    fig.colorbar(im, ax=axes[1].tolist(), label="Boundary depth (µm)",
                 shrink=0.8)
    fig.suptitle(f"Experiment 43; representative facet {facet_id}")
    fig.savefig(out / "experiment_43_masked_reconstruction.png", dpi=220,
                bbox_inches="tight")
    fig.savefig(out / "experiment_43_masked_reconstruction.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    volume, header = read_nrrd(args.volume)
    samples = pd.read_csv(args.samples)
    centers = pd.read_csv(args.centers)
    _, fy, fz = surface_frame_maps(volume, args.surface_threshold)
    aligned, uv, dn = extract_aligned_intensity(volume, centers, fy, fz)
    all_blocks = assign_all_blocks(centers, samples)
    (template, profile, gradient, bootstrap_edges, block_edges,
     intensity_metrics) = intensity_template_analysis(
        aligned, centers, all_blocks, uv, dn
    )
    np.savez_compressed(
        args.out / "experiment_43_repeat_aligned_template.npz",
        median_intensity_template=template, uv_normalized=uv,
        depth_normalized_by_pitch=dn, central_profile=profile,
        negative_gradient=-gradient, bootstrap_edge_norm=bootstrap_edges,
    )
    block_edges.to_csv(args.out / "experiment_43_template_block_stability.csv",
                       index=False)
    (args.out / "experiment_43_intensity_metrics.json").write_text(
        json.dumps(intensity_metrics, indent=2), encoding="utf-8"
    )
    make_intensity_figure(
        template, uv, dn, profile, gradient, bootstrap_edges, block_edges,
        intensity_metrics["median_pitch_vox_all_facets"], args.out,
    )

    predictions, facets, tuning = full_facet_transfer(samples)
    predictions.to_csv(args.out / "experiment_43_full_transfer_predictions.csv",
                       index=False)
    facets.to_csv(args.out / "experiment_43_full_transfer_facet_MAE.csv",
                  index=False)
    tuning.to_csv(args.out / "experiment_43_nested_tuning.csv", index=False)
    gap = gap_fill_validation(predictions)
    gap.to_csv(args.out / "experiment_43_gap_fill_facet_MAE.csv", index=False)
    gap_summary, comparisons = summarize_gap(gap)
    gap_summary.to_csv(args.out / "experiment_43_gap_fill_summary.csv",
                       index=False)
    comparisons.to_csv(args.out / "experiment_43_gap_fill_comparisons.csv",
                       index=False)
    make_reconstruction_figure(predictions, facets, gap, gap_summary, args.out)

    canonical_gain = float(facets["canonical_improvement_um"].median())
    block_gain = facets.groupby("cv_block")[
        "canonical_improvement_um"
    ].median().to_numpy()
    full_metrics = {
        "radial_median_facet_MAE_um": float(facets["radial_MAE_um"].median()),
        "canonical_median_facet_MAE_um": float(
            facets["canonical_MAE_um"].median()
        ),
        "canonical_median_paired_advantage_um": canonical_gain,
        "blocks_favouring_canonical": int(np.sum(block_gain > 0)),
        "exact_block_sign_flip_p": exact_sign_flip_p(block_gain),
    }
    (args.out / "experiment_43_full_transfer_metrics.json").write_text(
        json.dumps(full_metrics, indent=2), encoding="utf-8"
    )

    central_comparison = comparisons[
        (comparisons["mask"] == "central_disk")
        & (comparisons["baseline"] == "local_quadratic")
    ].iloc[0]
    half_comparison = comparisons[
        (comparisons["mask"] == "half_plane")
        & (comparisons["baseline"] == "local_quadratic")
    ].iloc[0]
    confirmation_comparison = comparisons[
        (comparisons["mask"] == "offset_disk_confirmation")
        & (comparisons["baseline"] == "local_quadratic")
    ].iloc[0]
    confirmation_rbf = comparisons[
        (comparisons["mask"] == "offset_disk_confirmation")
        & (comparisons["baseline"] == "local_rbf")
    ].iloc[0]
    confirmation_template = comparisons[
        (comparisons["mask"] == "offset_disk_confirmation")
        & (comparisons["baseline"] == "template_calibrated")
    ].iloc[0]
    unguarded_half = gap_summary[
        (gap_summary["mask"] == "half_plane")
        & (gap_summary["method"] == "hierarchical_template_residual")
    ].iloc[0]
    supported_gap = (
        confirmation_comparison["blocks_favouring_hierarchical"] >= 4
        and confirmation_comparison[
            "median_paired_hierarchical_advantage_um"
        ] > 0
    )
    if supported_gap:
        verdict = (
            "Repeat alignment supports a shared internal CT template, and the "
            "guarded hierarchical method passed the frozen off-centre-mask "
            "check within this specimen."
        )
    else:
        verdict = (
            "Repeat alignment supports a shared internal CT template, but the "
            "hierarchical method does not reliably outperform local "
            "interpolation when filling missing regions."
        )
    report = f"""# Experiment 43 — repeat-aligned internal reconstruction

## Verdict

{verdict}

## What was tested

All 116 threshold-stable facets were aligned by outer-surface normal and scaled
by facet spacing before their raw CT intensities were combined.  The 74
quality-controlled internal-edge maps were then used in five spatial-block
holdouts.  A canonical cross-block template was tested both on completely
held-out facets and after contiguous regions of the observed boundary were
deliberately hidden.

This experiment uses repeated internal CT evidence.  It is not a test of
reconstructing a completely invisible lens from its outer surface.

## Repeat-aligned intensity result

- shared template edge: **{intensity_metrics['template_edge_um_at_global_median_pitch']:.2f} µm**
- facet-bootstrap 95% interval: **{intensity_metrics['bootstrap_edge_um_ci95'][0]:.2f}–{intensity_metrics['bootstrap_edge_um_ci95'][1]:.2f} µm**
- leave-one-spatial-block template depths: **{block_edges['edge_depth_um_at_global_median_pitch'].min():.2f}–{block_edges['edge_depth_um_at_global_median_pitch'].max():.2f} µm**

The repeated CT signal therefore survives both facet bootstrapping and removal
of complete spatial regions, although its 43.46–61.75 µm interval is broad.
Together with the previously measured facet-versus-inter-facet AUC of 0.747,
this supports a shared image-domain boundary, not its final anatomical identity.

## Full-facet template transfer

- radial shape model: **{full_metrics['radial_median_facet_MAE_um']:.2f} µm**
- nonparametric repeat template: **{full_metrics['canonical_median_facet_MAE_um']:.2f} µm**
- median paired template advantage: **{full_metrics['canonical_median_paired_advantage_um']:.2f} µm**
- blocks favouring template: **{full_metrics['blocks_favouring_canonical']}/5**
- exact block sign-flip p: **{full_metrics['exact_block_sign_flip_p']:.4f}**

This transfer does not use absolute specimen coordinates.  It tests whether a
shared normalised optical-unit shape transfers between spatial regions.

## Deliberate gap filling

See `experiment_43_gap_fill_summary.csv` for every method and mask.  The primary
comparison is the hierarchical template against a quadratic surface fitted
only to the visible part of the same facet.

- central disk: median hierarchical advantage **{central_comparison['median_paired_hierarchical_advantage_um']:.2f} µm**, blocks **{int(central_comparison['blocks_favouring_hierarchical'])}/5**, p **{central_comparison['exact_block_sign_flip_p']:.4f}**
- half plane: median hierarchical advantage **{half_comparison['median_paired_hierarchical_advantage_um']:.2f} µm**, blocks **{int(half_comparison['blocks_favouring_hierarchical'])}/5**, p **{half_comparison['exact_block_sign_flip_p']:.4f}**
- frozen off-centre confirmation mask: median advantage **{confirmation_comparison['median_paired_hierarchical_advantage_um']:.2f} µm**, blocks **{int(confirmation_comparison['blocks_favouring_hierarchical'])}/5**, p **{confirmation_comparison['exact_block_sign_flip_p']:.4f}**

On that confirmation mask, the guarded method did **not** consistently beat
the flexible local RBF interpolator (paired advantage
**{confirmation_rbf['median_paired_hierarchical_advantage_um']:.2f} µm**;
**{int(confirmation_rbf['blocks_favouring_hierarchical'])}/5** blocks), but it
did improve on template calibration alone by
**{confirmation_template['median_paired_hierarchical_advantage_um']:.2f} µm**
across **{int(confirmation_template['blocks_favouring_hierarchical'])}/5**
blocks.

The initial unguarded residual interpolator failed during large extrapolation
(half-plane median MAE **{unguarded_half['median_facet_MAE_um']:.2f} µm**).  That
failure is retained in the output.  The guarded method was introduced
afterwards: corrections are bounded by visible residuals and decay with
distance from visible data.  Its central-, half-plane and diagonal-mask results
are exploratory.  The off-centre mask was defined only after the guard was
frozen and provides a limited confirmation check, but it is still from the same
Asaphus specimen.

## Claim boundary

Supported with limits: a shared, repeat-aligned internal CT boundary.  The
guarded method provides a candidate way to reconstruct locally missing
portions using other facets plus the visible part of the same boundary.

Not supported: anatomical identification as the proximal lens surface,
reconstruction of a wholly absent internal lens from outer geometry, or
generalisation to independent fossil specimens.

## Provenance

- volume: `{args.volume}`
- volume SHA-256: `{sha256(args.volume)}`
- samples SHA-256: `{sha256(args.samples)}`
- centres SHA-256: `{sha256(args.centers)}`
- NRRD sizes: `{header.get('sizes')}`
- NRRD directions: `{header.get('space directions')}`
- declared local RBF sigma: {RBF_SIGMA}
- declared local RBF alpha: {RBF_ALPHA}
- bootstrap resamples: {BOOTSTRAPS}
"""
    (args.out / "EXPERIMENT_43_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Reproducible outer-to-inner fossil lens reconstruction pipeline.

INPUT
-----
A cropped 3D scalar CT volume in NRRD format, with correct voxel spacing.
For the current Asaphus dataset:
    1652a_0000_1_cropped_CORRECT_3p7um.nrrd

WHAT THIS SCRIPT DOES
---------------------
1. Extracts the outer fossil eye surface at sub-voxel precision.
2. Detects repeated convex facet relief.
3. Keeps only facet centers stable across neighboring isovalues.
4. Estimates the first reproducible internal CT edge beneath each facet.
5. Builds inter-facet midpoint controls.
6. Tests extraction stability.
7. Hides complete spatial regions and predicts their internal edge
   from preserved outer geometry + local facet context.
8. Writes the complete sample table, CSV summaries, QC figures and Markdown.

IMPORTANT
---------
The default parameters reproduce the Asaphus analysis and are NOT universal.
For a new specimen, the eye ROI, surface intensity threshold, and internal-edge
depth window must be set from the new CT data before claiming results.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import gzip
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


def read_nrrd(path: Path):
    """Read gzip-compressed uint8 NRRD and return volume + header dict."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading NRRD header.")
            if line in (b"\n", b"\r\n"):
                body_offset = f.tell()
                break
            header_lines.append(line.decode("ascii").rstrip())

    hdr = {}
    for line in header_lines:
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            hdr[k.strip()] = v.strip()

    sizes = tuple(map(int, hdr["sizes"].split()))
    encoding = hdr.get("encoding", "raw").lower()

    with open(path, "rb") as f:
        f.seek(body_offset)
        if encoding in ("gzip", "gz"):
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                raw = gz.read()
        elif encoding == "raw":
            raw = f.read()
        else:
            raise ValueError(f"Unsupported NRRD encoding: {encoding}")

    if hdr.get("type", "").lower() not in {
        "uchar", "unsigned char", "uint8", "uint8_t"
    }:
        raise ValueError(
            "This reproducible version currently expects uint8 CT data. "
            f"NRRD type was: {hdr.get('type')}"
        )

    vol = np.frombuffer(raw, dtype=np.uint8).reshape(sizes, order="F")
    return vol, hdr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Input eye-crop NRRD.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory.")
    p.add_argument("--spacing-um", type=float, default=3.7,
                   help="Isotropic voxel spacing in micrometres.")

    # Asaphus frozen defaults
    p.add_argument("--surface-threshold", type=float, default=50.0)
    p.add_argument("--neighbor-thresholds", type=float, nargs="+",
                   default=[45, 55, 60, 65, 70])

    p.add_argument("--patch-y", type=int, nargs=2, default=[100, 450])
    p.add_argument("--patch-z", type=int, nargs=2, default=[300, 480])

    p.add_argument("--surface-sigma-vox", type=float, default=1.0)
    p.add_argument("--curvature-sigma-vox", type=float, default=15.0)
    p.add_argument("--peak-min-distance-vox", type=int, default=7)
    p.add_argument("--peak-relief-threshold-vox", type=float, default=1.2)
    p.add_argument("--persistence-distance-vox", type=float, default=4.0)
    p.add_argument("--persistence-min-hits", type=int, default=4)

    # Internal edge: current Asaphus frozen window = 9..19 vox
    p.add_argument("--edge-window-vox", type=float, nargs=2, default=[9, 19])
    p.add_argument("--edge-sigma-vox", type=float, default=1.5)
    p.add_argument("--edge-strength-min", type=float, default=1.5)

    # High-quality gate
    p.add_argument("--hq-coverage-min", type=float, default=0.50)
    p.add_argument("--hq-center-strength-min", type=float, default=2.5)
    p.add_argument("--hq-quadratic-rmse-max-vox", type=float, default=2.0)

    p.add_argument("--cv-blocks", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=20.0)

    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vol, hdr = read_nrrd(args.input)
    volf = vol.astype(np.float32)

    # ---------------- outer surface ----------------
    sm = ndi.gaussian_filter1d(
        volf, sigma=args.surface_sigma_vox, axis=0
    )

    def subvoxel_surface(T):
        above = sm >= T
        rev = np.argmax(above[::-1, :, :], axis=0)
        any_above = above.any(axis=0)
        x0 = (sm.shape[0] - 1 - rev).astype(np.int32)

        surf = np.full(any_above.shape, np.nan, dtype=np.float32)
        ys, zs = np.where(any_above)
        xs = x0[ys, zs]
        valid = xs < sm.shape[0] - 1

        yv, zv, xv = ys[valid], zs[valid], xs[valid]
        v0, v1 = sm[xv, yv, zv], sm[xv + 1, yv, zv]
        denom = v0 - v1
        frac = np.divide(
            v0 - T,
            denom,
            out=np.zeros_like(v0),
            where=np.abs(denom) > 1e-6,
        )
        surf[yv, zv] = xv + frac
        surf[ys[~valid], zs[~valid]] = xs[~valid]
        return surf

    def relief_and_points(T):
        surf = subvoxel_surface(T)
        valid = np.isfinite(surf)

        ind = ndi.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = surf[tuple(ind)]

        relief = filled - ndi.gaussian_filter(
            filled, sigma=args.curvature_sigma_vox
        )

        y0, y1 = args.patch_y
        z0, z1 = args.patch_z
        patch = relief[y0:y1, z0:z1]

        c = peak_local_max(
            patch,
            min_distance=args.peak_min_distance_vox,
            threshold_abs=args.peak_relief_threshold_vox,
            exclude_border=False,
        )

        yy = c[:, 0] + y0
        zz = c[:, 1] + z0
        xx = surf[yy, zz]
        keep = np.isfinite(xx)

        pts = np.c_[xx[keep], yy[keep], zz[keep]]
        return surf, filled, relief, pts

    Ts = [args.surface_threshold] + list(args.neighbor_thresholds)
    surfaces = {}
    points_by_T = {}
    threshold_rows = []

    for T in Ts:
        surf, filled, rel, pts = relief_and_points(T)
        surfaces[T] = (surf, filled, rel)
        points_by_T[T] = pts

        if len(pts) >= 2:
            nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
            med_nn_um = float(np.median(nn) * args.spacing_um)
            nn_cv = float(np.std(nn) / np.mean(nn))
        else:
            med_nn_um = np.nan
            nn_cv = np.nan

        threshold_rows.append({
            "isovalue": T,
            "candidate_count": len(pts),
            "median_nn_um": med_nn_um,
            "nn_cv": nn_cv,
        })

    pd.DataFrame(threshold_rows).to_csv(
        args.out / "surface_threshold_stability.csv", index=False
    )

    base = points_by_T[args.surface_threshold]
    hits = []

    for p0 in base:
        h = 0
        for T in args.neighbor_thresholds:
            pts = points_by_T[T]
            if len(pts) == 0:
                continue
            d, _ = cKDTree(pts).query(p0, k=1)
            h += int(d <= args.persistence_distance_vox)
        hits.append(h)

    hits = np.asarray(hits)
    robust_mask = hits >= args.persistence_min_hits
    robust_pts = base[robust_mask]

    if len(robust_pts) < 10:
        raise RuntimeError(
            f"Only {len(robust_pts)} robust facet candidates found. "
            "For a new specimen, adjust the eye patch / surface threshold."
        )

    rob_nn = cKDTree(robust_pts).query(robust_pts, k=2)[0][:, 1]

    surf, filled, relief = surfaces[args.surface_threshold]
    surf_smooth = ndi.gaussian_filter(filled, sigma=2.0)
    fy, fz = np.gradient(surf_smooth)
    lap = ndi.laplace(surf_smooth)

    centers = pd.DataFrame({
        "facet_id": np.arange(len(robust_pts)),
        "x_vox": robust_pts[:, 0],
        "y_vox": robust_pts[:, 1],
        "z_vox": robust_pts[:, 2],
        "x_um": robust_pts[:, 0] * args.spacing_um,
        "y_um": robust_pts[:, 1] * args.spacing_um,
        "z_um": robust_pts[:, 2] * args.spacing_um,
        "persistence_hits": hits[robust_mask],
        "nearest_neighbor_um": rob_nn * args.spacing_um,
    })
    centers.to_csv(args.out / "robust_facet_centers.csv", index=False)

    # ---------------- local volumes aligned to outer surface ----------------
    def local_volume(point, pitch_vox, uv_step=1.0,
                     dmin=-4.0, dmax=25.0, dstep=0.5):
        x, y, z = point
        yi, zi = int(round(y)), int(round(z))

        n = np.array([1.0, -fy[yi, zi], -fz[yi, zi]], float)
        n /= np.linalg.norm(n)

        t1 = np.array([fy[yi, zi], 1.0, 0.0], float)
        t1 -= n * np.dot(t1, n)
        t1 /= np.linalg.norm(t1)

        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2)

        uv_extent = min(8.0, max(4.0, 0.50 * pitch_vox))
        u = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
        v = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
        d = np.arange(dmin, dmax + 1e-9, dstep)

        U, V, D = np.meshgrid(u, v, d, indexing="ij")

        xyz = (
            np.array([x, y, z])[:, None, None, None]
            + t1[:, None, None, None] * U[None]
            + t2[:, None, None, None] * V[None]
            - n[:, None, None, None] * D[None]
        )

        arr = ndi.map_coordinates(
            volf,
            xyz.reshape(3, -1),
            order=1,
            mode="nearest",
        ).reshape(U.shape)

        return arr, u, v, d

    local = [
        local_volume(p, pitch)
        for p, pitch in zip(robust_pts, rob_nn)
    ]

    # Inter-facet midpoint controls
    yz = robust_pts[:, 1:3]
    tree = cKDTree(yz)
    _, inds = tree.query(yz, k=2)

    seen = set()
    midpoints = []

    for i, j in enumerate(inds[:, 1]):
        a, b = sorted((i, int(j)))
        if (a, b) in seen:
            continue
        seen.add((a, b))

        ymid, zmid = (yz[a] + yz[b]) / 2
        yi, zi = int(round(ymid)), int(round(zmid))

        if (
            0 <= yi < surf.shape[0]
            and 0 <= zi < surf.shape[1]
            and np.isfinite(surf[yi, zi])
        ):
            midpoints.append(
                np.array([surf[yi, zi], ymid, zmid])
            )

    midpoint_local = [
        local_volume(p, float(np.median(rob_nn)))
        for p in midpoints
    ]

    # ---------------- first reproducible internal edge ----------------
    def central_edge(tmp, sigma, window, rmax=2.5):
        arr, u, v, d = tmp
        U, V = np.meshgrid(u, v, indexing="ij")
        use = np.hypot(U, V) <= rmax

        prof = arr[use, :].mean(axis=0)
        ps = ndi.gaussian_filter1d(prof, sigma=sigma)
        grad = np.gradient(ps, d)

        sel = np.where(
            (d >= window[0]) & (d <= window[1])
        )[0]

        j = sel[np.argmin(grad[sel])]
        return float(d[j]), float(-grad[j]), prof

    edge_facet = [
        central_edge(
            x,
            args.edge_sigma_vox,
            args.edge_window_vox,
        )
        for x in local
    ]

    edge_mid = [
        central_edge(
            x,
            args.edge_sigma_vox,
            args.edge_window_vox,
        )
        for x in midpoint_local
    ]

    facet_depth = np.array([x[0] for x in edge_facet])
    facet_strength = np.array([x[1] for x in edge_facet])
    mid_strength = np.array([x[1] for x in edge_mid])

    u_stat, pval = mannwhitneyu(
        facet_strength, mid_strength, alternative="greater"
    )
    auc = u_stat / (
        len(facet_strength) * len(mid_strength)
    )

    # ---------------- 2D internal-edge map in each facet ----------------
    def edge_map(tmp, pitch_vox,
                 sigma=None, window=None,
                 strength_min=None):
        if sigma is None:
            sigma = args.edge_sigma_vox
        if window is None:
            window = args.edge_window_vox
        if strength_min is None:
            strength_min = args.edge_strength_min

        arr, u, v, d = tmp
        ps = ndi.gaussian_filter1d(
            arr, sigma=sigma, axis=2
        )
        grad = np.gradient(ps, d, axis=2)

        sel = np.where(
            (d >= window[0]) & (d <= window[1])
        )[0]

        jj = np.argmin(
            grad[:, :, sel], axis=2
        )
        edge_idx = sel[jj]

        depth = d[edge_idx]
        strength = -np.take_along_axis(
            grad,
            edge_idx[:, :, None],
            axis=2,
        )[:, :, 0]

        U, V = np.meshgrid(u, v, indexing="ij")
        rmax = min(8.0, 0.45 * pitch_vox)
        disk = U**2 + V**2 <= rmax**2
        valid = disk & (strength >= strength_min)

        pts = np.column_stack([
            U[valid], V[valid],
            depth[valid], strength[valid]
        ])
        coverage = (
            valid.sum() / max(disk.sum(), 1)
        )

        if len(pts) >= 20:
            uu, vv, dep, _ = pts.T
            X = np.c_[
                np.ones(len(uu)),
                uu, vv,
                uu**2, vv**2, uu*vv,
            ]
            coef = np.linalg.lstsq(
                X, dep, rcond=None
            )[0]
            pred = X @ coef
            rmse = float(
                np.sqrt(
                    np.mean((dep - pred)**2)
                )
            )
        else:
            rmse = np.nan

        cent = U**2 + V**2 <= 2.5**2

        return {
            "pts": pts,
            "coverage": float(coverage),
            "quadratic_rmse_vox": rmse,
            "center_strength": float(
                np.median(strength[cent])
            ),
            "center_depth_vox": float(
                np.median(depth[cent])
            ),
        }

    patches = [
        edge_map(tmp, pitch)
        for tmp, pitch in zip(local, rob_nn)
    ]

    manifest = []
    for i, p in enumerate(patches):
        manifest.append({
            "facet_id": i,
            "edge_depth_vox": facet_depth[i],
            "edge_depth_um":
                facet_depth[i] * args.spacing_um,
            "edge_strength": facet_strength[i],
            "coverage": p["coverage"],
            "quadratic_rmse_vox":
                p["quadratic_rmse_vox"],
            "nearest_neighbor_um":
                rob_nn[i] * args.spacing_um,
        })

    manifest = pd.DataFrame(manifest)

    good = (
        (manifest["coverage"] >= args.hq_coverage_min)
        & (
            manifest["edge_strength"]
            >= args.hq_center_strength_min
        )
        & (
            manifest["quadratic_rmse_vox"]
            <= args.hq_quadratic_rmse_max_vox
        )
    )

    manifest["high_quality"] = good
    good_ids = manifest.loc[
        good, "facet_id"
    ].to_numpy(int)

    manifest.to_csv(
        args.out / "internal_edge_facets.csv",
        index=False,
    )

    if len(good_ids) < args.cv_blocks:
        raise RuntimeError(
            f"Only {len(good_ids)} high-quality facets; "
            "not enough for spatial CV."
        )

    # ---------------- extraction stability ----------------
    alt_settings = [
        (1.0, [9, 19]),
        (2.0, [9, 19]),
        (1.5, [8, 20]),
        (1.5, [10, 18]),
    ]

    stability_rows = []

    def full_depth_strength(tmp, sigma, window):
        arr, u, v, d = tmp
        ps = ndi.gaussian_filter1d(
            arr, sigma=sigma, axis=2
        )
        grad = np.gradient(ps, d, axis=2)

        sel = np.where(
            (d >= window[0])
            & (d <= window[1])
        )[0]

        jj = np.argmin(
            grad[:, :, sel], axis=2
        )
        edge_idx = sel[jj]

        depth = d[edge_idx]
        strength = -np.take_along_axis(
            grad,
            edge_idx[:, :, None],
            axis=2,
        )[:, :, 0]

        return depth, strength

    for i in good_ids:
        tmp = local[i]
        arr, u, v, d = tmp
        U, V = np.meshgrid(
            u, v, indexing="ij"
        )
        rmax = min(8.0, 0.45 * rob_nn[i])
        disk = U**2 + V**2 <= rmax**2

        d0, s0 = full_depth_strength(
            tmp,
            args.edge_sigma_vox,
            args.edge_window_vox,
        )

        for sigma, window in alt_settings:
            da, sa = full_depth_strength(
                tmp, sigma, window
            )
            shared = (
                disk
                & (s0 >= args.edge_strength_min)
                & (sa >= args.edge_strength_min)
            )

            if shared.sum() == 0:
                continue

            diff = np.abs(
                da[shared] - d0[shared]
            )

            stability_rows.append({
                "facet_id": i,
                "sigma": sigma,
                "window":
                    f"{window[0]}-{window[1]}",
                "shared_points":
                    int(shared.sum()),
                "median_abs_shift_vox":
                    float(np.median(diff)),
                "p90_abs_shift_vox":
                    float(np.percentile(diff, 90)),
                "p90_abs_shift_um":
                    float(
                        np.percentile(diff, 90)
                        * args.spacing_um
                    ),
            })

    stability_df = pd.DataFrame(
        stability_rows
    )
    stability_df.to_csv(
        args.out / "internal_edge_stability.csv",
        index=False,
    )

    # ---------------- spatial held-out reconstruction ----------------
    outer_features = []

    for i, (x, y, z) in enumerate(robust_pts):
        yi, zi = int(round(y)), int(round(z))
        outer_features.append([
            x, y, z,
            relief[yi, zi],
            fy[yi, zi],
            fz[yi, zi],
            lap[yi, zi],
            rob_nn[i],
        ])

    outer_features = np.asarray(
        outer_features
    )

    rows = []

    for i in good_ids:
        p = patches[i]
        pitch = rob_nn[i]
        rscale = max(pitch / 2, 1e-6)

        for u0, v0, dep0, strength0 in p["pts"]:
            un, vn = u0 / rscale, v0 / rscale
            rn = np.hypot(un, vn)

            features = [
                un, vn,
                rn, rn**2,
                un**2, vn**2,
                un*vn,
                *outer_features[i],
            ]

            rows.append([
                i,
                *features,
                dep0 / pitch,
                dep0,
            ])

    cols = [
        "facet_id",
        "un", "vn", "rn", "rn2",
        "un2", "vn2", "uv",
        "surf_x", "y", "z",
        "relief", "fy", "fz",
        "lap", "pitch",
        "target_norm", "target_vox",
    ]

    df = pd.DataFrame(rows, columns=cols)

    feature_cols = [
        "un", "vn", "rn", "rn2",
        "un2", "vn2", "uv",
        "surf_x", "y", "z",
        "relief", "fy", "fz",
        "lap", "pitch",
    ]

    baseline_cols = [
        "un", "vn", "rn", "rn2",
        "un2", "vn2", "uv",
    ]

    unique_facets = np.array(
        sorted(df.facet_id.unique())
    )

    fc = np.array([
        [robust_pts[i, 1], robust_pts[i, 2]]
        for i in unique_facets
    ])

    groups = KMeans(
        n_clusters=args.cv_blocks,
        random_state=42,
        n_init=10,
    ).fit_predict(fc)

    gmap = dict(
        zip(unique_facets, groups)
    )
    dg = df.facet_id.map(gmap).to_numpy()
    df["cv_block"] = dg
    df.to_csv(
        args.out / "reconstruction_samples.csv",
        index=False,
    )

    def run_cv(cols_use):
        y = df.target_norm.to_numpy()
        pred = np.full(len(df), np.nan)

        cv = GroupKFold(
            n_splits=args.cv_blocks
        )

        for tr, te in cv.split(df, y, dg):
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=args.ridge_alpha),
            )
            model.fit(
                df.iloc[tr][cols_use],
                y[tr],
            )
            pred[te] = model.predict(
                df.iloc[te][cols_use]
            )

        pred_vox = (
            pred * df.pitch.to_numpy()
        )
        err_um = np.abs(
            pred_vox
            - df.target_vox.to_numpy()
        ) * args.spacing_um

        fmae = []
        fids = df.facet_id.to_numpy()

        for i in unique_facets:
            fmae.append(
                np.mean(err_um[fids == i])
            )

        return (
            pred_vox,
            err_um,
            np.asarray(fmae),
        )

    pred_full, err_full, fmae_full = run_cv(
        feature_cols
    )
    pred_base, err_base, fmae_base = run_cv(
        baseline_cols
    )

    median_depth_um = float(
        np.median(df.target_vox)
        * args.spacing_um
    )
    med_mae = float(
        np.median(fmae_full)
    )
    p90_mae = float(
        np.percentile(fmae_full, 90)
    )
    baseline = float(
        np.median(fmae_base)
    )

    summary = {
        "input": str(args.input),
        "volume_shape": list(vol.shape),
        "spacing_um": args.spacing_um,
        "surface_threshold":
            args.surface_threshold,
        "robust_facets":
            int(len(robust_pts)),
        "median_facet_spacing_um":
            float(np.median(rob_nn)
                  * args.spacing_um),
        "median_internal_edge_depth_um":
            float(np.median(facet_depth)
                  * args.spacing_um),
        "facet_center_median_edge_strength":
            float(np.median(facet_strength)),
        "interfacet_median_edge_strength":
            float(np.median(mid_strength)),
        "facet_vs_interfacet_auc":
            float(auc),
        "facet_vs_interfacet_p":
            float(pval),
        "high_quality_facets":
            int(len(unique_facets)),
        "heldout_median_facet_MAE_um":
            med_mae,
        "heldout_p90_facet_MAE_um":
            p90_mae,
        "normalized_median_error":
            med_mae / median_depth_um,
        "radial_only_baseline_MAE_um":
            baseline,
        "improvement_over_baseline_percent":
            float(
                (1 - med_mae / baseline) * 100
            ),
    }

    pd.DataFrame([summary]).to_csv(
        args.out / "reconstruction_summary.csv",
        index=False,
    )
    (args.out / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # ---------------- QC figures ----------------
    # Outer relief + high quality facets
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(
        relief, origin="lower", aspect="auto"
    )
    gp = robust_pts[good_ids]
    ax.scatter(
        gp[:, 2], gp[:, 1], s=14
    )
    ax.set_xlabel("surface z (vox)")
    ax.set_ylabel("surface y (vox)")
    ax.set_title(
        f"High-quality internal edges "
        f"(n={len(good_ids)})"
    )
    fig.colorbar(
        im, ax=ax,
        label="outer-surface relief (vox)"
    )
    fig.tight_layout()
    fig.savefig(
        args.out / "high_quality_facets.png",
        dpi=220,
    )
    plt.close(fig)

    # Profiles
    facet_profiles = np.stack(
        [x[2] for x in edge_facet]
    )
    mid_profiles = np.stack(
        [x[2] for x in edge_mid]
    )
    d_axis = local[0][3]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        d_axis,
        facet_profiles.mean(axis=0),
        label="facet centres",
    )
    ax.plot(
        d_axis,
        mid_profiles.mean(axis=0),
        label="inter-facet midpoints",
    )
    ax.axvline(
        np.median(facet_depth),
        linewidth=1.2,
    )
    ax.set_xlabel("inward distance (voxels)")
    ax.set_ylabel("mean CT intensity")
    ax.set_title("Facet-locked internal edge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.out / "facet_vs_interfacet_profiles.png",
        dpi=220,
    )
    plt.close(fig)

    # observed vs predicted
    obs_um = (
        df.target_vox.to_numpy()
        * args.spacing_um
    )
    pred_um = pred_full * args.spacing_um

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        obs_um, pred_um,
        s=4, alpha=0.35,
    )
    lo = min(obs_um.min(), pred_um.min())
    hi = max(obs_um.max(), pred_um.max())
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlabel(
        "observed internal-edge depth (µm)"
    )
    ax.set_ylabel(
        "spatially held-out prediction (µm)"
    )
    ax.set_title(
        "Blind spatial-block reconstruction"
    )
    fig.tight_layout()
    fig.savefig(
        args.out / "observed_vs_predicted.png",
        dpi=220,
    )
    plt.close(fig)

    md = f"""# Fossil lens reconstruction result

Input volume: `{args.input.name}`

- volume: {vol.shape[0]} × {vol.shape[1]} × {vol.shape[2]} voxels
- spacing: {args.spacing_um:.3f} µm
- robust facet candidates: **{len(robust_pts)}**
- median facet spacing: **{np.median(rob_nn)*args.spacing_um:.2f} µm**
- first facet-locked internal edge: **{np.median(facet_depth)*args.spacing_um:.2f} µm**
- facet/inter-facet edge-strength AUC: **{auc:.3f}**
- high-quality reconstruction facets: **{len(unique_facets)}**
- spatially held-out median facet MAE: **{med_mae:.2f} µm**
- p90 facet MAE: **{p90_mae:.2f} µm**
- normalized median error: **{100*med_mae/median_depth_um:.1f}%**
- radial-only baseline: **{baseline:.2f} µm**

Interpretation: the script establishes a repeatable facet-locked internal CT
boundary and tests how well outer geometry predicts that hidden boundary.
The anatomical name of that boundary must be validated independently before
calling it definitively the proximal lens surface.
"""
    (args.out / "SUMMARY.md").write_text(md)

    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the Asaphus subvoxel facet / candidate-interface analysis.

Input: 1652a_0000_1_cropped_CORRECT_3p7um.nrrd in the same directory.
Output CSV/MD files are written beside the input.

Important: the internal low-intensity surface is treated as a candidate
lens-subjacent/proximal interface, not as proven anatomical ground truth.
"""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "1652a_0000_1_cropped_CORRECT_3p7um.nrrd"
SPACING_UM = 3.7
SURFACE_SIGMA_X = 1.0
CURVATURE_SIGMA = 15.0
PATCH_Y = (100, 450)
PATCH_Z = (300, 480)
PEAK_MIN_DISTANCE = 7
PEAK_RELIEF_THRESHOLD = 1.2
CENTRAL_T = 50
NEIGHBOR_TS = [45, 55, 60, 65, 70]
PERSIST_DISTANCE = 4.0
PERSIST_MIN_HITS = 4


def read_nrrd(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if line in (b"\n", b"\r\n"):
                body_offset = f.tell(); break
            header.append(line.decode("ascii").rstrip())
    h = {}
    for line in header:
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1); h[k.strip()] = v.strip()
    sizes = tuple(map(int, h["sizes"].split()))
    with open(path, "rb") as f:
        f.seek(body_offset)
        with gzip.GzipFile(fileobj=f, mode="rb") as gz:
            raw = gz.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(sizes, order="F")


vol = read_nrrd(INPUT)
sm = ndi.gaussian_filter1d(vol.astype(np.float32), sigma=SURFACE_SIGMA_X, axis=0)


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
    frac = np.divide(v0 - T, denom, out=np.zeros_like(v0), where=np.abs(denom) > 1e-6)
    surf[yv, zv] = xv + frac
    surf[ys[~valid], zs[~valid]] = xs[~valid]
    return surf


def relief_and_points(T):
    s = subvoxel_surface(T)
    m = np.isfinite(s)
    ind = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
    filled = s[tuple(ind)]
    relief = filled - ndi.gaussian_filter(filled, sigma=CURVATURE_SIGMA)
    p = relief[PATCH_Y[0]:PATCH_Y[1], PATCH_Z[0]:PATCH_Z[1]]
    c = peak_local_max(p, min_distance=PEAK_MIN_DISTANCE,
                       threshold_abs=PEAK_RELIEF_THRESHOLD, exclude_border=False)
    yy = c[:, 0] + PATCH_Y[0]
    zz = c[:, 1] + PATCH_Z[0]
    xx = s[yy, zz]
    good = np.isfinite(xx)
    pts = np.c_[xx[good], yy[good], zz[good]]
    return s, filled, relief, pts


surfaces = {}
points_by_T = {}
stability_rows = []
for T in [45, 50, 55, 60, 65, 70]:
    s, sf, rel, pts = relief_and_points(T)
    surfaces[T] = (s, sf, rel)
    points_by_T[T] = pts
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    stability_rows.append({
        "isovalue_T": T, "candidate_count": len(pts),
        "median_nn_um": np.median(nn) * SPACING_UM,
        "nn_cv": np.std(nn) / np.mean(nn),
    })
pd.DataFrame(stability_rows).to_csv(ROOT / "36_asaphus_surface_threshold_stability.csv", index=False)

base = points_by_T[CENTRAL_T]
hits = []
for p in base:
    h = 0
    for T in NEIGHBOR_TS:
        d, _ = cKDTree(points_by_T[T]).query(p, k=1)
        h += int(d <= PERSIST_DISTANCE)
    hits.append(h)
hits = np.asarray(hits)
robust_mask = hits >= PERSIST_MIN_HITS
centers = base
robust_pts = base[robust_mask]
rob_nn = cKDTree(robust_pts).query(robust_pts, k=2)[0][:, 1]

s50, sf50, relief = surfaces[50]
surf_smooth = ndi.gaussian_filter(sf50, sigma=2)
fy, fz = np.gradient(surf_smooth)
lap = ndi.laplace(surf_smooth)

center_table = pd.DataFrame({
    "facet_id": np.where(robust_mask)[0],
    "x_vox": robust_pts[:, 0], "y_vox": robust_pts[:, 1], "z_vox": robust_pts[:, 2],
    "x_um": robust_pts[:, 0] * SPACING_UM,
    "y_um": robust_pts[:, 1] * SPACING_UM,
    "z_um": robust_pts[:, 2] * SPACING_UM,
    "persistence_hits_of_5_other_thresholds": hits[robust_mask],
    "nearest_neighbor_um": rob_nn * SPACING_UM,
})
center_table.to_csv(ROOT / "36_asaphus_robust_facet_centers.csv", index=False)

# All T50 centers: local geometry and centerline inward profiles.
nn_all = cKDTree(centers).query(centers, k=2)[0][:, 1]
d = np.linspace(-5, 40, 181)
profiles = []
for x, y, z in centers:
    yi, zi = int(round(y)), int(round(z))
    n = np.array([1.0, -fy[yi, zi], -fz[yi, zi]])
    n /= np.linalg.norm(n)
    q = np.array([x, y, z])[:, None] - n[:, None] * d[None, :]
    profiles.append(ndi.map_coordinates(vol.astype(np.float32), q, order=1, mode="nearest"))
profiles = np.asarray(profiles)
plateau = np.median(profiles[:, (d >= 4) & (d <= 10)], axis=1)
inner_min = np.min(profiles[:, (d >= 10) & (d <= 25)], axis=1)
ps = ndi.gaussian_filter1d(profiles, sigma=1.5, axis=1)
g = np.gradient(ps, d, axis=1)
kr = np.where((d >= 8) & (d <= 25))[0]
ki = kr[np.argmin(g[:, kr], axis=1)]
edge_d = d[ki]
edge_strength = -g[np.arange(len(profiles)), ki]
qual = robust_mask & (edge_strength >= 4.5) & (inner_min < 80) & (edge_d >= 8) & (edge_d <= 24)

outer_features = []
for i, (x, y, z) in enumerate(centers):
    yi, zi = int(round(y)), int(round(z))
    outer_features.append([x, y, z, relief[yi, zi], fy[yi, zi], fz[yi, zi], lap[yi, zi], nn_all[i]])
outer_features = np.asarray(outer_features)


def local_volume(idx, uv_extent=8, uv_step=1.0, dmax=25, dstep=0.5):
    x, y, z = centers[idx]
    yi, zi = int(round(y)), int(round(z))
    n = np.array([1.0, -fy[yi, zi], -fz[yi, zi]])
    n /= np.linalg.norm(n)
    t1 = np.array([fy[yi, zi], 1.0, 0.0])
    t1 -= n * np.dot(t1, n); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1); t2 /= np.linalg.norm(t2)
    u = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
    v = np.arange(-uv_extent, uv_extent + 1e-9, uv_step)
    dd = np.arange(0, dmax + 1e-9, dstep)
    U, V, D = np.meshgrid(u, v, dd, indexing="ij")
    p = (np.array([x, y, z])[:, None, None, None]
         + t1[:, None, None, None] * U[None]
         + t2[:, None, None, None] * V[None]
         - n[:, None, None, None] * D[None])
    arr = ndi.map_coordinates(vol.astype(np.float32), p.reshape(3, -1), order=1, mode="nearest").reshape(U.shape)
    return arr, u, v, dd


def candidate_patch(idx, threshold):
    arr, u, v, dd = local_volume(idx)
    mask = arr < threshold
    mask[:, :, dd < 5] = False
    lab, _ = ndi.label(mask)
    ci, cj = len(u)//2, len(v)//2
    kval = np.where(dd >= 5)[0]
    kk = kval[np.argmin(arr[ci, cj, kval])]
    label = lab[ci, cj, kk]
    if label == 0:
        return None
    comp = lab == label
    U, V = np.meshgrid(u, v, indexing="ij")
    disk = U**2 + V**2 <= 8**2
    exists = comp.any(axis=2) & disk
    top = np.full(U.shape, np.nan)
    for ii, jj in np.argwhere(exists):
        top[ii, jj] = dd[np.argmax(comp[ii, jj, :])]
    coverage = exists.sum() / disk.sum()
    pts = np.column_stack([U[exists], V[exists], top[exists]])
    if coverage < 0.15 or len(pts) < 20:
        return None
    return {"pts": pts, "coverage": coverage, "top": top}


def build_set(threshold):
    patches = {}
    for idx in np.where(qual)[0]:
        p = candidate_patch(int(idx), threshold)
        if p is not None:
            patches[int(idx)] = p
    rows = []
    for idx, p in patches.items():
        pitch = nn_all[idx]
        for u, v, dep in p["pts"]:
            rscale = max(pitch/2, 1e-6)
            un, vn = u/rscale, v/rscale
            rn = np.hypot(un, vn)
            f = [un, vn, rn, rn**2, un**2, vn**2, un*vn, *outer_features[idx]]
            rows.append([idx, *f, dep/pitch, dep])
    cols = ["facet_idx", "un", "vn", "rn", "rn2", "un2", "vn2", "uv",
            "surf_x", "y", "z", "relief", "fy", "fz", "lap", "pitch",
            "target_norm", "target_vox"]
    return patches, pd.DataFrame(rows, columns=cols)


feature_cols = ["un", "vn", "rn", "rn2", "un2", "vn2", "uv",
                "surf_x", "y", "z", "relief", "fy", "fz", "lap", "pitch"]


def spatial_cv(df):
    facet_ids = np.array(sorted(df.facet_idx.unique()))
    fc = np.array([[centers[i, 1], centers[i, 2]] for i in facet_ids])
    groups = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(fc)
    gmap = dict(zip(facet_ids, groups))
    dg = df.facet_idx.map(gmap).to_numpy()
    y = df.target_norm.to_numpy()
    pred = np.full(len(df), np.nan)
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(df, y, dg):
        model = make_pipeline(StandardScaler(), Ridge(alpha=20))
        model.fit(df.iloc[tr][feature_cols], y[tr])
        pred[te] = model.predict(df.iloc[te][feature_cols])
    pred_vox = pred * df.pitch.to_numpy()
    err_um = np.abs(pred_vox - df.target_vox.to_numpy()) * SPACING_UM
    fmae = [np.mean(err_um[df.facet_idx.to_numpy() == i]) for i in facet_ids]
    return {
        "n_facets": len(facet_ids), "n_points": len(df),
        "point_MAE_um": np.mean(err_um),
        "median_facet_MAE_um": np.median(fmae),
        "p90_facet_MAE_um": np.percentile(fmae, 90),
        "median_interface_depth_um": np.median(df.target_vox) * SPACING_UM,
        "normalized_median_error": np.median(fmae)/(np.median(df.target_vox)*SPACING_UM),
    }


results = []
for threshold in [70, 75]:
    patches, df = build_set(threshold)
    cv = spatial_cv(df)
    cv["intensity_threshold"] = threshold
    results.append(cv)
    pd.DataFrame([{"facet_id": k, "coverage": v["coverage"], "n_points": len(v["pts"])} for k, v in patches.items()]).to_csv(
        ROOT / f"37_patch_manifest_T{threshold}.csv", index=False)

pd.DataFrame(results).to_csv(ROOT / "37_asaphus_candidate_interface_cv.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))

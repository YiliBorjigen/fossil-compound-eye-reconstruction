"""
Strict leave-one-out blind missing-lens benchmark for ODA CT output.

Run from the ODA repository root:
    ./oda_env/bin/python research/missing_lens/04_strict_blind_benchmark.py

Expected input:
    tests/d_mauritiana_ct_stack/ommatidial_data.csv

The hidden lens is removed before a sphere is refit from the remaining
Cartesian centers. Hole candidates are obtained from Voronoi geometry,
boundary-like candidates are filtered, and nearby Voronoi vertices are
clustered into physical hole hypotheses.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree, ConvexHull, Delaunay
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/d_mauritiana_ct_stack/ommatidial_data.csv"
OUTDIR = ROOT / "research/missing_lens/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

MERGE_FACTOR = 0.40
ANGLE_GAP_LIMIT = 150

def parse_neighbors(text):
    return np.fromstring(str(text).strip("[]"), sep=" ", dtype=int)

def fit_sphere(xyz):
    A = np.column_stack([2 * xyz, np.ones(len(xyz))])
    b = np.sum(xyz ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(sol[3] + np.dot(center, center)))
    return center, radius

def surface_map(observed_xyz, truth_xyz=None):
    center, radius = fit_sphere(observed_xyz)
    v = observed_xyz - center
    r = np.linalg.norm(v, axis=1)
    theta = np.arccos(np.clip(v[:, 2] / r, -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    theta0 = float(np.median(theta))
    scale_phi = float(np.sin(theta0))
    points = np.column_stack([phi * scale_phi, theta])

    truepos = None
    if truth_xyz is not None:
        tv = truth_xyz - center
        tr = np.linalg.norm(tv)
        tt = np.arccos(np.clip(tv[2] / tr, -1, 1))
        tp = np.arctan2(tv[1], tv[0])
        truepos = np.array([tp * scale_phi, tt])
    return points, truepos, radius

def detect_one(df, target_label):
    truth = df.loc[df["label"] == target_label].iloc[0]
    observed = df.loc[df["label"] != target_label]

    points, truepos, sphere_radius = surface_map(
        observed[["x","y","z"]].to_numpy(float),
        truth[["x","y","z"]].to_numpy(float),
    )

    vor = Voronoi(points)
    candidates = vor.vertices

    hull = ConvexHull(points)
    hull_test = Delaunay(points[hull.vertices])
    candidates = candidates[hull_test.find_simplex(candidates) >= 0]

    tree = cKDTree(points)
    all_nn, _ = tree.query(points, k=2)
    nearest_other = all_nn[:, 1]

    distances, inds = tree.query(candidates, k=6)
    local_spacing = np.median(nearest_other[inds], axis=1)
    gap_score = distances[:, 0] / local_spacing

    nearby = points[inds]
    vectors = nearby - candidates[:, None, :]
    angles = np.sort(np.arctan2(vectors[:,:,1], vectors[:,:,0]), axis=1)
    circular_gaps = np.diff(
        np.concatenate([angles, angles[:, :1] + 2*np.pi], axis=1),
        axis=1,
    )
    max_angle_gap = np.degrees(circular_gaps.max(axis=1))

    keep = max_angle_gap <= ANGLE_GAP_LIMIT
    candidates = candidates[keep]
    local_spacing = local_spacing[keep]
    gap_score = gap_score[keep]

    merge_radius = MERGE_FACTOR * float(np.median(local_spacing))
    cluster_ids = DBSCAN(
        eps=merge_radius, min_samples=1
    ).fit_predict(candidates)
    n_clusters = int(cluster_ids.max()) + 1

    wsum = np.bincount(cluster_ids, weights=gap_score, minlength=n_clusters)
    xsum = np.bincount(
        cluster_ids, weights=gap_score*candidates[:,0], minlength=n_clusters
    )
    ysum = np.bincount(
        cluster_ids, weights=gap_score*candidates[:,1], minlength=n_clusters
    )
    counts = np.bincount(cluster_ids, minlength=n_clusters)

    max_score = np.full(n_clusters, -np.inf)
    np.maximum.at(max_score, cluster_ids, gap_score)

    centers = np.column_stack([xsum/wsum, ysum/wsum])
    order = np.argsort(max_score)[::-1]
    centers = centers[order]
    max_score = max_score[order]
    counts = counts[order]

    truth_neighbor_distances, _ = tree.query(truepos, k=6)
    truth_spacing = float(np.mean(truth_neighbor_distances))

    dtruth = np.linalg.norm(centers - truepos, axis=1)
    true_idx = int(np.argmin(dtruth))

    return {
        "label": int(target_label),
        "truth_rank": true_idx + 1,
        "truth_hole_norm_error": float(dtruth[true_idx] / truth_spacing),
        "top1_pred_norm_error": float(
            np.linalg.norm(centers[0]-truepos) / truth_spacing
        ),
        "truth_hole_score": float(max_score[true_idx]),
        "top_score": float(max_score[0]),
        "n_holes": n_clusters,
        "top_n_vertices": int(counts[0]),
        "truth_hole_n_vertices": int(counts[true_idx]),
        "sphere_radius": sphere_radius,
    }

df = pd.read_csv(DATA)

tmp = df.copy()
tmp["neighbor_list"] = tmp["neighbors"].apply(parse_neighbors)
tmp["n_neighbors"] = tmp["neighbor_list"].apply(len)
by_label = tmp.set_index("label")

targets = []
for _, row in tmp.iterrows():
    if row["n_neighbors"] != 6:
        continue
    ns = row["neighbor_list"]
    if all(
        n in by_label.index and by_label.loc[n, "n_neighbors"] == 6
        for n in ns
    ):
        targets.append(int(row["label"]))

results = pd.DataFrame([detect_one(df, label) for label in targets])
results.to_csv(OUTDIR / "04_strict_blind_benchmark_results.csv", index=False)

print("N tested:", len(results))
print("Top-1:", 100*(results["truth_rank"] == 1).mean(), "%")
print("Top-5:", 100*(results["truth_rank"] <= 5).mean(), "%")
print("Top-10:", 100*(results["truth_rank"] <= 10).mean(), "%")
print(
    "Median localization error:",
    100*results["truth_hole_norm_error"].median(),
    "% of local spacing",
)
print(
    "95th-percentile localization error:",
    100*results["truth_hole_norm_error"].quantile(.95),
    "% of local spacing",
)
print("\nFailures:")
print(
    results.loc[
        results["truth_rank"] != 1,
        ["label","truth_rank","truth_hole_norm_error",
         "truth_hole_score","top_score"]
    ].to_string(index=False)
)

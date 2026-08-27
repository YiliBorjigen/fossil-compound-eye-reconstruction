"""
Experiment 06: joint reconstruction of two adjacent missing ommatidia.

This is a stricter follow-up to Experiment 05.

Development cohort:
    200 adjacent strong-interior pairs selected with seed 20260817.
    This cohort was used while developing the joint method.

Independent holdout cohort:
    200 different adjacent strong-interior pairs selected from the remaining
    pairs with seed 20260818.

The blind detector never receives the hidden lenses' positions or neighbor
lists. The intact neighbor graph is used only to define which evaluation
pairs are truly adjacent.

Joint reconstruction idea:
1. Remove both adjacent lenses.
2. Refit a sphere from surviving Cartesian centers.
3. Detect the strongest damaged region with Voronoi geometry.
4. Build several plausible two-point initializations around that region.
5. Optimize BOTH inferred lenses simultaneously so that:
   - each has five surviving near-neighbors at approximately local spacing,
   - the two inferred lenses are approximately one local spacing apart,
   - the pair remains local to the detected damaged region.
6. Score only after optimization by revealing the two hidden true positions.

Run from ODA repository root:
    ./oda_env/bin/python research/missing_lens/06_joint_adjacent_pair_benchmark.py
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import Voronoi, cKDTree, ConvexHull, Delaunay
from sklearn.cluster import DBSCAN


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/d_mauritiana_ct_stack/ommatidial_data.csv"
OUTDIR = ROOT / "research/missing_lens/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVELOPMENT_SEED = 20260817
HOLDOUT_SEED = 20260818
COHORT_SIZE = 200

MERGE_FACTOR = 0.40
ANGLE_GAP_LIMIT = 150


def parse_neighbors(text):
    return np.fromstring(
        str(text).strip("[]"),
        sep=" ",
        dtype=int,
    )


def fit_sphere(xyz):
    """Least-squares sphere fitted only to surviving ommatidial centers."""
    A = np.column_stack([2 * xyz, np.ones(len(xyz))])
    b = np.sum(xyz ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    center = sol[:3]
    radius = float(
        np.sqrt(sol[3] + np.dot(center, center))
    )
    return center, radius


def surface_map(observed_xyz, truth_xyzs):
    """
    Map surviving 3D centers to a 2D spherical surface representation.

    Truth points are mapped using the same refitted sphere only for final
    evaluation; they are not supplied to the detector.
    """
    center, _ = fit_sphere(observed_xyz)

    vec = observed_xyz - center
    radius = np.linalg.norm(vec, axis=1)

    theta = np.arccos(
        np.clip(vec[:, 2] / radius, -1, 1)
    )
    phi = np.arctan2(vec[:, 1], vec[:, 0])

    phi_scale = float(np.sin(np.median(theta)))

    points = np.column_stack([
        phi * phi_scale,
        theta,
    ])

    truth_vec = np.atleast_2d(truth_xyzs) - center
    truth_radius = np.linalg.norm(truth_vec, axis=1)

    truth_theta = np.arccos(
        np.clip(
            truth_vec[:, 2] / truth_radius,
            -1,
            1,
        )
    )
    truth_phi = np.arctan2(
        truth_vec[:, 1],
        truth_vec[:, 0],
    )

    truth_points = np.column_stack([
        truth_phi * phi_scale,
        truth_theta,
    ])

    return points, truth_points


def hole_groups(points):
    """
    Find physical empty-region hypotheses from Voronoi vertices.

    Returns groups rather than only group centers because the internal shape
    of the strongest damaged region is useful for initializing a two-lens fit.
    """
    vor = Voronoi(points)
    candidates = vor.vertices

    hull = ConvexHull(points)
    hull_test = Delaunay(points[hull.vertices])

    candidates = candidates[
        hull_test.find_simplex(candidates) >= 0
    ]

    tree = cKDTree(points)

    nearest_distances, _ = tree.query(
        points,
        k=2,
    )
    nearest_other = nearest_distances[:, 1]

    distances, inds = tree.query(
        candidates,
        k=6,
    )

    local_spacing = np.median(
        nearest_other[inds],
        axis=1,
    )

    gap_score = (
        distances[:, 0] / local_spacing
    )

    nearby = points[inds]
    vectors = nearby - candidates[:, None, :]

    angles = np.sort(
        np.arctan2(
            vectors[:, :, 1],
            vectors[:, :, 0],
        ),
        axis=1,
    )

    circular_gaps = np.diff(
        np.concatenate(
            [
                angles,
                angles[:, :1] + 2 * np.pi,
            ],
            axis=1,
        ),
        axis=1,
    )

    max_angle_gap = np.degrees(
        circular_gaps.max(axis=1)
    )

    keep = max_angle_gap <= ANGLE_GAP_LIMIT

    candidates = candidates[keep]
    local_spacing = local_spacing[keep]
    gap_score = gap_score[keep]

    merge_radius = (
        MERGE_FACTOR * float(
            np.median(local_spacing)
        )
    )

    cluster_ids = DBSCAN(
        eps=merge_radius,
        min_samples=1,
    ).fit_predict(candidates)

    groups = []

    for cluster_id in np.unique(cluster_ids):
        use = cluster_ids == cluster_id

        vertices = candidates[use]
        scores = gap_score[use]

        center = np.average(
            vertices,
            axis=0,
            weights=scores,
        )

        groups.append({
            "score": float(scores.max()),
            "center": center,
            "vertices": vertices,
            "vertex_scores": scores,
        })

    groups.sort(
        key=lambda g: g["score"],
        reverse=True,
    )

    return groups, tree, nearest_other


def joint_pair_prediction(points):
    """
    Infer two missing positions simultaneously around the strongest damage.

    The objective is deliberately simple and interpretable:
      * each missing lens should have five surviving neighbors near spacing s;
      * the two missing lenses should be separated by about s;
      * their midpoint should remain near the detected damaged region.

    Multiple geometric initializations are tried to avoid depending on one
    arbitrary starting orientation.
    """
    groups, tree, nearest_other = hole_groups(points)

    strongest = groups[0]
    hole_center = strongest["center"]

    _, local_inds = tree.query(
        hole_center,
        k=10,
    )

    spacing = float(
        np.median(
            nearest_other[
                np.atleast_1d(local_inds)
            ]
        )
    )

    candidate_pool = []

    # Use Voronoi geometry from high-scoring nearby damaged regions.
    for group in groups[:20]:

        if (
            np.linalg.norm(
                group["center"] - hole_center
            ) <= 2.0 * spacing
            and group["score"] >= 0.70 * strongest["score"]
        ):
            candidate_pool.extend(
                list(group["vertices"])
            )
            candidate_pool.append(
                group["center"]
            )

    # Add an initialization along the long axis of the strongest
    # Voronoi-vertex cloud when that axis can be estimated.
    top_vertices = strongest["vertices"]

    if len(top_vertices) >= 2:
        centered = (
            top_vertices
            - top_vertices.mean(axis=0)
        )

        _, _, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        long_axis = vh[0]

        candidate_pool.extend([
            hole_center
            + 0.5 * spacing * long_axis,
            hole_center
            - 0.5 * spacing * long_axis,
        ])

    # Add several orientation-neutral split initializations.
    for angle in np.linspace(
        0,
        np.pi,
        6,
        endpoint=False,
    ):
        direction = np.array([
            np.cos(angle),
            np.sin(angle),
        ])

        candidate_pool.extend([
            hole_center
            + 0.5 * spacing * direction,
            hole_center
            - 0.5 * spacing * direction,
        ])

    candidate_pool = np.asarray(candidate_pool)

    # Remove numerically duplicate initial points.
    _, unique_inds = np.unique(
        np.round(
            candidate_pool / spacing,
            3,
        ),
        axis=0,
        return_index=True,
    )

    candidate_pool = candidate_pool[
        np.sort(unique_inds)
    ]

    def residual(parameters):
        p1 = parameters[:2]
        p2 = parameters[2:]

        # With two adjacent lenses missing, each inferred lens should retain
        # five surviving immediate neighbors; the sixth is the other inferred
        # missing lens.
        d1 = np.sort(
            tree.query(p1, k=5)[0] / spacing
        )
        d2 = np.sort(
            tree.query(p2, k=5)[0] / spacing
        )

        pair_distance = (
            np.linalg.norm(p1 - p2)
            / spacing
        )

        midpoint = (p1 + p2) / 2

        values = list(d1 - 1)
        values += list(d2 - 1)

        # Pair spacing is given slightly higher weight.
        values.append(
            1.4 * (pair_distance - 1)
        )

        # Mild locality regularizer: this is not a hard assumption that the
        # Voronoi center equals the pair midpoint.
        values += list(
            0.15
            * (midpoint - hole_center)
            / spacing
        )

        # Prevent an optimizer from wandering far from the damaged region.
        for point in (p1, p2):
            radial_distance = (
                np.linalg.norm(
                    point - hole_center
                )
                / spacing
            )

            if radial_distance > 1.2:
                values.append(
                    0.5 * (
                        radial_distance - 1.2
                    )
                )
            else:
                values.append(0)

        return np.asarray(values)

    initializations = []

    for i, j in combinations(
        range(len(candidate_pool)),
        2,
    ):
        distance = (
            np.linalg.norm(
                candidate_pool[i]
                - candidate_pool[j]
            )
            / spacing
        )

        midpoint_distance = (
            np.linalg.norm(
                (
                    candidate_pool[i]
                    + candidate_pool[j]
                ) / 2
                - hole_center
            )
            / spacing
        )

        if (
            0.55 <= distance <= 1.45
            and midpoint_distance < 0.75
        ):
            initial = np.r_[
                candidate_pool[i],
                candidate_pool[j],
            ]

            initial_loss = np.mean(
                residual(initial) ** 2
            )

            initializations.append(
                (initial_loss, initial)
            )

    initializations.sort(
        key=lambda item: item[0]
    )

    # Only optimize the most promising starts.
    initializations = initializations[:12]

    lower_bound = np.r_[
        hole_center - 1.5 * spacing,
        hole_center - 1.5 * spacing,
    ]

    upper_bound = np.r_[
        hole_center + 1.5 * spacing,
        hole_center + 1.5 * spacing,
    ]

    best = None

    for _, initial in initializations:

        fit = least_squares(
            residual,
            initial,
            bounds=(
                lower_bound,
                upper_bound,
            ),
            max_nfev=300,
        )

        loss = float(
            np.mean(
                residual(fit.x) ** 2
            )
        )

        if (
            best is None
            or loss < best["loss"]
        ):
            best = {
                "loss": loss,
                "parameters": fit.x,
            }

    predictions = (
        best["parameters"]
        .reshape(2, 2)
    )

    return predictions, best["loss"]


def greedy_pair_prediction(points):
    """
    Experiment-05 baseline: detect one hole, insert it, detect again.
    """
    work = points.copy()
    predictions = []

    for _ in range(2):
        groups, _, _ = hole_groups(work)

        prediction = groups[0]["center"]

        predictions.append(prediction)

        work = np.vstack([
            work,
            prediction,
        ])

    return np.asarray(predictions)


def evaluate_pair(df, pair, method):
    label_a, label_b = pair

    truth = df[
        df["label"].isin(
            [label_a, label_b]
        )
    ]

    observed = df[
        ~df["label"].isin(
            [label_a, label_b]
        )
    ]

    points, truth_points = surface_map(
        observed[
            ["x", "y", "z"]
        ].to_numpy(float),
        truth[
            ["x", "y", "z"]
        ].to_numpy(float),
    )

    tree = cKDTree(points)

    truth_neighbor_distances, _ = tree.query(
        truth_points,
        k=6,
    )

    evaluation_spacing = float(
        truth_neighbor_distances.mean()
    )

    if method == "joint":
        predictions, fit_loss = (
            joint_pair_prediction(points)
        )
    elif method == "greedy":
        predictions = (
            greedy_pair_prediction(points)
        )
        fit_loss = np.nan
    else:
        raise ValueError(method)

    cost = np.linalg.norm(
        predictions[:, None, :]
        - truth_points[None, :, :],
        axis=2,
    )

    predicted_inds, truth_inds = (
        linear_sum_assignment(cost)
    )

    errors = (
        cost[
            predicted_inds,
            truth_inds,
        ]
        / evaluation_spacing
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "method": method,
        "error_1": float(errors[0]),
        "error_2": float(errors[1]),
        "mean_error": float(
            errors.mean()
        ),
        "max_error": float(
            errors.max()
        ),
        "fit_loss": fit_loss,
    }


def make_adjacent_pairs(df):
    working = df.copy()

    working["neighbor_list"] = (
        working["neighbors"]
        .apply(parse_neighbors)
    )

    working["n_neighbors"] = (
        working["neighbor_list"]
        .apply(len)
    )

    by_label = working.set_index("label")

    strong = []

    for _, row in working.iterrows():

        if row["n_neighbors"] != 6:
            continue

        neighbors = row["neighbor_list"]

        if all(
            n in by_label.index
            and by_label.loc[
                n,
                "n_neighbors"
            ] == 6
            for n in neighbors
        ):
            strong.append(
                int(row["label"])
            )

    strong_set = set(strong)

    pairs = set()

    for label_a in strong:
        for label_b in by_label.loc[
            label_a,
            "neighbor_list",
        ]:
            label_b = int(label_b)

            if label_b in strong_set:
                pairs.add(
                    tuple(
                        sorted(
                            (
                                label_a,
                                label_b,
                            )
                        )
                    )
                )

    return sorted(pairs)


def summarize(results, name):
    max_error = results["max_error"]

    print(f"\n{name}")
    print("-" * len(name))

    print(
        "Both <10% spacing:",
        100 * (max_error < 0.10).mean(),
        "%"
    )

    print(
        "Both <20% spacing:",
        100 * (max_error < 0.20).mean(),
        "%"
    )

    print(
        "Both <30% spacing:",
        100 * (max_error < 0.30).mean(),
        "%"
    )

    print(
        "Median worst-lens error:",
        100 * max_error.median(),
        "%"
    )

    print(
        "90th-percentile worst-lens error:",
        100 * max_error.quantile(0.90),
        "%"
    )

    print(
        "95th-percentile worst-lens error:",
        100 * max_error.quantile(0.95),
        "%"
    )


df = pd.read_csv(DATA)

pairs = make_adjacent_pairs(df)

# Exact development cohort used in Experiment 05.
development_rng = np.random.default_rng(
    DEVELOPMENT_SEED
)

development_inds = np.sort(
    development_rng.choice(
        len(pairs),
        size=min(
            COHORT_SIZE,
            len(pairs),
        ),
        replace=False,
    )
)

development_pairs = [
    pairs[i]
    for i in development_inds
]

development_set = set(
    development_pairs
)

# Independent holdout: sampled only from pairs that were not in development.
remaining_pairs = [
    pair
    for pair in pairs
    if pair not in development_set
]

holdout_rng = np.random.default_rng(
    HOLDOUT_SEED
)

holdout_inds = np.sort(
    holdout_rng.choice(
        len(remaining_pairs),
        size=min(
            COHORT_SIZE,
            len(remaining_pairs),
        ),
        replace=False,
    )
)

holdout_pairs = [
    remaining_pairs[i]
    for i in holdout_inds
]


print("Adjacent strong-interior pairs:", len(pairs))
print("Development pairs:", len(development_pairs))
print("Independent holdout pairs:", len(holdout_pairs))


# The key confirmatory comparison is on the independent holdout set.
greedy_records = [
    evaluate_pair(
        df,
        pair,
        "greedy",
    )
    for pair in holdout_pairs
]

joint_records = [
    evaluate_pair(
        df,
        pair,
        "joint",
    )
    for pair in holdout_pairs
]

greedy_results = pd.DataFrame(
    greedy_records
)

joint_results = pd.DataFrame(
    joint_records
)


summarize(
    greedy_results,
    "GREEDY BASELINE — INDEPENDENT HOLDOUT",
)

summarize(
    joint_results,
    "JOINT METHOD — INDEPENDENT HOLDOUT",
)


greedy_results.to_csv(
    OUTDIR
    / "06_holdout_greedy_results.csv",
    index=False,
)

joint_results.to_csv(
    OUTDIR
    / "06_holdout_joint_results.csv",
    index=False,
)


comparison = pd.DataFrame({
    "metric": [
        "both_lt_10pct",
        "both_lt_20pct",
        "both_lt_30pct",
        "median_worst_error",
        "p90_worst_error",
        "p95_worst_error",
    ],
    "greedy": [
        (greedy_results["max_error"] < 0.10).mean(),
        (greedy_results["max_error"] < 0.20).mean(),
        (greedy_results["max_error"] < 0.30).mean(),
        greedy_results["max_error"].median(),
        greedy_results["max_error"].quantile(0.90),
        greedy_results["max_error"].quantile(0.95),
    ],
    "joint": [
        (joint_results["max_error"] < 0.10).mean(),
        (joint_results["max_error"] < 0.20).mean(),
        (joint_results["max_error"] < 0.30).mean(),
        joint_results["max_error"].median(),
        joint_results["max_error"].quantile(0.90),
        joint_results["max_error"].quantile(0.95),
    ],
})

comparison.to_csv(
    OUTDIR
    / "06_holdout_method_comparison.csv",
    index=False,
)


print(
    "\nSaved results in:",
    OUTDIR,
)

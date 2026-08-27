"""
Experiment 10: geometric model selection for 3–5 adjacent missing lenses.

Goal
----
Improve Experiment 08 by separating the problem into two parts:

1) infer the missing-lens count from the blind hole-score trajectory;
2) reconstruct all predicted missing lenses jointly rather than greedily.

Cohorts
-------
For each true hidden count (3, 4, 5), 120 connected strong-interior patches are
generated reproducibly.

Development:
    first 30 patches/count
    - train count classifier
    - freeze joint reconstruction method

Experiment-08 diagnostic cohort:
    next 50 patches/count (indices 30:80)
    - already used previously, NOT final validation here

Fresh holdout for Experiment 10:
    final 40 patches/count (indices 80:120)
    - 120 total patches
    - evaluated after the method is frozen

Count model
-----------
A multinomial logistic-regression classifier uses the first five blind
hole-insertion scores. This is intentionally simple and interpretable.

Joint position model
--------------------
For the predicted number k, initialize from the first k blind greedy hole
positions, then optimize all k positions together. For each inferred lens,
the six nearest distances to the union of:
    - surviving observed lenses
    - other inferred lenses
are encouraged to match the local lattice spacing.

Run from ODA repository root:
    python experiments/synthetic-centres/10_geometric_model_selection_3to5.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import Voronoi, cKDTree, ConvexHull, Delaunay
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/d_mauritiana_ct_stack/ommatidial_data.csv"
OUTDIR = ROOT / "research/missing_lens/results"
OUTDIR.mkdir(parents=True, exist_ok=True)

HIDDEN_COUNTS = (3, 4, 5)
PATCHES_PER_COUNT = 120
DEVELOPMENT_PER_COUNT = 30
DIAGNOSTIC_PER_COUNT = 50
FINAL_HOLDOUT_PER_COUNT = 40

MAX_INSERTIONS = 8
COUNT_FEATURES = 5

PATCH_SEED_BASE = 20260830

MERGE_FACTOR = 0.40
ANGLE_GAP_LIMIT = 150


def parse_neighbors(text):
    return np.fromstring(
        str(text).strip("[]"),
        sep=" ",
        dtype=int,
    )


def fit_sphere(xyz):
    A = np.column_stack([2 * xyz, np.ones(len(xyz))])
    b = np.sum(xyz ** 2, axis=1)

    solution, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    center = solution[:3]
    radius = float(
        np.sqrt(
            solution[3]
            + np.dot(center, center)
        )
    )

    return center, radius


def surface_map(observed_xyz, truth_xyzs):
    center, _ = fit_sphere(observed_xyz)

    vectors = observed_xyz - center
    radii = np.linalg.norm(vectors, axis=1)

    theta = np.arccos(
        np.clip(
            vectors[:, 2] / radii,
            -1,
            1,
        )
    )

    phi = np.arctan2(
        vectors[:, 1],
        vectors[:, 0],
    )

    phi_scale = float(
        np.sin(np.median(theta))
    )

    points = np.column_stack([
        phi * phi_scale,
        theta,
    ])

    truth_vectors = (
        np.atleast_2d(truth_xyzs)
        - center
    )

    truth_radii = np.linalg.norm(
        truth_vectors,
        axis=1,
    )

    truth_theta = np.arccos(
        np.clip(
            truth_vectors[:, 2]
            / truth_radii,
            -1,
            1,
        )
    )

    truth_phi = np.arctan2(
        truth_vectors[:, 1],
        truth_vectors[:, 0],
    )

    truth_points = np.column_stack([
        truth_phi * phi_scale,
        truth_theta,
    ])

    return points, truth_points


def strongest_physical_hole(points):
    voronoi = Voronoi(points)
    candidates = voronoi.vertices

    hull = ConvexHull(points)
    hull_test = Delaunay(
        points[hull.vertices]
    )

    candidates = candidates[
        hull_test.find_simplex(candidates) >= 0
    ]

    tree = cKDTree(points)

    nearest_distances, _ = tree.query(
        points,
        k=2,
    )

    nearest_other = (
        nearest_distances[:, 1]
    )

    distances, indices = tree.query(
        candidates,
        k=6,
    )

    local_spacing = np.median(
        nearest_other[indices],
        axis=1,
    )

    gap_score = (
        distances[:, 0]
        / local_spacing
    )

    nearby = points[indices]

    vectors = (
        nearby
        - candidates[:, None, :]
    )

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
                angles[:, :1]
                + 2 * np.pi,
            ],
            axis=1,
        ),
        axis=1,
    )

    max_angle_gap = np.degrees(
        circular_gaps.max(axis=1)
    )

    keep = (
        max_angle_gap
        <= ANGLE_GAP_LIMIT
    )

    candidates = candidates[keep]
    local_spacing = local_spacing[keep]
    gap_score = gap_score[keep]

    if len(candidates) == 0:
        return None

    merge_radius = (
        MERGE_FACTOR
        * float(
            np.median(local_spacing)
        )
    )

    cluster_ids = DBSCAN(
        eps=merge_radius,
        min_samples=1,
    ).fit_predict(candidates)

    best = None

    for cluster_id in np.unique(
        cluster_ids
    ):

        use = (
            cluster_ids
            == cluster_id
        )

        vertices = candidates[use]
        scores = gap_score[use]

        center = np.average(
            vertices,
            axis=0,
            weights=scores,
        )

        score = float(scores.max())

        if (
            best is None
            or score > best["score"]
        ):
            best = {
                "center": center,
                "score": score,
            }

    return best


def insertion_sequence(df, hidden_labels):
    truth = df[
        df["label"].isin(hidden_labels)
    ]

    observed = df[
        ~df["label"].isin(hidden_labels)
    ]

    points, truth_points = surface_map(
        observed[
            ["x", "y", "z"]
        ].to_numpy(float),

        truth[
            ["x", "y", "z"]
        ].to_numpy(float),
    )

    evaluation_tree = cKDTree(points)

    truth_neighbor_distances, _ = (
        evaluation_tree.query(
            truth_points,
            k=6,
        )
    )

    evaluation_spacing = float(
        truth_neighbor_distances.mean()
    )

    work = points.copy()

    predictions = []
    scores = []

    for _ in range(MAX_INSERTIONS):

        hole = strongest_physical_hole(
            work
        )

        if hole is None:
            break

        predictions.append(
            hole["center"]
        )

        scores.append(
            hole["score"]
        )

        work = np.vstack([
            work,
            hole["center"],
        ])

    return {
        "truth_points": truth_points,
        "observed_points": points,
        "greedy_predictions": np.asarray(predictions),
        "scores": np.asarray(scores),
        "spacing": evaluation_spacing,
    }


def build_strong_graph(df):
    working = df.copy()

    working["neighbor_list"] = (
        working["neighbors"]
        .apply(parse_neighbors)
    )

    working["n_neighbors"] = (
        working["neighbor_list"]
        .apply(len)
    )

    by_label = (
        working.set_index("label")
    )

    strong = []

    for _, row in working.iterrows():

        if row["n_neighbors"] != 6:
            continue

        neighbors = row[
            "neighbor_list"
        ]

        if all(
            neighbor in by_label.index
            and by_label.loc[
                neighbor,
                "n_neighbors",
            ] == 6
            for neighbor in neighbors
        ):
            strong.append(
                int(row["label"])
            )

    strong_set = set(strong)

    graph = {
        label: [
            int(neighbor)
            for neighbor
            in by_label.loc[
                label,
                "neighbor_list",
            ]
            if int(neighbor)
            in strong_set
        ]
        for label in strong
    }

    return graph


def sample_connected_patches(
    graph,
    size,
    n_patches,
    seed,
):
    rng = np.random.default_rng(seed)

    nodes = np.array(
        sorted(graph)
    )

    patches = set()

    attempts = 0

    while (
        len(patches) < n_patches
        and attempts < 200000
    ):
        attempts += 1

        seed_node = int(
            rng.choice(nodes)
        )

        patch = {seed_node}

        while len(patch) < size:

            frontier = set()

            for label in patch:

                frontier.update(
                    neighbor
                    for neighbor
                    in graph[label]
                    if neighbor
                    not in patch
                )

            if not frontier:
                break

            patch.add(
                int(
                    rng.choice(
                        sorted(frontier)
                    )
                )
            )

        if len(patch) == size:

            patches.add(
                tuple(sorted(patch))
            )

    if len(patches) < n_patches:

        raise RuntimeError(
            f"Could only sample "
            f"{len(patches)} patches "
            f"of size {size}."
        )

    return sorted(patches)


def score_features(sequence):
    scores = sequence["scores"]

    if len(scores) < COUNT_FEATURES:

        raise RuntimeError(
            "Not enough insertion scores."
        )

    return scores[
        :COUNT_FEATURES
    ]


def joint_optimize(
    points,
    greedy_predictions,
    predicted_count,
):
    """
    Jointly optimize all k inferred missing-lens positions.
    """
    k = int(predicted_count)

    initial = np.asarray(
        greedy_predictions[:k],
        dtype=float,
    )

    if len(initial) != k:

        raise RuntimeError(
            "Not enough greedy initial positions."
        )

    tree = cKDTree(points)

    damage_center = initial.mean(axis=0)

    _, local_indices = tree.query(
        damage_center,
        k=20,
    )

    local_points = points[
        np.atleast_1d(
            local_indices
        )
    ]

    nearest_distances, _ = (
        tree.query(
            local_points,
            k=2,
        )
    )

    spacing = float(
        np.median(
            nearest_distances[:, 1]
        )
    )

    def residual(parameters):
        inferred = (
            parameters.reshape(k, 2)
        )

        values = []

        for index, point in enumerate(
            inferred
        ):
            observed_distances = (
                tree.query(
                    point,
                    k=8,
                )[0]
            )

            other_inferred = np.delete(
                inferred,
                index,
                axis=0,
            )

            inferred_distances = (
                np.linalg.norm(
                    other_inferred
                    - point,
                    axis=1,
                )
            )

            all_distances = np.r_[
                observed_distances,
                inferred_distances,
            ]

            nearest_six = (
                np.sort(all_distances)[:6]
                / spacing
            )

            values.extend(
                nearest_six - 1
            )

            radial_distance = (
                np.linalg.norm(
                    point
                    - damage_center
                )
                / spacing
            )

            if radial_distance > 2.0:
                values.append(
                    0.5
                    * (
                        radial_distance
                        - 2.0
                    )
                )
            else:
                values.append(0)

        # Prevent inferred lenses from collapsing onto one another.
        for i in range(k):
            for j in range(i + 1, k):

                pair_distance = (
                    np.linalg.norm(
                        inferred[i]
                        - inferred[j]
                    )
                    / spacing
                )

                values.append(
                    1.5
                    * max(
                        0,
                        0.75
                        - pair_distance,
                    )
                )

        # Mild centroid locality constraint.
        values.extend(
            0.05
            * (
                inferred.mean(axis=0)
                - damage_center
            )
            / spacing
        )

        return np.asarray(values)

    # Bounds must contain every greedy initialization. On some harder
    # multi-hole cases a late greedy candidate can lie >3 local spacings
    # from the mean of the initial guesses, which made scipy reject x0 as
    # infeasible before optimization even started. Expand the box only as
    # much as needed to contain the full initial configuration.
    max_initial_offset = float(
        np.max(
            np.abs(
                initial
                - damage_center
            )
        )
    )

    half_width = max(
        3.0 * spacing,
        max_initial_offset
        + 0.5 * spacing,
    )

    lower_bound = np.tile(
        damage_center
        - half_width,
        k,
    )

    upper_bound = np.tile(
        damage_center
        + half_width,
        k,
    )

    starts = [
        initial.ravel()
    ]

    # Deterministic small perturbations avoid dependence on one greedy start.
    rng = np.random.default_rng(0)

    for _ in range(0):

        perturbed = (
            initial
            + rng.normal(
                0,
                0.10 * spacing,
                size=initial.shape,
            )
        )

        starts.append(
            perturbed.ravel()
        )

    best = None

    for start in starts:

        # Small perturbations can land exactly outside a bound by floating-
        # point roundoff. Clip only for numerical feasibility; the dynamic
        # bounds above already contain the unperturbed starting geometry.
        start = np.clip(
            start,
            lower_bound + 1e-12,
            upper_bound - 1e-12,
        )

        fit = least_squares(
            residual,
            start,
            bounds=(
                lower_bound,
                upper_bound,
            ),
            max_nfev=120,
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
                "positions": (
                    fit.x.reshape(k, 2)
                ),
            }

    return best


def evaluate_sequence(
    sequence,
    true_count,
    predicted_count,
):
    fit = joint_optimize(
        points=sequence[
            "observed_points"
        ],
        greedy_predictions=sequence[
            "greedy_predictions"
        ],
        predicted_count=predicted_count,
    )

    predictions = fit[
        "positions"
    ]

    truth_points = sequence[
        "truth_points"
    ]

    spacing = sequence[
        "spacing"
    ]

    cost = np.linalg.norm(
        predictions[:, None, :]
        - truth_points[None, :, :],
        axis=2,
    ) / spacing

    predicted_indices, truth_indices = (
        linear_sum_assignment(cost)
    )

    matched_errors = cost[
        predicted_indices,
        truth_indices,
    ]

    count_correct = (
        predicted_count
        == true_count
    )

    return {
        "true_count": true_count,
        "predicted_count": int(
            predicted_count
        ),
        "count_correct": (
            count_correct
        ),
        "truth_recovered_lt10": (
            int(
                (
                    matched_errors
                    < 0.10
                ).sum()
            )
            / true_count
        ),
        "truth_recovered_lt20": (
            int(
                (
                    matched_errors
                    < 0.20
                ).sum()
            )
            / true_count
        ),
        "truth_recovered_lt30": (
            int(
                (
                    matched_errors
                    < 0.30
                ).sum()
            )
            / true_count
        ),
        "worst_error_if_exact": (
            float(
                matched_errors.max()
            )
            if count_correct
            else np.nan
        ),
        "fit_loss": fit["loss"],
    }




# ============================================================
# Experiment 10: geometric model selection vs score classifier
# ============================================================

FRESH_SEED_BASE = 20261000
FRESH_PER_COUNT = 40
LAMBDA_GRID = np.arange(0.0, 0.0201, 0.0005)


def evaluate_positions(sequence, positions, true_count, predicted_count):
    truth_points = sequence["truth_points"]
    spacing = sequence["spacing"]

    cost = np.linalg.norm(
        positions[:, None, :] - truth_points[None, :, :], axis=2
    ) / spacing

    pred_idx, truth_idx = linear_sum_assignment(cost)
    matched_errors = cost[pred_idx, truth_idx]

    return {
        "true_count": int(true_count),
        "predicted_count": int(predicted_count),
        "count_correct": int(predicted_count) == int(true_count),
        "truth_recovered_lt10": float((matched_errors < 0.10).sum() / true_count),
        "truth_recovered_lt20": float((matched_errors < 0.20).sum() / true_count),
        "truth_recovered_lt30": float((matched_errors < 0.30).sum() / true_count),
        "worst_error_if_exact": (
            float(matched_errors.max()) if predicted_count == true_count else np.nan
        ),
    }


def candidate_fits(sequence):
    out = {}
    for k in HIDDEN_COUNTS:
        fit = joint_optimize(
            points=sequence["observed_points"],
            greedy_predictions=sequence["greedy_predictions"],
            predicted_count=k,
        )
        out[int(k)] = fit
    return out


# Load data and recreate the original development patches only.
df = pd.read_csv(DATA)
graph = build_strong_graph(df)

old_patches = {
    hidden_count: sample_connected_patches(
        graph=graph,
        size=hidden_count,
        n_patches=PATCHES_PER_COUNT,
        seed=PATCH_SEED_BASE + hidden_count,
    )
    for hidden_count in HIDDEN_COUNTS
}

print("Building Experiment-10 development cohort...")

dev_rows = []
dev_score_features = []
dev_truth = []

for hidden_count in HIDDEN_COUNTS:
    for patch in old_patches[hidden_count][:DEVELOPMENT_PER_COUNT]:
        sequence = insertion_sequence(df, patch)
        fits = candidate_fits(sequence)

        row = {"true_count": hidden_count, "labels": " ".join(map(str, patch))}
        for k in HIDDEN_COUNTS:
            row[f"loss_k{k}"] = fits[k]["loss"]
        dev_rows.append(row)

        dev_score_features.append(score_features(sequence))
        dev_truth.append(hidden_count)


dev = pd.DataFrame(dev_rows)
dev_truth = np.asarray(dev_truth)
dev_score_features = np.asarray(dev_score_features)

# Freeze the geometric complexity penalty using development data only.
penalty_records = []
for lam in LAMBDA_GRID:
    predictions = []
    for _, row in dev.iterrows():
        scored = [
            (row[f"loss_k{k}"] + lam * (k - 3), k)
            for k in HIDDEN_COUNTS
        ]
        predictions.append(min(scored)[1])
    predictions = np.asarray(predictions)
    penalty_records.append({
        "lambda": float(lam),
        "accuracy": float((predictions == dev_truth).mean()),
        "mae": float(np.abs(predictions - dev_truth).mean()),
    })

penalty_df = pd.DataFrame(penalty_records).sort_values(
    ["accuracy", "mae", "lambda"], ascending=[False, True, True]
).reset_index(drop=True)

chosen_lambda = float(penalty_df.iloc[0]["lambda"])

# Train the Experiment-09 score classifier on the SAME development cohort,
# solely as a head-to-head comparator on the new fresh holdout.
feature_names = [f"score_{i}" for i in range(1, COUNT_FEATURES + 1)]
count_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
count_model.fit(dev_score_features, dev_truth)

print("\nDEVELOPMENT")
print("-----------")
print("Chosen geometric complexity penalty lambda:", chosen_lambda)
print("Geometric model-selection development accuracy:",
      100 * penalty_df.iloc[0]["accuracy"], "%")
print("Score-classifier development accuracy:",
      100 * (count_model.predict(dev_score_features) == dev_truth).mean(), "%")

# Create entirely new connected patches and explicitly exclude all 120 patches
# per count used in Experiments 08/09.
fresh_patches = {}
for hidden_count in HIDDEN_COUNTS:
    used = set(old_patches[hidden_count])
    candidates = sample_connected_patches(
        graph=graph,
        size=hidden_count,
        n_patches=220,
        seed=FRESH_SEED_BASE + hidden_count,
    )
    fresh = [patch for patch in candidates if patch not in used]
    if len(fresh) < FRESH_PER_COUNT:
        raise RuntimeError("Not enough fresh non-overlapping patches.")
    fresh_patches[hidden_count] = fresh[:FRESH_PER_COUNT]

print("\nEvaluating brand-new Experiment-10 holdout...")

geo_records = []
clf_records = []

for hidden_count in HIDDEN_COUNTS:
    for patch in fresh_patches[hidden_count]:
        sequence = insertion_sequence(df, patch)
        fits = candidate_fits(sequence)

        # Geometric model selection: choose k by joint fit + frozen complexity.
        geo_scored = [
            (fits[k]["loss"] + chosen_lambda * (k - 3), k)
            for k in HIDDEN_COUNTS
        ]
        geo_k = min(geo_scored)[1]
        geo_eval = evaluate_positions(
            sequence, fits[geo_k]["positions"], hidden_count, geo_k
        )
        geo_eval["labels"] = " ".join(map(str, patch))
        geo_eval["selected_loss"] = float(fits[geo_k]["loss"])
        geo_records.append(geo_eval)

        # Experiment-09-style score classifier, same cohort and same candidate
        # position fits, to isolate count-selection performance.
        clf_k = int(count_model.predict(score_features(sequence).reshape(1, -1))[0])
        clf_eval = evaluate_positions(
            sequence, fits[clf_k]["positions"], hidden_count, clf_k
        )
        clf_eval["labels"] = " ".join(map(str, patch))
        clf_eval["selected_loss"] = float(fits[clf_k]["loss"])
        clf_records.append(clf_eval)


geo = pd.DataFrame(geo_records)
clf = pd.DataFrame(clf_records)

geo.to_csv(OUTDIR / "10_geometric_model_selection_holdout.csv", index=False)
clf.to_csv(OUTDIR / "10_score_classifier_comparator_holdout.csv", index=False)
penalty_df.to_csv(OUTDIR / "10_development_penalty_scan.csv", index=False)


def report(results, title):
    print(f"\n{title}")
    print("-" * len(title))
    print("N patches:", len(results))
    print("Exact count accuracy:", 100 * results["count_correct"].mean(), "%")
    print("Truth recovered within 10% spacing:",
          100 * results["truth_recovered_lt10"].mean(), "%")
    print("Truth recovered within 20% spacing:",
          100 * results["truth_recovered_lt20"].mean(), "%")
    print("Truth recovered within 30% spacing:",
          100 * results["truth_recovered_lt30"].mean(), "%")
    print("Count confusion matrix (rows=true 3/4/5; cols=pred 3/4/5):")
    print(confusion_matrix(
        results["true_count"], results["predicted_count"], labels=[3,4,5]
    ))
    for hidden_count in HIDDEN_COUNTS:
        subset = results[results["true_count"] == hidden_count]
        exact = subset[subset["count_correct"]]["worst_error_if_exact"].dropna()
        print(f"  true {hidden_count}: count accuracy =",
              100 * subset["count_correct"].mean(), "%")
        if len(exact):
            print("    median worst-lens error (exact-count) =",
                  100 * exact.median(), "%")
            print("    90th percentile worst-lens error =",
                  100 * exact.quantile(0.90), "%")


report(geo, "GEOMETRIC MODEL SELECTION — BRAND-NEW HOLDOUT")
report(clf, "SCORE CLASSIFIER COMPARATOR — SAME HOLDOUT")

print("\nSaved results in:", OUTDIR)

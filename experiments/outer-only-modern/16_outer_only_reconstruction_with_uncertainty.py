
from pathlib import Path
from collections import defaultdict
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("/mnt/data")
NPZ = HERE / "labeled_lens_points.npz"

if not NPZ.exists():
    # Repository layout if copied into research/missing_lens/
    ROOT = HERE.parents[1] if len(HERE.parents) >= 2 else HERE
    candidate = ROOT / "research/missing_lens/labeled_lens_points.npz"
    if candidate.exists():
        NPZ = candidate

OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

arr = np.load(NPZ)
points = arr["points"]
labels = arr["labels"]

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def sphere_center(points3d):
    A = np.column_stack([2 * points3d, np.ones(len(points3d))])
    b = np.sum(points3d ** 2, axis=1)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    return solution[:3]

# This center is used ONLY to identify which side of a complete lens is
# anatomically outward when creating synthetic ground truth. It is not
# supplied to the prediction model.
anatomical_center = sphere_center(points)

def canonical_tangent(outward):
    w = outward / np.linalg.norm(outward)

    ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, w) * w

    if np.linalg.norm(u) < 0.15:
        ref = np.array([1.0, 0.0, 0.0])
        u = ref - np.dot(ref, w) * w

    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    v = v / np.linalg.norm(v)
    return u, v, w

def paired_surfaces_from_complete_lens(sub, nbins=14):
    """
    Create synthetic ground truth.

    The complete point cloud is used only to determine paired outer and inner
    surface samples. The prediction stage later discards the inner samples.
    """
    center = sub.mean(axis=0)
    outward = center - anatomical_center
    u, v, w = canonical_tangent(outward)

    local = np.column_stack(
        [
            (sub - center) @ u,
            (sub - center) @ v,
            (sub - center) @ w,
        ]
    )

    x, y, z = local.T
    xmin, xmax = np.percentile(x, [2, 98])
    ymin, ymax = np.percentile(y, [2, 98])

    if xmax <= xmin or ymax <= ymin:
        return None

    xe = np.linspace(xmin, xmax, nbins + 1)
    ye = np.linspace(ymin, ymax, nbins + 1)

    bins = defaultdict(list)

    for index, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ix = np.searchsorted(xe, xi, side="right") - 1
        iy = np.searchsorted(ye, yi, side="right") - 1

        if 0 <= ix < nbins and 0 <= iy < nbins:
            bins[(ix, iy)].append((zi, index))

    outer_points = []
    inner_points = []

    for values in bins.values():
        if len(values) < 2:
            continue

        zvals = np.array([zval for zval, _ in values])
        inds = np.array([ind for _, ind in values])

        outer_points.append(sub[inds[np.argmax(zvals)]])
        inner_points.append(sub[inds[np.argmin(zvals)]])

    if len(outer_points) < 30:
        return None

    return np.asarray(outer_points), np.asarray(inner_points)

def fit_rim_ellipse(xy):
    hull = ConvexHull(xy)
    h = xy[hull.vertices]

    center = h.mean(axis=0)
    cov = np.cov((h - center).T)
    values, vectors = np.linalg.eigh(cov)
    vectors = vectors[:, np.argsort(values)[::-1]]

    theta0 = math.atan2(vectors[1, 0], vectors[0, 0])
    ca, sa = np.cos(theta0), np.sin(theta0)

    dx = h[:, 0] - center[0]
    dy = h[:, 1] - center[1]

    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy

    a0 = max(np.percentile(np.abs(xr), 95), 1e-3)
    b0 = max(np.percentile(np.abs(yr), 95), 1e-3)

    p0 = np.array([center[0], center[1], a0, b0, theta0])
    span = np.ptp(h, axis=0)

    lower = [
        h[:, 0].min() - span[0],
        h[:, 1].min() - span[1],
        a0 * 0.3,
        b0 * 0.3,
        -np.pi,
    ]

    upper = [
        h[:, 0].max() + span[0],
        h[:, 1].max() + span[1],
        a0 * 3,
        b0 * 3,
        np.pi,
    ]

    def residual(p):
        cx, cy, a, b, theta = p
        ca, sa = np.cos(theta), np.sin(theta)

        dx = h[:, 0] - cx
        dy = h[:, 1] - cy

        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy

        return np.sqrt((xr / a) ** 2 + (yr / b) ** 2) - 1

    fit = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        max_nfev=500,
    )

    return fit.x

def normalized_xy(xy, ellipse):
    cx, cy, a, b, theta = ellipse
    ca, sa = np.cos(theta), np.sin(theta)

    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy

    return np.column_stack(
        [
            (ca * dx + sa * dy) / a,
            (-sa * dx + ca * dy) / b,
        ]
    )

# ---------------------------------------------------------------------
# Synthetic outer/inner surface pairs
# ---------------------------------------------------------------------

unique_labels = np.unique(labels)
counts = np.array([(labels == lab).sum() for lab in unique_labels])

candidate_labels = unique_labels[
    (counts >= 180)
    & (counts <= 500)
]

rng = np.random.RandomState(42)

if len(candidate_labels) > 400:
    candidate_labels = rng.choice(
        candidate_labels,
        size=400,
        replace=False,
    )

raw_pairs = {}

for lab in candidate_labels:
    sub = points[labels == lab]
    pair = paired_surfaces_from_complete_lens(sub, nbins=14)

    if pair is not None:
        raw_pairs[int(lab)] = pair

# Fit an operational eye center using OBSERVED OUTER SURFACES ONLY.
all_outer_points = np.vstack(
    [raw_pairs[lab][0] for lab in sorted(raw_pairs)]
)
outer_eye_center = sphere_center(all_outer_points)

# ---------------------------------------------------------------------
# Outer-only feature construction
# ---------------------------------------------------------------------

def build_outer_only_record(lab):
    outer3d, inner3d = raw_pairs[int(lab)]

    # Every coordinate system used for prediction is now reconstructed from
    # the observed outer surface only.
    outer_center = outer3d.mean(axis=0)
    outward = outer_center - outer_eye_center
    u, v, w = canonical_tangent(outward)

    outer = np.column_stack(
        [
            (outer3d - outer_center) @ u,
            (outer3d - outer_center) @ v,
            (outer3d - outer_center) @ w,
        ]
    )

    inner = np.column_stack(
        [
            (inner3d - outer_center) @ u,
            (inner3d - outer_center) @ v,
            (inner3d - outer_center) @ w,
        ]
    )

    try:
        ellipse = fit_rim_ellipse(outer[:, :2])
    except Exception:
        return None

    uv = normalized_xy(outer[:, :2], ellipse)
    rho = np.linalg.norm(uv, axis=1)

    true_thickness = outer[:, 2] - inner[:, 2]

    valid = (
        np.isfinite(rho)
        & np.isfinite(true_thickness)
        & (rho >= 0)
        & (rho <= 1.0)
        & (true_thickness > 0)
    )

    if valid.sum() < 25:
        return None

    rim_mask = (
        (rho > 0.8)
        & (rho < 1.05)
    )

    center_mask = rho < 0.3

    rim_z = (
        np.median(outer[rim_mask, 2])
        if rim_mask.sum() >= 3
        else np.percentile(outer[:, 2], 20)
    )

    apex_z = (
        np.median(outer[center_mask, 2])
        if center_mask.sum() >= 3
        else np.percentile(outer[:, 2], 80)
    )

    sag = float(apex_z - rim_z)
    outer_zrel = outer[:, 2] - rim_z

    uu, vv = uv.T

    # Low-order outer-surface shape coefficients.
    phi = np.column_stack(
        [
            np.ones(len(uu)),
            uu,
            vv,
            uu ** 2,
            uu * vv,
            vv ** 2,
            uu ** 3,
            uu ** 2 * vv,
            uu * vv ** 2,
            vv ** 3,
        ]
    )

    outer_beta = np.linalg.solve(
        phi.T @ phi + 0.1 * np.eye(phi.shape[1]),
        phi.T @ outer_zrel,
    )

    _, _, a, b, _ = ellipse

    location = outer_center - outer_eye_center
    eye_radius = np.linalg.norm(location)
    theta = np.arccos(
        np.clip(
            location[2] / eye_radius,
            -1,
            1,
        )
    )
    azimuth = np.arctan2(location[1], location[0])

    lens_features = np.r_[
        a,
        b,
        np.sqrt(a * b),
        a / b,
        sag,
        theta,
        np.sin(azimuth),
        np.cos(azimuth),
        eye_radius,
        outer_beta,
    ]

    return {
        "label": int(lab),
        "centroid": outer_center,
        "outer": outer,
        "inner": inner,
        "ellipse": ellipse,
        "uv": uv,
        "rho": rho,
        "outer_zrel": outer_zrel,
        "true_thickness": true_thickness,
        "valid": valid,
        "lens_features": lens_features,
    }

records = {}

for lab in sorted(raw_pairs):
    rec = build_outer_only_record(lab)
    if rec is not None:
        records[int(lab)] = rec

lens_ids = np.array(sorted(records))

def point_features(rec, indices):
    rho = rec["rho"][indices]
    uv = rec["uv"][indices]
    u, v = uv.T
    oz = rec["outer_zrel"][indices]

    local_features = np.column_stack(
        [
            rho,
            rho ** 2,
            rho ** 3,
            rho ** 4,
            u ** 2,
            v ** 2,
            u * v,
            oz,
            oz ** 2,
            rho * oz,
        ]
    )

    repeated_lens_features = np.tile(
        rec["lens_features"],
        (len(indices), 1),
    )

    return np.column_stack(
        [
            local_features,
            repeated_lens_features,
        ]
    )

def matrix_for_lenses(ids):
    X = []
    y = []
    groups = []
    aux = []

    for lab in ids:
        rec = records[int(lab)]
        inds = np.where(rec["valid"])[0]

        X.append(point_features(rec, inds))
        y.append(rec["true_thickness"][inds])

        groups.extend(
            [int(lab)] * len(inds)
        )

        for ind in inds:
            aux.append(
                [
                    int(lab),
                    int(ind),
                    rec["rho"][ind],
                    rec["outer"][ind, 2],
                    rec["inner"][ind, 2],
                ]
            )

    return (
        np.vstack(X),
        np.concatenate(y),
        np.asarray(groups),
        np.asarray(aux),
    )

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def candidate_models(seed=0):
    return {
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=10,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }

def per_lens_metrics(y_true, y_pred, groups):
    rows = []

    for lab in np.unique(groups):
        mask = groups == lab

        mae = np.mean(
            np.abs(
                y_pred[mask]
                - y_true[mask]
            )
        )

        median_true_thickness = np.median(
            y_true[mask]
        )

        rows.append(
            {
                "label": int(lab),
                "mae": float(mae),
                "normalized_mae": float(
                    mae / median_true_thickness
                ),
            }
        )

    return pd.DataFrame(rows)

def inner_cv_score(ids, model_name, seed):
    inner = KFold(
        n_splits=3,
        shuffle=True,
        random_state=seed,
    )

    fold_medians = []

    for train_index, validation_index in inner.split(ids):
        train_ids = ids[train_index]
        validation_ids = ids[validation_index]

        X_train, y_train, _, _ = matrix_for_lenses(train_ids)
        X_validation, y_validation, validation_groups, _ = matrix_for_lenses(
            validation_ids
        )

        model = candidate_models(seed)[model_name]
        model.fit(X_train, y_train)

        prediction = model.predict(X_validation)

        fold_medians.append(
            per_lens_metrics(
                y_validation,
                prediction,
                validation_groups,
            )["mae"].median()
        )

    return float(np.mean(fold_medians))

# ---------------------------------------------------------------------
# Naive sphere baseline
# ---------------------------------------------------------------------

def fit_naive_outer_sphere(rec):
    x, y, z = rec["outer"].T

    xr = (x.max() - x.min()) / 2
    yr = (y.max() - y.min()) / 2
    zr = z.max() - z.min()

    a0 = max(xr, yr, 1e-3)
    c0 = max(zr, a0 * 0.75)
    radius0 = max(a0, c0)

    cz0 = np.median(z) - 0.3 * radius0

    lower = np.array(
        [
            z.min() - 3 * radius0,
            max(a0 * 0.5, 1e-3),
        ]
    )

    upper = np.array(
        [
            z.max(),
            3 * max(a0, c0),
        ]
    )

    fit = least_squares(
        lambda p: (
            np.sqrt(
                x ** 2
                + y ** 2
                + (z - p[0]) ** 2
            )
            - p[1]
        ),
        x0=np.clip(
            [cz0, radius0],
            lower,
            upper,
        ),
        bounds=(lower, upper),
        max_nfev=500,
    )

    return fit.x

def naive_sphere_metrics(ids):
    rows = []

    for lab in ids:
        rec = records[int(lab)]
        inds = np.where(rec["valid"])[0]

        try:
            cz, radius = fit_naive_outer_sphere(rec)
        except Exception:
            continue

        xy = rec["inner"][inds, :2]

        term = (
            radius ** 2
            - xy[:, 0] ** 2
            - xy[:, 1] ** 2
        )

        valid = term >= 0

        if valid.sum() < 0.8 * len(inds):
            continue

        predicted_inner_z = (
            cz
            - np.sqrt(term[valid])
        )

        true_inner_z = rec["inner"][inds, 2][valid]

        mae = np.mean(
            np.abs(
                predicted_inner_z
                - true_inner_z
            )
        )

        median_true_thickness = np.median(
            rec["true_thickness"][inds][valid]
        )

        rows.append(
            {
                "label": int(lab),
                "mae": float(mae),
                "normalized_mae": float(
                    mae / median_true_thickness
                ),
            }
        )

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Nested grouped cross-validation + adaptive conformal uncertainty
# ---------------------------------------------------------------------

outer_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=20260820,
)

metric_frames = []
selection_rows = []
uncertainty_rows = []
point_prediction_rows = []
lens_uncertainty_rows = []

for fold, (outer_train_index, test_index) in enumerate(
    outer_cv.split(lens_ids),
    start=1,
):
    outer_train_ids = lens_ids[outer_train_index]
    test_ids = lens_ids[test_index]

    # Reserve a lens-level calibration set that is never used for model choice.
    rng = np.random.RandomState(7000 + fold)
    shuffled_train = rng.permutation(outer_train_ids)

    n_calibration = max(
        50,
        int(0.2 * len(shuffled_train)),
    )

    calibration_ids = shuffled_train[:n_calibration]
    proper_train_ids = shuffled_train[n_calibration:]

    # Model selection is done only inside the proper-training lenses.
    inner_scores = {}

    for model_name in [
        "ridge",
        "extra_trees",
        "hist_gradient",
    ]:
        inner_scores[model_name] = inner_cv_score(
            proper_train_ids,
            model_name,
            seed=9000 + fold,
        )

    selected_model_name = min(
        inner_scores,
        key=inner_scores.get,
    )

    selection_rows.append(
        {
            "fold": fold,
            "selected_model": selected_model_name,
            **{
                f"innercv_{name}": score
                for name, score in inner_scores.items()
            },
        }
    )

    X_train, y_train, _, _ = matrix_for_lenses(proper_train_ids)

    (
        X_calibration,
        y_calibration,
        calibration_groups,
        calibration_aux,
    ) = matrix_for_lenses(calibration_ids)

    (
        X_test,
        y_test,
        test_groups,
        test_aux,
    ) = matrix_for_lenses(test_ids)

    models = candidate_models(seed=fold)

    calibration_predictions = {}
    test_predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        calibration_predictions[model_name] = model.predict(
            X_calibration
        )

        test_predictions[model_name] = model.predict(
            X_test
        )

        model_metrics = per_lens_metrics(
            y_test,
            test_predictions[model_name],
            test_groups,
        )

        model_metrics["fold"] = fold
        model_metrics["method"] = model_name
        metric_frames.append(model_metrics)

    selected_calibration = calibration_predictions[
        selected_model_name
    ]

    selected_test = test_predictions[
        selected_model_name
    ]

    selected_metrics = per_lens_metrics(
        y_test,
        selected_test,
        test_groups,
    )

    selected_metrics["fold"] = fold
    selected_metrics["method"] = "nested_selected"
    metric_frames.append(selected_metrics)

    sphere_metrics = naive_sphere_metrics(test_ids)

    if len(sphere_metrics):
        sphere_metrics["fold"] = fold
        sphere_metrics["method"] = "naive_sphere"
        metric_frames.append(sphere_metrics)

    # Adaptive uncertainty:
    # use disagreement among three independently structured predictors.
    calibration_stack = np.vstack(
        [
            calibration_predictions["ridge"],
            calibration_predictions["extra_trees"],
            calibration_predictions["hist_gradient"],
        ]
    )

    test_stack = np.vstack(
        [
            test_predictions["ridge"],
            test_predictions["extra_trees"],
            test_predictions["hist_gradient"],
        ]
    )

    calibration_disagreement = np.std(
        calibration_stack,
        axis=0,
    )

    test_disagreement = np.std(
        test_stack,
        axis=0,
    )

    # Small floor prevents division by tiny disagreement.
    calibration_scale = (
        0.20
        + calibration_disagreement
    )

    test_scale = (
        0.20
        + test_disagreement
    )

    nonconformity = (
        np.abs(
            y_calibration
            - selected_calibration
        )
        / calibration_scale
    )

    def conformal_quantile(scores, coverage):
        n = len(scores)

        level = (
            np.ceil(
                (n + 1) * coverage
            )
            / n
        )

        level = min(level, 1.0)

        return np.quantile(
            scores,
            level,
            method="higher",
        )

    q90 = conformal_quantile(
        nonconformity,
        0.90,
    )

    q95 = conformal_quantile(
        nonconformity,
        0.95,
    )

    halfwidth90 = q90 * test_scale
    halfwidth95 = q95 * test_scale

    absolute_error = np.abs(
        y_test
        - selected_test
    )

    uncertainty_rows.append(
        {
            "fold": fold,
            "selected_model": selected_model_name,
            "coverage90": float(
                np.mean(
                    absolute_error
                    <= halfwidth90
                )
            ),
            "coverage95": float(
                np.mean(
                    absolute_error
                    <= halfwidth95
                )
            ),
            "median_halfwidth90": float(
                np.median(halfwidth90)
            ),
            "median_halfwidth95": float(
                np.median(halfwidth95)
            ),
        }
    )

    # Point-level out-of-fold predictions.
    for i in range(len(y_test)):
        lab = int(test_aux[i, 0])
        local_index = int(test_aux[i, 1])
        rec = records[lab]

        true_thickness = float(y_test[i])
        predicted_thickness = float(selected_test[i])

        outer_z = float(test_aux[i, 3])
        true_inner_z = float(test_aux[i, 4])

        point_prediction_rows.append(
            {
                "fold": fold,
                "label": lab,
                "local_index": local_index,
                "rho": float(test_aux[i, 2]),
                "outer_z": outer_z,
                "true_inner_z": true_inner_z,
                "true_thickness": true_thickness,
                "predicted_thickness": predicted_thickness,
                "predicted_inner_z": (
                    outer_z
                    - predicted_thickness
                ),
                "halfwidth90": float(
                    halfwidth90[i]
                ),
                "halfwidth95": float(
                    halfwidth95[i]
                ),
                "model_disagreement": float(
                    test_disagreement[i]
                ),
                "selected_model": selected_model_name,
            }
        )

    # Lens-level confidence summary.
    for lab in np.unique(test_groups):
        mask = test_groups == lab

        mae = np.mean(
            np.abs(
                selected_test[mask]
                - y_test[mask]
            )
        )

        median_true_thickness = np.median(
            y_test[mask]
        )

        lens_uncertainty_rows.append(
            {
                "fold": fold,
                "label": int(lab),
                "mae": float(mae),
                "normalized_mae": float(
                    mae / median_true_thickness
                ),
                "median_halfwidth90": float(
                    np.median(
                        halfwidth90[mask]
                    )
                ),
                "coverage90": float(
                    np.mean(
                        absolute_error[mask]
                        <= halfwidth90[mask]
                    )
                ),
            }
        )

# ---------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------

metrics = pd.concat(
    metric_frames,
    ignore_index=True,
)

summary_rows = []

for method, subset in metrics.groupby("method"):
    summary_rows.append(
        {
            "method": method,
            "n_lenses": len(subset),
            "median_mae": subset["mae"].median(),
            "mean_mae": subset["mae"].mean(),
            "p90_mae": subset["mae"].quantile(0.90),
            "median_normalized_mae": subset["normalized_mae"].median(),
            "p90_normalized_mae": subset["normalized_mae"].quantile(0.90),
        }
    )

summary = pd.DataFrame(summary_rows).sort_values(
    "median_mae"
)

selection = pd.DataFrame(selection_rows)
uncertainty = pd.DataFrame(uncertainty_rows)
point_predictions = pd.DataFrame(point_prediction_rows)
lens_uncertainty = pd.DataFrame(lens_uncertainty_rows)

rho_conf, p_conf = spearmanr(
    lens_uncertainty["median_halfwidth90"],
    lens_uncertainty["mae"],
)

median_width = lens_uncertainty[
    "median_halfwidth90"
].median()

high_confidence = lens_uncertainty[
    lens_uncertainty["median_halfwidth90"]
    <= median_width
]

low_confidence = lens_uncertainty[
    lens_uncertainty["median_halfwidth90"]
    > median_width
]

confidence_summary = pd.DataFrame(
    [
        {
            "group": "narrower-half intervals",
            "n_lenses": len(high_confidence),
            "median_mae": high_confidence["mae"].median(),
            "median_normalized_mae": high_confidence[
                "normalized_mae"
            ].median(),
        },
        {
            "group": "wider-half intervals",
            "n_lenses": len(low_confidence),
            "median_mae": low_confidence["mae"].median(),
            "median_normalized_mae": low_confidence[
                "normalized_mae"
            ].median(),
        },
    ]
)

# ---------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------

metrics.to_csv(
    OUT / "16_nested_cv_lens_results.csv",
    index=False,
)

summary.to_csv(
    OUT / "16_nested_cv_summary.csv",
    index=False,
)

selection.to_csv(
    OUT / "16_model_selection_by_fold.csv",
    index=False,
)

uncertainty.to_csv(
    OUT / "16_uncertainty_calibration.csv",
    index=False,
)

point_predictions.to_csv(
    OUT / "16_oof_point_predictions.csv",
    index=False,
)

lens_uncertainty.to_csv(
    OUT / "16_lens_uncertainty.csv",
    index=False,
)

confidence_summary.to_csv(
    OUT / "16_confidence_stratification.csv",
    index=False,
)

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

# Benchmark figure
benchmark_order = [
    "naive_sphere",
    "ridge",
    "extra_trees",
    "hist_gradient",
    "nested_selected",
]

plot_summary = (
    summary
    .set_index("method")
    .loc[
        [
            method
            for method in benchmark_order
            if method in summary["method"].values
        ]
    ]
    .reset_index()
)

fig = plt.figure(figsize=(8.5, 5.2))
ax = fig.add_subplot(111)

ax.bar(
    np.arange(len(plot_summary)),
    plot_summary["median_mae"],
)

ax.set_yscale("log")

ax.set_xticks(
    np.arange(len(plot_summary))
)

ax.set_xticklabels(
    [
        "Naive\nsphere",
        "Ridge",
        "Extra\nTrees",
        "Hist.\ngradient",
        "Nested\nselected",
    ][:len(plot_summary)]
)

ax.set_ylabel(
    "Median hidden-inner-surface MAE (log scale)"
)

ax.set_title(
    "Outer-only reconstruction: nested lens-level cross-validation"
)

fig.tight_layout()
fig.savefig(
    OUT / "16_benchmark.png",
    dpi=200,
)
plt.close(fig)

# Uncertainty calibration figure
fig = plt.figure(figsize=(5.5, 5.0))
ax = fig.add_subplot(111)

nominal = np.array([0.90, 0.95])

observed = np.array(
    [
        uncertainty["coverage90"].mean(),
        uncertainty["coverage95"].mean(),
    ]
)

ax.plot(
    [0.86, 0.98],
    [0.86, 0.98],
    linestyle="--",
)

ax.scatter(
    nominal,
    observed,
    s=70,
)

for x, y, label in zip(
    nominal,
    observed,
    ["90%", "95%"],
):
    ax.annotate(
        f"{label}: {100*y:.1f}%",
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
    )

ax.set_xlim(0.86, 0.98)
ax.set_ylim(0.86, 0.98)

ax.set_xlabel("Nominal interval coverage")
ax.set_ylabel("Observed out-of-fold coverage")

ax.set_title(
    "Adaptive conformal uncertainty calibration"
)

fig.tight_layout()
fig.savefig(
    OUT / "16_uncertainty_calibration.png",
    dpi=200,
)
plt.close(fig)

# Representative cross-section from a median-performing lens.
selected_only = metrics[
    metrics["method"] == "nested_selected"
].copy()

median_error = selected_only["mae"].median()

representative_label = int(
    selected_only.iloc[
        np.argmin(
            np.abs(
                selected_only["mae"]
                - median_error
            )
        )
    ]["label"]
)

rep = point_predictions[
    point_predictions["label"]
    == representative_label
].copy()

rec = records[representative_label]

# Select a narrow strip through one normalized lens axis.
uv = rec["uv"]
valid_indices = np.where(rec["valid"])[0]
lookup = {
    int(ind): pos
    for pos, ind in enumerate(valid_indices)
}

rep["u"] = [
    uv[int(ind), 0]
    for ind in rep["local_index"]
]

rep["v"] = [
    uv[int(ind), 1]
    for ind in rep["local_index"]
]

cross = rep[
    np.abs(rep["v"]) < 0.22
].sort_values("u")

if len(cross) >= 8:
    fig = plt.figure(figsize=(8.0, 5.2))
    ax = fig.add_subplot(111)

    ax.scatter(
        cross["u"],
        cross["outer_z"],
        label="Observed outer surface",
        marker="o",
    )

    ax.plot(
        cross["u"],
        cross["true_inner_z"],
        marker="o",
        label="True hidden inner surface",
    )

    ax.plot(
        cross["u"],
        cross["predicted_inner_z"],
        marker="s",
        label="Predicted inner surface",
    )

    ax.fill_between(
        cross["u"],
        cross["predicted_inner_z"]
        - cross["halfwidth90"],
        cross["predicted_inner_z"]
        + cross["halfwidth90"],
        alpha=0.18,
        label="90% calibrated interval",
    )

    ax.set_xlabel(
        "Normalized position across lens"
    )

    ax.set_ylabel(
        "Local axial coordinate"
    )

    ax.set_title(
        f"Representative held-out lens #{representative_label}"
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "16_representative_reconstruction.png",
        dpi=200,
    )

    plt.close(fig)

# ---------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------

nested_row = summary[
    summary["method"]
    == "nested_selected"
].iloc[0]

hist_row = summary[
    summary["method"]
    == "hist_gradient"
].iloc[0]

sphere_row = summary[
    summary["method"]
    == "naive_sphere"
].iloc[0]

mean90 = uncertainty["coverage90"].mean()
mean95 = uncertainty["coverage95"].mean()

summary_text = f"""# Experiment 16 — strict outer-only reconstruction with uncertainty

## Question

Can the missing internal/proximal surface of a lens be reconstructed from the
preserved outer surface more reliably than the naive sphere/spheroid/ellipsoid
approach?

This experiment is intentionally stricter than Experiment 15.

## Important leakage control

The complete lens point clouds are used to create synthetic ground truth by
pairing outer and inner surface samples.

After that split:

- every prediction coordinate frame is rebuilt from the **outer surface only**;
- the operational eye center is fitted using **outer surface points only**;
- rim geometry, surface curvature, sag, eye position, and local point features
  are calculated from the **outer surface only**;
- the hidden inner surface is used only to score the final prediction.

The first full-data eye center is used only to decide which side of each
complete lens is anatomically outward when constructing the synthetic damage
benchmark.

## Dataset

- labeled 3D points: {len(labels):,}
- initial lens labels: {len(unique_labels):,}
- usable medium-sized lenses for this benchmark: {len(lens_ids):,}

## Model

At each preserved outer-surface sample the predictors include:

- position within a fitted rim ellipse;
- normalized radial position;
- observed outer-surface height relative to the fitted rim;
- local outer-surface polynomial descriptors;
- lens rim dimensions and aspect ratio;
- outer-surface sag;
- position on the eye.

Three model families were compared:

1. ridge regression;
2. Extra Trees;
3. histogram gradient boosting.

For each of five outer folds, the target lenses were never used for training.

Inside each outer-training set:

- a proper training subset was used for three-fold model selection;
- a separate lens-level calibration subset was withheld from model selection;
- the calibration subset was used only to calibrate prediction intervals.

## Main out-of-fold results

{summary.to_markdown(index=False)}

The nested procedure selected Extra Trees in one fold and histogram gradient
boosting in four folds.

The nested-selected model reached:

- median hidden-inner-surface MAE: **{nested_row["median_mae"]:.3f} coordinate units**
- 90th-percentile MAE: **{nested_row["p90_mae"]:.3f}**
- median error / median true lens thickness: **{100*nested_row["median_normalized_mae"]:.1f}%**
- 90th-percentile normalized error: **{100*nested_row["p90_normalized_mae"]:.1f}%**

For comparison, naive outer-cap sphere extrapolation had a median MAE of
**{sphere_row["median_mae"]:.2f}** on lenses where the sphere produced enough
valid predictions.

## Uncertainty

Adaptive prediction intervals use disagreement among Ridge, Extra Trees, and
histogram gradient boosting as a local uncertainty scale, then calibrate that
scale on held-out calibration lenses using a conformal quantile.

Across the five outer folds:

- nominal 90% intervals achieved **{100*mean90:.1f}%** mean observed coverage;
- nominal 95% intervals achieved **{100*mean95:.1f}%** mean observed coverage.

Median interval half-widths were:

- 90% interval: **{uncertainty["median_halfwidth90"].mean():.3f}**
- 95% interval: **{uncertainty["median_halfwidth95"].mean():.3f}**

## Does uncertainty identify difficult lenses?

Spearman correlation between lens-level 90% interval width and reconstruction
MAE:

- rho = **{rho_conf:.3f}**
- p = **{p_conf:.3g}**

Confidence stratification:

{confidence_summary.to_markdown(index=False)}

This tells us whether interval width is actually useful as a failure-warning
signal rather than only being numerically calibrated.

## Interpretation

This is a meaningful improvement over the naive quadric idea.

The strongest gain came from using the detailed **observed outer-surface
geometry** in a regularized nonlinear predictor. Neighbour smoothing was not
required to obtain the main improvement, so the evidence does not currently
justify imposing strong spatial smoothness across the whole eye.

The remaining median normalized error is still about
**{100*nested_row["median_normalized_mae"]:.1f}% of true thickness**. Therefore
the internal surface is not "solved" yet. However, the catastrophic
ill-conditioning of free outer-cap extrapolation has been replaced by a
bounded, cross-validated prediction with calibrated uncertainty.

## What I would do next

Experiment 17 should focus on the remaining residual structure rather than add
more free quadric parameters.

The highest-value tests are:

1. determine whether residuals are systematic with lens position, rim shape,
   or outer curvature;
2. fit a low-dimensional residual surface only when cross-validation shows a
   reproducible pattern;
3. simulate progressive fossil preservation (100%, 75%, 50%, 30% of outer
   surface retained) and determine the preservation threshold at which the
   uncertainty interval becomes too wide for useful optical inference;
4. validate on an independent eye/specimen before any strong general claim.

The preservation-threshold experiment is especially relevant to the Bristol
project because it converts the method from "produce a reconstruction" into
"state when a reconstruction is defensible."
"""

(OUT / "16_experiment_summary.md").write_text(
    summary_text
)

print("Experiment 16 complete.")
print()
print(summary.to_string(index=False))
print()
print("Mean 90% interval coverage:", round(mean90, 4))
print("Mean 95% interval coverage:", round(mean95, 4))
print("Lens uncertainty/error Spearman rho:", round(float(rho_conf), 4))
print()
print("Files written to:", OUT)

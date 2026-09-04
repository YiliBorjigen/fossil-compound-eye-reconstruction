#!/usr/bin/env python3
"""Target-blind geometry for Experiment 63 distal corneal-lens caps.

This module is deliberately independent of proximal surfaces and prediction
targets.  Its public loader accepts the small, sealed Stage-1 distal artifacts;
all eye registration, frames, quality control, and position descriptors are
then derived from those distal points alone.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import ConvexHull, QhullError, cKDTree, distance


SEALED_DISTAL_SCHEMA_VERSION = "experiment63.sealed-distal.v2"
SEALED_DISTAL_KEYS = frozenset(
    {
        "schema_version",
        "lens_index",
        "points_zyx",
        "spacing_um",
        "config_json",
        "config_sha256",
    }
)

# These values are frozen before any Maike proximal target or prediction error
# is inspected.  Keep this dictionary JSON-native so its canonical hash is
# portable across Python/numpy versions.
DEFAULT_CONFIG: dict[str, Any] = {
    "distal_scale_statistic": "q90_radius_um",
    "distal_scale_min_um": 3.0,
    "distal_scale_max_um": 13.0,
    "min_sealed_distal_cap_points": 25,
    "distal_fit_rmse_max_um": 2.5,
    "quadratic_design_condition_max": 1_000_000.0,
    "fixed_point_max_iterations": 20,
    "lateral_bin_um": 0.325,
    "min_axial_span_um": 0.650,
    "connectivity": 26,
    "candidate_seeds_per_voxel": 1,
}

SCALE_QUANTILE = 0.90
CENTRAL_ECCENTRICITY_NN_FRACTION = 0.5
ROBUST_HUBER_TUNING = 1.345
ROBUST_MAX_ITERATIONS = 25
ROBUST_TOLERANCE = 1.0e-10
NUMERICAL_TOLERANCE = 1.0e-10
POSITION_NEIGHBOUR_ORDERS = (1, 3, 6)
POSITION_DISTANCE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

POSITION_FEATURE_NAMES = (
    "position_radius_um",
    "position_boundary_distance_um",
    "position_nn1_um",
    "position_nn3_um",
    "position_nn6_um",
    "position_pairwise_q10_um",
    "position_pairwise_q25_um",
    "position_pairwise_q50_um",
    "position_pairwise_q75_um",
    "position_pairwise_q90_um",
)
CONTROL_FEATURE_NAMES = POSITION_FEATURE_NAMES + ("distal_scale_um",)


class DistalGeometryError(ValueError):
    """Raised when a distal artifact or eye geometry violates the contract."""


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"configuration contains a non-JSON value: {type(value).__name__}")


def _normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config is not None:
        unknown = sorted(set(config) - set(DEFAULT_CONFIG))
        if unknown:
            raise DistalGeometryError(f"unknown configuration key(s): {', '.join(unknown)}")
        merged.update(_json_native(config))

    positive_integer_keys = (
        "min_sealed_distal_cap_points",
        "fixed_point_max_iterations",
        "connectivity",
        "candidate_seeds_per_voxel",
    )
    for key in positive_integer_keys:
        if isinstance(merged[key], bool) or int(merged[key]) != merged[key] or int(merged[key]) < 1:
            raise DistalGeometryError(f"{key} must be a positive integer")
        merged[key] = int(merged[key])

    finite_nonnegative_keys = (
        "distal_scale_min_um",
        "distal_scale_max_um",
        "distal_fit_rmse_max_um",
        "quadratic_design_condition_max",
        "lateral_bin_um",
        "min_axial_span_um",
    )
    for key in finite_nonnegative_keys:
        number = float(merged[key])
        if not math.isfinite(number) or number < 0.0:
            raise DistalGeometryError(f"{key} must be finite and non-negative")
        merged[key] = number

    if merged["distal_scale_min_um"] > merged["distal_scale_max_um"]:
        raise DistalGeometryError("distal_scale_min_um must not exceed distal_scale_max_um")
    if merged["distal_scale_statistic"] != "q90_radius_um":
        raise DistalGeometryError("distal_scale_statistic must be 'q90_radius_um'")
    if merged["connectivity"] != 26:
        raise DistalGeometryError("connectivity must remain frozen at 26")
    if merged["candidate_seeds_per_voxel"] != 1:
        raise DistalGeometryError("candidate_seeds_per_voxel must remain frozen at 1")
    return merged


def canonical_config_json(config: Mapping[str, Any] | None = None) -> str:
    """Return the canonical, fully populated JSON representation of *config*."""

    merged = _normalise_config(config)
    return json.dumps(merged, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(config: Mapping[str, Any] | None = None) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def _scalar(array: np.ndarray, name: str) -> Any:
    if array.shape != ():
        raise DistalGeometryError(f"sealed field {name!r} must be scalar")
    return array.item()


def load_sealed_distal(
    path: str | Path,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and load one immutable Stage-1 distal-cap ``.npz`` artifact.

    Object arrays and pickle loading are forbidden.  Coordinates are stored as
    integer source-array ``zyx`` indices; this loader is the only conversion to
    physical ``xyz`` used by the Stage-2 geometry code.
    """

    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".npz" or not artifact_path.is_file():
        raise DistalGeometryError(f"sealed distal artifact does not exist: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            names = frozenset(archive.files)
            if names != SEALED_DISTAL_KEYS:
                missing = sorted(SEALED_DISTAL_KEYS - names)
                extra = sorted(names - SEALED_DISTAL_KEYS)
                raise DistalGeometryError(
                    f"sealed distal keys differ from contract; missing={missing}, extra={extra}"
                )
            schema_version = str(_scalar(archive["schema_version"], "schema_version"))
            lens_index_raw = _scalar(archive["lens_index"], "lens_index")
            points_zyx = np.asarray(archive["points_zyx"])
            spacing_um = np.asarray(archive["spacing_um"])
            stored_config_json = str(_scalar(archive["config_json"], "config_json"))
            stored_config_sha256 = str(_scalar(archive["config_sha256"], "config_sha256"))
    except (OSError, ValueError) as exc:
        if isinstance(exc, DistalGeometryError):
            raise
        raise DistalGeometryError(f"cannot read sealed distal artifact {artifact_path}: {exc}") from exc

    if schema_version != SEALED_DISTAL_SCHEMA_VERSION:
        raise DistalGeometryError(
            f"unsupported sealed distal schema {schema_version!r}; "
            f"expected {SEALED_DISTAL_SCHEMA_VERSION!r}"
        )
    if isinstance(lens_index_raw, (bool, np.bool_)):
        raise DistalGeometryError("lens_index must be a non-negative integer")
    try:
        lens_index = int(lens_index_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DistalGeometryError("lens_index must be a non-negative integer") from exc
    if lens_index < 0 or lens_index != lens_index_raw:
        raise DistalGeometryError("lens_index must be a non-negative integer")
    if points_zyx.dtype != np.dtype(np.int32) or points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise DistalGeometryError("points_zyx must have dtype int32 and shape (n, 3)")
    if np.any(points_zyx < 0):
        raise DistalGeometryError("points_zyx must contain non-negative source-array indices")
    if spacing_um.dtype != np.dtype(np.float64) or spacing_um.shape != (3,):
        raise DistalGeometryError("spacing_um must have dtype float64 and shape (3,)")
    if not np.all(np.isfinite(spacing_um)) or np.any(spacing_um <= 0.0):
        raise DistalGeometryError("spacing_um must be finite and strictly positive")

    try:
        parsed_config = json.loads(stored_config_json)
    except json.JSONDecodeError as exc:
        raise DistalGeometryError("config_json is not valid JSON") from exc
    if not isinstance(parsed_config, dict):
        raise DistalGeometryError("config_json must encode an object")
    canonical_stored = json.dumps(
        _json_native(parsed_config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if canonical_stored != stored_config_json:
        raise DistalGeometryError("config_json is not canonical JSON")
    forbidden_fragments = ("target", "proximal", "prediction", "error", "tip", "oda_axis")
    lowered_config = stored_config_json.casefold()
    found_forbidden = [item for item in forbidden_fragments if item in lowered_config]
    if found_forbidden:
        raise DistalGeometryError(
            "sealed config contains forbidden Stage-2 content: " + ", ".join(found_forbidden)
        )
    required_top_level = {
        "analysis_scope",
        "isolation_basis",
        "threshold_config",
        "predictor_pipeline_config",
    }
    if set(parsed_config) != required_top_level:
        raise DistalGeometryError("sealed config has unexpected top-level keys")
    if parsed_config["analysis_scope"] != "conditional_on_oracle_distal_surface_localization":
        raise DistalGeometryError("sealed config has the wrong analysis_scope")
    if parsed_config["isolation_basis"] != "stage2_reads_only_sha256_sealed_distal_artifacts":
        raise DistalGeometryError("sealed config has the wrong isolation_basis")
    predictor_config = parsed_config["predictor_pipeline_config"]
    predictor_keys = {
        "fixed_point_policy",
        "original_spacing_um",
        "distal_split",
        "canonical_grid",
    }
    if not isinstance(predictor_config, dict) or set(predictor_config) != predictor_keys:
        raise DistalGeometryError("sealed predictor_pipeline_config has unexpected keys")
    if predictor_config["fixed_point_policy"] != "monotone_drop_only_no_reentry":
        raise DistalGeometryError("sealed fixed_point_policy is not drop-only")
    if predictor_config["original_spacing_um"] != [0.325, 0.325, 0.325]:
        raise DistalGeometryError("sealed original_spacing_um differs from the frozen Maike spacing")
    if (
        predictor_config["distal_split"]
        != "largest_26_component_boundary_deterministic_1d_two_means"
    ):
        raise DistalGeometryError("sealed distal_split differs from the frozen Stage-1 method")
    if predictor_config["canonical_grid"] != "experiment57_disk_radius_0.65_step_0.13":
        raise DistalGeometryError("sealed canonical_grid differs from the frozen grid")
    threshold_config = parsed_config["threshold_config"]
    if not isinstance(threshold_config, dict):
        raise DistalGeometryError("sealed threshold_config must be an object")
    if canonical_config_json(threshold_config) != json.dumps(
        threshold_config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ):
        raise DistalGeometryError("sealed threshold_config is incomplete or non-canonical")
    calculated_config_sha256 = hashlib.sha256(stored_config_json.encode("utf-8")).hexdigest()
    if stored_config_sha256 != calculated_config_sha256:
        raise DistalGeometryError("config_sha256 does not match config_json")
    if expected_config_sha256 is not None and stored_config_sha256 != expected_config_sha256:
        raise DistalGeometryError(
            f"config hash mismatch: got {stored_config_sha256}, expected {expected_config_sha256}"
        )

    file_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    points_xyz_um = points_zyx[:, ::-1].astype(np.float64) * spacing_um[::-1]
    return {
        "schema_version": schema_version,
        "lens_index": lens_index,
        "lens_id": lens_index,
        "points_zyx": points_zyx.copy(),
        "spacing_um": spacing_um.copy(),
        "points_xyz_um": points_xyz_um,
        "config": _normalise_config(threshold_config),
        "sealed_config": parsed_config,
        "config_json": stored_config_json,
        "config_sha256": stored_config_sha256,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256,
        "sealed_distal_artifact": True,
        "stage1_eligible": True,
    }


def _unit(vector: np.ndarray, tolerance: float, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= tolerance:
        raise DistalGeometryError(f"cannot normalise {name}")
    return np.asarray(vector, dtype=np.float64) / norm


def _canonical_tangent(normal: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    # Pick the Cartesian axis least parallel to the normal.  This is used only
    # for standalone fitting and the explicitly marked central-frame fallback.
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    u = _unit(reference - np.dot(reference, normal) * normal, tolerance, "tangent axis")
    v = _unit(np.cross(normal, u), tolerance, "second tangent axis")
    return u, v


def _validate_points(points_xyz_um: Any, *, allow_empty: bool = False) -> np.ndarray:
    points = np.asarray(points_xyz_um)
    if points.ndim != 2 or points.shape[1] != 3:
        raise DistalGeometryError("distal points must have shape (n, 3)")
    if not np.issubdtype(points.dtype, np.number):
        raise DistalGeometryError("distal points must be numeric")
    points = points.astype(np.float64, copy=False)
    if (len(points) == 0 and not allow_empty) or not np.all(np.isfinite(points)):
        requirement = "finite" if allow_empty else "non-empty and finite"
        raise DistalGeometryError(f"distal points must be {requirement}")
    return points


def _robust_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float, float]:
    design = np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))
    condition = float(np.linalg.cond(design)) if len(design) >= 6 else math.inf
    if len(design) < 6 or np.linalg.matrix_rank(design) < 6:
        return np.full(6, np.nan), math.inf, condition

    beta = np.linalg.lstsq(design, z, rcond=None)[0]
    tuning = ROBUST_HUBER_TUNING
    tolerance = ROBUST_TOLERANCE
    for _ in range(ROBUST_MAX_ITERATIONS):
        residual = z - design @ beta
        centre = float(np.median(residual))
        mad = float(np.median(np.abs(residual - centre)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= np.finfo(np.float64).eps:
            break
        cutoff = tuning * robust_sigma
        absolute = np.abs(residual - centre)
        weights = np.ones_like(absolute)
        large = absolute > cutoff
        weights[large] = cutoff / absolute[large]
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root_weights[:, None], z * root_weights, rcond=None)[0]
        if float(np.linalg.norm(updated - beta)) <= tolerance * (1.0 + float(np.linalg.norm(beta))):
            beta = updated
            break
        beta = updated
    residual = z - design @ beta
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return beta, rmse, condition


def fit_robust_quadratic(
    x: Any,
    y: Any,
    z: Any,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Public shared robust fit for an already selected quadratic cap.

    ``x`` and ``y`` are the caller's normalized coordinates.  The returned
    coefficients correspond to ``[1, x, y, x², xy, y²]``.  Both Arthur and
    Maike target adapters use this literal function after applying their
    distal-q90 support mask, preventing source/test fitting drift.
    """

    cfg = _normalise_config(config)
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    z_array = np.asarray(z, dtype=np.float64)
    if any(array.ndim != 1 for array in (x_array, y_array, z_array)):
        raise DistalGeometryError("robust quadratic x, y, and z must be one-dimensional")
    if not len(x_array) == len(y_array) == len(z_array) or len(x_array) == 0:
        raise DistalGeometryError("robust quadratic x, y, and z must have equal nonzero length")
    if not all(np.all(np.isfinite(array)) for array in (x_array, y_array, z_array)):
        raise DistalGeometryError("robust quadratic inputs must be finite")
    beta, rmse, condition = _robust_quadratic(x_array, y_array, z_array, cfg)
    return {
        "beta": beta,
        "coefficients": beta,
        "rmse": rmse,
        "rmse_um": rmse,
        "condition": condition,
        "support": int(len(x_array)),
    }


def _principal_curvatures(beta: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(beta)):
        return np.full(2, np.nan)
    fx, fy = float(beta[1]), float(beta[2])
    first = np.array([[1.0 + fx * fx, fx * fy], [fx * fy, 1.0 + fy * fy]])
    normaliser = math.sqrt(1.0 + fx * fx + fy * fy)
    second = np.array([[2.0 * beta[3], beta[4]], [beta[4], 2.0 * beta[5]]]) / normaliser
    try:
        values = eigh(second, first, eigvals_only=True, check_finite=True)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan)
    return np.sort(np.asarray(values, dtype=np.float64))


def _fit_with_frame(
    points_xyz_um: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    points = _validate_points(points_xyz_um)
    origin = np.mean(points, axis=0)
    centred = points - origin
    x = centred @ u
    y = centred @ v
    z = centred @ w
    radial_distance = np.sqrt(x * x + y * y)
    scale = float(np.quantile(radial_distance, SCALE_QUANTILE))
    if scale > NUMERICAL_TOLERANCE and math.isfinite(scale):
        central_cap = radial_distance <= scale + NUMERICAL_TOLERANCE
        beta_normalised, rmse, condition = _robust_quadratic(
            x[central_cap] / scale,
            y[central_cap] / scale,
            z[central_cap],
            config,
        )
        conversion = np.array(
            [1.0, scale, scale, scale * scale, scale * scale, scale * scale],
            dtype=np.float64,
        )
        beta = beta_normalised / conversion
    else:
        central_cap = np.ones(len(points), dtype=bool)
        beta_normalised = np.full(6, np.nan)
        beta = np.full(6, np.nan)
        rmse = math.inf
        condition = math.inf
    curvatures = _principal_curvatures(beta)
    gradient = float(np.hypot(beta[1], beta[2])) if np.all(np.isfinite(beta[:3])) else math.nan
    normalised_rmse = rmse / scale if scale > 0.0 and math.isfinite(rmse) else math.inf
    return {
        "raw_support": int(len(points)),
        # The frozen support gate is the Stage-1 distal-cap support.  The q90
        # central subset is recorded separately; changing the gate to that
        # post-scale count would silently remove an additional source lens.
        "support": int(len(points)),
        "fit_support": int(np.count_nonzero(central_cap)),
        "origin": origin,
        "outward": w.copy(),
        "u": u.copy(),
        "v": v.copy(),
        "w": w.copy(),
        "beta": beta,
        "quadratic_beta": beta,
        "quadratic_beta_normalized": beta_normalised,
        "rmse_um": rmse,
        "quadratic_rmse_um": rmse,
        "condition": condition,
        "quadratic_condition": condition,
        "scale_um": scale,
        "gradient_magnitude": gradient,
        "curvature_eigenvalues": curvatures,
        "normalised_rmse": normalised_rmse,
        "shape_features": np.array(
            [gradient, curvatures[0], curvatures[1], normalised_rmse], dtype=np.float64
        ),
    }


def fit_distal_cap(points_xyz_um: Any) -> dict[str, Any]:
    """Fit a robust local quadratic to one distal cap without eye context."""

    points = _validate_points(points_xyz_um)
    origin = np.mean(points, axis=0)
    centred = points - origin
    covariance = centred.T @ centred / float(len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    largest_component = int(np.argmax(np.abs(normal)))
    if normal[largest_component] < 0.0:
        normal = -normal
    config = _normalise_config(None)
    u, v = _canonical_tangent(normal, NUMERICAL_TOLERANCE)
    return _fit_with_frame(points, u, v, normal, config)


def _record_lens_index(record: Mapping[str, Any], fallback: int) -> int | str:
    for key in ("lens_index", "lens_id", "landmark_id"):
        if key in record:
            value = record[key]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                return value
            return str(value)
    return fallback


def _record_points(record: Mapping[str, Any], *, allow_empty: bool = False) -> np.ndarray:
    for key in ("points_xyz_um", "distal_points", "outer", "points"):
        if key in record:
            return _validate_points(record[key], allow_empty=allow_empty)
    if "points_zyx" in record and "spacing_um" in record:
        points_zyx = np.asarray(record["points_zyx"])
        spacing = np.asarray(record["spacing_um"], dtype=np.float64)
        if points_zyx.ndim != 2 or points_zyx.shape[1] != 3 or spacing.shape != (3,):
            raise DistalGeometryError("invalid points_zyx/spacing_um record")
        return _validate_points(
            points_zyx[:, ::-1].astype(np.float64) * spacing[::-1], allow_empty=allow_empty
        )
    raise DistalGeometryError("record has no distal point field")


def _sphere_fit(origins: np.ndarray, tolerance: float) -> tuple[np.ndarray, float]:
    if len(origins) < 4:
        raise DistalGeometryError("at least four distal origins are required for an eye sphere")
    mean = np.mean(origins, axis=0)
    centred = origins - mean
    design = np.column_stack((2.0 * centred, np.ones(len(centred))))
    rhs = np.sum(centred * centred, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 4:
        raise DistalGeometryError("distal origins do not define a non-degenerate eye sphere")
    centre = mean + solution[:3]
    radii = np.linalg.norm(origins - centre, axis=1)
    radius = float(np.median(radii))
    if not math.isfinite(radius) or radius <= tolerance:
        raise DistalGeometryError("fitted eye sphere radius is invalid")
    return centre, radius


def _log_map_positions(
    origins: np.ndarray,
    sphere_centre: np.ndarray,
    sphere_radius: float,
    pole_direction: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    basis_u, basis_v = _canonical_tangent(pole_direction, tolerance)
    radial = origins - sphere_centre
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    cosine = np.clip(radial @ pole_direction, -1.0, 1.0)
    angles = np.arccos(cosine)
    tangent = radial - cosine[:, None] * pole_direction
    tangent_norm = np.linalg.norm(tangent, axis=1)
    directions = np.zeros_like(tangent)
    valid = tangent_norm > tolerance
    directions[valid] = tangent[valid] / tangent_norm[valid, None]
    coordinates = np.column_stack((directions @ basis_u, directions @ basis_v))
    coordinates *= (sphere_radius * angles)[:, None]
    return coordinates


def _distance_to_segments(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
    vectors = ends - starts
    squared = np.sum(vectors * vectors, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        parameter = np.sum((point - starts) * vectors, axis=1) / squared
    parameter = np.where(squared > 0.0, np.clip(parameter, 0.0, 1.0), 0.0)
    closest = starts + parameter[:, None] * vectors
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def _position_feature_matrix(
    coordinates: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    reference_coordinates = coordinates[reference_indices]
    all_to_reference = distance.cdist(coordinates, reference_coordinates)
    # Exclude self distances wherever an evaluation lens belongs to the cohort.
    reference_lookup = {int(record_index): column for column, record_index in enumerate(reference_indices)}
    features = np.full((len(coordinates), len(POSITION_FEATURE_NAMES)), np.nan, dtype=np.float64)

    hull_starts: np.ndarray | None = None
    hull_ends: np.ndarray | None = None
    if len(reference_coordinates) >= 3:
        try:
            hull = ConvexHull(reference_coordinates)
            vertices = reference_coordinates[hull.vertices]
            hull_starts = vertices
            hull_ends = np.roll(vertices, -1, axis=0)
        except QhullError:
            hull_starts = None
            hull_ends = None

    orders = POSITION_NEIGHBOUR_ORDERS
    quantiles = POSITION_DISTANCE_QUANTILES
    for row in range(len(coordinates)):
        distances = all_to_reference[row].copy()
        self_column = reference_lookup.get(row)
        if self_column is not None:
            distances = np.delete(distances, self_column)
        finite = np.sort(distances[np.isfinite(distances)])
        features[row, 0] = float(np.linalg.norm(coordinates[row]))
        if hull_starts is None or hull_ends is None:
            features[row, 1] = 0.0
        else:
            features[row, 1] = _distance_to_segments(coordinates[row], hull_starts, hull_ends)
        for offset, order in enumerate(orders, start=2):
            features[row, offset] = finite[order - 1] if len(finite) >= order else math.nan
        start = 2 + len(orders)
        if len(finite):
            features[row, start : start + len(quantiles)] = np.quantile(finite, quantiles)
    return features


def derive_eye_frames_from_origins(
    origins_xyz_um: Any,
    reference_indices: Sequence[int] | np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive sphere, pole, frames, and central flags from distal origins only."""

    cfg = _normalise_config(config)
    origins = _validate_points(origins_xyz_um)
    if reference_indices is None:
        references = np.arange(len(origins), dtype=np.int64)
    else:
        references = np.asarray(reference_indices, dtype=np.int64)
        if references.ndim != 1 or len(references) == 0:
            raise DistalGeometryError("reference_indices must be a non-empty 1-D sequence")
        if np.any(references < 0) or np.any(references >= len(origins)):
            raise DistalGeometryError("reference_indices contain an out-of-range index")
        if len(np.unique(references)) != len(references):
            raise DistalGeometryError("reference_indices must be unique")
    tolerance = NUMERICAL_TOLERANCE
    sphere_centre, sphere_radius = _sphere_fit(origins[references], tolerance)
    reference_radial = origins[references] - sphere_centre
    reference_radial /= np.linalg.norm(reference_radial, axis=1)[:, None]
    pole_direction = _unit(np.mean(reference_radial, axis=0), tolerance, "eye pole")

    radial = origins - sphere_centre
    radii = np.linalg.norm(radial, axis=1)
    if np.any(~np.isfinite(radii)) or np.any(radii <= tolerance):
        raise DistalGeometryError("a distal origin coincides with the eye sphere centre")
    outward = radial / radii[:, None]
    coordinates = _log_map_positions(
        origins, sphere_centre, sphere_radius, pole_direction, tolerance
    )
    eccentricity = np.linalg.norm(coordinates, axis=1)

    if len(references) < 2:
        raise DistalGeometryError("at least two reference origins are required")
    nearest_distances, _ = cKDTree(coordinates[references]).query(
        coordinates[references], k=2, workers=1
    )
    median_nn = float(np.median(nearest_distances[:, 1]))
    if not math.isfinite(median_nn) or median_nn <= tolerance:
        raise DistalGeometryError("reference origins have an invalid nearest-neighbour spacing")
    central_threshold = CENTRAL_ECCENTRICITY_NN_FRACTION * median_nn
    central = eccentricity <= central_threshold

    u = np.empty_like(outward)
    v = np.empty_like(outward)
    pole_projection = outward @ pole_direction
    poleward = pole_direction[None, :] - pole_projection[:, None] * outward
    poleward_norm = np.linalg.norm(poleward, axis=1)
    fallback = poleward_norm <= tolerance
    regular = ~fallback
    u[regular] = poleward[regular] / poleward_norm[regular, None]
    v[regular] = np.cross(outward[regular], u[regular])
    v_norm = np.linalg.norm(v[regular], axis=1)
    if np.any(~np.isfinite(v_norm)) or np.any(v_norm <= tolerance):
        raise DistalGeometryError("cannot construct regular transverse lens frames")
    v[regular] /= v_norm[:, None]
    for index in np.flatnonzero(fallback):
        u[index], v[index] = _canonical_tangent(outward[index], tolerance)

    return {
        "sphere_centre": sphere_centre,
        "sphere_radius": sphere_radius,
        "pole_direction": pole_direction,
        "origins": origins.copy(),
        "outward": outward,
        "u": u,
        "v": v,
        "w": outward.copy(),
        "eccentricity_um": eccentricity,
        "median_nearest_neighbour_um": median_nn,
        "central_threshold_um": central_threshold,
        "central": central,
        "numerical_fallback": fallback,
        "positions_2d_um": coordinates,
        "reference_indices": references,
    }


def _qc_reasons(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(metrics["support"]) < int(config["min_sealed_distal_cap_points"]):
        reasons.append("support_below_minimum")
    scale = float(metrics["scale_um"])
    if not math.isfinite(scale):
        reasons.append("scale_nonfinite")
    elif scale < float(config["distal_scale_min_um"]):
        reasons.append("scale_below_minimum")
    elif scale > float(config["distal_scale_max_um"]):
        reasons.append("scale_above_maximum")
    rmse = float(metrics["quadratic_rmse_um"])
    if not math.isfinite(rmse):
        reasons.append("quadratic_rmse_nonfinite")
    elif rmse > float(config["distal_fit_rmse_max_um"]):
        reasons.append("quadratic_rmse_above_maximum")
    condition = float(metrics["quadratic_condition"])
    if not math.isfinite(condition):
        reasons.append("quadratic_condition_nonfinite")
    elif condition > float(config["quadratic_design_condition_max"]):
        reasons.append("quadratic_condition_above_maximum")
    return reasons


def derive_distal_only_eye_geometry(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
    reference_indices: Sequence[int] | np.ndarray | None = None,
    eligible_indices: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """Derive all Stage-2 quantities from distal point records.

    ``reference_indices`` selects the distal-QC cohort that defines the global
    eye geometry while still returning frames for every supplied record.  It is
    used by the stability audit and never consults proximal target status.
    """

    cfg = _normalise_config(config)
    if len(records) == 0:
        raise DistalGeometryError("at least one distal record is required")
    if eligible_indices is not None:
        if reference_indices is not None:
            raise DistalGeometryError("use either eligible_indices or reference_indices, not both")
        identifier_to_position: dict[int | str, int] = {}
        for position, record in enumerate(records):
            identifier = _record_lens_index(record, position)
            if identifier in identifier_to_position:
                raise DistalGeometryError("record lens identifiers must be unique")
            identifier_to_position[identifier] = position
        requested = list(eligible_indices)
        if len(set(requested)) != len(requested):
            raise DistalGeometryError("eligible_indices must be unique")
        missing = [identifier for identifier in requested if identifier not in identifier_to_position]
        if missing:
            raise DistalGeometryError(f"eligible lens identifier(s) not found: {missing}")
        selected_positions = [identifier_to_position[identifier] for identifier in requested]
        selected_records = [records[position] for position in selected_positions]
        result = derive_distal_only_eye_geometry(selected_records, config=cfg)
        result["source_record_indices"] = selected_positions
        result["eligible_indices"] = requested
        return result
    points = [_record_points(record) for record in records]
    origins = np.asarray([np.mean(item, axis=0) for item in points], dtype=np.float64)
    frames = derive_eye_frames_from_origins(origins, reference_indices=reference_indices, config=cfg)
    references = np.asarray(frames["reference_indices"], dtype=np.int64)

    per_lens: list[dict[str, Any]] = []
    scales = np.empty(len(records), dtype=np.float64)
    for index, (record, lens_points) in enumerate(zip(records, points, strict=True)):
        metrics = _fit_with_frame(
            lens_points,
            frames["u"][index],
            frames["v"][index],
            frames["w"][index],
            cfg,
        )
        reasons = _qc_reasons(metrics, cfg)
        metrics.update(
            {
                "record_index": index,
                "lens_index": _record_lens_index(record, index),
                "central": bool(frames["central"][index]),
                "eccentricity_um": float(frames["eccentricity_um"][index]),
                "numerical_fallback": bool(frames["numerical_fallback"][index]),
                "distal_qc_pass": not reasons,
                "distal_qc_reasons": reasons,
                "distal_qc_reason": "pass" if not reasons else ";".join(reasons),
            }
        )
        scales[index] = float(metrics["scale_um"])
        per_lens.append(metrics)

    position_features = _position_feature_matrix(
        frames["positions_2d_um"], references
    )
    for index, metrics in enumerate(per_lens):
        metrics["position_2d_um"] = frames["positions_2d_um"][index].copy()
        metrics["position_features"] = position_features[index].copy()
        metrics["position_feature_names"] = POSITION_FEATURE_NAMES
        metrics["control_features"] = np.concatenate(
            (position_features[index], np.asarray([scales[index]], dtype=np.float64))
        )
        metrics["control_feature_names"] = CONTROL_FEATURE_NAMES
        metrics["geometry_features"] = metrics["control_features"].copy()

    return {
        "config": cfg,
        "config_json": canonical_config_json(cfg),
        "config_sha256": config_sha256(cfg),
        "sphere_centre": frames["sphere_centre"],
        "sphere_radius": frames["sphere_radius"],
        "pole_direction": frames["pole_direction"],
        "median_nearest_neighbour_um": frames["median_nearest_neighbour_um"],
        "central_threshold_um": frames["central_threshold_um"],
        "reference_indices": references,
        "position_feature_names": POSITION_FEATURE_NAMES,
        "control_feature_names": CONTROL_FEATURE_NAMES,
        "per_lens": per_lens,
    }


def _stage1_eligible(record: Mapping[str, Any]) -> bool:
    for key in ("stage1_eligible", "stage1_pass", "oracle_distal_localized"):
        if key in record:
            return bool(record[key])
    return True


def _serialisable_metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "support",
        "scale_um",
        "quadratic_rmse_um",
        "quadratic_condition",
        "gradient_magnitude",
        "normalised_rmse",
        "central",
        "eccentricity_um",
        "numerical_fallback",
        "distal_qc_pass",
        "distal_qc_reasons",
    )
    return {key: _json_native(metrics[key]) for key in keys}


def run_monotone_fixed_point(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen distal QC as a deterministic, drop-only fixed point."""

    cfg = _normalise_config(config)
    initial: list[int] = []
    dropped_iteration: dict[int, int] = {}
    dropped_reasons: dict[int, list[str]] = {}
    last_metrics: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        explicitly_eligible = _stage1_eligible(record)
        points = _record_points(record, allow_empty=True)
        support = int(len(points))
        if not explicitly_eligible:
            dropped_iteration[index] = 0
            dropped_reasons[index] = ["stage1_ineligible"]
        elif support < int(cfg["min_sealed_distal_cap_points"]):
            dropped_iteration[index] = 0
            dropped_reasons[index] = ["support_below_minimum"]
            last_metrics[index] = {
                "support": support,
                "scale_um": math.nan,
                "quadratic_rmse_um": math.nan,
                "quadratic_condition": math.nan,
                "gradient_magnitude": math.nan,
                "normalised_rmse": math.nan,
                "central": False,
                "eccentricity_um": math.nan,
                "numerical_fallback": False,
                "distal_qc_pass": False,
                "distal_qc_reasons": ["support_below_minimum"],
            }
        else:
            initial.append(index)
    if len(initial) < 4:
        raise DistalGeometryError("fewer than four Stage-1-eligible distal records")
    active = list(initial)
    iteration_diagnostics: list[dict[str, Any]] = []
    eligible_counts = [len(initial)]
    converged = False

    for iteration_number in range(1, int(cfg["fixed_point_max_iterations"]) + 1):
        active_records = [records[index] for index in active]
        geometry = derive_distal_only_eye_geometry(active_records, config=cfg)
        keep: list[int] = []
        dropped_now: list[int] = []
        reason_counter: Counter[str] = Counter()
        for local_index, metrics in enumerate(geometry["per_lens"]):
            original_index = active[local_index]
            last_metrics[original_index] = metrics
            reasons = list(metrics["distal_qc_reasons"])
            if reasons:
                dropped_now.append(original_index)
                dropped_iteration[original_index] = iteration_number
                dropped_reasons[original_index] = reasons
                reason_counter.update(reasons)
            else:
                keep.append(original_index)
        iteration_diagnostics.append(
            {
                "iteration": iteration_number,
                "start_count": len(active),
                "kept_count": len(keep),
                "dropped_count": len(dropped_now),
                "dropped_record_indices": dropped_now,
                "dropped_lens_indices": [
                    _record_lens_index(records[index], index) for index in dropped_now
                ],
                "reason_counts": dict(sorted(reason_counter.items())),
            }
        )
        eligible_counts.append(len(keep))
        if not dropped_now:
            converged = True
            active = keep
            break
        active = keep
        if len(active) < 4:
            raise DistalGeometryError(
                "distal QC left fewer than four lenses before the fixed point converged"
            )

    final_geometry: dict[str, Any] | None = None
    if converged:
        # The no-drop iteration already represents the final cohort, but derive
        # once explicitly so returned indices and descriptors cannot refer to a
        # prior cohort by accident.
        final_geometry = derive_distal_only_eye_geometry(
            [records[index] for index in active], config=cfg
        )
        for local_index, metrics in enumerate(final_geometry["per_lens"]):
            last_metrics[active[local_index]] = metrics

    per_lens: list[dict[str, Any]] = []
    active_set = set(active) if converged else set()
    initial_set = set(initial)
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "record_index": index,
            "lens_index": _record_lens_index(record, index),
            "stage1_eligible": index in initial_set,
            "eligible": index in active_set,
            "dropped_iteration": dropped_iteration.get(index),
            "drop_reasons": dropped_reasons.get(index, []),
            "reasons": dropped_reasons.get(index, []),
        }
        if index in last_metrics:
            summary = _serialisable_metric_summary(last_metrics[index])
            row["metrics"] = summary
            row.update(summary)
        per_lens.append(row)

    return {
        "config": cfg,
        "config_json": canonical_config_json(cfg),
        "config_sha256": config_sha256(cfg),
        "initial_record_indices": initial,
        "eligible_record_indices": active if converged else [],
        "eligible_indices": [
            _record_lens_index(records[index], index) for index in active
        ]
        if converged
        else [],
        "eligible_records": [records[index] for index in active] if converged else [],
        "iterations": len(iteration_diagnostics),
        "iteration_diagnostics": iteration_diagnostics,
        "eligible_counts": eligible_counts,
        "per_lens": per_lens,
        "converged": converged,
        "max_iterations": int(cfg["fixed_point_max_iterations"]),
        "readded_count": 0,
        "geometry": final_geometry,
        "target_blind": True,
        "reentry_allowed": False,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "CONTROL_FEATURE_NAMES",
    "POSITION_FEATURE_NAMES",
    "SEALED_DISTAL_SCHEMA_VERSION",
    "DistalGeometryError",
    "canonical_config_json",
    "config_sha256",
    "derive_distal_only_eye_geometry",
    "derive_eye_frames_from_origins",
    "fit_distal_cap",
    "fit_robust_quadratic",
    "load_sealed_distal",
    "run_monotone_fixed_point",
]

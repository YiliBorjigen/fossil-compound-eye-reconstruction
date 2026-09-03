#!/usr/bin/env python3
"""Experiment 59: contiguous proximal-loss reconstruction from neighbours.

Complete geometry and matched tips are used first for oracle layer separation.
After that target-construction step, mask geometry is fixed from retained
distal-cap centroids without consulting target availability, quality, depth or
prediction error. Each eye contributes eight separated seeds and nested
graph-radius 1, 2 and 3 patches. Every proximal surface in a patch is hidden;
visible proximal surfaces elsewhere in that eye may then be used by the
within-eye reconstruction methods.

This is a transductive, same-eye benchmark for patchy loss. It is not an
outer-only reconstruction and it is not a cross-specimen population model.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_57_outer_only_validation import (
    RIDGE_ALPHAS,
    SEED,
    canonical_grid,
    polynomial_design,
    sphere_center,
)
from experiment_58_cross_volume_confirmation import load_manifest, prepare_all


GRAPH_NEIGHBOURS = 6
GRAPH_EDGE_FACTOR = 1.60
BOUNDARY_GAP_DEGREES = 125.0
SEEDS_PER_EYE = 8
MIN_BOUNDARY_HOPS = 5
MIN_SEED_SEPARATION_HOPS = 7
PATCH_RADII = (1, 2, 3)
PRIMARY_RADIUS = 2
N_NEIGHBOURS = 6
DISTANCE_POWER = 2.0
MAIN_METHOD = "six_neighbor_depth_shared_shape"
EXPECTED_VOLUMES = {"20231107", "20240530", "20240701"}
METHOD_ORDER = [
    "axisymmetric_ellipsoid",
    "same_eye_template",
    "same_eye_outer_ridge",
    "nearest_visible_surface",
    "six_neighbor_surface",
    "graph_harmonic_depth_shared_shape",
    MAIN_METHOD,
]


@dataclass(frozen=True)
class OuterGraph:
    """A connected surface graph derived only from retained distal centroids."""

    points: np.ndarray
    adjacency: tuple[np.ndarray, ...]
    edge_lengths: tuple[np.ndarray, ...]
    weighted: csr_matrix
    local_pitch: np.ndarray
    boundary: np.ndarray


@dataclass
class EyeProtocol:
    volume: str
    eye: int
    records: list[dict]
    graph: OuterGraph
    seeds: np.ndarray
    masks: dict[int, dict[int, np.ndarray]]
    metric_distances: np.ndarray


def grouped_indices(records: list[dict], fields: list[str]):
    groups: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        key = tuple(record[field] for field in fields)
        groups.setdefault(key, []).append(index)
    return sorted(groups.items())


def graph_distances(
    adjacency: Sequence[np.ndarray], sources: int | Iterable[int]
) -> np.ndarray:
    """Minimum unweighted graph distance from one or more source nodes."""
    if np.isscalar(sources):
        source_array = np.asarray([sources], dtype=int)
    else:
        source_array = np.asarray(list(sources), dtype=int)
    if source_array.size == 0:
        raise ValueError("At least one graph source is required")

    result = np.full(len(adjacency), np.inf)
    queue: deque[int] = deque()
    for source in np.unique(source_array):
        if source < 0 or source >= len(adjacency):
            raise IndexError("Graph source lies outside the graph")
        result[source] = 0.0
        queue.append(int(source))
    while queue:
        node = queue.popleft()
        candidate = result[node] + 1.0
        for other in adjacency[node]:
            if not np.isfinite(result[other]):
                result[other] = candidate
                queue.append(int(other))
    return result


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    first = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(first) < 0.15:
        reference = np.array([1.0, 0.0, 0.0])
        first = reference - np.dot(reference, normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def detect_boundary(
    points: np.ndarray,
    adjacency: Sequence[np.ndarray],
    normals: np.ndarray,
) -> np.ndarray:
    """Find the observable eye margin from neighbour-free tangent sectors."""
    threshold = np.deg2rad(BOUNDARY_GAP_DEGREES)
    boundary = np.zeros(len(points), dtype=bool)
    for node, neighbours in enumerate(adjacency):
        if len(neighbours) < 3:
            boundary[node] = True
            continue
        first, second = tangent_basis(normals[node])
        vectors = points[neighbours] - points[node]
        x = vectors @ first
        y = vectors @ second
        usable = np.hypot(x, y) > 1e-12
        if np.sum(usable) < 3:
            boundary[node] = True
            continue
        angles = np.sort(np.arctan2(y[usable], x[usable]))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        boundary[node] = bool(np.max(gaps) > threshold)
    return boundary


def build_outer_graph(points: np.ndarray) -> OuterGraph:
    """Build a symmetrised, locally length-pruned distal-only kNN graph."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 50:
        raise ValueError("Each eye needs at least 50 finite distal centroids")
    if not np.all(np.isfinite(points)):
        raise ValueError("Distal centroids contain non-finite values")

    distances, neighbours = cKDTree(points).query(
        points, k=GRAPH_NEIGHBOURS + 1, workers=-1
    )
    distances = np.asarray(distances)[:, 1:]
    neighbours = np.asarray(neighbours, dtype=int)[:, 1:]
    local_pitch = np.median(distances, axis=1)
    if np.any(local_pitch <= 0.0):
        raise ValueError("Distal centroids must be distinct")

    edge_sets: list[set[int]] = [set() for _ in range(len(points))]
    for node in range(len(points)):
        for other, length in zip(neighbours[node], distances[node]):
            limit = GRAPH_EDGE_FACTOR * 0.5 * (
                local_pitch[node] + local_pitch[other]
            )
            if length <= limit:
                edge_sets[node].add(int(other))
                edge_sets[int(other)].add(node)

    adjacency = []
    edge_lengths = []
    rows = []
    columns = []
    values = []
    for node, connected in enumerate(edge_sets):
        ordered = np.asarray(sorted(connected), dtype=int)
        lengths = np.linalg.norm(points[ordered] - points[node], axis=1)
        adjacency.append(ordered)
        edge_lengths.append(lengths)
        rows.extend([node] * len(ordered))
        columns.extend(ordered.tolist())
        values.extend(lengths.tolist())

    adjacency_tuple = tuple(adjacency)
    if not np.all(np.isfinite(graph_distances(adjacency_tuple, 0))):
        raise ValueError("Length-pruned distal graph is disconnected")
    weighted = csr_matrix((values, (rows, columns)), shape=(len(points), len(points)))

    centre = sphere_center(points)
    normals = points - centre
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    boundary = detect_boundary(points, adjacency_tuple, normals)
    if not np.any(boundary):
        raise ValueError("No distal-only eye boundary was detected")

    return OuterGraph(
        points=points,
        adjacency=adjacency_tuple,
        edge_lengths=tuple(edge_lengths),
        weighted=weighted,
        local_pitch=local_pitch,
        boundary=boundary,
    )


def coordinate_rank(points: np.ndarray) -> np.ndarray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    rank = np.empty(len(points), dtype=int)
    rank[order] = np.arange(len(points))
    return rank


def select_seeds(graph: OuterGraph) -> np.ndarray:
    """Deterministic farthest-point seeds with an intact radius-four guard."""
    distance_to_boundary = graph_distances(
        graph.adjacency, np.flatnonzero(graph.boundary)
    )
    candidates = np.flatnonzero(distance_to_boundary >= MIN_BOUNDARY_HOPS)
    if not len(candidates):
        raise ValueError("No nodes meet the boundary guard requirement")
    rank = coordinate_rank(graph.points)

    def choose(pool: np.ndarray, primary: np.ndarray) -> int:
        tied = pool[primary[pool] == np.max(primary[pool])]
        tied = tied[
            distance_to_boundary[tied] == np.max(distance_to_boundary[tied])
        ]
        return int(tied[np.argmin(rank[tied])])

    selected = [choose(candidates, distance_to_boundary)]
    distance_to_selected = graph_distances(graph.adjacency, selected[0])
    while len(selected) < SEEDS_PER_EYE:
        pool = candidates[
            distance_to_selected[candidates] >= MIN_SEED_SEPARATION_HOPS
        ]
        if not len(pool):
            break
        chosen = choose(pool, distance_to_selected)
        selected.append(chosen)
        distance_to_selected = np.minimum(
            distance_to_selected, graph_distances(graph.adjacency, chosen)
        )
    if len(selected) != SEEDS_PER_EYE:
        raise ValueError(
            f"Only {len(selected)} of {SEEDS_PER_EYE} required seeds satisfy "
            "the frozen separation and boundary rules"
        )
    return np.asarray(selected, dtype=int)


def nested_masks(
    graph: OuterGraph, seeds: np.ndarray
) -> dict[int, dict[int, np.ndarray]]:
    masks = {}
    for seed in seeds:
        hops = graph_distances(graph.adjacency, int(seed))
        masks[int(seed)] = {
            radius: np.flatnonzero(hops <= radius) for radius in PATCH_RADII
        }
    largest = [set(masks[int(seed)][max(PATCH_RADII)]) for seed in seeds]
    for first in range(len(largest)):
        for second in range(first + 1, len(largest)):
            if not largest[first].isdisjoint(largest[second]):
                raise AssertionError("Frozen radius-three masks overlap")
    return masks


def build_protocols(records: list[dict]) -> tuple[list[EyeProtocol], list[dict]]:
    protocols = []
    diagnostics = []
    for (volume, eye), indices in grouped_indices(records, ["volume", "eye"]):
        eye_records = [records[index] for index in indices]
        points = np.vstack([record["outer_origin"] for record in eye_records])
        graph = build_outer_graph(points)
        seeds = select_seeds(graph)
        masks = nested_masks(graph, seeds)
        metric_distances = dijkstra(graph.weighted, directed=False)
        protocols.append(
            EyeProtocol(
                volume=str(volume),
                eye=int(eye),
                records=eye_records,
                graph=graph,
                seeds=seeds,
                masks=masks,
                metric_distances=metric_distances,
            )
        )

        degrees = np.asarray([len(values) for values in graph.adjacency])
        boundary_hops = graph_distances(
            graph.adjacency, np.flatnonzero(graph.boundary)
        )
        separation = []
        for first in range(len(seeds)):
            hops = graph_distances(graph.adjacency, int(seeds[first]))
            separation.extend(hops[seeds[first + 1 :]].tolist())
        diagnostics.append(
            {
                "volume": str(volume),
                "eye": int(eye),
                "outer_qc_nodes": len(eye_records),
                "target_resolvable_nodes": int(
                    np.sum([record["target_resolvable"] for record in eye_records])
                ),
                "target_qc_pass_nodes": int(
                    np.sum([record["target_qc_pass"] for record in eye_records])
                ),
                "undirected_edges": int(np.sum(degrees) // 2),
                "degree_q05_q50_q95": [
                    float(value) for value in np.quantile(degrees, [0.05, 0.5, 0.95])
                ],
                "median_local_pitch_um": float(np.median(graph.local_pitch)),
                "boundary_nodes": int(np.sum(graph.boundary)),
                "max_boundary_distance_hops": int(np.max(boundary_hops)),
                "minimum_seed_separation_hops": int(np.min(separation)),
                "mask_sizes_by_radius": {
                    str(radius): [len(masks[int(seed)][radius]) for seed in seeds]
                    for radius in PATCH_RADII
                },
                "seed_landmark_ids": [
                    int(eye_records[int(seed)]["landmark_id"]) for seed in seeds
                ],
                "mask_landmark_ids_by_radius": {
                    str(radius): [
                        [
                            int(eye_records[int(node)]["landmark_id"])
                            for node in masks[int(seed)][radius]
                        ]
                        for seed in seeds
                    ]
                    for radius in PATCH_RADII
                },
            }
        )
    eyes_by_volume = {}
    for protocol in protocols:
        eyes_by_volume.setdefault(protocol.volume, set()).add(protocol.eye)
    if any(eyes != {0, 1} for eyes in eyes_by_volume.values()):
        raise ValueError("Experiment 59 requires both eyes (0 and 1) per volume")
    return protocols, diagnostics


def neighbour_weights(
    protocol: EyeProtocol, train_nodes: np.ndarray, test_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(train_nodes) < N_NEIGHBOURS:
        raise ValueError("Too few visible QC targets for neighbour prediction")
    selected_nodes = np.empty((len(test_nodes), N_NEIGHBOURS), dtype=int)
    selected_distances = np.empty((len(test_nodes), N_NEIGHBOURS), dtype=float)
    for row, node in enumerate(test_nodes):
        distances = protocol.metric_distances[node, train_nodes]
        order = np.lexsort((train_nodes, distances))[:N_NEIGHBOURS]
        selected_nodes[row] = train_nodes[order]
        selected_distances[row] = distances[order]
    weights = 1.0 / np.maximum(selected_distances, 1e-9) ** DISTANCE_POWER
    weights /= weights.sum(axis=1, keepdims=True)
    return selected_nodes, selected_distances, weights


def training_shape(train_thickness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.median(train_thickness, axis=1)
    shared_shape = np.median(train_thickness - depth[:, None], axis=0)
    return depth, shared_shape


def harmonic_depth(
    protocol: EyeProtocol,
    train_nodes: np.ndarray,
    train_depth: np.ndarray,
    test_nodes: np.ndarray,
) -> np.ndarray:
    """Dirichlet graph-Laplacian interpolation from visible target depths."""
    graph = protocol.graph
    rows = []
    columns = []
    values = []
    for node, (neighbours, lengths) in enumerate(
        zip(graph.adjacency, graph.edge_lengths)
    ):
        rows.extend([node] * len(neighbours))
        columns.extend(neighbours.tolist())
        values.extend((1.0 / np.maximum(lengths, 1e-9) ** 2).tolist())
    weights = csr_matrix(
        (values, (rows, columns)), shape=(len(graph.points), len(graph.points))
    )
    laplacian = diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
    observed = np.zeros(len(graph.points), dtype=bool)
    observed[train_nodes] = True
    unknown_nodes = np.flatnonzero(~observed)
    system = laplacian[unknown_nodes][:, unknown_nodes]
    rhs = weights[unknown_nodes][:, train_nodes] @ train_depth
    solution = spsolve(system, rhs)
    output = np.empty(len(graph.points), dtype=float)
    output[train_nodes] = train_depth
    output[unknown_nodes] = solution
    if not np.all(np.isfinite(output[test_nodes])):
        raise ValueError("Graph-harmonic interpolation produced non-finite depth")
    return output[test_nodes]


def predict_methods(
    protocol: EyeProtocol, train_nodes: np.ndarray, test_nodes: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train_records = [protocol.records[node] for node in train_nodes]
    test_records = [protocol.records[node] for node in test_nodes]
    train_thickness = np.vstack([record["thickness"] for record in train_records])
    train_depth, shared_shape = training_shape(train_thickness)
    selected, selected_distances, weights = neighbour_weights(
        protocol, train_nodes, test_nodes
    )
    node_to_train_row = {int(node): row for row, node in enumerate(train_nodes)}
    selected_rows = np.asarray(
        [[node_to_train_row[int(node)] for node in row] for row in selected]
    )
    local_depth = np.sum(weights * train_depth[selected_rows], axis=1)
    local_surface = np.einsum(
        "ij,ijk->ik", weights, train_thickness[selected_rows]
    )
    harmonic = harmonic_depth(
        protocol, train_nodes, train_depth, test_nodes
    )[:, None] + shared_shape[None, :]

    x_train = np.vstack([record["outer_features"] for record in train_records])
    x_test = np.vstack([record["outer_features"] for record in test_records])
    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_absolute_error"),
    )
    ridge.fit(x_train, train_thickness)

    predictions = {
        "axisymmetric_ellipsoid": np.vstack(
            [record["outer_grid"] - record["ellipsoid_inner"] for record in test_records]
        ),
        "same_eye_template": np.tile(
            np.median(train_thickness, axis=0), (len(test_nodes), 1)
        ),
        "same_eye_outer_ridge": ridge.predict(x_test),
        "nearest_visible_surface": train_thickness[selected_rows[:, 0]],
        "six_neighbor_surface": local_surface,
        "graph_harmonic_depth_shared_shape": harmonic,
        MAIN_METHOD: local_depth[:, None] + shared_shape[None, :],
    }
    return predictions, selected_distances


def predict_main(
    protocol: EyeProtocol, train_nodes: np.ndarray, test_nodes: np.ndarray
) -> np.ndarray:
    train_thickness = np.vstack(
        [protocol.records[node]["thickness"] for node in train_nodes]
    )
    train_depth, shared_shape = training_shape(train_thickness)
    selected, _, weights = neighbour_weights(protocol, train_nodes, test_nodes)
    node_to_train_row = {int(node): row for row, node in enumerate(train_nodes)}
    selected_rows = np.asarray(
        [[node_to_train_row[int(node)] for node in row] for row in selected]
    )
    local_depth = np.sum(weights * train_depth[selected_rows], axis=1)
    return local_depth[:, None] + shared_shape[None, :]


def surface_normal_error_degrees(record: dict, predicted_thickness: np.ndarray) -> float:
    grid_x, grid_y = canonical_grid()
    design = polynomial_design(grid_x, grid_y)
    predicted_inner = record["outer_grid"] - predicted_thickness
    predicted_beta, *_ = np.linalg.lstsq(design, predicted_inner, rcond=None)
    truth_beta, *_ = np.linalg.lstsq(design, record["inner_grid"], rcond=None)

    def normals(beta: np.ndarray) -> np.ndarray:
        dx = (
            beta[1] + 2.0 * beta[3] * grid_x + beta[4] * grid_y
        ) / record["scale"]
        dy = (
            beta[2] + beta[4] * grid_x + 2.0 * beta[5] * grid_y
        ) / record["scale"]
        result = np.column_stack([-dx, -dy, np.ones(len(dx))])
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    cosine = np.einsum("ij,ij->i", normals(predicted_beta), normals(truth_beta))
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.median(angle))


def run_validation(
    protocols: list[EyeProtocol],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    mask_rows = []
    grid_x, grid_y = canonical_grid()
    centre_index = int(np.argmin(grid_x * grid_x + grid_y * grid_y))

    for protocol in protocols:
        qc = np.asarray(
            [record["target_qc_pass"] for record in protocol.records], dtype=bool
        )
        for seed_rank, seed in enumerate(protocol.seeds):
            for radius in PATCH_RADII:
                test_nodes = protocol.masks[int(seed)][radius]
                hidden = np.zeros(len(protocol.records), dtype=bool)
                hidden[test_nodes] = True
                train_nodes = np.flatnonzero(qc & ~hidden)
                predictions, neighbour_distances = predict_methods(
                    protocol, train_nodes, test_nodes
                )
                hops_to_visible = graph_distances(protocol.graph.adjacency, train_nodes)
                test_records = [protocol.records[node] for node in test_nodes]
                reasons = pd.Series(
                    [record["target_reason"] for record in test_records]
                ).value_counts()
                mask_rows.append(
                    {
                        "volume": protocol.volume,
                        "eye": protocol.eye,
                        "seed_rank": seed_rank,
                        "radius_hops": radius,
                        "outer_hidden_nodes": len(test_nodes),
                        "target_resolvable_nodes": int(
                            np.sum([record["target_resolvable"] for record in test_records])
                        ),
                        "primary_qc_scored_nodes": int(
                            np.sum([record["target_qc_pass"] for record in test_records])
                        ),
                        "visible_qc_training_nodes": len(train_nodes),
                        "unresolvable_nodes": int(
                            np.sum(
                                [not record["target_resolvable"] for record in test_records]
                            )
                        ),
                        "resolvable_qc_fail_nodes": int(
                            np.sum(
                                [
                                    record["target_resolvable"]
                                    and not record["target_qc_pass"]
                                    for record in test_records
                                ]
                            )
                        ),
                        "target_reason_counts": json.dumps(
                            {str(key): int(value) for key, value in reasons.items()},
                            sort_keys=True,
                        ),
                    }
                )

                for test_row, (node, record) in enumerate(zip(test_nodes, test_records)):
                    if not record["target_resolvable"]:
                        continue
                    truth = record["thickness"]
                    target_depth = float(np.median(truth))
                    for method in METHOD_ORDER:
                        predicted = predictions[method][test_row]
                        error = np.abs(predicted - truth)
                        metric_rows.append(
                            {
                                "volume": protocol.volume,
                                "eye": protocol.eye,
                                "seed_rank": seed_rank,
                                "radius_hops": radius,
                                "landmark_id": record["landmark_id"],
                                "method": method,
                                "target_qc_pass": bool(record["target_qc_pass"]),
                                "target_reason": record["target_reason"],
                                "mae_um": float(np.mean(error)),
                                "p90_error_um": float(np.quantile(error, 0.90)),
                                "central_depth_abs_error_um": float(
                                    abs(predicted[centre_index] - truth[centre_index])
                                ),
                                "median_normal_error_degrees": surface_normal_error_degrees(
                                    record, predicted
                                ),
                                "target_depth_um": target_depth,
                                "normalized_mae": (
                                    float(np.mean(error) / target_depth)
                                    if target_depth > 0.0
                                    else float("nan")
                                ),
                                "hops_to_nearest_visible_target": int(
                                    hops_to_visible[node]
                                ),
                                "median_six_neighbor_distance_um": float(
                                    np.median(neighbour_distances[test_row])
                                ),
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(mask_rows)


def run_nested_calibration(protocols: list[EyeProtocol]) -> pd.DataFrame:
    """Calibrate the primary radius-two test from the other seven patches."""
    rows = []
    for protocol in protocols:
        qc = np.asarray(
            [record["target_qc_pass"] for record in protocol.records], dtype=bool
        )
        for seed_rank, test_seed in enumerate(protocol.seeds):
            test_nodes = protocol.masks[int(test_seed)][PRIMARY_RADIUS]
            test_hidden = np.zeros(len(protocol.records), dtype=bool)
            test_hidden[test_nodes] = True
            calibration_errors = []
            for calibration_seed in protocol.seeds:
                if calibration_seed == test_seed:
                    continue
                validation_nodes = protocol.masks[int(calibration_seed)][PRIMARY_RADIUS]
                validation_nodes = validation_nodes[qc[validation_nodes]]
                calibration_hidden = np.zeros(len(protocol.records), dtype=bool)
                calibration_hidden[
                    protocol.masks[int(calibration_seed)][PRIMARY_RADIUS]
                ] = True
                train_nodes = np.flatnonzero(qc & ~test_hidden & ~calibration_hidden)
                prediction = predict_main(protocol, train_nodes, validation_nodes)
                truth = np.vstack(
                    [protocol.records[node]["thickness"] for node in validation_nodes]
                )
                calibration_errors.extend(np.abs(prediction - truth).ravel().tolist())
            half_width = float(
                np.quantile(calibration_errors, 0.90, method="higher")
            )
            score_nodes = test_nodes[qc[test_nodes]]
            train_nodes = np.flatnonzero(qc & ~test_hidden)
            prediction = predict_main(protocol, train_nodes, score_nodes)
            truth = np.vstack(
                [protocol.records[node]["thickness"] for node in score_nodes]
            )
            covered = np.abs(prediction - truth) <= half_width
            rows.append(
                {
                    "volume": protocol.volume,
                    "eye": protocol.eye,
                    "seed_rank": seed_rank,
                    "radius_hops": PRIMARY_RADIUS,
                    "calibration_patches": SEEDS_PER_EYE - 1,
                    "calibration_points": len(calibration_errors),
                    "calibrated_half_width_um": half_width,
                    "heldout_qc_lenses": len(score_nodes),
                    "point_coverage": float(np.mean(covered)),
                    "lenses_with_at_least_90pct_point_coverage": int(
                        np.sum(np.mean(covered, axis=1) >= 0.90)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summary_row(
    cohort: str, volume: str, radius: int, method: str, group: pd.DataFrame
) -> dict:
    return {
        "cohort": cohort,
        "volume": volume,
        "radius_hops": int(radius),
        "method": method,
        "n_eyes": group[["volume", "eye"]].drop_duplicates().shape[0],
        "n_patches": group[["volume", "eye", "seed_rank"]].drop_duplicates().shape[0],
        "n_lenses": len(group),
        "median_lens_mae_um": group["mae_um"].median(),
        "p90_lens_mae_um": group["mae_um"].quantile(0.90),
        "median_normalized_mae": group["normalized_mae"].median(),
        "median_central_depth_abs_error_um": group[
            "central_depth_abs_error_um"
        ].median(),
        "median_normal_error_degrees": group[
            "median_normal_error_degrees"
        ].median(),
        "median_target_depth_um": group["target_depth_um"].median(),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cohorts = {
        "primary_target_qc": metrics[metrics["target_qc_pass"]],
        "all_resolvable_sensitivity": metrics,
    }
    for cohort, frame in cohorts.items():
        for (volume, radius, method), group in frame.groupby(
            ["volume", "radius_hops", "method"], sort=True
        ):
            rows.append(summary_row(cohort, volume, radius, method, group))
        for (radius, method), group in frame.groupby(
            ["radius_hops", "method"], sort=True
        ):
            rows.append(
                summary_row(cohort, "pooled_descriptive", radius, method, group)
            )
    return pd.DataFrame(rows)


def summarize_masks(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics["target_qc_pass"]]
    return (
        primary.groupby(
            ["volume", "eye", "seed_rank", "radius_hops", "method"],
            as_index=False,
        )
        .agg(
            scored_lenses=("landmark_id", "size"),
            median_lens_mae_um=("mae_um", "median"),
            p90_lens_mae_um=("mae_um", lambda values: values.quantile(0.90)),
            median_normalized_mae=("normalized_mae", "median"),
        )
    )


def compare_at_eye_level(mask_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = mask_summary[mask_summary["radius_hops"] == PRIMARY_RADIUS]
    eye_method = (
        primary.groupby(["volume", "eye", "method"], as_index=False)
        .agg(median_patch_mae_um=("median_lens_mae_um", "median"))
    )
    pivot = eye_method.pivot(
        index=["volume", "eye"], columns="method", values="median_patch_mae_um"
    )
    detail_rows = []
    summary_rows = []
    for reference in [method for method in METHOD_ORDER if method != MAIN_METHOD]:
        advantage = pivot[reference] - pivot[MAIN_METHOD]
        for (volume, eye), value in advantage.items():
            detail_rows.append(
                {
                    "volume": volume,
                    "eye": eye,
                    "method": MAIN_METHOD,
                    "reference": reference,
                    "reference_minus_main_mae_um": float(value),
                }
            )
        volume_advantage = advantage.groupby(level="volume").median()
        summary_rows.append(
            {
                "method": MAIN_METHOD,
                "reference": reference,
                "eyes": len(advantage),
                "median_eye_advantage_um": float(np.median(advantage)),
                "eyes_favouring_main": int(np.sum(advantage > 0.0)),
                "volumes": len(volume_advantage),
                "median_volume_advantage_um": float(np.median(volume_advantage)),
                "volumes_favouring_main": int(np.sum(volume_advantage > 0.0)),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def summarize_volume_comparisons(eye_comparisons: pd.DataFrame) -> pd.DataFrame:
    """Keep the three-volume evidential unit visible in the output."""
    return (
        eye_comparisons.groupby(
            ["volume", "method", "reference"], as_index=False
        )
        .agg(
            eyes=("eye", "size"),
            median_reference_minus_main_mae_um=(
                "reference_minus_main_mae_um",
                "median",
            ),
            eyes_favouring_main=(
                "reference_minus_main_mae_um",
                lambda values: int(np.sum(values > 0.0)),
            ),
        )
    )


def summarize_rings(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics["target_qc_pass"] & (metrics["radius_hops"] == PRIMARY_RADIUS)
    ]
    return (
        primary.groupby(
            ["volume", "method", "hops_to_nearest_visible_target"],
            as_index=False,
        )
        .agg(
            n_lenses=("landmark_id", "size"),
            median_lens_mae_um=("mae_um", "median"),
            p90_lens_mae_um=("mae_um", lambda values: values.quantile(0.90)),
        )
    )


def calibration_row(volume: str, group: pd.DataFrame) -> dict:
    lenses_with_90pct_coverage = group[
        "lenses_with_at_least_90pct_point_coverage"
    ].sum()
    heldout_qc_lenses = group["heldout_qc_lenses"].sum()
    return {
        "volume": volume,
        "method": MAIN_METHOD,
        "radius_hops": PRIMARY_RADIUS,
        "patches": len(group),
        "median_calibrated_half_width_um": group[
            "calibrated_half_width_um"
        ].median(),
        "point_coverage": np.average(
            group["point_coverage"], weights=group["heldout_qc_lenses"]
        ),
        "lenses_with_at_least_90pct_point_coverage": lenses_with_90pct_coverage,
        "fraction_lenses_with_at_least_90pct_point_coverage": (
            lenses_with_90pct_coverage / heldout_qc_lenses
        ),
        "heldout_qc_lenses": heldout_qc_lenses,
    }


def summarize_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    rows = [
        calibration_row(volume, group)
        for volume, group in calibration.groupby("volume", sort=True)
    ]
    rows.append(calibration_row("pooled_descriptive", calibration))
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, output: Path) -> None:
    primary = summary[
        (summary["cohort"] == "primary_target_qc")
        & (summary["volume"] != "pooled_descriptive")
    ]
    volumes = sorted(primary["volume"].unique())
    colours = {
        "20231107": "#555555",
        "20240530": "#C56A1A",
        "20240701": "#E5A04B",
    }
    key_methods = [
        "axisymmetric_ellipsoid",
        "same_eye_template",
        "nearest_visible_surface",
        "six_neighbor_surface",
        MAIN_METHOD,
        "graph_harmonic_depth_shared_shape",
    ]
    labels = {
        "axisymmetric_ellipsoid": "Outer-only\nellipsoid",
        "same_eye_template": "Same-eye\ntemplate",
        "nearest_visible_surface": "Nearest visible\nsurface",
        "six_neighbor_surface": "Six-neighbour\nfull surface",
        MAIN_METHOD: "Six-neighbour\nprimary",
        "graph_harmonic_depth_shared_shape": "Graph-harmonic\nsecondary",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 4.8), constrained_layout=True)

    x = np.arange(len(key_methods))
    for volume in volumes:
        values = []
        for method in key_methods:
            row = primary[
                (primary["volume"] == volume)
                & (primary["radius_hops"] == PRIMARY_RADIUS)
                & (primary["method"] == method)
            ]
            values.append(float(row["median_lens_mae_um"].iloc[0]))
        axes[0].plot(
            x, values, marker="o", linewidth=1.7,
            color=colours.get(volume), label=volume,
        )
    axes[0].set_xticks(x, [labels[method] for method in key_methods])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median axial canonical-grid MAE (µm, log scale)")
    axes[0].set_title("Primary 19-lens-scale patches — selected methods")
    axes[0].legend(frameon=False)

    for volume in volumes:
        rows = primary[
            (primary["volume"] == volume) & (primary["method"] == MAIN_METHOD)
        ].sort_values("radius_hops")
        axes[1].plot(
            rows["radius_hops"], rows["median_lens_mae_um"], marker="o",
            linewidth=1.7, color=colours.get(volume), label=volume,
        )
    axes[1].set_xticks(
        PATCH_RADII,
        ["radius 1\n≈7 lenses", "radius 2\n≈19 lenses", "radius 3\n≈37 lenses"],
    )
    axes[1].set_ylabel("Median axial canonical-grid MAE (µm)")
    axes[1].set_title("Sensitivity to contiguous loss size")
    figure.suptitle(
        "Experiment 59 — same-eye neighbour reconstruction of patchy loss",
        fontsize=12,
    )
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest)
    manifest_volumes = [str(row["volume"]) for row in manifest_rows]
    if len(manifest_volumes) != len(set(manifest_volumes)):
        raise ValueError("Experiment 59 manifest contains duplicate volume labels")
    if set(manifest_volumes) != EXPECTED_VOLUMES:
        raise ValueError(
            "Experiment 59 requires exactly 20231107, 20240530 and 20240701"
        )
    records, source_diagnostics = prepare_all(
        manifest_rows, include_invalid_targets=True
    )
    protocols, graph_diagnostics = build_protocols(records)
    metrics, masks = run_validation(protocols)
    calibration = run_nested_calibration(protocols)
    summary = summarize_metrics(metrics)
    mask_summary = summarize_masks(metrics)
    eye_comparisons, comparisons = compare_at_eye_level(mask_summary)
    volume_comparisons = summarize_volume_comparisons(eye_comparisons)
    rings = summarize_rings(metrics)
    calibration_summary = summarize_calibration(calibration)

    metrics.to_csv(args.output / "experiment_59_per_lens_metrics.csv", index=False)
    masks.to_csv(args.output / "experiment_59_mask_counts.csv", index=False)
    summary.to_csv(args.output / "experiment_59_summary.csv", index=False)
    mask_summary.to_csv(args.output / "experiment_59_patch_summary.csv", index=False)
    eye_comparisons.to_csv(
        args.output / "experiment_59_eye_comparisons.csv", index=False
    )
    comparisons.to_csv(args.output / "experiment_59_comparisons.csv", index=False)
    volume_comparisons.to_csv(
        args.output / "experiment_59_volume_comparisons.csv", index=False
    )
    rings.to_csv(args.output / "experiment_59_ring_summary.csv", index=False)
    calibration.to_csv(
        args.output / "experiment_59_calibration_by_patch.csv", index=False
    )
    calibration_summary.to_csv(
        args.output / "experiment_59_calibration_summary.csv", index=False
    )
    (args.output / "experiment_59_diagnostics.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "graph_neighbours": GRAPH_NEIGHBOURS,
                    "graph_edge_factor": GRAPH_EDGE_FACTOR,
                    "boundary_gap_degrees": BOUNDARY_GAP_DEGREES,
                    "seeds_per_eye": SEEDS_PER_EYE,
                    "minimum_boundary_hops": MIN_BOUNDARY_HOPS,
                    "minimum_seed_separation_hops": MIN_SEED_SEPARATION_HOPS,
                    "patch_radii_hops": PATCH_RADII,
                    "primary_radius_hops": PRIMARY_RADIUS,
                    "prediction_neighbours": N_NEIGHBOURS,
                    "distance_power": DISTANCE_POWER,
                    "random_seed_for_upstream_processing": SEED,
                    "mask_input": "outer-QC distal-cap centroids only",
                    "training_labels": "same-eye target-QC proximal surfaces outside mask",
                    "primary_scoring": "target-resolvable and target-QC-pass nodes inside outer-defined mask",
                    "sensitivity_scoring": "all target-resolvable nodes inside outer-defined mask",
                },
                "sources": source_diagnostics,
                "eyes": graph_diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    plot_results(summary, args.output / "experiment_59_comparison.png")

    printed = summary[
        (summary["cohort"] == "primary_target_qc")
        & (summary["radius_hops"] == PRIMARY_RADIUS)
        & (
            summary["method"].isin(
                [MAIN_METHOD, "same_eye_template", "axisymmetric_ellipsoid"]
            )
        )
    ]
    print(printed.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)
    print(calibration_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

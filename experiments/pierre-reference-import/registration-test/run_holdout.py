#!/usr/bin/env python3
"""Register supplied Apis data and evaluate fixed blocked direction predictors.

Inputs stay read-only. Requires numpy, scipy and Pillow. No MATLAB execution.
The six author runs are one eye; this is a within-eye test, not fossil validation.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import cKDTree


def nodes(v):
    if isinstance(v, dict):
        if 'type' in v:
            yield v
        yield from nodes(v.get('children', []))
    elif isinstance(v, (list, np.ndarray)):
        for child in v:
            yield from nodes(child)


def figure_arrays(z, member):
    with warnings.catch_warnings(record=True):
        fig = loadmat(io.BytesIO(z.read(member)), simplify_cells=True)
    arrays = []
    for n in nodes(fig['hgS_070000']):
        p = n.get('properties', {})
        if all(k in p for k in ('XData', 'YData', 'ZData')):
            arrays.append(np.column_stack([np.atleast_1d(p[k]) for k in ('XData', 'YData', 'ZData')]))
    return arrays


def unit(x):
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError('Undefined direction')
    return x / length


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(unit(a) * unit(b), axis=-1)), 0, 1)))


def summary(x):
    return {'n': len(x), 'median_deg': float(np.median(x)),
            'mean_deg': float(np.mean(x)), 'p90_deg': float(np.quantile(x, 0.9))}


def normals(surface, query, interior, k=256):
    """Quadratic local surface fit; use only the visible corneal point cloud."""
    tree = cKDTree(surface)
    _, inds = tree.query(query, k=k, workers=2)
    result = []
    for q, ids in zip(query, inds):
        delta = surface[ids] - q
        _, basis = np.linalg.eigh(np.cov(delta.T))
        tangent = basis[:, 1:]
        u, v = (delta @ tangent).T
        scale = max(float(np.sqrt(np.mean(u*u + v*v))), 1e-12)
        u, v = u / scale, v / scale
        design = np.column_stack([np.ones(k), u, v, u*u, u*v, v*v])
        fit = np.linalg.lstsq(design, delta @ basis[:, 0], rcond=None)[0]
        n = basis[:, 0] - tangent @ (fit[1:3] / scale)
        n = unit(n)
        result.append(n if np.dot(n, q - interior) >= 0 else -n)
    return np.asarray(result)


def features(x):
    u, v = x.T
    return np.column_stack([np.ones(len(x)), u, v, u*u, u*v, v*v])


def ridge(x, y):
    penalty = np.eye(x.shape[1]) * (0.001 * len(x))
    penalty[0, 0] = 0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(inputs, output):
    archive = inputs / 'Archive(3).zip'
    mask_archive = inputs / '60185_AM_only_CE.zip'
    with zipfile.ZipFile(archive) as z:
        registration = figure_arrays(z, 'Results04/CleanManualConeFig.fig')
        plotted_surface = next(a for a in registration if a.shape == (124560, 3)).astype(float)
        paired = [a.astype(float) for a in registration if a.shape == (3253, 3)]
        dst, src = paired
        design = np.column_stack([src, np.ones(len(src))])
        transform = np.linalg.lstsq(design[::2], dst[::2], rcond=None)[0]
        heldout_transform_error = float(np.max(np.linalg.norm(design[1::2] @ transform - dst[1::2], axis=1)))
        auto = figure_arrays(z, 'Results04/CleanConeViewingAxes.fig')
        segments = np.asarray([a for a in auto if a.shape == (2, 3)])
        auto_centres = next(a for a in auto if a.shape == (4423, 3))
        if not np.array_equal(auto_centres, segments[:, 0]):
            raise ValueError('Axis starts do not match cone centres')
        auto_axes = unit(segments[:, 1] - segments[:, 0])
        # Run 04's manual figure is an author registration/debug plot. Run 06
        # contains the 101 manual voxel clouds, in the same coordinate frame.
        manual_plot = figure_arrays(z, 'Results06/CleanManualConeFig.fig')
        manual_clouds = [a.astype(float) for a in manual_plot if 10 < len(a) < 10000]
        manual_surface = next(a for a in manual_plot if len(a) == 24912)
        if len(manual_clouds) != 101:
            raise ValueError('Expected the supplied 101 manual cone clouds')
    raw_surface = []
    interior_sum, interior_count = np.zeros(3), 0
    with zipfile.ZipFile(mask_archive) as z:
        for name in sorted(z.namelist()):
            if '__MACOSX' in name or not name.endswith('.tif'):
                continue
            depth = int(name.rsplit('_', 1)[1].split('.')[0]) + 1
            image = np.asarray(Image.open(io.BytesIO(z.read(name))))
            y, x = np.nonzero(image == 2)
            raw_surface.append(np.column_stack([x+1, y+1, np.full(len(x), depth)]))
            inside = image == 1
            count = int(inside.sum())
            interior_count += count
            interior_sum += [np.dot(inside.sum(0), np.arange(image.shape[1])+1),
                             np.dot(inside.sum(1), np.arange(image.shape[0])+1), count*depth]
    raw_surface = np.concatenate(raw_surface)
    surface = np.column_stack([raw_surface, np.ones(len(raw_surface))]) @ transform
    interior = np.r_[interior_sum / interior_count, 1] @ transform
    tree = cKDTree(surface)
    forward = tree.query(plotted_surface, workers=2)[0]
    reverse = cKDTree(plotted_surface).query(surface, workers=2)[0]
    manual_frame_error = float(tree.query(manual_surface, workers=2)[0].max())
    if max(forward.max(), reverse.max(), heldout_transform_error, manual_frame_error) > 1e-5:
        raise ValueError('Source coordinate registration failed')

    manual_centres, manual_axes, ratios = [], [], []
    for cloud in manual_clouds:
        values, vectors = np.linalg.eigh(np.cov(cloud.T))
        manual_centres.append(cloud.mean(0))
        manual_axes.append(vectors[:, -1])
        ratios.append(values[-1] / values[-2])
    manual_centres, manual_axes = np.asarray(manual_centres), np.asarray(manual_axes)
    # Cone centres locate evaluation sites only. Predictors receive surface
    # positions and surface normals, never hidden cone depth or target axes.
    auto_sites = surface[tree.query(auto_centres, workers=2)[1]]
    manual_sites = surface[tree.query(manual_centres, workers=2)[1]]
    auto_normals = normals(surface, auto_sites, interior)
    manual_normals = normals(surface, manual_sites, interior)
    auto_axes *= np.where(np.sum(auto_axes * auto_normals, axis=1) >= 0, 1., -1.)[:, None]

    origin = surface.mean(0)
    _, basis = np.linalg.eigh(np.cov(surface.T))
    basis = basis[:, ::-1]
    # Fix PCA signs deterministically, using only the visible surface.
    for col in range(3):
        if basis[np.argmax(abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    uv_surface = (surface-origin) @ basis[:, :2]
    cuts = np.median(uv_surface, axis=0)
    scale = uv_surface.std(0)
    auto_uv = (auto_sites-origin) @ basis[:, :2]
    manual_uv = (manual_sites-origin) @ basis[:, :2]
    x_auto, x_manual = features((auto_uv-cuts)/scale), features((manual_uv-cuts)/scale)
    auto_block = (auto_uv[:, 0] >= cuts[0]).astype(int) + 2*(auto_uv[:, 1] >= cuts[1]).astype(int)
    manual_block = (manual_uv[:, 0] >= cuts[0]).astype(int) + 2*(manual_uv[:, 1] >= cuts[1]).astype(int)
    predicted = {name: np.full((101, 3), np.nan) for name in
                 ['position_only', 'surface_plus_tilt', 'surface_global_rotation']}
    folds = []
    for block in range(4):
        signs = np.array([1 if block & 1 else -1, 1 if block & 2 else -1])
        distance = np.linalg.norm(np.maximum(-((auto_uv-cuts)*signs), 0), axis=1)
        train = distance > 80.0
        test = manual_block == block
        if not test.any() or train.sum() < 20 or np.any(train & (auto_block == block)):
            raise ValueError('Invalid spatial holdout')
        position = ridge(x_auto[train], auto_axes[train])
        residual = ridge(x_auto[train], auto_axes[train]-auto_normals[train])
        # Added after the primary run as a simpler-bias diagnostic. This is
        # not parameter tuning; it tests whether one rigid rotation suffices.
        u, _, vt = np.linalg.svd(auto_normals[train].T @ auto_axes[train])
        handedness = np.eye(3)
        handedness[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ handedness @ vt
        predicted['position_only'][test] = unit(x_manual[test] @ position)
        predicted['surface_plus_tilt'][test] = unit(manual_normals[test] + x_manual[test] @ residual)
        predicted['surface_global_rotation'][test] = manual_normals[test] @ rotation
        folds.append({'block': block, 'training_auto_cones': int(train.sum()),
                      'hidden_auto_cones': int((auto_block == block).sum()),
                      'manual_test_cones': int(test.sum())})
    errors = {'surface_normal': angle(manual_normals, manual_axes)}
    errors.update({k: angle(v, manual_axes) for k, v in predicted.items()})
    for fold in folds:
        test = manual_block == fold['block']
        fold['errors'] = {k: summary(v[test]) for k, v in errors.items()}
    results = {
        'source_hashes': {p.name: file_hash(p) for p in [archive, mask_archive]},
        'registration': {'paired_landmarks': len(src), 'fit_landmarks': len(src[::2]),
            'heldout_landmarks': len(src[1::2]), 'affine_row_matrix': transform.tolist(),
            'linear_singular_values': np.linalg.svd(transform[:3])[1].tolist(),
            'heldout_max_error': heldout_transform_error, 'surface_points': len(surface),
            'surface_bidirectional_max_error': float(max(forward.max(), reverse.max())),
            'manual_plot_surface_max_error': manual_frame_error},
        'protocol': {'specimens': 1, 'auto_training_source_run': 4, 'manual_voxel_clouds': 101,
            'manual_source_figure': 'Results06/CleanManualConeFig.fig',
            'normal_neighbours': 256, 'folds': 4, 'buffer_plot_units': 80,
            'ridge_penalty_per_training_cone': 0.001, 'polynomial_degree': 2,
            'evaluation_sites': 'Nearest corneal point to each supplied manual cone centre',
            'training_targets': 'Author automatic directions outside the hidden quadrant and buffer',
            'evaluation_targets': 'Leading PCA axes of author manual voxel clouds',
            'angles': 'Unsigned axis angle in degrees',
            'parameter_selection': 'Fixed before the first error calculation; no search or tuning'},
        'additional_control': 'One global rotation of normals fitted per training fold; added after viewing primary results to test whether simple orientation bias explains the gain.',
        'manual_axis_eigenvalue_ratio_range': [float(min(ratios)), float(max(ratios))],
        'overall': {k: summary(v) for k, v in errors.items()}, 'folds': folds,
        'surface_plus_tilt_better_than_normal_count': int((errors['surface_plus_tilt'] < errors['surface_normal']).sum()),
        'surface_plus_tilt_better_than_position_count': int((errors['surface_plus_tilt'] < errors['position_only']).sum()),
        'normal_neighbourhood_sensitivity': {str(k): summary(angle(normals(surface, manual_sites, interior, k), manual_axes)) for k in [64, 1024]},
        'limitations': [
            'One modern eye, not independent specimen replication or fossil validation.',
            'Automatic labels and author curation predate this holdout; upstream processing was not rerun with regions hidden.',
            'Manual labels are an anatomical reference, not error-free ground truth or independent raw-image validation.',
            'Known cone centres provide evaluation-site correspondence. Facet detection and whole-eye reconstruction are not tested.',
            'Learned models use internal directions from other regions of this same eye; surface-only cross-eye transfer is not tested.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    with (output/'per_cone_errors.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['manual_figure_cloud_index_1based', 'block'] + [k+'_deg' for k in errors])
        for i in range(101):
            w.writerow([i+1, int(manual_block[i])] + [float(v[i]) for v in errors.values()])
    print(json.dumps({'registration_max_error': results['registration']['surface_bidirectional_max_error'],
                      'overall': results['overall'], 'folds': folds}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output)

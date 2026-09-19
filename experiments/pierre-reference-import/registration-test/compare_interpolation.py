#!/usr/bin/env python3
"""Add fixed interpolation controls to the existing three-eye direction test.

Read-only input archives; no MATLAB execution or full-stack extraction.
Settings are in interpolation-protocol.json. Earlier code/results stay intact.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import scipy
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

import run_holdout as frozen
import replicate_holdout as replication


HERE = Path(__file__).resolve().parent


def load_apis(inputs):
    """Load the already verified honeybee frame, preserving cloud order."""
    archive = inputs / 'Archive(3).zip'
    mask_archive = inputs / '60185_AM_only_CE.zip'
    with zipfile.ZipFile(archive) as z:
        registration = frozen.figure_arrays(z, 'Results04/CleanManualConeFig.fig')
        plotted_surface = next(a for a in registration if a.shape == (124560, 3)).astype(float)
        dst, src = [a.astype(float) for a in registration if a.shape == (3253, 3)]
        design = np.column_stack([src, np.ones(len(src))])
        transform = np.linalg.lstsq(design[::2], dst[::2], rcond=None)[0]
        landmark_error = np.linalg.norm(design[1::2] @ transform-dst[1::2], axis=1).max()
        auto = frozen.figure_arrays(z, 'Results04/CleanConeViewingAxes.fig')
        segments = np.asarray([a for a in auto if a.shape == (2, 3)])
        auto_centres = next(a for a in auto if a.shape == (4423, 3))
        if not np.array_equal(auto_centres, segments[:, 0]):
            raise ValueError('Automatic axes and centres differ')
        manual = frozen.figure_arrays(z, 'Results06/CleanManualConeFig.fig')
        clouds = [a.astype(float) for a in manual if 10 < len(a) < 10000]
        manual_surface = next(a for a in manual if len(a) == 24912)
        if len(clouds) != 101:
            raise ValueError('Unexpected honeybee manual cloud count')
    raw, interior_sum, interior_count = [], np.zeros(3), 0
    with zipfile.ZipFile(mask_archive) as z:
        for name in sorted(z.namelist()):
            if '__MACOSX' in name or not name.endswith('.tif'):
                continue
            depth = int(name.rsplit('_', 1)[1].split('.')[0]) + 1
            a = np.asarray(Image.open(io.BytesIO(z.read(name))))
            y, x = np.nonzero(a == 2)
            raw.append(np.column_stack([x+1, y+1, np.full(len(x), depth)]))
            inside = a == 1
            count = int(inside.sum())
            interior_count += count
            interior_sum += [np.dot(inside.sum(0), np.arange(a.shape[1])+1),
                             np.dot(inside.sum(1), np.arange(a.shape[0])+1), count*depth]
    raw = np.concatenate(raw)
    surface = np.column_stack([raw, np.ones(len(raw))]) @ transform
    interior = np.r_[interior_sum/interior_count, 1] @ transform
    tree = cKDTree(surface)
    registration_error = float(max(landmark_error, tree.query(plotted_surface)[0].max(),
                                   cKDTree(plotted_surface).query(surface)[0].max(),
                                   tree.query(manual_surface)[0].max()))
    if registration_error > 1e-5:
        raise ValueError('Honeybee coordinate registration failed')
    centres, axes = [], []
    for cloud in clouds:
        _, vectors = np.linalg.eigh(np.cov(cloud.T))
        centres.append(cloud.mean(0))
        axes.append(vectors[:, -1])
    metadata = {'specimen': 'apis', 'automatic_cones': 4423, 'manual_cones': 101,
                'registration_max_error': registration_error,
                'source_hashes': {p.name: frozen.file_hash(p) for p in [archive, mask_archive]}}
    return (surface, interior, auto_centres, frozen.unit(segments[:, 1]-segments[:, 0]),
            np.asarray(centres), np.asarray(axes), metadata)


def unique_targets(sites, values):
    unique, inverse, count = np.unique(sites, axis=0, return_inverse=True, return_counts=True)
    total = np.zeros((len(unique), values.shape[1]))
    np.add.at(total, inverse, values)
    return unique, total/count[:, None]


def interpolate(train, values, query, method, settings):
    sites, targets = unique_targets(train, values)
    if method == 'idw16':
        distance, indices = cKDTree(sites).query(query, k=settings['neighbours'])
        if np.any(distance <= 0):
            raise ValueError('Held-out site coincides with a training site')
        weights = distance**(-settings['inverse_distance_power'])
        weights /= weights.sum(axis=1, keepdims=True)
        result = (targets[indices]*weights[:, :, None]).sum(axis=1)
    else:
        scipy_settings = dict(settings)
        scipy_settings['neighbors'] = scipy_settings.pop('neighbours')
        result = RBFInterpolator(sites, targets, **scipy_settings)(query)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite interpolated vectors')
    return result, len(sites)


def evaluate(data, protocol, old_csv):
    surface, interior, auto_centres, auto_axes, centres, axes = data
    tree = cKDTree(surface)
    auto_sites = surface[tree.query(auto_centres, workers=2)[1]]
    sites = surface[tree.query(centres, workers=2)[1]]
    auto_normals = frozen.normals(surface, auto_sites, interior)
    normals = frozen.normals(surface, sites, interior)
    auto_axes = auto_axes*np.where((auto_axes*auto_normals).sum(1) >= 0, 1., -1.)[:, None]
    origin = surface.mean(0)
    _, basis = np.linalg.eigh(np.cov(surface.T))
    basis = basis[:, ::-1]
    for col in range(3):
        if basis[np.argmax(abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    surface_uv = (surface-origin) @ basis[:, :2]
    cuts, scale = np.median(surface_uv, axis=0), surface_uv.std(0)
    auto_uv, uv = (auto_sites-origin) @ basis[:, :2], (sites-origin) @ basis[:, :2]
    a, q = (auto_uv-cuts)/scale, (uv-cuts)/scale
    x_auto, x_manual = frozen.features(a), frozen.features(q)
    auto_block = (auto_uv[:, 0] >= cuts[0]).astype(int)+2*(auto_uv[:, 1] >= cuts[1]).astype(int)
    block_id = (uv[:, 0] >= cuts[0]).astype(int)+2*(uv[:, 1] >= cuts[1]).astype(int)
    methods = ['position_only', 'surface_plus_tilt', 'surface_global_rotation']
    methods += [name+suffix for name in protocol['comparators'] for suffix in ['_direct', '_residual']]
    prediction = {name: np.full_like(centres, np.nan) for name in methods}
    folds = []
    for block in range(4):
        signs = np.array([1 if block & 1 else -1, 1 if block & 2 else -1])
        distance = np.linalg.norm(np.maximum(-((auto_uv-cuts)*signs), 0), axis=1)
        train, test = distance > 80., block_id == block
        if not test.any() or train.sum() < 64 or np.any(train & (auto_block == block)):
            raise ValueError('Invalid spatial holdout')
        prediction['position_only'][test] = frozen.unit(x_manual[test] @ frozen.ridge(x_auto[train], auto_axes[train]))
        prediction['surface_plus_tilt'][test] = frozen.unit(normals[test]+x_manual[test] @ frozen.ridge(x_auto[train], auto_axes[train]-auto_normals[train]))
        u, _, vt = np.linalg.svd(auto_normals[train].T @ auto_axes[train])
        handedness = np.eye(3)
        handedness[-1, -1] = np.linalg.det(u @ vt)
        prediction['surface_global_rotation'][test] = normals[test] @ (u @ handedness @ vt)
        # Identical neighbourhoods and settings for direct axes and residuals.
        # Both target sets come exclusively from the training side of the split.
        values = np.column_stack([auto_axes[train], auto_axes[train]-auto_normals[train]])
        for name, settings in protocol['comparators'].items():
            result, n_unique = interpolate(a[train], values, q[test], name, settings)
            prediction[name+'_direct'][test] = frozen.unit(result[:, :3])
            prediction[name+'_residual'][test] = frozen.unit(normals[test]+result[:, 3:])
        folds.append({'block': block, 'training_auto_cones': int(train.sum()),
                      'unique_training_surface_sites': n_unique,
                      'hidden_auto_cones': int((auto_block == block).sum()),
                      'manual_test_cones': int(test.sum())})
    errors = {'surface_normal': frozen.angle(normals, axes)}
    errors.update({name: frozen.angle(value, axes) for name, value in prediction.items()})
    # A meaningful regression gate: every original per-cone result, not just
    # the rounded medians, must agree with the already committed experiment.
    with old_csv.open() as f:
        old = list(csv.DictReader(f))
    if len(old) != len(axes) or not np.array_equal(block_id, [int(row['block']) for row in old]):
        raise ValueError('Changed reference ordering or folds')
    old_methods = ['surface_normal', 'position_only', 'surface_plus_tilt', 'surface_global_rotation']
    discrepancy = {name: float(np.max(np.abs(errors[name]-np.array([float(row[name+'_deg']) for row in old])))) for name in old_methods}
    if max(discrepancy.values()) > 1e-7:
        raise ValueError('Original results changed: '+str(discrepancy))
    for fold in folds:
        fold['errors'] = {name: frozen.summary(e[block_id == fold['block']]) for name, e in errors.items()}
    result = {'overall': {name: frozen.summary(e) for name, e in errors.items()},
              'folds': folds, 'original_per_cone_max_difference_deg': discrepancy,
              'existing_correction_better_than_comparator_count': {
                  name: int((errors['surface_plus_tilt'] < errors[name]).sum())
                  for name in methods if name != 'surface_plus_tilt'}}
    return result, errors, block_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Refuse to overwrite a previous run; history is additive.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError('Output directory must be empty')
    protocol_path = HERE/'interpolation-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    combined = {'protocol': protocol, 'protocol_sha256': frozen.file_hash(protocol_path),
                'source_code_sha256': {p.name: frozen.file_hash(p) for p in
                    [Path(__file__), Path(frozen.__file__), Path(replication.__file__)]},
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                'specimens': []}
    for specimen in protocol['specimens']:
        *data, metadata = load_apis(args.inputs) if specimen == 'apis' else replication.load_source(args.inputs, specimen)
        reference = HERE/'results/per_cone_errors.csv' if specimen == 'apis' else HERE/'replication-results'/(specimen+'_per_cone_errors.csv')
        result, errors, blocks = evaluate(data, protocol, reference)
        result.update(metadata)
        combined['specimens'].append(result)
        with (args.output/(specimen+'_per_cone_errors.csv')).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['manual_figure_cloud_index_1based', 'block']+[name+'_deg' for name in errors])
            for i, block in enumerate(blocks):
                writer.writerow([i+1, int(block)]+[float(e[i]) for e in errors.values()])
        print(json.dumps({'specimen': specimen, 'overall': result['overall'],
                          'original_per_cone_max_difference_deg': result['original_per_cone_max_difference_deg']}), flush=True)
    (args.output/'comparison.json').write_text(json.dumps(combined, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

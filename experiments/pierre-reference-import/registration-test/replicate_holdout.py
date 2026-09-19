#!/usr/bin/env python3
"""Repeat the fixed honeybee holdout protocol in the two other supplied eyes.

Uses the original normal, feature, ridge and angle functions unchanged.
No hyperparameter selection, training across specimens or MATLAB execution.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import run_holdout as frozen


CONFIG = {
    'bombus': ('77970 Bombus terrestris.zip', '77970_BT_F_only_CE', 3783, 102),
    'pieris': ('Pieris napi.zip', 'butterfly_only_CE', 4334, 104),
}


def load_source(inputs, specimen):
    archive_name, mask_folder, n_auto, n_manual = CONFIG[specimen]
    path = inputs / archive_name
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if '__MACOSX' not in n]
        manual_name = next(n for n in members if n.endswith('/Results/CleanManualConeFig.fig'))
        auto_name = next(n for n in members if n.endswith('/Results/CleanConeViewingAxes.fig'))
        manual_arrays = frozen.figure_arrays(z, manual_name)
        plotted_surface = max(manual_arrays, key=len).astype(float)
        clouds = [a.astype(float) for a in manual_arrays if 10 < len(a) < 10000]
        if len(clouds) != n_manual:
            raise ValueError('Unexpected number of manual reference clouds')
        auto_arrays = frozen.figure_arrays(z, auto_name)
        segments = np.asarray([a for a in auto_arrays if a.shape == (2, 3)])
        auto_centres = next(a for a in auto_arrays if a.shape == (n_auto, 3))
        if not np.array_equal(auto_centres, segments[:, 0]):
            raise ValueError('Automatic axis starts differ from centres')
        raw, interior_sum, interior_count = [], np.zeros(3), 0
        names = sorted(n for n in members if '/'+mask_folder+'/' in n and n.lower().endswith(('.tif', '.tiff')))
        for depth, name in enumerate(names, 1):
            image = np.asarray(Image.open(io.BytesIO(z.read(name))))
            y, x = np.nonzero(image == 2)
            raw.append(np.column_stack([x+1, y+1, np.full(len(x), depth)]))
            inside = image == 1
            count = int(inside.sum())
            interior_count += count
            interior_sum += [np.dot(inside.sum(0), np.arange(image.shape[1])+1),
                             np.dot(inside.sum(1), np.arange(image.shape[0])+1), count*depth]
    raw = np.concatenate(raw)
    # Source figures contain every fifth corneal voxel, in z/y/x traversal
    # order with x changing fastest. Verify this using geometry alone.
    landmarks = raw[::5]
    if landmarks.shape != plotted_surface.shape:
        raise ValueError('Unexpected surface sampling')
    design = np.column_stack([landmarks, np.ones(len(landmarks))])
    transform = np.linalg.lstsq(design[::2], plotted_surface[::2], rcond=None)[0]
    heldout_error = np.linalg.norm(design[1::2] @ transform - plotted_surface[1::2], axis=1)
    singular_values = np.linalg.svd(transform[:3])[1]
    if heldout_error.max() > 1e-5 or not np.allclose(singular_values, 4, atol=3e-5, rtol=0):
        raise ValueError('Rigid surface registration failed')
    surface = np.column_stack([raw, np.ones(len(raw))]) @ transform
    interior = np.r_[interior_sum/interior_count, 1] @ transform
    manual_centres, manual_axes, eigen_ratios = [], [], []
    for cloud in clouds:
        val, vec = np.linalg.eigh(np.cov(cloud.T))
        manual_centres.append(cloud.mean(0))
        manual_axes.append(vec[:, -1])
        eigen_ratios.append(val[-1]/val[-2])
    meta = {
        'specimen': specimen, 'archive': archive_name, 'archive_sha256': frozen.file_hash(path),
        'manual_figure': manual_name, 'automatic_figure': auto_name,
        'automatic_cones': n_auto, 'manual_cones': n_manual,
        'surface_voxels': len(raw), 'plotted_landmarks': len(landmarks),
        'registration_fit_landmarks': len(landmarks[::2]),
        'registration_heldout_landmarks': len(landmarks[1::2]),
        'registration_heldout_max_error': float(heldout_error.max()),
        'linear_singular_values': singular_values.tolist(),
        'affine_row_matrix': transform.tolist(),
        'manual_axis_eigenvalue_ratio_range': [float(min(eigen_ratios)), float(max(eigen_ratios))],
        'registration_scope': 'Author figure plots every fifth mask surface voxel. The transform is fitted on alternating plotted points and checked on the other half; full raw surface is transformed for normal estimation.',
    }
    return (surface, interior, auto_centres, frozen.unit(segments[:, 1]-segments[:, 0]),
            np.asarray(manual_centres), np.asarray(manual_axes), meta)


def evaluate(surface, interior, auto_centres, auto_axes, manual_centres, manual_axes):
    tree = cKDTree(surface)
    auto_sites = surface[tree.query(auto_centres, workers=2)[1]]
    manual_sites = surface[tree.query(manual_centres, workers=2)[1]]
    auto_normals = frozen.normals(surface, auto_sites, interior)
    manual_normals = frozen.normals(surface, manual_sites, interior)
    auto_axes = auto_axes * np.where(np.sum(auto_axes*auto_normals, axis=1) >= 0, 1., -1.)[:, None]
    origin = surface.mean(0)
    _, basis = np.linalg.eigh(np.cov(surface.T))
    basis = basis[:, ::-1]
    for col in range(3):
        if basis[np.argmax(abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    uv_surface = (surface-origin) @ basis[:, :2]
    cuts, scale = np.median(uv_surface, axis=0), uv_surface.std(0)
    auto_uv = (auto_sites-origin) @ basis[:, :2]
    manual_uv = (manual_sites-origin) @ basis[:, :2]
    x_auto = frozen.features((auto_uv-cuts)/scale)
    x_manual = frozen.features((manual_uv-cuts)/scale)
    auto_block = (auto_uv[:, 0] >= cuts[0]).astype(int) + 2*(auto_uv[:, 1] >= cuts[1]).astype(int)
    manual_block = (manual_uv[:, 0] >= cuts[0]).astype(int) + 2*(manual_uv[:, 1] >= cuts[1]).astype(int)
    prediction = {k: np.full_like(manual_centres, np.nan) for k in
                  ['position_only', 'surface_plus_tilt', 'surface_global_rotation']}
    folds = []
    for block in range(4):
        signs = np.array([1 if block & 1 else -1, 1 if block & 2 else -1])
        distance = np.linalg.norm(np.maximum(-((auto_uv-cuts)*signs), 0), axis=1)
        train, test = distance > 80., manual_block == block
        if not test.any() or train.sum() < 20 or np.any(train & (auto_block == block)):
            raise ValueError('Invalid spatial holdout')
        position = frozen.ridge(x_auto[train], auto_axes[train])
        residual = frozen.ridge(x_auto[train], auto_axes[train]-auto_normals[train])
        u, _, vt = np.linalg.svd(auto_normals[train].T @ auto_axes[train])
        handedness = np.eye(3)
        handedness[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ handedness @ vt
        prediction['position_only'][test] = frozen.unit(x_manual[test] @ position)
        prediction['surface_plus_tilt'][test] = frozen.unit(manual_normals[test]+x_manual[test] @ residual)
        prediction['surface_global_rotation'][test] = manual_normals[test] @ rotation
        folds.append({'block': block, 'manual_test_cones': int(test.sum()),
                      'training_auto_cones': int(train.sum()),
                      'hidden_auto_cones': int((auto_block == block).sum())})
    errors = {'surface_normal': frozen.angle(manual_normals, manual_axes)}
    errors.update({k: frozen.angle(v, manual_axes) for k, v in prediction.items()})
    for fold in folds:
        fold['errors'] = {k: frozen.summary(v[manual_block == fold['block']]) for k, v in errors.items()}
    out = {'overall': {k: frozen.summary(v) for k, v in errors.items()}, 'folds': folds,
           'correction_better_than_position_count': int((errors['surface_plus_tilt'] < errors['position_only']).sum()),
           'correction_better_than_rotation_count': int((errors['surface_plus_tilt'] < errors['surface_global_rotation']).sum())}
    return out, errors, manual_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = {
        'protocol_source_sha256': frozen.file_hash(Path(frozen.__file__)),
        'fixed_settings': {'normal_neighbours': 256, 'spatial_folds': 4,
                           'buffer_plot_units': 80, 'polynomial_degree': 2,
                           'ridge_penalty_per_training_cone': 0.001},
        'scope': 'Unchanged within-eye procedure in two further specimens; each fit learns from visible directions in its own eye. No frozen coefficients transferred between eyes.',
        'specimens': [],
    }
    for specimen in CONFIG:
        *data, meta = load_source(args.inputs, specimen)
        result, errors, blocks = evaluate(*data)
        result.update(meta)
        combined['specimens'].append(result)
        with (args.output/(specimen+'_per_cone_errors.csv')).open('w') as f:
            w = csv.writer(f)
            w.writerow(['manual_figure_cloud_index_1based', 'block']+[k+'_deg' for k in errors])
            for i in range(len(blocks)):
                w.writerow([i+1, int(blocks[i])]+[float(v[i]) for v in errors.values()])
        print(json.dumps({'specimen': specimen, 'overall': result['overall'],
                          'registration_max_error': result['registration_heldout_max_error']}), flush=True)
    (args.output/'replication.json').write_text(json.dumps(combined, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

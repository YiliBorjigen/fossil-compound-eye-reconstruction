#!/usr/bin/env python3
"""Read the additional author-supplied archives without expanding TIFF stacks.

Requires numpy, scipy, h5py. No MATLAB code is executed. Source arrays are
not exported; the output contains file checks and descriptive counts only.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import warnings
import zipfile

import h5py
import numpy as np
from scipy.io import loadmat


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def usable(name):
    return not name.startswith('__MACOSX/') and not PurePosixPath(name).name.startswith('.')


def nodes(value):
    if isinstance(value, dict):
        if 'type' in value:
            yield value
        yield from nodes(value.get('children', []))
    elif isinstance(value, (list, np.ndarray)):
        for child in value:
            yield from nodes(child)


def inspect(directory):
    reports, runs, comparisons = [], [], []
    names = ['Archive.zip', 'Archive 2.zip', 'Archive(1).zip',
             'Archive(2).zip', 'Archive(3).zip', 'Archive(4).zip',
             '60185_AM_only_CE.zip']
    comparison_arrays = {}
    for name in names:
        path = directory / name
        with zipfile.ZipFile(path) as z:
            members = [i for i in z.infolist() if not i.is_dir() and usable(i.filename)]
            reports.append({
                'archive': name, 'sha256': digest(path), 'archive_bytes': path.stat().st_size,
                'source_files': len(members),
                'uncompressed_file_bytes': sum(i.file_size for i in members),
                'extensions': dict(Counter(PurePosixPath(i.filename).suffix for i in members)),
                'matlab_scripts': [i.filename for i in members if i.filename.endswith('.m')],
            })
            for item in members:
                member = item.filename
                if member.endswith('/trustedConeCenters.mat'):
                    with h5py.File(io.BytesIO(z.read(member))) as h:
                        flags = h['trustedConeCenters'][()].ravel()
                    if not np.isin(flags, [0, 1]).all():
                        raise ValueError('Unexpected selection values: ' + member)
                    folder = str(PurePosixPath(member).parent)
                    figure_name = folder + '/CleanConeViewingAxes.fig'
                    with warnings.catch_warnings(record=True) as caught:
                        figure = loadmat(io.BytesIO(z.read(figure_name)), simplify_cells=True)
                    properties = [n['properties'] for n in nodes(figure['hgS_070000'])
                                  if n['type'] == 'graph2d.lineseries']
                    points = [np.column_stack([p[k] for k in ['XData', 'YData', 'ZData']])
                              for p in properties if all(k in p for k in ['XData', 'YData', 'ZData'])]
                    segments = np.asarray([a for a in points if a.shape == (2, 3)])
                    centres = [a for a in points if a.shape == (len(segments), 3)]
                    if len(centres) != 1 or not len(segments):
                        raise ValueError('Ambiguous plotted centres: ' + figure_name)
                    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
                    angular_name = folder + '/AngularDifManualAutoCone.fig'
                    with warnings.catch_warnings(record=True):
                        angular = loadmat(io.BytesIO(z.read(angular_name)), simplify_cells=True)
                    coloured_points = []
                    for node in nodes(angular['hgS_070000']):
                        if node['type'] != 'specgraph.scattergroup':
                            continue
                        p = node['properties']
                        n = np.asarray(p.get('XData', [])).size
                        if n > 3 and np.asarray(p.get('CData', [])).size == n:
                            coloured_points.append(n)
                    if len(coloured_points) != 1:
                        raise ValueError('Ambiguous angular-discrepancy scatter: ' + angular_name)
                    runs.append({
                        'archive': name, 'run': int(folder.removeprefix('Results')),
                        'selection_member': member, 'selection_entries': len(flags),
                        'selection_entries_marked_keep': int(np.count_nonzero(flags)),
                        'figure_member': figure_name, 'plotted_axis_segments': len(segments),
                        'all_segments_finite': bool(np.isfinite(segments).all()),
                        'all_segments_nonzero': bool((lengths > 0).all()),
                        'line_starts_equal_plotted_centres': bool(np.array_equal(segments[:, 0], centres[0])),
                        'display_line_length_range': [float(lengths.min()), float(lengths.max())],
                        'loadmat_warnings': sorted(set(str(w.message) for w in caught)),
                        'angular_comparison_figure': angular_name,
                        'angular_comparison_points': coloured_points[0],
                    })
                elif 'comparisonAutoToManualCones' in member and member.endswith('.mat'):
                    with h5py.File(io.BytesIO(z.read(member))) as h:
                        a = h['ToSave'][()].T
                    comparison_arrays[PurePosixPath(member).name] = a
                    comparisons.append({
                        'member': member, 'rows_columns': list(a.shape),
                        'all_finite': bool(np.isfinite(a).all()),
                        'second_column_values': np.unique(a[:, 1]).tolist(),
                        'second_column_ones': int(np.count_nonzero(a[:, 1] == 1)),
                        'column_meanings': 'Not established by this inspection; no accuracy is reported.',
                    })
    grid_path = directory / 'CCorientationOnDistanceBasedGrid.mat'
    with h5py.File(grid_path) as h:
        grid = h['CCorientationOnDistanceBasedGrid'][()].T
    csv_path = directory / 'measured_ManyPoints.csv'
    measurements = np.loadtxt(csv_path, delimiter=',')
    calculated_lengths = np.linalg.norm(measurements[:, 1:4] - measurements[:, 4:7], axis=1)
    return {
        'archives': reports, 'runs': sorted(runs, key=lambda r: r['run']),
        'comparisons': comparisons,
        'unnumbered_comparison_equals_02': bool(np.array_equal(
            comparison_arrays['comparisonAutoToManualCones.mat'],
            comparison_arrays['comparisonAutoToManualCones02.mat'])),
        'orientation_grid': {
            'file': grid_path.name, 'sha256': digest(grid_path),
            'rows_columns': list(grid.shape),
            'finite_entries_per_column': np.isfinite(grid).sum(axis=0).tolist(),
            'fully_finite_rows': int(np.isfinite(grid).all(axis=1).sum()),
        },
        'endpoint_measurements': {
            'file': csv_path.name, 'sha256': digest(csv_path),
            'rows_columns': list(measurements.shape),
            'all_finite': bool(np.isfinite(measurements).all()),
            'maximum_difference_first_column_vs_endpoint_distance':
                float(abs(calculated_lengths - measurements[:, 0]).max()),
            'units_and_registration': 'Not established by the headerless file.',
        },
        'interpretation': {
            'source_study': 'https://doi.org/10.1186/s40850-021-00101-w',
            'runs': 'The 1/2/4/6/9/12 region settings match the published Apis experiment; do not count them as six independent specimens.',
            'directions': 'Numeric centres and display segments are readable. Segment lengths are display scaling, not cone lengths. Registration to TIFF coordinates and original object IDs remains unresolved.',
            'selection': 'Keep flags are not final unique cone counts and have not been applied to TIFF objects.',
            'status': 'Source-data inspection. No reconstruction test or new prediction accuracy.',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upload_directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inspect(args.upload_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'runs': len(result['runs']), 'plotted_axes_per_run':
                      [r['plotted_axis_segments'] for r in result['runs']],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()

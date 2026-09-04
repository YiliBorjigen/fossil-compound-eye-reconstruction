#!/usr/bin/env python3
"""Create the frozen, outcome-blind Experiment 63 instance-QC sample.

The renderer deliberately has a very small input surface.  It reads one
distal-only sampling table and, for the selected lenses, only the corresponding
instance and sealed-distal NPZ archives.  It never opens the fitted-lens archive
or any target, proximal, prediction, error, or model output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = "experiment63.instance-qc-sample.v1"
SAMPLING_ALGORITHM = "experiment63_instance_qc_v1"
SAMPLE_DIRECTORY_NAME = "instance_qc_visual_sample"
EXPECTED_SAMPLE_SIZE = 32
N_RADIAL_STRATA = 4
N_SCALE_STRATA = 4
N_PER_CELL = 2

SAMPLING_FIELDS = (
    "eye_id",
    "lens_index",
    "seed_id",
    "distal_eligible",
    "position_u_um",
    "position_v_um",
    "distal_scale_um",
    "instance_relpath",
    "sealed_distal_relpath",
)

INSTANCE_MEMBERS = frozenset(
    {
        "schema_version",
        "lens_index",
        "full_assigned_points_zyx",
        "main_component_points_zyx",
        "component_sizes_descending",
        "spacing_um",
        "seed_source_zyx",
        "oda_axis_source_zyx",
        "config_json",
        "config_sha256",
    }
)
INSTANCE_ACCESSED_MEMBERS = (
    "schema_version",
    "lens_index",
    "full_assigned_points_zyx",
    "main_component_points_zyx",
    "spacing_um",
)
SEALED_DISTAL_MEMBERS = frozenset(
    {
        "schema_version",
        "lens_index",
        "points_zyx",
        "spacing_um",
        "config_json",
        "config_sha256",
    }
)
SEALED_DISTAL_ACCESSED_MEMBERS = (
    "schema_version",
    "lens_index",
    "points_zyx",
    "spacing_um",
)
FORBIDDEN_INPUT_ROLES = (
    "fitted lens NPZ",
    "proximal surface or target",
    "prediction",
    "error or score",
    "trained model or model output",
)
FORBIDDEN_NAME_TOKENS = (
    "proximal",
    "target",
    "prediction",
    "predicted",
    "error",
    "score",
    "model",
)


class QCError(RuntimeError):
    """Raised when the frozen QC contract is not satisfied."""


@dataclass(frozen=True)
class SamplingRow:
    eye_id: str
    lens_index: int
    seed_id: str
    position_u_um: float
    position_v_um: float
    distal_scale_um: float
    instance_relpath: str
    sealed_distal_relpath: str

    @property
    def radius_um(self) -> float:
        return math.hypot(self.position_u_um, self.position_v_um)


@dataclass(frozen=True)
class SelectedRow:
    row: SamplingRow
    radial_rank: int
    radial_stratum: int
    scale_rank_within_radial: int
    scale_stratum: int
    selection_sha256: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    try:
        display_path = path.relative_to(relative_to).as_posix() if relative_to else path.name
    except ValueError as exc:
        raise QCError(f"Artifact escapes its declared root: {path}") from exc
    return {
        "relative_path": display_path,
        "sha256": _sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise QCError(f"Row {row_number}: {field} must be literal 'true' or 'false'")


def _parse_finite_float(value: str, *, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise QCError(f"Row {row_number}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise QCError(f"Row {row_number}: {field} must be finite")
    return result


def read_sampling_table(path: Path, *, eye_id: str) -> tuple[list[SamplingRow], int]:
    """Read only the frozen distal-only sampling fields.

    Requiring an exact header is intentional: adding a target-bearing column to
    this file must stop the visual-QC workflow, even if a future implementation
    would otherwise ignore that column.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLING_FIELDS:
            raise QCError(
                "Sampling table header is not the exact outcome-blind allowlist; "
                f"expected {SAMPLING_FIELDS}, got {tuple(reader.fieldnames or ())}"
            )
        eligible: list[SamplingRow] = []
        n_rows = 0
        seen_indices: set[int] = set()
        seen_seed_ids: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            n_rows += 1
            if raw["eye_id"] != eye_id:
                raise QCError(
                    f"Row {row_number}: eye_id {raw['eye_id']!r} does not match {eye_id!r}"
                )
            try:
                lens_index = int(raw["lens_index"])
            except ValueError as exc:
                raise QCError(f"Row {row_number}: lens_index is not an integer") from exc
            if lens_index < 0 or lens_index in seen_indices:
                raise QCError(f"Row {row_number}: duplicate or negative lens_index {lens_index}")
            seed_id = raw["seed_id"]
            if not seed_id or seed_id in seen_seed_ids:
                raise QCError(f"Row {row_number}: empty or duplicate seed_id {seed_id!r}")
            seen_indices.add(lens_index)
            seen_seed_ids.add(seed_id)
            is_eligible = _parse_bool(
                raw["distal_eligible"], field="distal_eligible", row_number=row_number
            )
            if not is_eligible:
                continue
            row = SamplingRow(
                    eye_id=eye_id,
                    lens_index=lens_index,
                    seed_id=seed_id,
                    position_u_um=_parse_finite_float(
                        raw["position_u_um"], field="position_u_um", row_number=row_number
                    ),
                    position_v_um=_parse_finite_float(
                        raw["position_v_um"], field="position_v_um", row_number=row_number
                    ),
                    distal_scale_um=_parse_finite_float(
                        raw["distal_scale_um"], field="distal_scale_um", row_number=row_number
                    ),
                    instance_relpath=raw["instance_relpath"],
                    sealed_distal_relpath=raw["sealed_distal_relpath"],
                )
            if row.distal_scale_um <= 0:
                raise QCError(f"Row {row_number}: distal_scale_um must be positive")
            eligible.append(row)
    if n_rows == 0:
        raise QCError("Sampling table is empty")
    if sorted(seen_indices) != list(range(n_rows)):
        raise QCError("Sampling table must contain every lens_index exactly once from 0 to N-1")
    return eligible, n_rows


def _stable_stratum(rank: int, population_size: int, n_strata: int) -> int:
    if not (0 <= rank < population_size):
        raise QCError("Internal stable-rank error")
    return min(n_strata - 1, (rank * n_strata) // population_size)


def select_frozen_sample(rows: Sequence[SamplingRow]) -> list[SelectedRow]:
    """Select exactly two hash-minimal rows from every 4x4 stable-rank cell."""

    radial_order = sorted(rows, key=lambda r: (r.radius_um, r.lens_index, r.seed_id))
    ranked: list[tuple[SamplingRow, int, int]] = []
    for radial_rank, row in enumerate(radial_order):
        radial_stratum = _stable_stratum(radial_rank, len(radial_order), N_RADIAL_STRATA)
        ranked.append((row, radial_rank, radial_stratum))

    cells: dict[tuple[int, int], list[SelectedRow]] = {
        (radial, scale): []
        for radial in range(N_RADIAL_STRATA)
        for scale in range(N_SCALE_STRATA)
    }
    for radial_stratum in range(N_RADIAL_STRATA):
        members = [item for item in ranked if item[2] == radial_stratum]
        members.sort(key=lambda item: (item[0].distal_scale_um, item[0].lens_index, item[0].seed_id))
        for scale_rank, (row, radial_rank, _) in enumerate(members):
            scale_stratum = _stable_stratum(scale_rank, len(members), N_SCALE_STRATA)
            token = (
                f"{SAMPLING_ALGORITHM}|{row.eye_id}|{row.lens_index}|{row.seed_id}"
            ).encode("utf-8")
            selected = SelectedRow(
                row=row,
                radial_rank=radial_rank,
                radial_stratum=radial_stratum,
                scale_rank_within_radial=scale_rank,
                scale_stratum=scale_stratum,
                selection_sha256=hashlib.sha256(token).hexdigest(),
            )
            cells[(radial_stratum, scale_stratum)].append(selected)

    shortages = {cell: len(values) for cell, values in cells.items() if len(values) < N_PER_CELL}
    if shortages:
        raise QCError(f"Cannot construct the frozen 32-lens sample; undersized cells: {shortages}")

    result: list[SelectedRow] = []
    for cell in sorted(cells):
        chosen = sorted(
            cells[cell], key=lambda item: (item.selection_sha256, item.row.lens_index, item.row.seed_id)
        )[:N_PER_CELL]
        result.extend(chosen)
    if len(result) != EXPECTED_SAMPLE_SIZE:
        raise QCError(f"Frozen sampling produced {len(result)} rows instead of 32")
    return result


def _resolve_exact_artifact(
    bundle_root: Path, relpath: str, *, prefix: str, lens_index: int
) -> Path:
    pure = PurePosixPath(relpath)
    expected = PurePosixPath(prefix) / f"lens_{lens_index:06d}.npz"
    if pure != expected or pure.is_absolute() or ".." in pure.parts:
        raise QCError(
            f"Lens {lens_index}: {prefix} path must be exactly {expected.as_posix()!r}"
        )
    root = bundle_root.resolve()
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root / prefix)
    except ValueError as exc:
        raise QCError(f"Lens {lens_index}: artifact escapes the {prefix} root") from exc
    if not path.is_file():
        raise QCError(f"Lens {lens_index}: missing artifact {path}")
    return path


def _np_scalar_text(array: np.ndarray, *, name: str) -> str:
    value = np.asarray(array)
    if value.shape != ():
        raise QCError(f"{name} must be a scalar string")
    return str(value.item())


def _np_scalar_int(array: np.ndarray, *, name: str) -> int:
    value = np.asarray(array)
    if value.shape != () or value.dtype.kind not in "iu":
        raise QCError(f"{name} must be a scalar integer")
    return int(value.item())


def _validate_points(array: np.ndarray, *, name: str, require_nonempty: bool) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[1] != 3 or value.dtype.kind not in "iu":
        raise QCError(f"{name} must be an integer N x 3 array")
    if require_nonempty and value.shape[0] == 0:
        raise QCError(f"{name} is empty for a distal-QC-eligible sampled lens")
    return value.astype(np.int64, copy=False)


def _load_instance_for_render(path: Path, *, lens_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        members = frozenset(archive.files)
        if members != INSTANCE_MEMBERS:
            raise QCError(f"Unexpected instance NPZ members in {path.name}: {sorted(members)}")
        if any(token in name.lower() for name in members for token in FORBIDDEN_NAME_TOKENS):
            raise QCError(f"Forbidden outcome-bearing member name in {path.name}")
        schema = _np_scalar_text(archive["schema_version"], name="schema_version")
        stored_index = _np_scalar_int(archive["lens_index"], name="lens_index")
        full = _validate_points(
            archive["full_assigned_points_zyx"], name="full_assigned_points_zyx", require_nonempty=True
        )
        main = _validate_points(
            archive["main_component_points_zyx"], name="main_component_points_zyx", require_nonempty=True
        )
        spacing = np.asarray(archive["spacing_um"], dtype=np.float64)
    if schema != "experiment63.instance.v2":
        raise QCError(f"Wrong instance schema {schema!r}")
    if stored_index != lens_index:
        raise QCError(f"Instance lens index {stored_index} does not match {lens_index}")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise QCError("spacing_um must contain three positive finite values")
    full_set = {tuple(row) for row in full.tolist()}
    if len(full_set) != len(full):
        raise QCError(f"Lens {lens_index}: duplicate coordinates in full assigned mask")
    if any(tuple(row) not in full_set for row in main.tolist()):
        raise QCError(f"Lens {lens_index}: dominant component is not a subset of assigned mask")
    return full, main, spacing, sorted(members)


def _load_distal_for_render(path: Path, *, lens_index: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        members = frozenset(archive.files)
        if members != SEALED_DISTAL_MEMBERS:
            raise QCError(f"Unexpected sealed-distal NPZ members in {path.name}: {sorted(members)}")
        if any(token in name.lower() for name in members for token in FORBIDDEN_NAME_TOKENS):
            raise QCError(f"Forbidden outcome-bearing member name in {path.name}")
        schema = _np_scalar_text(archive["schema_version"], name="schema_version")
        stored_index = _np_scalar_int(archive["lens_index"], name="lens_index")
        points = _validate_points(archive["points_zyx"], name="points_zyx", require_nonempty=True)
        spacing = np.asarray(archive["spacing_um"], dtype=np.float64)
    if schema != "experiment63.sealed-distal.v2":
        raise QCError(f"Wrong sealed-distal schema {schema!r}")
    if stored_index != lens_index:
        raise QCError(f"Sealed-distal lens index {stored_index} does not match {lens_index}")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise QCError("spacing_um must contain three positive finite values")
    return points, spacing, sorted(members)


def _plot_projection(
    axis: plt.Axes,
    full_xyz: np.ndarray,
    main_xyz: np.ndarray,
    distal_xyz: np.ndarray,
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
) -> None:
    a, b = dims
    axis.scatter(full_xyz[:, a], full_xyz[:, b], s=1.1, c="#777777", alpha=0.9, linewidths=0, rasterized=True)
    axis.scatter(main_xyz[:, a], main_xyz[:, b], s=0.8, c="#ffffff", alpha=0.95, linewidths=0, rasterized=True)
    axis.scatter(distal_xyz[:, a], distal_xyz[:, b], s=1.6, c="#00d6e5", alpha=1.0, linewidths=0, rasterized=True)
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#111111")
    axis.tick_params(labelsize=7)


def _render_lens(
    output_path: Path,
    *,
    selected: SelectedRow,
    full_zyx: np.ndarray,
    main_zyx: np.ndarray,
    distal_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> None:
    full_xyz = full_zyx[:, ::-1] * spacing_zyx[::-1]
    main_xyz = main_zyx[:, ::-1] * spacing_zyx[::-1]
    distal_xyz = distal_zyx[:, ::-1] * spacing_zyx[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _plot_projection(axes[0], full_xyz, main_xyz, distal_xyz, (0, 1), ("x (µm)", "y (µm)"), "XY / project z")
    _plot_projection(axes[1], full_xyz, main_xyz, distal_xyz, (0, 2), ("x (µm)", "z (µm)"), "XZ / project y")
    _plot_projection(axes[2], full_xyz, main_xyz, distal_xyz, (1, 2), ("y (µm)", "z (µm)"), "YZ / project x")
    fig.suptitle(
        f"{selected.row.eye_id} · lens {selected.row.lens_index} · seed {selected.row.seed_id} · "
        f"radial {selected.radial_stratum} / scale {selected.scale_stratum}\n"
        "assigned mask: grey · dominant component: white · sealed distal: cyan",
        fontsize=10,
    )
    fig.savefig(
        output_path,
        dpi=170,
        facecolor="white",
        metadata={"Software": "Experiment 63 outcome-blind QC renderer"},
    )
    plt.close(fig)


def render_eye_sample(
    *,
    eye_id: str,
    bundle_root: Path,
    sampling_table: Path,
    output_dir: Path,
) -> Path:
    """Render one immutable 32-lens sample and return its manifest path."""

    if not eye_id or any(character in eye_id for character in "/\\"):
        raise QCError("eye_id must be a nonempty path-safe identifier")
    bundle_root = bundle_root.resolve()
    sampling_table = sampling_table.resolve()
    output_dir = output_dir.resolve()
    expected_table = bundle_root / "distal_qc_sampling.csv"
    if sampling_table != expected_table:
        raise QCError(f"Sampling input must be exactly {expected_table}")
    expected_output_dir = bundle_root / SAMPLE_DIRECTORY_NAME
    if output_dir != expected_output_dir:
        raise QCError(f"QC sample output must be exactly {expected_output_dir}")
    if output_dir.exists():
        raise QCError(f"Refusing to overwrite existing QC sample directory: {output_dir}")

    eligible, n_inventory_rows = read_sampling_table(sampling_table, eye_id=eye_id)
    selected_rows = select_frozen_sample(eligible)
    output_parent = output_dir.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_parent))
    try:
        renders_dir = staging / "renders"
        renders_dir.mkdir()
        sample_entries: list[dict[str, Any]] = []
        for ordinal, selected in enumerate(selected_rows):
            row = selected.row
            instance_path = _resolve_exact_artifact(
                bundle_root, row.instance_relpath, prefix="instances", lens_index=row.lens_index
            )
            distal_path = _resolve_exact_artifact(
                bundle_root,
                row.sealed_distal_relpath,
                prefix="sealed_distal",
                lens_index=row.lens_index,
            )
            full, main, instance_spacing, instance_members = _load_instance_for_render(
                instance_path, lens_index=row.lens_index
            )
            distal, distal_spacing, distal_members = _load_distal_for_render(
                distal_path, lens_index=row.lens_index
            )
            if not np.array_equal(instance_spacing, distal_spacing):
                raise QCError(f"Lens {row.lens_index}: inconsistent instance/distal spacing")
            main_set = {tuple(point) for point in main.tolist()}
            if any(tuple(point) not in main_set for point in distal.tolist()):
                raise QCError(
                    f"Lens {row.lens_index}: sealed distal points are not a subset of the dominant component"
                )
            render_name = f"sample_{ordinal:02d}_r{selected.radial_stratum}_s{selected.scale_stratum}_lens_{row.lens_index:06d}.png"
            render_path = renders_dir / render_name
            _render_lens(
                render_path,
                selected=selected,
                full_zyx=full,
                main_zyx=main,
                distal_zyx=distal,
                spacing_zyx=instance_spacing,
            )
            sample_entries.append(
                {
                    "ordinal": ordinal,
                    "eye_id": row.eye_id,
                    "lens_index": row.lens_index,
                    "seed_id": row.seed_id,
                    "radius_um": row.radius_um,
                    "distal_scale_um": row.distal_scale_um,
                    "radial_rank": selected.radial_rank,
                    "radial_stratum": selected.radial_stratum,
                    "scale_rank_within_radial": selected.scale_rank_within_radial,
                    "scale_stratum": selected.scale_stratum,
                    "selection_sha256": selected.selection_sha256,
                    "instance_artifact": {
                        **_artifact_binding(instance_path, relative_to=bundle_root),
                        "members_present": instance_members,
                        "members_accessed": list(INSTANCE_ACCESSED_MEMBERS),
                    },
                    "sealed_distal_artifact": {
                        **_artifact_binding(distal_path, relative_to=bundle_root),
                        "members_present": distal_members,
                        "members_accessed": list(SEALED_DISTAL_ACCESSED_MEMBERS),
                    },
                    "render": _artifact_binding(render_path, relative_to=staging),
                    "point_counts": {
                        "full_assigned": int(len(full)),
                        "dominant_component": int(len(main)),
                        "sealed_distal": int(len(distal)),
                    },
                }
            )

        cell_counts = {
            f"r{radial}_s{scale}": sum(
                1
                for item in sample_entries
                if item["radial_stratum"] == radial and item["scale_stratum"] == scale
            )
            for radial in range(N_RADIAL_STRATA)
            for scale in range(N_SCALE_STRATA)
        }
        if set(cell_counts.values()) != {N_PER_CELL}:
            raise QCError(f"Internal sample coverage failure: {cell_counts}")
        renderer_path = Path(__file__).resolve()
        repository_root = renderer_path.parents[2]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "eye_id": eye_id,
            "review_scope": "stratified_sample_only",
            "all_instances_manually_reviewed": False,
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "sampling_rule": {
                "eligible_population": "final distal-QC rows only",
                "radial_measure": "hypot(position_u_um,position_v_um)",
                "radial_strata": N_RADIAL_STRATA,
                "scale_measure": "distal_scale_um within radial stratum",
                "scale_strata_per_radial_stratum": N_SCALE_STRATA,
                "rows_per_cell": N_PER_CELL,
                "tie_break_order": "numeric measure, lens_index, seed_id",
                "selection_order": "smallest sha256(experiment63_instance_qc_v1|eye_id|lens_index|seed_id)",
            },
            "n_inventory_rows": n_inventory_rows,
            "n_distal_qc_eligible": len(eligible),
            "n_selected": len(sample_entries),
            "cell_counts": cell_counts,
            "outcome_blindness": {
                "sampling_table_exact_field_allowlist": list(SAMPLING_FIELDS),
                "instance_members_accessed": list(INSTANCE_ACCESSED_MEMBERS),
                "sealed_distal_members_accessed": list(SEALED_DISTAL_ACCESSED_MEMBERS),
                "forbidden_input_roles": list(FORBIDDEN_INPUT_ROLES),
                "fitted_lens_npz_opened": False,
                "proximal_target_prediction_error_or_model_data_opened": False,
            },
            "sampling_table": _artifact_binding(sampling_table, relative_to=bundle_root),
            "renderer_code": {
                "relative_path": renderer_path.relative_to(repository_root).as_posix(),
                "sha256": _sha256_path(renderer_path),
                "size_bytes": renderer_path.stat().st_size,
            },
            "samples": sample_entries,
        }
        manifest_path = staging / "sample_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        return output_dir / "sample_manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye-id", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--sampling-table",
        type=Path,
        help="Defaults to BUNDLE_ROOT/distal_qc_sampling.csv and cannot point elsewhere.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Defaults to BUNDLE_ROOT/{SAMPLE_DIRECTORY_NAME} and cannot point elsewhere.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sampling_table = args.sampling_table or (args.bundle_root / "distal_qc_sampling.csv")
    output_dir = args.output_dir or (args.bundle_root / SAMPLE_DIRECTORY_NAME)
    try:
        manifest = render_eye_sample(
            eye_id=args.eye_id,
            bundle_root=args.bundle_root,
            sampling_table=sampling_table,
            output_dir=output_dir,
        )
    except QCError as exc:
        print(f"QC sample generation failed: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

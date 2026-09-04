from __future__ import annotations

import json
import unittest

import numpy as np

import experiment_64_robust_distal_core as robust_core


def _smooth_cap(step: float = 0.5, radius: float = 5.0) -> np.ndarray:
    coordinates = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float64)
    xy = np.array(
        [
            (x, y)
            for x in coordinates
            for y in coordinates
            if x * x + y * y <= radius * radius + 1.0e-12
        ],
        dtype=np.float64,
    )
    x = xy[:, 0]
    y = xy[:, 1]
    # A mildly asymmetric, smooth quadratic cap avoids relying on a repeated
    # radial ring or an arbitrary tangent-axis orientation in invariance tests.
    z = 0.030 * x * x + 0.018 * x * y + 0.045 * y * y + 0.012 * x - 0.007 * y
    return np.column_stack((x, y, z))


def _cap_with_attached_prong() -> tuple[np.ndarray, np.ndarray]:
    cap = _smooth_cap()
    prong_x = np.arange(5.25, 12.01, 0.25, dtype=np.float64)
    prong_y = np.zeros_like(prong_x)
    # Continue the cap height smoothly from its +x edge.  The first prong
    # coordinate is only one native grid step from the cap and is therefore an
    # attached peripheral spur, not a distant satellite cluster.
    prong_z = 0.030 * prong_x * prong_x + 0.012 * prong_x
    prong = np.column_stack((prong_x, prong_y, prong_z))
    points = np.vstack((cap, prong))
    # The first two coordinates are the physical connector at the cap edge;
    # the asserted prong tail begins beyond that attachment segment.
    tail = np.flatnonzero(points[:, 0] >= 5.75)
    return points, tail


def _cap_with_attached_axial_prong() -> tuple[np.ndarray, np.ndarray]:
    cap = _smooth_cap()
    prong_z = np.arange(0.25, 7.01, 0.25, dtype=np.float64)
    prong = np.column_stack(
        (
            np.full_like(prong_z, 0.13),
            np.full_like(prong_z, 0.17),
            prong_z,
        )
    )
    points = np.vstack((cap, prong))
    # Coordinates below this threshold form the connector; the long axial tail
    # is what a 10% robust trim is designed to reject.
    tail = np.flatnonzero(points[:, 2] >= 5.75)
    return points, tail


def _minimum_supported_cap() -> np.ndarray:
    cap = _smooth_cap()
    indices = np.linspace(0, len(cap) - 1, 33, dtype=np.int64)
    return cap[indices]


def _proper_rotation() -> np.ndarray:
    axis = np.asarray([0.31, -0.72, 0.62], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.83
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


class RobustDistalCoreContractTests(unittest.TestCase):
    def test_config_is_complete_canonical_and_hashed(self) -> None:
        config_json = robust_core.canonical_robust_core_config_json()
        self.assertEqual(json.dumps(json.loads(config_json), sort_keys=True, separators=(",", ":")), config_json)
        first = robust_core.robust_core_config_sha256()
        second = robust_core.robust_core_config_sha256(dict(robust_core.DEFAULT_ROBUST_CORE_CONFIG))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "unknown"):
            robust_core.normalise_robust_core_config({"unfrozen_option": 1})
        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "must remain frozen"):
            robust_core.normalise_robust_core_config({"minimum_retained_points": 28})
        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "must remain frozen"):
            robust_core.normalise_robust_core_config({"pca_trim_quantile": 0.80})

    def test_minimum_support_and_invalid_geometry_fail_closed(self) -> None:
        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "at least 33"):
            robust_core.select_robust_distal_core(_smooth_cap()[:32])

        duplicated = _minimum_supported_cap()
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "duplicate"):
            robust_core.select_robust_distal_core(duplicated)

        line = np.column_stack((np.arange(33, dtype=np.float64), np.zeros((33, 2))))
        with self.assertRaisesRegex(robust_core.RobustDistalCoreError, "not identifiable"):
            robust_core.select_robust_distal_core(line)

        result = robust_core.select_robust_distal_core(_minimum_supported_cap())
        self.assertGreaterEqual(int(np.count_nonzero(result["retained_mask"])), 27)
        self.assertGreaterEqual(result["diagnostics"]["downstream_q90_support_lower_bound"], 25)

    def test_q90_boundary_ties_are_included_without_axis_tie_breaking(self) -> None:
        inner_angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        outer_angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + 0.07
        centre = np.asarray([[0.0, 0.0, 0.0]])
        inner = np.column_stack(
            (np.cos(inner_angles), np.sin(inner_angles), np.full(24, 0.03))
        )
        outer = np.column_stack(
            (2.0 * np.cos(outer_angles), 2.0 * np.sin(outer_angles), np.full(8, 0.12))
        )
        result = robust_core.select_robust_distal_core(np.vstack((centre, inner, outer)))

        # q90 requests 30/33.  All eight outer-ring points are equal within the
        # frozen scalar tolerance, so all 33 are retained instead of choosing
        # three by an orientation-dependent coordinate order.
        self.assertEqual(result["diagnostics"]["pca_trim_requested_count"], 30)
        self.assertEqual(result["diagnostics"]["pca_trim_tie_expansion_count"], 3)
        self.assertEqual(result["diagnostics"]["lateral_q90_tie_expansion_count"], 3)
        self.assertEqual(result["diagnostics"]["retained_support"], 33)

    def test_smooth_cap_retains_a_contiguous_central_domain(self) -> None:
        cap = _smooth_cap()
        result = robust_core.select_robust_distal_core(cap)
        mask = result["retained_mask"]
        radius = np.linalg.norm(cap[:, :2], axis=1)

        self.assertTrue(np.all(mask[radius <= 3.5]))
        self.assertTrue(np.all(~mask[radius >= 4.9]))
        self.assertGreaterEqual(np.count_nonzero(mask), int(np.ceil(0.80 * len(cap))))
        self.assertEqual(
            result["diagnostics"]["retained_point_count"], int(np.count_nonzero(mask))
        )
        self.assertEqual(result["diagnostics"]["input_support"], len(cap))
        self.assertEqual(result["diagnostics"]["retained_support"], int(np.count_nonzero(mask)))
        self.assertAlmostEqual(
            result["diagnostics"]["retained_fraction"], np.count_nonzero(mask) / len(cap)
        )
        projector = result["tangent_projector"]
        np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-12)
        self.assertAlmostEqual(float(np.trace(projector)), 2.0, places=12)

    def test_attached_peripheral_prong_is_removed(self) -> None:
        points, prong_tail_indices = _cap_with_attached_prong()
        result = robust_core.select_robust_distal_core(points)
        mask = result["retained_mask"]

        self.assertFalse(np.any(mask[prong_tail_indices]))
        self.assertGreater(result["diagnostics"]["rejected_point_count"], len(prong_tail_indices))
        self.assertLess(result["diagnostics"]["lateral_core_threshold_um"], 5.75)

    def test_attached_axial_prong_is_removed_by_intersection(self) -> None:
        points, prong_tail_indices = _cap_with_attached_axial_prong()
        result = robust_core.select_robust_distal_core(points)

        # The axial prong lies in the lateral q90 but outside the Euclidean
        # q90, so the explicit intersection excludes it.
        self.assertFalse(np.any(result["retained_mask"][prong_tail_indices]))
        self.assertTrue(np.all(result["lateral_q90_mask"][prong_tail_indices]))
        self.assertFalse(np.any(result["euclidean_q90_mask"][prong_tail_indices]))
        self.assertGreater(result["diagnostics"]["lateral_only_count"], 0)
        self.assertGreater(result["diagnostics"]["full_axial_distance_quantiles_um"]["q99"], 0.5)
        self.assertGreater(result["diagnostics"]["single_sheet_p99_axial_over_lateral_q90"], 0.1)

    def test_permutation_invariance_and_original_mask_alignment(self) -> None:
        points, _ = _cap_with_attached_prong()
        baseline = robust_core.select_robust_distal_core(points)
        rng = np.random.default_rng(6401)
        permutation = rng.permutation(len(points))
        permuted = robust_core.select_robust_distal_core(points[permutation])

        restored_mask = np.empty(len(points), dtype=np.bool_)
        restored_mask[permutation] = permuted["retained_mask"]
        np.testing.assert_array_equal(restored_mask, baseline["retained_mask"])
        np.testing.assert_array_equal(
            permuted["retained_points_xyz_um"], baseline["retained_points_xyz_um"]
        )
        np.testing.assert_allclose(
            permuted["geometric_median_xyz_um"], baseline["geometric_median_xyz_um"], atol=0.0
        )
        self.assertEqual(permuted["diagnostics"], baseline["diagnostics"])

    def test_rotation_and_translation_equivariance(self) -> None:
        points, _ = _cap_with_attached_prong()
        rotation = _proper_rotation()
        translation = np.asarray([81.2, -37.4, 12.8], dtype=np.float64)
        transformed_points = points @ rotation.T + translation

        baseline = robust_core.select_robust_distal_core(points)
        transformed = robust_core.select_robust_distal_core(transformed_points)

        np.testing.assert_array_equal(transformed["retained_mask"], baseline["retained_mask"])
        np.testing.assert_allclose(
            transformed["geometric_median_xyz_um"],
            baseline["geometric_median_xyz_um"] @ rotation.T + translation,
            rtol=0.0,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(
            transformed["tangent_projector"],
            rotation @ baseline["tangent_projector"] @ rotation.T,
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            transformed["lateral_radius_um"], baseline["lateral_radius_um"], rtol=0.0, atol=2.0e-11
        )
        self.assertEqual(
            transformed["diagnostics"]["retained_point_count"],
            baseline["diagnostics"]["retained_point_count"],
        )


if __name__ == "__main__":
    unittest.main()

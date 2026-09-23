from __future__ import annotations

import unittest

from epiaudio.analysis import frontier


class FrontierGeometryTests(unittest.TestCase):
    def _point(
        self, compute: float, model: float, data: float, run: int = 0
    ) -> frontier.CodeLengthPoint:
        return frontier.CodeLengthPoint(
            compute_flops=compute, model_bits=model, data_bits=data, run_index=run
        )

    def test_hull_keeps_negative_points(self):
        points = [
            self._point(1e14, -5.0, 100.0, run=0),
            self._point(1e15, 10.0, 60.0, run=1),
            self._point(1e16, 20.0, 30.0, run=2),
        ]
        hull = frontier.lower_convex_hull(points, reduce=False)
        self.assertTrue(any(point.model_bits < 0 for point in hull))

    def test_non_finite_and_non_positive_compute_dropped(self):
        points = [
            self._point(float("inf"), 1.0, 1.0, run=0),
            self._point(0.0, 1.0, 1.0, run=1),
            self._point(1e15, 1.0, 1.0, run=2),
            self._point(1e16, 0.5, 0.5, run=3),
        ]
        hull = frontier.lower_convex_hull(points, reduce=False)
        self.assertTrue(all(point.compute_flops > 0 for point in hull))
        self.assertEqual(len(hull), 2)

    def test_median_reduction_gives_one_point_per_run(self):
        points = [
            self._point(1e14, 30.0, 100.0, run=0),
            self._point(2e14, 25.0, 90.0, run=0),
            self._point(3e14, 20.0, 80.0, run=0),
            self._point(1e16, 5.0, 20.0, run=1),
        ]
        hull = frontier.lower_convex_hull(points, reduce=True)
        self.assertEqual(len({point.run_index for point in hull}), len(hull))

    def test_endpoint_is_the_highest_compute_point(self):
        points = [
            self._point(1e14, 30.0, 100.0, run=0),
            self._point(1e16, 5.0, 20.0, run=1),
        ]
        endpoint = frontier.frontier_endpoint(
            frontier.lower_convex_hull(points, reduce=False)
        )
        assert endpoint is not None
        self.assertEqual(endpoint.compute_flops, 1e16)

    def test_headroom_slope_detects_a_still_rising_cell(self):
        hull = [
            self._point(1e14, 10.0, 1.0, 0),
            self._point(1e15, 20.0, 1.0, 1),
            self._point(1e16, 30.0, 1.0, 2),
            self._point(1e17, 40.0, 1.0, 3),
        ]
        slope = frontier.headroom_slope(hull)
        assert slope is not None
        self.assertAlmostEqual(slope, 10.0, places=6)

    def test_headroom_slope_is_flat_for_a_saturated_cell(self):
        hull = [
            self._point(1e14, 30.0, 1.0, 0),
            self._point(1e15, 30.0, 1.0, 1),
            self._point(1e16, 30.0, 1.0, 2),
        ]
        slope = frontier.headroom_slope(hull)
        self.assertIsNotNone(slope)
        assert slope is not None
        self.assertAlmostEqual(slope, 0.0)

    def test_pos_only_excludes_a_broken_point_that_would_dominate_the_hull(self):
        """A single very negative point can otherwise BE the whole hull.

        With no filter, the broken point at low compute has such a low total
        that nothing at higher compute ever improves on it, so it is the
        entire frontier -- exactly the case pos_only exists for.
        """
        broken = self._point(1e14, -50.0, 10.0, run=0)
        fine = self._point(1e15, 20.0, 5.0, run=1)

        without_filter = frontier.lower_convex_hull([broken, fine], reduce=False)
        self.assertEqual(without_filter, [broken])

        with_filter = frontier.lower_convex_hull(
            [broken, fine], reduce=False, pos_only=True
        )
        self.assertEqual(with_filter, [fine])

    def test_pos_only_excludes_exactly_zero_model_bits(self):
        """pos_only means strictly positive: a real model cannot code zero
        bits of structure, so zero has no special case to preserve.
        """
        zero = self._point(1e14, 0.0, 10.0, run=0)
        fine = self._point(1e15, 20.0, 5.0, run=1)
        hull = frontier.lower_convex_hull([zero, fine], reduce=False, pos_only=True)
        self.assertFalse(any(point.model_bits == 0.0 for point in hull))

    def test_summarize_frontier_forwards_pos_only_without_hiding_the_count(self):
        """negative_point_count stays honest even when pos_only shrank the hull."""
        broken = self._point(1e14, -50.0, 10.0, run=0)
        fine = self._point(1e15, 20.0, 5.0, run=1)
        summary = frontier.summarize_frontier(
            [broken, fine], inference_tokens=1000, pos_only=True
        )
        self.assertEqual(summary.point_count, 1)
        self.assertEqual(summary.negative_point_count, 1)
        self.assertEqual(summary.endpoint_model_bits, 20.0)

    def test_summary_flags_saturation_and_counts_negatives(self):
        points = [
            self._point(1e14, -1.0, 100.0, 0),
            self._point(1e15, 30.0, 60.0, 1),
            self._point(1e16, 30.0, 30.0, 2),
        ]
        summary = frontier.summarize_frontier(points, inference_tokens=6_480_000)
        self.assertEqual(summary.negative_point_count, 1)
        self.assertEqual(summary.inference_tokens, 6_480_000)
        self.assertIsNotNone(summary.endpoint_model_bits)

    def test_empty_input_summarizes_without_raising(self):
        summary = frontier.summarize_frontier([], inference_tokens=1000)
        self.assertEqual(summary.point_count, 0)
        self.assertIsNone(summary.endpoint_model_bits)
        self.assertIsNone(summary.endpoint_total_bits)
        self.assertFalse(summary.endpoint_is_valid)
        self.assertIsNone(summary.endpoint_run_index)

    def test_endpoint_run_index_identifies_which_run_supplied_the_endpoint(self):
        points = [
            self._point(1e14, 30.0, 100.0, run=7),
            self._point(1e16, 5.0, 20.0, run=2),
        ]
        summary = frontier.summarize_frontier(points, inference_tokens=1000)
        self.assertEqual(summary.endpoint_run_index, 2)

if __name__ == "__main__":
    unittest.main()

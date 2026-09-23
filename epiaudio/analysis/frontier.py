"""Pareto-frontier geometry that tolerates non-positive epiplexity.

The frontier is the lower convex hull of ``(compute, K(M,X))`` pooled across
every run and evaluation checkpoint in a sweep, following Appendix B.1 and
Figure 2b. The algorithm here is the same monotone chain as
``sweep.py:_lower_convex_hull`` and ``chess_order.ipynb``'s
``compute_lower_convex_hull``, including the median-per-run reduction that stops
one run's trajectory from supplying many frontier points.

Two deliberate differences from upstream's notebook path:

* **Linear space only.** ``interp_loglog`` interpolates through ``np.log10``,
  which turns non-positive ``K(M)`` into NaN for the following ``dropna()`` to
  remove. Nothing here takes a logarithm of a code length.
* **The frontier is rebuilt, not reused.** Because ``K(M,X)`` depends on the
  inference-set size, changing that size changes which points win. A frontier
  restandardized after the fact is not the frontier for the new size, so callers
  must rebuild from the pooled point cloud -- which is what
  ``sweep_outputs/*__all_run_data.csv`` preserves.

The endpoint alone is a poor summary, so :func:`headroom_slope` reports how fast
epiplexity is still rising at the compute ceiling. That distinguishes a cell
that has saturated from one whose structure is merely unreached, which is the
distinction Figure 3 draws between ECA rules 15, 30 and 54, and it is the
honest answer to "was the token budget enough?".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log10

DEFAULT_HEADROOM_POINTS = 4


@dataclass(frozen=True)
class CodeLengthPoint:
    """One (compute, model bits, data bits) measurement, from anywhere in a
    sweep's pooled point cloud.

    Not necessarily a point *on* the frontier: :func:`lower_convex_hull` takes
    a pool of these and keeps only the ones that survive, so most instances
    that ever exist are candidates that get rejected, not points that end up
    on a frontier. The name says what is measured, not where it ends up.
    """

    compute_flops: float
    model_bits: float
    data_bits: float
    run_index: int

    @property
    def total_bits(self) -> float:
        return self.model_bits + self.data_bits


def _cross(
    origin: CodeLengthPoint, first: CodeLengthPoint, second: CodeLengthPoint
) -> float:
    return (first.compute_flops - origin.compute_flops) * (
        second.total_bits - origin.total_bits
    ) - (first.total_bits - origin.total_bits) * (
        second.compute_flops - origin.compute_flops
    )


def _median_point_per_run(hull: Sequence[CodeLengthPoint]) -> list[CodeLengthPoint]:
    """Collapse each run's hull points to the single one closest to its own
    median compute value, so one run's trajectory cannot supply multiple
    frontier points.
    """
    by_run: dict[int, list[CodeLengthPoint]] = {}
    for point in hull:
        by_run.setdefault(point.run_index, []).append(point)

    chosen: list[CodeLengthPoint] = []
    for run_points in by_run.values():
        computes = [point.compute_flops for point in run_points]
        middle = sorted(computes)[len(computes) // 2]
        index = min(range(len(computes)), key=lambda i: abs(computes[i] - middle))
        chosen.append(run_points[index])
    return chosen


def lower_convex_hull(
    points: Sequence[CodeLengthPoint],
    *,
    tolerance: float = 0.0,
    reduce: bool = True,
    pos_only: bool = False,
) -> list[CodeLengthPoint]:
    """Lower convex hull minimising total bits as compute grows.

    ``tolerance`` is the relative slack from ``sweep.py``: a point whose total
    exceeds the best total seen so far by more than
    ``tolerance * max(best, candidate)`` is skipped. ``reduce=False`` keeps every
    hull point, which is useful for inspecting a single cell but overweights
    long trajectories when comparing cells; the default ``reduce=True``
    collapses each run to its one point nearest its own median compute
    (:func:`_median_point_per_run`).

    Non-finite coordinates are dropped, since they cannot be ordered; negative
    bit counts are kept by default, since their whole point is to be visible.

    ``pos_only=True`` additionally drops any point that is not strictly
    ``model_bits > 0`` before the hull is built. A trained model coding zero
    bits of structure is not a real outcome to preserve a special case for --
    it exists in the data only as a broken measurement -- so the name means
    what it says: positive only. This exists for a caller who has a specific
    reason not to let a known-broken point sit among the top few points a
    downstream statistic reads -- :func:`headroom_slope` fits a line through
    them, and one negative point in that handful would bias the fitted slope.
    It is not a way to make bad data disappear from a report: the count this
    excludes is still whatever ``negative_point_count`` reports on the full,
    unfiltered pool, and any result computed with ``pos_only=True`` should say
    so.
    """
    usable = [
        point
        for point in points
        if isfinite(point.compute_flops)
        and isfinite(point.total_bits)
        and point.compute_flops > 0
        and (not pos_only or point.model_bits > 0.0)
    ]
    if len(usable) < 2:
        return sorted(usable, key=lambda point: point.compute_flops)

    ordered = sorted(usable, key=lambda point: point.compute_flops)
    hull: list[CodeLengthPoint] = []
    for point in ordered:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        if hull:
            best = min(candidate.total_bits for candidate in hull)
            slack = tolerance * max(abs(best), abs(point.total_bits))
            if point.total_bits - best > slack:
                continue
        hull.append(point)

    if reduce:
        hull = _median_point_per_run(hull)
    return sorted(hull, key=lambda point: point.compute_flops)


def frontier_endpoint(hull: Sequence[CodeLengthPoint]) -> CodeLengthPoint | None:
    """Highest-compute point on the frontier, or ``None`` for an empty hull.

    This is the value ``sweep_summary.py`` reports. It is only meaningful
    alongside the inference-set size it was computed at and the headroom slope
    at the same point.
    """
    if not hull:
        return None
    return max(hull, key=lambda point: point.compute_flops)


def headroom_slope(
    hull: Sequence[CodeLengthPoint],
    *,
    points: int = DEFAULT_HEADROOM_POINTS,
) -> float | None:
    """Whether the frontier's endpoint is still climbing, and how fast.

    The problem this answers: the endpoint (:func:`frontier_endpoint`, the
    highest-compute hull point) is a single number, and by itself it cannot
    say whether that number is a settled estimate or just wherever the compute
    budget happened to run out. Two cells can have very different endpoints
    for that reason alone, with nothing to do with the audio -- exactly the
    "was the token budget enough?" question Figure 3 answers for ECA rules 15,
    30 and 54 by showing some still rising and some flat.

    Concretely: take the ``points`` highest-compute hull points (default 4),
    and fit a least-squares line through ``model_bits`` against
    ``log10(compute_flops)``. The returned slope is that line's slope, i.e.
    ``d K(M) / d log10(compute)`` estimated at the high-compute end. Near zero
    means epiplexity has stopped changing across that stretch of compute, so
    the endpoint is a real estimate. Clearly positive means the cell was still
    absorbing structure when the budget ran out, so the endpoint is a lower
    bound, not a converged value, and must not be compared against a saturated
    cell's endpoint as though both mean the same thing. See
    ``FrontierSummary.is_saturated`` for the scale-free yes/no built on top of
    this number.

    Returns ``None`` when fewer than two usable points remain (too few to fit
    a line) or the compute values do not vary (a vertical fit is undefined).
    """
    if points < 2:
        raise ValueError("points must be at least 2")
    tail = sorted(hull, key=lambda point: point.compute_flops)[-points:]
    xs = [log10(point.compute_flops) for point in tail if point.compute_flops > 0]
    ys = [point.model_bits for point in tail if point.compute_flops > 0]
    if len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / variance


@dataclass(frozen=True)
class FrontierSummary:
    """A cell's frontier reduced to reportable numbers."""

    inference_tokens: int
    point_count: int
    endpoint_compute_flops: float | None
    endpoint_model_bits: float | None
    endpoint_data_bits: float | None
    endpoint_run_index: int | None
    headroom_bits_per_decade: float | None
    negative_point_count: int

    @property
    def endpoint_total_bits(self) -> float | None:
        if self.endpoint_model_bits is None or self.endpoint_data_bits is None:
            return None
        return self.endpoint_model_bits + self.endpoint_data_bits

    @property
    def endpoint_is_valid(self) -> bool:
        return self.endpoint_model_bits is not None and self.endpoint_model_bits >= 0.0

    @property
    def is_saturated(self) -> bool | None:
        """Whether epiplexity has stopped rising with compute.

        ``None`` when the slope could not be estimated. The comparison is
        against 1% of the endpoint value per decade of compute, which keeps the
        test scale-free across cells whose absolute bit counts differ by orders
        of magnitude.
        """
        if self.headroom_bits_per_decade is None or self.endpoint_model_bits is None:
            return None
        if self.endpoint_model_bits <= 0:
            return None
        return self.headroom_bits_per_decade <= 0.01 * self.endpoint_model_bits


def summarize_frontier(
    points: Sequence[CodeLengthPoint],
    *,
    inference_tokens: int,
    tolerance: float = 0.0,
    reduce: bool = True,
    headroom_points: int = DEFAULT_HEADROOM_POINTS,
    pos_only: bool = False,
) -> FrontierSummary:
    """Build the frontier and reduce it to a reportable summary.

    ``pos_only`` forwards to :func:`lower_convex_hull`; see there for what it
    does and why. ``negative_point_count`` always counts against the full,
    unfiltered ``points``, regardless of ``pos_only``, so the fact that a cell
    had broken measurements is never lost even when the hull itself was built
    without them.
    """
    hull = lower_convex_hull(points, tolerance=tolerance, reduce=reduce, pos_only=pos_only)
    endpoint = frontier_endpoint(hull)
    return FrontierSummary(
        inference_tokens=inference_tokens,
        point_count=len(hull),
        endpoint_compute_flops=None if endpoint is None else endpoint.compute_flops,
        endpoint_model_bits=None if endpoint is None else endpoint.model_bits,
        endpoint_data_bits=None if endpoint is None else endpoint.data_bits,
        endpoint_run_index=None if endpoint is None else endpoint.run_index,
        headroom_bits_per_decade=headroom_slope(hull, points=headroom_points),
        negative_point_count=sum(1 for point in points if point.model_bits < 0),
    )

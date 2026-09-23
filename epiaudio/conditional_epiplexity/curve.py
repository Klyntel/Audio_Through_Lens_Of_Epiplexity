"""Serialization helpers for conditional epiplexity evaluation curves."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


MetricValue = float | int
REPORTED_POINT_COUNT = "reported_point_count"
REPORTED_POINT_FIELDS = (
    "optimizer_steps",
    "training_flops_estimate",
    "conditional_epiplexity_valid",
    "conditional_epiplexity_bits",
    "two_part_code_bits",
    "ema_accuracy",
    "ema_macro_f1",
    "ema_weighted_f1",
)


def _metric_value(value: object, name: str) -> MetricValue:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Reported point metric '{name}' must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Reported point metric '{name}' must be finite.")
    return value if isinstance(value, int) else numeric


def flatten_reported_points(
    points: Sequence[Mapping[str, object]],
) -> dict[str, MetricValue]:
    """Flatten selected curve fields for checkpoints and Comet summaries."""
    flattened: dict[str, MetricValue] = {REPORTED_POINT_COUNT: len(points)}
    for index, point in enumerate(points):
        for name in REPORTED_POINT_FIELDS:
            if name not in point:
                raise ValueError(f"Reported point is missing metric '{name}'.")
            flattened[f"reported_point_{index:03d}_{name}"] = _metric_value(
                point[name],
                name,
            )
    return flattened


def extract_reported_points(
    metrics: Mapping[str, object],
) -> tuple[dict[str, MetricValue], ...]:
    """Restore curve points flattened by :func:`flatten_reported_points`."""
    count_value = metrics.get(REPORTED_POINT_COUNT)
    if count_value is None:
        return ()
    numeric_count = (
        float(count_value)
        if isinstance(count_value, Real) and not isinstance(count_value, bool)
        else math.nan
    )
    if (
        not math.isfinite(numeric_count)
        or not numeric_count.is_integer()
        or numeric_count < 0
    ):
        raise ValueError(f"{REPORTED_POINT_COUNT} must be a non-negative integer.")

    points = []
    for index in range(int(numeric_count)):
        point = {}
        for name in REPORTED_POINT_FIELDS:
            key = f"reported_point_{index:03d}_{name}"
            if key not in metrics:
                raise ValueError(f"Run is missing reported point metric '{key}'.")
            point[name] = _metric_value(metrics[key], key)
        points.append(point)
    return tuple(points)

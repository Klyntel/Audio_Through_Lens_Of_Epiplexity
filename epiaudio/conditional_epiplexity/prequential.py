"""Conditional prequential code-length accounting.

For a one-pass sequence of multiclass examples ``(X_i, Y_i)``, the online
code is the label NLL measured by ``P_i(Y_i | X_i)`` before updating the
teacher on that example. Following Equation 8 and Appendix B.1 of the
epiplexity paper, the final-model code over the observed labels is estimated
from the final raw teacher's mean NLL on unseen IID data.

All losses in this module use nats. Bit conversion happens only on the
returned estimate, keeping the training and estimator units explicit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real


_STATE_VERSION = 1
_NATS_PER_BIT = math.log(2.0)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


@dataclass(frozen=True)
class ConditionalPrequentialEstimate:
    """A conditional prequential estimate at one training position."""

    label_count: int
    online_label_code_nats: float
    reference_label_nll_nats_per_label: float
    reference_label_code_nats: float
    conditional_epiplexity_nats: float

    @property
    def online_label_nll_nats_per_label(self) -> float:
        return self.online_label_code_nats / self.label_count

    @property
    def online_label_nll_bits_per_label(self) -> float:
        return self.online_label_nll_nats_per_label / _NATS_PER_BIT

    @property
    def reference_label_nll_bits_per_label(self) -> float:
        return self.reference_label_nll_nats_per_label / _NATS_PER_BIT

    @property
    def online_label_code_bits(self) -> float:
        return self.online_label_code_nats / _NATS_PER_BIT

    @property
    def reference_label_code_bits(self) -> float:
        return self.reference_label_code_nats / _NATS_PER_BIT

    @property
    def conditional_epiplexity_bits(self) -> float:
        return self.conditional_epiplexity_nats / _NATS_PER_BIT

    @property
    def conditional_epiplexity_bits_per_label(self) -> float:
        return self.conditional_epiplexity_bits / self.label_count


@dataclass
class ConditionalPrequentialEstimator:
    """Accumulate the pre-update label code for a one-pass teacher.

    Call :meth:`update` once per optimizer step using the label NLL sum and
    label count aggregated across all DDP ranks. The estimator stores only
    sufficient statistics, so it can be checkpointed without retaining
    batches or a sampled loss curve.
    """

    online_label_nll_nats: float = 0.0
    label_count: int = 0

    def __post_init__(self) -> None:
        self.online_label_nll_nats = _finite_nonnegative(
            self.online_label_nll_nats,
            "online_label_nll_nats",
        )
        self.label_count = _integer(self.label_count, "label_count", minimum=0)

    def update(self, label_nll_sum_nats: float, label_count: int) -> None:
        """Add one globally aggregated pre-update label-NLL observation."""
        nll_sum = _finite_nonnegative(label_nll_sum_nats, "label_nll_sum_nats")
        count = _integer(label_count, "label_count", minimum=1)
        updated_nll = self.online_label_nll_nats + nll_sum
        if not math.isfinite(updated_nll):
            raise ValueError("Accumulated online label NLL must remain finite.")
        self.online_label_nll_nats = updated_nll
        self.label_count += count

    def estimate(
        self,
        reference_label_nll_nats_per_label: float,
    ) -> ConditionalPrequentialEstimate:
        """Subtract the final raw teacher's unseen-data label code.

        A negative result is retained rather than clamped. It can occur because
        the final-model term is estimated from a finite held-out sample.
        """
        if self.label_count == 0:
            raise ValueError("At least one label observation is required.")
        reference_nll = _finite_nonnegative(
            reference_label_nll_nats_per_label,
            "reference_label_nll_nats_per_label",
        )
        reference_code = reference_nll * self.label_count
        if not math.isfinite(reference_code):
            raise ValueError("Reference label code must remain finite.")
        epiplexity = self.online_label_nll_nats - reference_code
        return ConditionalPrequentialEstimate(
            label_count=self.label_count,
            online_label_code_nats=self.online_label_nll_nats,
            reference_label_nll_nats_per_label=reference_nll,
            reference_label_code_nats=reference_code,
            conditional_epiplexity_nats=epiplexity,
        )

    def state_dict(self) -> dict[str, float | int]:
        """Return a versioned, checkpoint-safe estimator state."""
        return {
            "version": _STATE_VERSION,
            "online_label_nll_nats": self.online_label_nll_nats,
            "label_count": self.label_count,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ConditionalPrequentialEstimator:
        """Restore and validate state produced by :meth:`state_dict`."""
        expected_keys = {"version", "online_label_nll_nats", "label_count"}
        actual_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(str(key) for key in actual_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected keys: {', '.join(unexpected)}")
            raise ValueError(f"Invalid estimator state ({'; '.join(details)}).")
        version = _integer(state["version"], "version", minimum=1)
        if version != _STATE_VERSION:
            raise ValueError(f"Unsupported estimator state version {version}.")
        online_nll = _finite_nonnegative(
            state["online_label_nll_nats"],
            "online_label_nll_nats",
        )
        label_count = _integer(state["label_count"], "label_count", minimum=0)
        return cls(
            online_label_nll_nats=online_nll,
            label_count=label_count,
        )

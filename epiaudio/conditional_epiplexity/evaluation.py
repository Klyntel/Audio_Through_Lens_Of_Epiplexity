"""Evaluation summaries for conditional multiclass epiplexity experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real


_NATS_PER_BIT = math.log(2.0)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


@dataclass(frozen=True)
class ConditionalClassificationMetrics:
    """Label-code and classification metrics for one evaluated model."""

    label_count: int
    label_code_nats: float
    accuracy: float
    macro_f1: float
    weighted_f1: float

    @property
    def label_nll_nats_per_label(self) -> float:
        return self.label_code_nats / self.label_count

    @property
    def label_code_bits(self) -> float:
        return self.label_code_nats / _NATS_PER_BIT

    @property
    def label_nll_bits_per_label(self) -> float:
        return self.label_nll_nats_per_label / _NATS_PER_BIT

    def as_dict(self, prefix: str) -> dict[str, float | int]:
        """Return stable metric names suitable for checkpoints and Comet."""
        return {
            f"{prefix}_label_count": self.label_count,
            f"{prefix}_label_code_nats": self.label_code_nats,
            f"{prefix}_label_code_bits": self.label_code_bits,
            f"{prefix}_label_nll_nats_per_label": self.label_nll_nats_per_label,
            f"{prefix}_label_nll_bits_per_label": self.label_nll_bits_per_label,
            f"{prefix}_accuracy": self.accuracy,
            f"{prefix}_macro_f1": self.macro_f1,
            f"{prefix}_weighted_f1": self.weighted_f1,
        }


@dataclass(frozen=True)
class LabelPriorBaseline:
    """Unconditional empirical and uniform label-code baselines."""

    label_count: int
    num_classes: int
    entropy_nats_per_label: float

    @property
    def entropy_bits_per_label(self) -> float:
        return self.entropy_nats_per_label / _NATS_PER_BIT

    @property
    def entropy_code_nats(self) -> float:
        return self.entropy_nats_per_label * self.label_count

    @property
    def entropy_code_bits(self) -> float:
        return self.entropy_code_nats / _NATS_PER_BIT

    @property
    def uniform_nll_nats_per_label(self) -> float:
        return math.log(self.num_classes)

    @property
    def uniform_nll_bits_per_label(self) -> float:
        return self.uniform_nll_nats_per_label / _NATS_PER_BIT

    @property
    def uniform_code_nats(self) -> float:
        return self.uniform_nll_nats_per_label * self.label_count

    @property
    def uniform_code_bits(self) -> float:
        return self.uniform_code_nats / _NATS_PER_BIT


def summarize_classification(
    label_code_nats: float,
    confusion_matrix: Sequence[Sequence[int]],
) -> ConditionalClassificationMetrics:
    """Compute accuracy and F1 scores from an explicit class confusion matrix.

    Macro F1 includes every class in the configured vocabulary. A class with no
    true or predicted examples receives F1 zero, matching ``zero_division=0``.
    """
    code_nats = _finite_nonnegative(label_code_nats, "label_code_nats")
    num_classes = len(confusion_matrix)
    if num_classes == 0 or any(len(row) != num_classes for row in confusion_matrix):
        raise ValueError("confusion_matrix must be a non-empty square matrix.")

    matrix = tuple(
        tuple(
            _count(value, f"confusion_matrix[{row}][{column}]")
            for column, value in enumerate(values)
        )
        for row, values in enumerate(confusion_matrix)
    )
    support = tuple(sum(row) for row in matrix)
    predicted = tuple(
        sum(matrix[row][column] for row in range(num_classes))
        for column in range(num_classes)
    )
    label_count = sum(support)
    if label_count == 0:
        raise ValueError("Evaluation must contain at least one label.")

    true_positives = tuple(matrix[index][index] for index in range(num_classes))
    per_class_f1 = tuple(
        2.0 * true_positive / denominator if denominator else 0.0
        for true_positive, denominator in (
            (true_positives[index], support[index] + predicted[index])
            for index in range(num_classes)
        )
    )
    return ConditionalClassificationMetrics(
        label_count=label_count,
        label_code_nats=code_nats,
        accuracy=sum(true_positives) / label_count,
        macro_f1=sum(per_class_f1) / num_classes,
        weighted_f1=(
            sum(
                score * count
                for score, count in zip(per_class_f1, support, strict=True)
            )
            / label_count
        ),
    )


def summarize_label_prior(class_counts: Sequence[int]) -> LabelPriorBaseline:
    """Compute the empirical marginal label entropy and uniform-code baseline."""
    if not class_counts:
        raise ValueError("class_counts must contain at least one class.")
    counts = tuple(
        _count(value, f"class_counts[{index}]")
        for index, value in enumerate(class_counts)
    )
    label_count = sum(counts)
    if label_count == 0:
        raise ValueError("class_counts must contain at least one label.")
    probabilities = (count / label_count for count in counts if count)
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return LabelPriorBaseline(
        label_count=label_count,
        num_classes=len(counts),
        entropy_nats_per_label=entropy,
    )

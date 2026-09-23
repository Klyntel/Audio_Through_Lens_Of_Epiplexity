"""Loss-curve shape analysis and diagnosis of negative epiplexity.

Negative ``K(M)`` is widespread across our sweep cells, and the estimator's
algebra says exactly what it means.

``train_torch.py`` accumulates the online code from a constant number of tokens
per step, so with ``n`` steps and ``D = n * tokens_per_step``::

    K(X)    = mean_over_trajectory(L) * D / (1e6 * ln 2)
    K(X|M)  = mean_over_tail_window(L) * D / (1e6 * ln 2)
    K(M)    = K(X) - K(X|M)
            = D * [mean_over_trajectory(L) - mean_over_tail_window(L)] / (1e6 * ln 2)

Two consequences follow, and both are used below.

*The sign is a statement about curve shape, nothing else.* ``K(M) < 0`` holds if
and only if the final evaluation window's mean loss exceeds the mean loss over
the whole trajectory. For a non-increasing curve the trajectory mean is always
at least the tail mean, so **a monotonically decreasing loss curve can never
produce negative epiplexity**. Every negative cell therefore has a demonstrably
non-monotone loss curve, which makes the diagnosis a direct check rather than
speculation.

*The sign is independent of the inference-set size.* Restandardizing
``K(X|M)`` changes which frontier point is selected but not whether any given
``K(M)`` is negative, so the two defects this package addresses are separate and
have separate fixes.

*A shape classification only explains the value it was computed from.* ``K(M)``
is logged once per evaluation checkpoint, not once per run, and
``classify_shape`` reduces whatever sequence it is given to one label that can
only speak to the ``K(M)`` that same sequence would reconstruct (via
:func:`implied_model_bits`). Passing the wrong scope -- a run's whole
trajectory to explain one earlier checkpoint's value -- lets a later event get
blamed for an earlier one; see ``diagnose``'s ``train_tokens`` guard below,
which exists specifically to catch this.

Upstream's analysis pipeline cannot represent these points at all:
``chess_order.ipynb``'s ``interp_loglog`` interpolates ``K(M)`` through
``np.log10``, which maps non-positive values to NaN, and the subsequent
``dropna()`` deletes them. That is lossy near zero, which is harmless when
``S_T`` is a large fraction of the total and destructive in our regime. Nothing
here interpolates in log space, and nothing here drops a negative point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import log

NATS_PER_BIT = log(2.0)

# Defaults are starting points for the diagnosis, not tuned constants; every
# threshold is exposed so a report can state the values it used.
#
# DEFAULT_TAIL_FRACTION is a fallback only, for curves whose eval cadence
# was not recorded; pass `tail_fraction=tail_fraction_for(num_evals)` whenever
# it was (see that function for why 0.1 is not the shipped estimator's window).
DEFAULT_TAIL_FRACTION = 0.1
DEFAULT_CEILING_TOLERANCE = 0.02
DEFAULT_RISE_TOLERANCE = 0.05
DEFAULT_PLATEAU_TOLERANCE = 0.1
DEFAULT_NEAR_ZERO_TOLERANCE = 0.01
DEFAULT_MONOTONE_TOLERANCE = 0.01
# Relative disagreement, between a reconstructed and a supplied K(M), above
# which `losses` is presumed not to be the sequence that actually produced
# `model_bits`. Deliberately generous: ordinary slop between this module's
# fractional tail window and the shipped estimator's non-overlapping one should
# stay well under this; a genuinely wrong-scoped sequence (a different run, or a
# span that overshoots or undershoots the checkpoint being explained) routinely
# differs by much more, often in sign as well as magnitude.
DEFAULT_SCOPE_MISMATCH_TOLERANCE = 0.5


class LossCurveShape(StrEnum):
    """Coarse shape of a training loss curve.

    Classification is exhaustive by construction, but every member other than
    ``OTHER`` is a positive claim checked directly against the curve; nothing
    is inferred by elimination. ``OTHER`` is the honest name for "matched none
    of the checks below," not a guess at what the curve probably looks like.
    """

    DECREASING = "decreasing"
    """Non-increasing within tolerance; the well-behaved case."""

    PLATEAU = "plateau"
    """Real descent, and the tail has settled back near the best loss reached."""

    LATE_RISE = "late_rise"
    """Tail drifts upward relative to mid-training."""

    MINIMUM_NEAR_CEILING = "minimum_near_ceiling"
    """The best loss the run ever reached stayed within tolerance of the
    uniform-code ceiling. This checks only the curve's minimum, not its shape
    elsewhere -- a curve that swings wildly but happens to bottom out near the
    ceiling once gets this label too, and correctly so: whatever the rest of
    the curve did, the model was never meaningfully better than chance.
    """

    OTHER = "other"
    """Matches none of the above. A real, expected outcome for some curves,
    not a defect in this list -- inspect the curve directly before drawing a
    conclusion from it.
    """


class NegativityMechanism(StrEnum):
    """Why a cell's as-implemented ``K(M)`` came out below zero."""

    NOT_NEGATIVE = "not_negative"
    """``K(M) >= 0``; no diagnosis needed."""

    NO_LEARNABLE_STRUCTURE = "no_learnable_structure"
    """A real property of the cell; see
    ``LossCurveShape.MINIMUM_NEAR_CEILING`` for exactly what this checks.
    """

    LATE_DIVERGENCE = "late_divergence"
    """Loss rose late in training. A training-configuration failure.

    muP transfers an optimal learning rate across model *width*, not across data
    distributions, and ``lr=2`` under a constant schedule was tuned on text and
    chess. Cells landing here should be re-run at two or three learning rates
    before any conclusion is drawn about the data.
    """

    SIGN_UNSTABLE_NEAR_ZERO = "sign_unstable_near_zero"
    """``|K(M)|`` is negligible against ``K(X)``; the sign is noise.

    These cells carry no information about the data beyond "structure is below
    the estimator's resolution" and must not be read as rankings.
    """

    UNEXPLAINED = "unexplained"
    """Negative without matching any mechanism above; needs manual review."""

    SCOPE_MISMATCH = "scope_mismatch"
    """``losses`` does not reconstruct the supplied ``model_bits`` within
    tolerance, so it is likely the wrong sequence for this value -- most often
    a run's whole trajectory passed in to explain one specific checkpoint's
    ``K(M)``. Not a claim about the audio or the training run at all; nothing
    else about this diagnosis should be trusted until the caller passes the
    correct slice of losses.
    """


def tail_fraction_for(num_evals: int) -> float:
    """Tail fraction that reproduces the shipped estimator's averaging window.

    ``train_torch.py`` accumulates ``train_loss_sum`` and resets it at every
    evaluation, so the final-model term is the mean loss over one inter-eval
    interval -- ``1 / num_evals`` of training, or 0.05 at the ``num_evals=20``
    used across ``epiaudio/sweeps/``.

    That window is larger than it first appears: at T=31.25M tokens it averages
    1.56M tokens, roughly twelve times the 131,072-token evaluation set. The
    as-implemented estimator is therefore *less* noisy than a held-out estimate,
    not more. What it carries instead is bias -- it averages over the models
    within the window rather than the final model, all of which are worse, so it
    understates the descent and biases ``K(M)`` low.
    """
    if num_evals <= 0:
        raise ValueError("num_evals must be positive")
    return 1.0 / num_evals


def ceiling_loss_nats(vocab_size: int) -> float:
    """Uniform-code loss in nats, ``ln V``, the no-learning reference."""
    if vocab_size <= 1:
        raise ValueError("vocab_size must exceed 1")
    return log(vocab_size)


def trajectory_mean(losses: Sequence[float]) -> float:
    """Mean loss over the whole trajectory."""
    if not losses:
        raise ValueError("losses must be non-empty")
    return sum(losses) / len(losses)


def tail_window(
    losses: Sequence[float],
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> Sequence[float]:
    """Final ``tail_fraction`` of the trajectory, at least one point."""
    if not losses:
        raise ValueError("losses must be non-empty")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    size = max(1, round(len(losses) * tail_fraction))
    return losses[-size:]


def tail_mean(
    losses: Sequence[float],
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> float:
    """Mean loss over the final window."""
    window = tail_window(losses, tail_fraction)
    return sum(window) / len(window)


def _model_bits_from_means(
    trajectory_mean_nats: float, tail_mean_nats: float, train_tokens: float
) -> float:
    """Scale a (trajectory - tail) loss gap to bits.

    The one place this formula is written; both :func:`implied_model_bits`
    (given a raw loss sequence) and :func:`diagnose` (given a
    :class:`CurveStatistics` it already built) call this rather than each
    recomputing the trajectory and tail means independently.
    """
    if train_tokens < 0:
        raise ValueError("train_tokens must be non-negative")
    return (trajectory_mean_nats - tail_mean_nats) * train_tokens / NATS_PER_BIT


def implied_model_bits(
    losses: Sequence[float],
    train_tokens: float,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> float:
    """Reproduce the shipped ``K(M)`` estimator from a loss curve, in bits.

    Exact when the recorded losses are the per-step pre-update means and the
    token count per step is constant, which is how ``train_torch.py`` runs. Its
    purpose is to let the diagnosis be validated against a synthetic curve whose
    answer is known analytically.
    """
    return _model_bits_from_means(
        trajectory_mean(losses), tail_mean(losses, tail_fraction), train_tokens
    )


def is_non_increasing(
    losses: Sequence[float],
    tolerance: float = DEFAULT_MONOTONE_TOLERANCE,
) -> bool:
    """Whether the curve never rises by more than ``tolerance`` (relative).

    The tolerance is relative to the value at each step, so it absorbs
    step-to-step noise without absorbing a sustained upward drift.
    """
    if not losses:
        raise ValueError("losses must be non-empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return all(
        later <= earlier * (1.0 + tolerance)
        for earlier, later in zip(losses, losses[1:])
    )


@dataclass(frozen=True)
class CurveStatistics:
    """Shape statistics behind a diagnosis, kept so a report can show its work."""

    steps: int
    first_loss: float
    min_loss: float
    final_loss: float
    trajectory_mean: float
    tail_mean: float
    ceiling_nats: float

    @property
    def total_descent(self) -> float:
        """How far the curve fell from its starting value to its minimum."""
        return self.first_loss - self.min_loss

    @property
    def tail_drift(self) -> float:
        """Tail mean minus trajectory mean; positive means a late rise."""
        return self.tail_mean - self.trajectory_mean

    @property
    def ceiling_gap(self) -> float:
        """How far below the uniform-code ceiling the curve ever got."""
        return self.ceiling_nats - self.min_loss

    @property
    def relative_ceiling_gap(self) -> float:
        return self.ceiling_gap / self.ceiling_nats


def summarize_curve(
    losses: Sequence[float],
    *,
    vocab_size: int,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
) -> CurveStatistics:
    """Reduce a loss curve to the statistics the diagnosis needs."""
    if not losses:
        raise ValueError("losses must be non-empty")
    return CurveStatistics(
        steps=len(losses),
        first_loss=losses[0],
        min_loss=min(losses),
        final_loss=losses[-1],
        trajectory_mean=trajectory_mean(losses),
        tail_mean=tail_mean(losses, tail_fraction),
        ceiling_nats=ceiling_loss_nats(vocab_size),
    )


def classify_shape(
    losses: Sequence[float],
    stats: CurveStatistics,
    *,
    ceiling_tolerance: float = DEFAULT_CEILING_TOLERANCE,
    rise_tolerance: float = DEFAULT_RISE_TOLERANCE,
    monotone_tolerance: float = DEFAULT_MONOTONE_TOLERANCE,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
) -> LossCurveShape:
    """Assign a shape by checking each condition directly, never by elimination.

    Every branch below is a positive claim verified against the curve. The
    ceiling check comes first because it only inspects the curve's minimum: a
    curve whose best point never gets meaningfully below ``ln V`` is
    uninformative regardless of how the rest of it behaves, and it would be
    misleading to call that curve "decreasing" just because it happened to
    drift down by noise. ``OTHER`` is what a curve gets when none of the
    positive checks match; earlier versions of this function returned
    ``PLATEAU`` here unconditionally, which mislabeled any curve that was
    merely not one of the other three, whether or not it actually settled
    anywhere.
    """
    if stats.relative_ceiling_gap <= ceiling_tolerance:
        return LossCurveShape.MINIMUM_NEAR_CEILING
    if stats.total_descent > 0 and stats.tail_drift > rise_tolerance * stats.total_descent:
        return LossCurveShape.LATE_RISE
    if is_non_increasing(losses, monotone_tolerance):
        return LossCurveShape.DECREASING
    if (
        stats.total_descent > 0
        and abs(stats.tail_mean - stats.min_loss) <= plateau_tolerance * stats.total_descent
    ):
        return LossCurveShape.PLATEAU
    return LossCurveShape.OTHER


@dataclass(frozen=True)
class NegativityDiagnosis:
    """A cell's negativity verdict with the evidence that produced it."""

    mechanism: NegativityMechanism
    shape: LossCurveShape
    statistics: CurveStatistics
    model_bits: float
    online_code_bits: float | None
    reconstructed_model_bits: float | None = None
    """``implied_model_bits`` computed from the same ``losses`` and
    ``tail_fraction``, when ``train_tokens`` was supplied to :func:`diagnose`.
    Kept alongside ``model_bits`` so a scope mismatch can be inspected rather
    than just asserted; see ``NegativityMechanism.SCOPE_MISMATCH``.
    """

    @property
    def is_negative(self) -> bool:
        return self.model_bits < 0.0

    @property
    def is_data_property(self) -> bool:
        """Whether the verdict says something about the corpus.

        Only ``NO_LEARNABLE_STRUCTURE`` does. ``LATE_DIVERGENCE`` is a training
        failure and ``SIGN_UNSTABLE_NEAR_ZERO`` is a resolution limit; neither
        supports a claim about the audio.
        """
        return self.mechanism is NegativityMechanism.NO_LEARNABLE_STRUCTURE


def diagnose(
    losses: Sequence[float],
    *,
    model_bits: float,
    vocab_size: int,
    online_code_bits: float | None = None,
    train_tokens: float | None = None,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    ceiling_tolerance: float = DEFAULT_CEILING_TOLERANCE,
    rise_tolerance: float = DEFAULT_RISE_TOLERANCE,
    monotone_tolerance: float = DEFAULT_MONOTONE_TOLERANCE,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    near_zero_tolerance: float = DEFAULT_NEAR_ZERO_TOLERANCE,
    scope_mismatch_tolerance: float = DEFAULT_SCOPE_MISMATCH_TOLERANCE,
) -> NegativityDiagnosis:
    """Classify one cell's loss curve and, if ``K(M) < 0``, say why.

    ``online_code_bits`` is ``K(X)``. Without it the near-zero test cannot run,
    so a small negative value that is really below the estimator's resolution
    will fall through to ``UNEXPLAINED`` rather than being silently accepted.

    ``train_tokens`` is the token count that produced ``model_bits`` -- the
    same ``D`` used when it was originally computed, not the run's eventual
    total if ``losses`` is a prefix. When supplied, ``losses`` is cross-checked
    against ``model_bits`` via :func:`implied_model_bits`, and a shape-based
    mechanism is only named when they agree within ``scope_mismatch_tolerance``.
    This is what catches a run's whole trajectory being passed in to explain an
    earlier checkpoint's ``K(M)``: the two would very likely disagree, since a
    late-training event the earlier checkpoint never saw would otherwise get
    blamed for it. Omitting ``train_tokens`` skips this check entirely, which
    is why it is optional rather than required.
    """
    stats = summarize_curve(
        losses, vocab_size=vocab_size, tail_fraction=tail_fraction
    )
    shape = classify_shape(
        losses,
        stats,
        ceiling_tolerance=ceiling_tolerance,
        rise_tolerance=rise_tolerance,
        monotone_tolerance=monotone_tolerance,
        plateau_tolerance=plateau_tolerance,
    )

    reconstructed_model_bits = None
    scope_mismatch = False
    if train_tokens is not None:
        # stats already has trajectory_mean/tail_mean for this tail_fraction;
        # reuse them rather than have implied_model_bits recompute both from
        # losses a second time.
        reconstructed_model_bits = _model_bits_from_means(
            stats.trajectory_mean, stats.tail_mean, train_tokens
        )
        if model_bits < 0.0:
            scale = max(abs(model_bits), abs(reconstructed_model_bits), 1.0)
            disagreement = abs(reconstructed_model_bits - model_bits) / scale
            scope_mismatch = disagreement > scope_mismatch_tolerance

    if model_bits >= 0.0:
        mechanism = NegativityMechanism.NOT_NEGATIVE
    elif scope_mismatch:
        mechanism = NegativityMechanism.SCOPE_MISMATCH
    elif shape is LossCurveShape.MINIMUM_NEAR_CEILING:
        mechanism = NegativityMechanism.NO_LEARNABLE_STRUCTURE
    elif shape is LossCurveShape.LATE_RISE:
        mechanism = NegativityMechanism.LATE_DIVERGENCE
    elif (
        online_code_bits is not None
        and online_code_bits > 0
        and abs(model_bits) / online_code_bits <= near_zero_tolerance
    ):
        mechanism = NegativityMechanism.SIGN_UNSTABLE_NEAR_ZERO
    else:
        mechanism = NegativityMechanism.UNEXPLAINED

    return NegativityDiagnosis(
        mechanism=mechanism,
        shape=shape,
        statistics=stats,
        model_bits=model_bits,
        online_code_bits=online_code_bits,
        reconstructed_model_bits=reconstructed_model_bits,
    )

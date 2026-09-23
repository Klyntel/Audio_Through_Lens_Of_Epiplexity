"""Epiplexity estimators, restandardized so cells can be compared.

One unit mismatch is worth flagging up front: ``train_torch.py`` logs every
``K(...)`` metric to Comet in **megabits**, while ``sweep.py`` writes
``sweep_outputs/*.csv`` in **bits** -- mixing the two silently rescales a
result by a million. Nothing here converts between them, since neither this
package's CSV loader nor anything downstream of it ever reads Comet directly;
multiply or divide by 1e6 directly if a Comet-logged value is ever needed
alongside one of these.

Two things about the shipped pipeline motivate the rest of this module.

**The inference-set size is fixed per run, by design.** ``sweep.py:516`` sets
``test_tokens`` from the test split's own token count, then forms
``K(X|M) = eval_loss * test_tokens / ln 2``. That is the correct choice, not
an arbitrary one: ``K(X|M)`` is the code length of the set actually held out
for this cell, so T_eval is supposed to track that set's real size rather than
some target chosen independently of it. Upstream instead treats the size as an
analysis-time knob (``chess_order.ipynb``'s ``set_test_tokens``), picking it
explicitly at 1B, 5B and 1T tokens to study how epiplexity moves as the
described set grows -- Section 6.3's point that asymptotic epiplexity is
capped by dataset size. :func:`restandardize_data_bits` and
:func:`restandardize_data_bits_by_clips` below exist for that comparison, not
to correct anything: each cell's own ``K(X|M)`` is already correct for its own
test-split size, and restandardizing only matters when a report wants to ask
upstream's question -- how does epiplexity change with the size of the set
being coded -- rather than take each cell's number as given.

The inference set is a *counterfactual* budget: the amount of data we would code
with the trained model, estimated from the mean held-out loss. It need not exist
on disk, which is why upstream can project to 1T tokens from a much smaller
evaluation.

**Equation 8's final-model term needs one substitution, not a re-evaluation.**
Read literally, ``Σ (log 1/P_i(Z_i) - log 1/P_M(Z_i))`` needs the *final* model
``P_M`` scored on every one of the ``M`` training tokens; nothing in this
pipeline does that, and nothing needs to. Because the ``Z_i`` are drawn i.i.d.,
Section 4.1 licenses replacing the whole second sum with one scalar times
``M`` ("the code length for the model can be visualized as the area under the
loss curve above the final loss"), so only a single estimate of the final
model's expected loss on a fresh sample is required, not ``M`` separate
re-scorings of old tokens.

The two functions below supply that one scalar two different ways, and neither
re-evaluates a frozen model on historical data. The as-implemented reading
(``picodo/train.py:773-779``, ported verbatim in ``train_torch.py:1371-1378``)
averages the ordinary *online* loss already computed at each step -- the model
as it stood at that step, on a fresh, never-reused batch -- over the last
evaluation window. Re-reading that accumulation directly confirms this: the
window holds several near-final, still-improving models, not one repeated, so
the average is biased high (they are all slightly worse than the truly final
model) and ``K(M)`` biased low; see ``curves.tail_fraction_for`` for the size
of that window. The held-out alternative below supplies the same scalar from
genuinely unseen validation data instead. Both are legitimate estimates of the
one quantity Section 4.1 licenses, using different data to get there; this
module computes both so a report can state both rather than pick one silently.
"""

from __future__ import annotations

from math import log

from epiaudio.analysis.units import tokens_per_clip

NATS_PER_BIT = log(2.0)


def code_bits_from_loss(loss_nats_per_token: float, tokens: float) -> float:
    """Code length in bits for ``tokens`` tokens at a mean loss in nats."""
    if loss_nats_per_token < 0:
        raise ValueError("loss must be non-negative")
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    return loss_nats_per_token * tokens / NATS_PER_BIT


def loss_from_code_bits(code_bits: float, tokens: float) -> float:
    """Invert :func:`code_bits_from_loss` to recover a mean loss in nats.

    Used to recover the held-out loss that produced a stored ``K(X|M)`` column
    when the raw metric is no longer available.
    """
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    return code_bits * NATS_PER_BIT / tokens


def restandardize_data_bits(
    data_bits: float,
    stored_inference_tokens: int,
    target_inference_tokens: int,
) -> float:
    """Rescale a stored ``K(X|M)`` to a different inference-set size.

    ``K(X|M) = eval_loss * D_inference / ln 2`` is linear in the inference-set
    size, so restandardizing is an exact rescaling and needs no retraining --
    only the size that was originally used, which is not currently recorded in
    the CSVs (see the ``inference_tokens`` field of
    :class:`~epiaudio.analysis.inventory.SweepManifest`).
    """
    if stored_inference_tokens <= 0 or target_inference_tokens <= 0:
        raise ValueError("inference token counts must be positive")
    return data_bits * (target_inference_tokens / stored_inference_tokens)


def restandardize_data_bits_by_clips(
    data_bits: float,
    *,
    stored_clips: int,
    target_clips: int,
    tokenizer: str,
) -> float:
    """:func:`restandardize_data_bits`, sized in clips instead of tokens.

    Clip count is the axis that is actually comparable across tokenizers;
    a raw token target is not, since ``TOKENIZER_SPECS`` gives tokens-per-clip
    rates that differ 8x between EnCodec and DAC. Converting both endpoints
    through the same tokenizer's rate before rescaling means "restandardize to
    the same 10,000 clips of held-out audio" produces a comparable answer for
    every tokenizer, where "restandardize to the same 10,000 tokens" would not.
    """
    if stored_clips <= 0 or target_clips <= 0:
        raise ValueError("clip counts must be positive")
    rate = tokens_per_clip(tokenizer)
    return restandardize_data_bits(data_bits, stored_clips * rate, target_clips * rate)


def epiplexity_from_eval_loss(
    online_code_bits: float,
    eval_loss_nats: float,
    train_tokens: float,
) -> float:
    """Model description length using the held-out loss as the final-model term.

    ``K(M) = K(X) - eval_loss * D_train / ln 2``, where ``K(X)`` is the
    cumulative online (prequential) code length that ``train_torch.py`` logs.

    Like the as-implemented reading, this substitutes a single scalar for
    Equation 8's ``P_M`` term rather than re-evaluating anything on historical
    tokens (see the module docstring); it draws that scalar from genuinely
    held-out data instead of a recent training window. That removes the
    near-final-models bias the as-implemented reading carries, at the cost of
    higher variance: the held-out set is a fixed 131,072 tokens in the checked-in
    sweeps, well under the roughly 1.56M tokens the as-implemented window
    typically averages over (``curves.tail_fraction_for``). It is a **deviation
    from the reference implementation** and must be reported alongside the
    as-implemented value, never substituted for it silently.
    """
    return online_code_bits - code_bits_from_loss(eval_loss_nats, train_tokens)


def two_part_code_bits(model_bits: float, data_bits: float) -> float:
    """``K(M,X) = K(M) + K(X|M)``, the quantity the frontier minimises."""
    return model_bits + data_bits


def structural_fraction(model_bits: float, data_bits: float) -> float | None:
    """``S_T / (S_T + H_T)`` -- the Figure 8a decomposition.

    Returns ``None`` when the total is non-positive or the model term is
    negative, since the ratio has no reading in those cases. Both terms must
    already be at the same inference-set size.
    """
    total = two_part_code_bits(model_bits, data_bits)
    if total <= 0 or model_bits < 0:
        return None
    return model_bits / total


def relative_precision_required(model_bits: float, online_code_bits: float) -> float | None:
    """How accurate the final-model loss term must be to resolve ``K(M)``.

    ``K(M)`` is a difference of two large, nearly equal numbers: the online code
    length and the final-model code length over the same training tokens. The
    returned ratio ``|K(M)| / K(X)`` is the relative error in the second term
    that would consume the entire signal, so an estimate ten times better than
    this is needed for roughly 10% accuracy on ``K(M)``.

    For low-structure data this is punishing. If ``S_T`` is around 1% of the
    total information -- the regime the paper reports for CIFAR-5M, where over
    99% of the content is random -- then the final-loss term needs about 0.1%
    relative accuracy. A single-window training-loss estimate is not that good,
    which is the mechanism behind widespread near-zero and negative ``K(M)``.
    Requential coding avoids the problem structurally: Equation 9 is a *sum* of
    KL terms rather than a difference, so it cannot go negative.
    """
    if online_code_bits <= 0:
        return None
    return abs(model_bits) / online_code_bits


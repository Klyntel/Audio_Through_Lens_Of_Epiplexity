"""Unit conversions and parameter accounting for epiplexity analysis.

Raw token counts are not a valid basis for comparing sweep cells. Two facts
force every conversion in this module.

**A token is not a fixed amount of audio.** ``TOKENIZER_SPECS`` pins each
tokenizer's output for a five-second window, and the rates differ by 8x
(EnCodec 375 tokens per 5 s, DAC 3000). A fixed ``T`` token budget therefore
buys 115.7 h of audio under EnCodec and 14.5 h under DAC, so two cells at the
same ``T`` have seen very different corpora.

``T`` is fixed in tokens by design, not by oversight. True duration parity
(every codec seeing the same corpus, with no clip left unused even for the
largest datasets) would need ``T`` on the order of 1e9 tokens, which would
stretch a single sweep from hours to multiple days; the team weighed that
cost against the exposure confound and kept ``T`` fixed. This module's job is
to make the resulting exposure difference visible and interpretable at
analysis time, not to argue ``T`` should change.

Reading an *earlier checkpoint of the same already-run trajectory* lets a
faster-token-rate codec be read at a slower codec's audio duration without a
new run: every checked-in sweep uses ``opt.schedule=const`` with
``warmup_tokens=16384`` against ``T=31,250,000`` (0.05%), so there is no
learning-rate decay to bias an early read, and a checkpoint at token count
``k`` is a reasonable proxy for what a run with ``T=k`` would have produced.
This is a documented assumption to check per sweep, not a property this
module verifies.

That does not make duration-matching free, though, and matching it trades one
mismatch for another rather than removing it: at equal model size, compute is
``6*N*tokens``, so reading EnCodec at the checkpoint matching DAC's full
14.5 h budget means comparing EnCodec at about 3.9M tokens of compute against
DAC's full 31.25M -- an 8x COMPUTE mismatch, the exact size of the duration
mismatch it was meant to remove, just moved to a different axis. Token
matching (the sweeps as run) and duration matching are two different,
non-substitutable comparisons, not one fixed and one broken: token matching
asks "how much structure per unit of compute, regardless of audio heard,"
duration matching asks "how much structure per hour of audio, regardless of
compute spent." Neither answer stands in for the other, so a report using
either must say which question it is answering and name what it left
unmatched, rather than presenting one as *the* comparable number.

**Epiplexity and entropy scale with different things.** In
``epiplexity/notebooks/scaling_laws.ipynb`` the asymptotic form
``b/(1-b) * D_0^b * D^(1-b)`` is evaluated at ``D`` = compute-optimal TRAINING
tokens, so ``S_T`` grows sublinearly with training tokens. ``H_T`` instead grows
linearly with the inference-set size, being a mean loss times that size. Section
4.4 states the trends: ``S_T`` rises with the compute budget while ``H_T`` falls,
and in the infinite-compute limit ``S_inf`` grows with ``|X|`` because that limit
trains on the whole set.

Two consequences. Bits-per-second is the right unit for ``H_T`` and the wrong
unit for ``S_T``, since ``S_T`` is not a per-token property of the coded set at
all. And at a FIXED training budget, changing the inference-set size does not
move ``S_T`` through any scaling law; it changes only which point minimises the
two-part code, and so which point's ``S_T`` is reported.

Model-size helpers mirror ``epiaudio/train.py``'s own derivation rather than
the cube-root approximation used in ``epiplexity/notebooks/scaling_laws.ipynb``.
That approximation exists upstream because published scaling laws only report
a single ``N``; we configure the model ourselves, so we can be exact.
"""

from __future__ import annotations

from math import log2, sqrt

from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS, TokenizerSpec

# Every entry in TOKENIZER_SPECS describes one five-second window.
SPEC_WINDOW_SECONDS = 5.0

# epiaudio/train.py:143 rounds d_model to a multiple of 64 and refuses to build
# a model below that floor.
D_MODEL_GRANULARITY = 64
MIN_D_MODEL = 64

# Non-embedding parameters per layer per d_model**2 for this architecture:
# 4 * d**2 attention (q, k, v, out) + 8 * d**2 MLP (fc1 4d, fc2 4d).
PARAMS_PER_LAYER_PER_D_SQUARED = 12


def spec_for(tokenizer: str) -> TokenizerSpec:
    """Return the tokenizer spec, with a readable error for unknown names."""
    try:
        return TOKENIZER_SPECS[tokenizer]
    except KeyError:
        known = ", ".join(sorted(TOKENIZER_SPECS))
        raise KeyError(f"unknown tokenizer {tokenizer!r}; known: {known}") from None


def tokens_per_clip(tokenizer: str) -> int:
    """Tokens emitted for one five-second clip."""
    return spec_for(tokenizer).sequence_length


def tokens_per_second(tokenizer: str) -> float:
    """Token rate in tokens per second of audio.

    EnCodec 75.0, SQ-Codec 355.6, XCodec 400.0, DAC 600.0.
    """
    return tokens_per_clip(tokenizer) / SPEC_WINDOW_SECONDS


def tokens_to_seconds(tokens: float, tokenizer: str) -> float:
    """Convert a token count to seconds of audio."""
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    return tokens / tokens_per_second(tokenizer)


def seconds_to_tokens(seconds: float, tokenizer: str) -> int:
    """Convert seconds of audio to a token count, rounded to a whole token.

    SQ-Codec's rate (355.6 tokens/s) is not an integer, so the result is
    rounded; at the durations used for analysis the rounding error is far below
    one part in 10^6.
    """
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    return round(seconds * tokens_per_second(tokenizer))


def max_bits_per_token(tokenizer: str) -> float:
    """Uniform-code length per token, ``log2(V)``.

    This is the loss ceiling a model that has learned nothing would sit at, and
    the reference point for the "flat at the ceiling" diagnosis in
    :mod:`epiaudio.analysis.curves`. EnCodec/DAC/XCodec give 10.0 bits;
    SQ-Codec's V=117,649 gives 16.84.
    """
    spec = spec_for(tokenizer)
    if spec.vocab_size is None:
        raise ValueError(
            f"tokenizer {tokenizer!r} has no vocabulary size; it is a continuous "
            "feature extractor and cannot be used for discrete code lengths"
        )
    return log2(spec.vocab_size)


def nominal_bitrate_bits_per_second(tokenizer: str) -> float:
    """Uniform-code bitrate of the token stream, ``tokens/s * log2(V)``.

    Useful as a sanity denominator: the four tokenizers span 0.75 kbps
    (EnCodec) to 6.0 kbps (DAC), so absolute code lengths are dominated by this
    choice before any property of the audio enters.
    """
    return tokens_per_second(tokenizer) * max_bits_per_token(tokenizer)


def d_model_for(params_millions: float, depth: int) -> int:
    """Reproduce ``epiaudio/train.py``'s width derivation from (P, N).

    ``train.py:143`` computes
    ``D = round(sqrt(P * 1e6 / N / 12) / 64) * 64``
    and exits when the result falls below 64. ``model.P`` is therefore a
    *non-embedding* parameter budget, not a total, which is why embedding
    parameters have to be added back explicitly below.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if params_millions <= 0:
        raise ValueError("params_millions must be positive")
    raw = sqrt(params_millions * 1e6 / depth / PARAMS_PER_LAYER_PER_D_SQUARED)
    d_model = round(raw / D_MODEL_GRANULARITY) * D_MODEL_GRANULARITY
    if d_model < MIN_D_MODEL:
        raise ValueError(
            f"P={params_millions} at depth={depth} implies d_model={d_model}, "
            f"below the {MIN_D_MODEL} floor train.py enforces; this grid point "
            "would have exited rather than produced a run"
        )
    return d_model


def non_embedding_params(depth: int, d_model: int) -> int:
    """Transformer-block parameters, ``12 * N * D**2``."""
    if depth <= 0 or d_model <= 0:
        raise ValueError("depth and d_model must be positive")
    return PARAMS_PER_LAYER_PER_D_SQUARED * depth * d_model * d_model


def embedding_params(
    vocab_size: int,
    context_length: int,
    d_model: int,
    *,
    tie_readout: bool = False,
) -> int:
    """Token embedding, positional embedding, and readout parameters.

    ``model_torch.py`` builds ``nn.Embedding(V, D)``, ``nn.Embedding(L, D)`` and an
    untied ``nn.Linear(D, V, bias=False)`` readout, which is why ``tie_readout``
    defaults to False and a large vocabulary is charged twice. Pass
    ``tie_readout=True`` only when modelling a variant that shares the two.
    """
    if min(vocab_size, context_length, d_model) <= 0:
        raise ValueError("vocab_size, context_length and d_model must be positive")
    total = (vocab_size + context_length) * d_model
    if not tie_readout:
        total += vocab_size * d_model
    return total


def flop_bearing_params(
    params_millions: float,
    depth: int,
    vocab_size: int,
) -> int:
    """Parameters that scale with a matmul per token, for a secondary compute axis.

    This is NOT a claim that ``train_torch.py``'s logged compute is wrong. It
    computes ``6 * num_params * tokens`` with no adjustment for vocabulary, which
    is exactly the plain, unmodified ``6ND`` formula the epiplexity paper's main
    text cites (Section 4: "training a model with N parameters on D tokens takes
    time approximately 6ND (Kaplan et al., 2020)"), with no vocab-size term
    either -- the paper's main text does not adjust for this, and neither does
    this repo's compute logging. That is the correct default, and remains the
    primary, paper-faithful number to report.

    The reason a secondary axis is still worth computing is Kaplan et al. 2020's
    own definition of ``N``: they define it as *non-embedding* parameters
    specifically, given by ``N = 12 * n_layer * d_model**2`` (identical to
    :func:`non_embedding_params` here -- reproducing it against GPT-3's own
    published (n_layer=96, d_model=12288) gives 173.95B against a published
    175B, an independent check that the formula is the same one). Their stated
    reason is that excluding embeddings gives cleaner empirical power-law fits,
    not a FLOPs argument -- but their own per-token compute breakdown (Table 1)
    separately costs the de-embedding (readout) layer at ``2 * d_model *
    n_vocab`` FLOPs, distinct from the embedding lookup, which is not costed at
    all because it is an index into a table rather than a matmul. Folding
    everything into a blanket ``6N`` is an approximation that is only accurate
    when that readout term is small relative to ``N``, which holds for GPT-3
    (617.6M / 173.95B = 0.355%) and for three of our four codecs (0.69% at
    N=12, P=160 for EnCodec, XCodec, DAC) but not for SQ-Codec, where the
    readout alone is 79.8% of the block parameter count -- a regime the 6ND
    shortcut was never validated against, because no GPT-scale model has a
    vocabulary anywhere near that large relative to its width.

    So: report the plain logged compute as the primary, paper-faithful axis.
    Use this function only as a secondary, EpiAudio-specific sensitivity check,
    motivated by the fact that our small-model/large-vocabulary sweep cells
    (chiefly SQ-Codec) sit well outside the regime Kaplan's approximation was
    validated in. It excludes the token and positional embedding tables (index
    lookups, not matmuls) and keeps the transformer blocks and the untied
    readout (a real ``D x V`` matmul per token).
    """
    d_model = d_model_for(params_millions, depth)
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return non_embedding_params(depth, d_model) + vocab_size * d_model


def analytic_total_params(
    params_millions: float,
    depth: int,
    vocab_size: int,
    context_length: int,
    *,
    tie_readout: bool = False,
) -> int:
    """Embedding-inclusive parameter count implied by a grid point.

    Prefer the ``num_params`` value logged to Comet, which is measured from the
    real parameter tree. Use this when planning a sweep, or when reconstructing
    a run whose Comet record is unavailable.
    """
    d_model = d_model_for(params_millions, depth)
    return non_embedding_params(depth, d_model) + embedding_params(
        vocab_size, context_length, d_model, tie_readout=tie_readout
    )


def embedding_parameter_fraction(
    params_millions: float,
    depth: int,
    vocab_size: int,
    context_length: int,
    *,
    tie_readout: bool = False,
) -> float:
    """Fraction of total parameters spent on embeddings and the readout.

    This is why the (N, P) grid does not span a comparable range across
    tokenizers. ``model.P`` budgets only the transformer blocks, so a large
    vocabulary is added on top of it rather than competing with it.

    ``model_torch.py`` builds an untied readout (``nn.Linear(D, V, bias=False)``),
    so a large vocabulary is paid for twice: once in the token embedding table
    and again in the output projection. At N=12, P=160 the width rounds to 1024,
    giving 151.0M block parameters. EnCodec (V=1024, L=375) adds 2.4M for 1.6% of
    its 153.5M total; SQ-Codec (V=117,649, L=1778) adds 242.8M for 61.7% of its
    393.8M total, a 2.57x larger model at the same nominal grid point.

    Note this is a PARAMETER ratio, not a FLOPs ratio; see
    :func:`flop_bearing_params`. Both totals were checked against an
    instantiated ``TransformerDecoder`` rather than derived on paper.
    """
    d_model = d_model_for(params_millions, depth)
    embed = embedding_params(
        vocab_size, context_length, d_model, tie_readout=tie_readout
    )
    total = analytic_total_params(
        params_millions, depth, vocab_size, context_length, tie_readout=tie_readout
    )
    return embed / total


def repeat_factor(
    train_clips: int,
    tokenizer: str,
    token_budget: int,
) -> float:
    """How many times the training stream is replayed to reach ``token_budget``.

    ``prepare_audio.py`` samples one random five-second window per clip under a
    fixed seed and writes it once, so the effective corpus is
    ``train_clips * 5 s`` and a second epoch replays *identical* tokens. Any
    value above 1.0 breaks the prequential estimator's assumption that each
    coded token is unseen.

    The factor is tokenizer-dependent for the same corpus: EnCodec needs 83,333
    clips to reach T=31.25M while DAC needs only 10,417, so a corpus can repeat
    13x under one tokenizer and 1.6x under another.
    """
    if train_clips <= 0:
        raise ValueError("train_clips must be positive")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    available = train_clips * tokens_per_clip(tokenizer)
    return token_budget / available


def clips_required(token_budget: int, tokenizer: str) -> int:
    """Clips needed to cover ``token_budget`` exactly once."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    per_clip = tokens_per_clip(tokenizer)
    return -(-token_budget // per_clip)  # ceiling division

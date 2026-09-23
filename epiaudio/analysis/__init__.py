"""Comparable re-analysis of EpiAudio sweep outputs.

The sweeps report one epiplexity value per (dataset, tokenizer) cell, and those
values are not comparable to each other as stored. The differences come from the
measurement pipeline rather than from the audio, and each module here addresses
one of them:

* :mod:`units` for the fact that a token is not a fixed amount of audio, that
  ``model.P`` budgets only the transformer blocks, and that ``6ND`` charges
  embedding lookups as if they were matmuls;
* :mod:`estimators` for restandardizing a cell's data term to a common
  inference-set size, and for the two available readings of Equation 8's
  final-model term;
* :mod:`curves` for negative ``K(M)``, which is a statement about loss-curve
  shape and is invisible in the upstream analysis path;
* :mod:`frontier` for hull geometry that tolerates non-positive code lengths, and
  for reporting whether a cell had saturated at the compute ceiling.

One further module turns those conventions into a table: :mod:`pipeline`
fetches sweeps, recovers each one's provenance, and builds the comparable
table row by row, raising rather than defaulting when the inference-set size
is unrecoverable.

Import the modules rather than their contents, so a call site says which concern
it is reaching for: ``units.tokens_per_second(...)``, not a flat namespace of
several dozen names.

Nothing here trains or modifies a sweep artefact. :mod:`pipeline` is the one
exception to "contacts Comet": it fetches and caches a negative endpoint's
loss curve to diagnose it, opt-in at the call site and behind a
:class:`~epiaudio.analysis.pipeline.CometMetricFetcher` protocol so a test
never needs a real Comet connection. See ``README.md`` in this directory for
the full rationale and the CLI.
"""

from __future__ import annotations

from epiaudio.analysis import (
    curves,
    estimators,
    frontier,
    pipeline,
    units,
)

__all__ = [
    "curves",
    "estimators",
    "frontier",
    "pipeline",
    "units",
]

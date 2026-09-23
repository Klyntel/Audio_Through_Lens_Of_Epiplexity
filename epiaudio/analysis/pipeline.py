"""Fetch sweeps, build one comparable table, print and optionally save it.

    uv run python -m epiaudio.analysis

The whole job is three steps, run once per sweep and collected into a plain
list of dict rows -- nothing here holds a sweep's facts in an object on the
way to becoming a table row:

  1. ``load_sweep_facts``  -- dataset, tokenizer, inference-set size, read
                              from the sweep's saved config.
  2. ``load_points``       -- the pooled (compute, model_bits, data_bits)
                              cloud, at each cell's own recorded inference-set
                              size.
  3. ``build_row``         -- rebuild the frontier from its non-negative
                              points, diagnose a negative endpoint where one
                              is still reported, and return one flat dict.

``main`` appends each cell's row to a list and calls ``pd.DataFrame(rows)``
once, at the end.

The token budget a sweep used is the fixed comparison axis (team decision,
2026-08-25): a cell is always reported at its own recorded inference-set
size, never restandardized to match another cell's. ``K(X|M)`` is the code
length of the set actually held out for that cell, so reporting it at that
size is reporting the real measurement; ``entropy_bits_per_second`` gives the
duration-invariant rate the row also carries, which is how cells with
different native inference-set sizes get compared without inventing a common
one. Earlier revisions offered a ``--inference-hours`` flag that restandardized
every cell's data term to one common duration; removed, since fixing a
duration rather than the token budget was never the right axis for this
grid (ethanalwaise-del, PR #121).

Three further facts about the raw sweep outputs force the
``load_sweep_facts`` / ``load_points`` steps to do real work rather than a
plain CSV read:

* the **inference-set size** used to build ``K(X|M)`` is not in the CSVs at
  all -- it is measured at runtime from ``test.bin`` -- so a cell's data term
  cannot be reported, or its per-second rate computed, without either
  supplying that size explicitly or measuring it from
  ``<data_root>/<dataset>/test.bin``. A sweep missing it is skipped rather
  than guessed at.
* the **dataset and tokenizer** are not columns either; the tokenizer is
  encoded in the ``ds_path`` suffix, and the sweep's ``__sweep_config.yaml``
  records the rest (``T``, ``model.V``, ``model.L``).
* **a negative point never disappears from the record, even though it is
  excluded from ranking by default.** A single deeply negative point --
  a measurement failure, not a real result -- can otherwise dominate a
  cell's whole frontier and become its reported endpoint by itself, so
  ``build_row`` builds each cell's ranked frontier from its non-negative
  points only (``--keep-negative-points`` opts back into the unfiltered
  view). ``negative_points``/``negative_point_fraction`` always count
  against the full, unfiltered point cloud regardless, so a cell with
  broken measurements still reports that fact even when its endpoint
  excludes them.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from epiaudio.analysis.smooth_loss import get_smooth_code_points
from epiaudio.analysis import units
from epiaudio.analysis.curves import diagnose
from epiaudio.analysis.estimators import restandardize_data_bits, structural_fraction
from epiaudio.analysis.frontier import CodeLengthPoint, summarize_frontier
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS
from epiaudio.utils import (
    RUN_INDEX_COLUMN,
    COMPUTE_COLUMN,
    DATA_BITS_COLUMN,
    MODEL_BITS_COLUMN,
    fetch_sweep_records,
    load_run_comet_provenance
)

DEFAULT_OUTPUT_DIR = Path("sweep_outputs")
DEFAULT_DATA_ROOT = Path("data")

# prepare_audio.py writes uint16 token ids unless the vocabulary exceeds what
# uint16 holds, matching train_torch.py's reader (see sweep.py:513-515).
UINT16_VOCAB_LIMIT = 65536

SWEEP_CONFIG_SUFFIX = "__sweep_config.yaml"

class MissingProvenanceError(RuntimeError):
    """Raised when a sweep cannot be made comparable to the others."""


# --------------------------------------------------------------------------
# Step 1: sweep facts (dataset, tokenizer, inference-set size, ...) as a
# plain dict -- nothing computed here is a method, everything downstream
# reads a key.
# --------------------------------------------------------------------------


def token_dtype(vocab_size: int) -> type[np.unsignedinteger[Any]]:
    """Token id dtype ``prepare_audio.py`` used for this vocabulary."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return np.uint16 if vocab_size <= UINT16_VOCAB_LIMIT else np.uint32


def count_bin_tokens(path: Path, vocab_size: int) -> int:
    """Token count of a ``.bin`` split, read as a memmap length."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(len(np.memmap(path, dtype=token_dtype(vocab_size), mode="r")))


def infer_tokenizer(ds_path: str | Path) -> str | None:
    """Recover the tokenizer name from a dataset path suffix.

    Mirrors ``sweep_preflight._infer_tokenizer``: ``data/birdset_hsn_encodec``
    resolves to ``encodec``, preferring the longest match so a name that is a
    suffix of another cannot shadow it.
    """
    name = Path(str(ds_path)).name
    matches = [key for key in TOKENIZER_SPECS if name.endswith(f"_{key}")]
    if not matches:
        return None
    return max(matches, key=len)


def infer_dataset(ds_path: str | Path, tokenizer: str) -> str:
    """Strip the tokenizer suffix to recover the corpus name."""
    return Path(str(ds_path)).name.removesuffix(f"_{tokenizer}")


def _infer_sweep_name(sweep_config_path: str | Path) -> str:
    return Path(sweep_config_path).name.removesuffix(SWEEP_CONFIG_SUFFIX)

def _sweep_parameters(config_path: str | Path) -> dict[str, Any]:
    """Load a sweep config's ``parameters`` block as a plain dict.

    Deliberately avoids ``OmegaConf.select`` for these keys. Sweep parameter
    names contain literal dots (``"model.N"``), and ``select`` reads a dot as a
    path separator, so ``select(cfg, "parameters.model.N")`` looks for a nested
    ``model`` node and silently returns ``None``. Converting to a container first
    keeps the keys intact.

    ``resolve=False``, matching ``sweep.py``'s own ``_parse_command`` (its
    comment: "resolve=False keeps wandb templating tokens like ${env} as
    literal strings instead of trying to resolve them as interpolations").
    Every real ``__sweep_config.yaml`` carries the sweep's whole original
    YAML, ``command: [${env}, python, ...]`` included, since
    ``sweep.py:_save_sweep_config`` saves ``self.cfg`` unmodified. wandb's
    ``${env}``/``${program}``/``${args}`` reuse OmegaConf's own interpolation
    syntax for an unrelated purpose, so ``resolve=True`` here raises
    ``InterpolationKeyError`` on that block on every real sweep output --
    confirmed against a fixture shaped like ``sweeps/birdset_hsn.yaml``.
    Resolution was never needed for ``parameters`` itself: no sweep config in
    this repo uses ``${...}`` inside that block, only literal ``value``/
    ``values`` entries, and dotted keys survive ``to_container`` either way,
    since only ``.select``'s own path parsing treats a dot specially.
    """
    container = OmegaConf.to_container(OmegaConf.load(Path(config_path)), resolve=False)
    if not isinstance(container, dict):
        raise MissingProvenanceError(f"{config_path} is not a mapping")
    parameters = container.get("parameters")
    if not isinstance(parameters, dict):
        raise MissingProvenanceError(f"{config_path} records no parameters block")
    return {str(key): value for key, value in parameters.items()}


def _sweep_parameter(parameters: Mapping[str, Any], key: str) -> Any:
    """Read one wandb-style sweep parameter, whether ``value`` or ``values``."""
    node = parameters.get(key)
    if not isinstance(node, Mapping):
        return None
    if "value" in node:
        return node["value"]
    return node.get("values")


def _discover_sweep_configs(output_dir: str | Path, sweep_prefix: str) -> list[str]:
    output_path = Path(output_dir)
    paths = []

    for path in output_path.glob(f"*{SWEEP_CONFIG_SUFFIX}"):
        if not path.name.startswith(sweep_prefix):
            continue
        paths.append(path)

    return paths

def build_sweep_facts(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    inference_tokens: int | None = None,
    train_clips: int | None = None,
) -> dict[str, Any]:
    """Build one sweep's facts dict from a saved ``__sweep_config.yaml``.

    Everything except the inference-set size comes from the config. That size is
    taken from ``inference_tokens`` if given, otherwise measured from
    ``<data_root>/<ds_path>/test.bin``. If neither is available the cell's data
    term (``K(X|M)``) has no denominator to report a per-second rate against,
    so :class:`MissingProvenanceError` is raised rather than a default being
    invented; this is about reporting the cell's own measurement completely,
    not about restandardizing it to match another cell's.

    A sweep whose grid spans several ``ds_path`` values covers several cells and
    must be split before it can be described by one facts dict.

    The returned dict always has ``sweep_name``, ``dataset``, ``tokenizer``,
    ``inference_tokens``, ``token_budget``, ``vocab_size``, ``context_length``
    and ``train_clips`` (``None`` unless supplied).
    """
    parameters = _sweep_parameters(config_path)

    ds_paths = _sweep_parameter(parameters, "ds_path")
    if ds_paths is None:
        raise MissingProvenanceError(f"{config_path} records no ds_path")
    candidates = [ds_paths] if isinstance(ds_paths, str) else [str(v) for v in ds_paths]
    if len(candidates) != 1:
        raise MissingProvenanceError(
            f"{config_path} sweeps {len(candidates)} ds_path values "
            f"({candidates}); one facts dict describes one dataset/tokenizer "
            "cell, so split the outputs per ds_path first"
        )
    ds_path = candidates[0]

    tokenizer = infer_tokenizer(ds_path)
    if tokenizer is None:
        raise MissingProvenanceError(
            f"cannot infer a known tokenizer from ds_path {ds_path!r}"
        )
    spec = TOKENIZER_SPECS[tokenizer]
    if spec.vocab_size is None:
        raise MissingProvenanceError(
            f"tokenizer {tokenizer!r} has no vocabulary size and cannot be "
            "analysed as a discrete code"
        )

    token_budget = _sweep_parameter(parameters, "T")
    if token_budget is None:
        raise MissingProvenanceError(f"{config_path} records no T")
    token_budget = int(token_budget)
    if token_budget <= 0:
        raise MissingProvenanceError(f"{config_path} records a non-positive T")

    context_length = _sweep_parameter(parameters, "model.L") or spec.sequence_length

    sweep_name = _infer_sweep_name(config_path)
    resolved_inference = inference_tokens
    if resolved_inference is None and data_root is not None:
        test_path = Path(data_root) / Path(ds_path).name / "test.bin"
        if test_path.is_file():
            resolved_inference = count_bin_tokens(test_path, spec.vocab_size)
    if resolved_inference is None:
        raise MissingProvenanceError(
            f"sweep {sweep_name!r} has no recorded inference-set size. The "
            "stored K(X|M) column was built as eval_loss * test_tokens / ln 2, "
            "so without test_tokens the cell has no duration to report its "
            "own entropy rate against. Supply inference_tokens explicitly, or "
            f"point data_root at the directory holding {Path(ds_path).name}/test.bin."
        )
    resolved_inference = int(resolved_inference)
    if resolved_inference <= 0:
        raise MissingProvenanceError(f"sweep {sweep_name!r} has a non-positive test_tokens")

    return {
        "sweep_name": sweep_name,
        "dataset": infer_dataset(ds_path, tokenizer),
        "tokenizer": tokenizer,
        "inference_tokens": resolved_inference,
        "token_budget": token_budget,
        "vocab_size": int(spec.vocab_size),
        "context_length": int(context_length),
        "train_clips": train_clips,
    }


def load_sweep_facts(
    sweep_config_path: str,
    *,
    data_root: str | Path | None = None,
    inference_tokens: int | None = None,
    train_clips: int | None = None,
) -> dict[str, Any]:
    """Build a sweep's facts dict from the saved sweep config."""
    return build_sweep_facts(
        sweep_config_path,
        data_root=data_root,
        inference_tokens=inference_tokens,
        train_clips=train_clips,
    )


def audio_hours_seen(sweep_facts: Mapping[str, Any]) -> float:
    """Hours of audio the token budget corresponds to.

    Cells are only comparable at matched values of this, not at matched
    ``token_budget``: a fixed token count buys 8x more audio under EnCodec
    than under DAC.
    """
    seconds = units.tokens_to_seconds(sweep_facts["token_budget"], sweep_facts["tokenizer"])
    return seconds / 3600.0


def repeat_factor_for(sweep_facts: Mapping[str, Any]) -> float | None:
    """Replays of the training stream, or ``None`` without a clip count."""
    train_clips = sweep_facts.get("train_clips")
    if train_clips is None:
        return None
    return units.repeat_factor(train_clips, sweep_facts["tokenizer"], sweep_facts["token_budget"])


def prequential_assumption_holds(sweep_facts: Mapping[str, Any]) -> bool | None:
    """Whether every coded token was unseen when it was coded."""
    factor = repeat_factor_for(sweep_facts)
    return None if factor is None else factor <= 1.0


def target_inference_tokens_for(sweep_facts: Mapping[str, Any], inference_seconds: float) -> int:
    """Token count matching ``inference_seconds`` for this cell's tokenizer.

    A plain unit conversion, not a policy about what to hold fixed: the token
    budget is the fixed comparison axis (team decision, 2026-08-25), and this
    is only ever called with a cell's own native size (``units.
    tokens_to_seconds`` of its recorded ``inference_tokens``) to express that
    size as a token count for reporting.
    """
    return units.seconds_to_tokens(inference_seconds, sweep_facts["tokenizer"])


# --------------------------------------------------------------------------
# Step 2: the pooled point cloud, at each cell's native inference-set size.
# --------------------------------------------------------------------------


def load_points(
    sweep_name: str,
    sweep_facts: Mapping[str, Any],
    *,
    target_inference_tokens: int | None = None,
) -> list[CodeLengthPoint]:
    """Read every pooled point from a sweep, restandardized only if asked.

    ``main`` never passes ``target_inference_tokens`` (the token budget is the
    fixed comparison axis, team decision 2026-08-25 -- see the module
    docstring), so every CLI-produced table is read at each cell's own
    native size. The parameter stays available for a caller that has a
    specific, narrower reason to restandardize a single sweep's own points
    (:func:`estimators.restandardize_data_bits` is the formula, and remains
    a correct one -- what changed is only that this package's own report no
    longer offers restandardization as a comparison mode). Passing it rescales
    the data term to that size exactly, since ``K(X|M)`` is linear in it; the
    frontier must then be rebuilt from these points, because changing the
    data term changes which points win, so reusing the stored
    ``__pareto_front_data.csv`` would be the frontier for the old size.
    """
    frame = fetch_sweep_records(sweep_name, test_tokens=sweep_facts["inference_tokens"])

    # Column access by name rather than `itertuples`: pandas mangles field names
    # that are not valid identifiers, so `K(X|M)` would arrive as `_4`.
    run_indices = frame[RUN_INDEX_COLUMN].to_numpy()
    computes = frame[COMPUTE_COLUMN].to_numpy(dtype=float)
    model_bits = frame[MODEL_BITS_COLUMN].to_numpy(dtype=float)
    data_bits = frame[DATA_BITS_COLUMN].to_numpy(dtype=float)

    points: list[CodeLengthPoint] = []
    for run_index, compute, model, data in zip(run_indices, computes, model_bits, data_bits):
        if target_inference_tokens is not None:
            data = restandardize_data_bits(
                float(data), sweep_facts["inference_tokens"], target_inference_tokens
            )
        points.append(
            CodeLengthPoint(
                compute_flops=float(compute),
                model_bits=float(model),
                data_bits=float(data),
                run_index=int(run_index),
            )
        )
    return points


# --------------------------------------------------------------------------
# Optional: an endpoint's loss curve, for diagnose(). sweep.py never writes
# the raw per-checkpoint losses to *__all_run_data.csv -- only the derived
# K(M)/K(X|M) survive -- so recovering them means going back to Comet, the
# one place they still exist. This step is additive and self-contained: a
# sweep whose CSV predates the columns below simply gets no diagnosis, not
# an error, and every fetch is cached after its first call so re-running
# the analysis (even at a different --inference-hours, selecting a
# different run as the endpoint) never re-queries a run it already has.
# --------------------------------------------------------------------------

COMET_PROVENANCE_COLUMNS = (
    "experiment_key",
    "train_student",
    "teacher_ema_active",
    "student_ema_active",
)

# train_torch.py divides every K(...) metric by 1e6 before logging (see
# estimators.py's module docstring); this is the inverse, applied once here
# for the K(X) value fetched back off Comet.
COMET_BITS_PER_MEGABIT = 1e6


class CometMetricFetcher(Protocol):
    """What fetching a diagnosis input needs from a Comet client -- narrow
    enough to fake in tests, with no real network call or comet_ml install."""

    def __call__(self, experiment_key: str, metric_name: str) -> list[dict[str, Any]]: ...


def _fetch_comet_metric(experiment_key: str, metric_name: str) -> list[dict[str, Any]]:
    """One metric's full logged history for one Comet experiment.

    Deferred import: comet_ml is only needed on an actual cache miss, so
    this module stays importable -- and every existing test keeps passing
    -- without comet_ml installed at all.
    """
    import comet_ml

    api_experiment = comet_ml.APIExperiment(
        previous_experiment=experiment_key,
        api_key=os.environ.get("COMET_ML_API"),
    )
    return api_experiment.get_metrics(metric_name) or []


def _select_loss_metric(train_student: bool, teacher_ema_active: bool, student_ema_active: bool) -> str:
    """The eval-loss metric name sweep.py logs for this run's configuration.

    Mirrors sweep.py's GridSweeperWithFrontier._fetch_epiplexity: requential
    runs (train_student) log the student's eval loss, prequential runs the
    teacher's; either falls back from the ema_* variant to the plain metric
    when that model's EMA isn't configured (sweep.py's _ema_active: a
    positive EMA decay must actually be set for the ema_* metric to exist).
    Ported rather than imported, since GridSweeperWithFrontier's version is
    a bound method on live sweep state; kept behaviourally identical so the
    two stay checkable against each other rather than silently drifting.
    """
    if train_student:
        return "ema_student_eval_loss" if student_ema_active else "student_eval_loss"
    return "ema_teacher_eval_loss" if teacher_ema_active else "teacher_eval_loss"


def _endpoint_losses_cache_path(output_dir: str | Path, sweep_name: str) -> Path:
    return Path(output_dir) / f"{sweep_name}__endpoint_losses_cache.json"


def _read_endpoint_losses_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    return {int(run_index): value for run_index, value in json.loads(cache_path.read_text()).items()}


def _write_endpoint_losses_cache(cache_path: Path, cache: Mapping[int, Mapping[str, Any]]) -> None:
    payload = {str(run_index): dict(value) for run_index, value in cache.items()}
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fetch_endpoint_losses(
    sweep_name: str,
    run_index: int,
    *,
    output_dir: str | Path,
    fetch_metric: CometMetricFetcher | None = None,
) -> tuple[list[float], float | None] | None:
    """A run's loss trajectory and cumulative online code length, cached.

    Returns None when this sweep has no Comet provenance for run_index (an
    older CSV, or a run whose eval step never logged) -- diagnosis is
    skipped for that cell, not guessed at. Otherwise queries Comet only on
    a genuine cache miss, writing the result to a companion file next to
    the sweep's existing CSVs so every later call for the same
    (sweep, run_index) reads the cache instead, regardless of which
    invocation or --inference-hours asked for it.

    ``fetch_metric`` defaults to ``None`` rather than binding
    ``_fetch_comet_metric`` directly, so ``unittest.mock.patch.object(pipeline,
    "_fetch_comet_metric", ...)`` -- the natural way to fake Comet for a
    caller that never threads a fetcher through, like ``main``, or here in
    ``build_row`` -- takes effect. A default bound at definition time would
    keep calling the original function regardless of what gets patched.
    """
    fetch_metric = fetch_metric or _fetch_comet_metric
    provenance = load_run_comet_provenance(sweep_name)
    if provenance is None or run_index not in provenance:
        return None

    cache_path = _endpoint_losses_cache_path(output_dir, sweep_name)
    cache = _read_endpoint_losses_cache(cache_path)
    cached = cache.get(run_index)
    if cached is not None:
        return list(cached["losses"]), cached["online_code_bits"]

    run = provenance[run_index]
    loss_metric = _select_loss_metric(
        run["train_student"], run["teacher_ema_active"], run["student_ema_active"]
    )
    loss_records = sorted(fetch_metric(run["experiment_key"], loss_metric), key=lambda r: r["timestamp"])
    losses = [float(r["metricValue"]) for r in loss_records]

    kx_records = fetch_metric(run["experiment_key"], "K(X)")
    online_code_bits = (
        None
        if not kx_records
        else float(max(kx_records, key=lambda r: r["timestamp"])["metricValue"]) * COMET_BITS_PER_MEGABIT
    )

    cache[run_index] = {"losses": losses, "online_code_bits": online_code_bits}
    _write_endpoint_losses_cache(cache_path, cache)
    return losses, online_code_bits


# --------------------------------------------------------------------------
# Step 3: one flat row per cell.
# --------------------------------------------------------------------------


def build_provenance_row(sweep_facts: Mapping[str, Any]) -> dict[str, Any]:
    """Raw sweep facts every figure has to be read against.

    Nothing here depends on a result: it exists so a reader can check, for any
    comparison, whether the cells being compared saw comparable audio and
    whether their coded tokens were fresh.
    """
    tokenizer = sweep_facts["tokenizer"]
    return {
        "sweep_name": sweep_facts["sweep_name"],
        "dataset": sweep_facts["dataset"],
        "tokenizer": tokenizer,
        "vocab_size": sweep_facts["vocab_size"],
        "context_length": sweep_facts["context_length"],
        "tokens_per_second": units.tokens_per_second(tokenizer),
        "max_bits_per_token": units.max_bits_per_token(tokenizer),
        "token_budget": sweep_facts["token_budget"],
        "audio_hours_seen": audio_hours_seen(sweep_facts),
        "train_clips": sweep_facts.get("train_clips"),
        "clips_required": units.clips_required(sweep_facts["token_budget"], tokenizer),
        "repeat_factor": repeat_factor_for(sweep_facts),
        "prequential_assumption_holds": prequential_assumption_holds(sweep_facts),
        "stored_inference_tokens": sweep_facts["inference_tokens"],
        "stored_inference_hours": (
            units.tokens_to_seconds(sweep_facts["inference_tokens"], tokenizer) / 3600.0
        ),
    }


def build_row(
    sweep_facts: Mapping[str, Any],
    points: Sequence[CodeLengthPoint],
    *,
    inference_seconds: float,
    pos_only: bool = True,
    endpoint_losses: Sequence[float] | None = None,
    online_code_bits: float | None = None,
    sweep_name: str,
    output_dir: str | Path | None = None,
    fetch_metric: CometMetricFetcher | None = None,
) -> dict[str, Any]:
    """Rebuild one cell's frontier at ``inference_seconds`` and flatten it to a row.

    ``inference_seconds`` is a cell's own native size in every caller this
    package ships (the token budget is the fixed comparison axis, not
    duration -- team decision 2026-08-25); ``points`` must already correspond
    to it (see :func:`load_points`), since this only derives the token count
    for reporting, so a mismatch would show up in the row rather than be
    silently corrected.

    ``pos_only`` defaults to ``True``: a cell's frontier and endpoint are built
    from its non-negative points only, since a single deeply negative point --
    a measurement failure, not a real result -- can otherwise dominate the
    whole hull and become the reported endpoint by itself (see
    :func:`frontier.lower_convex_hull`). This does not make negative points
    disappear from the record: ``negative_points``/``negative_point_fraction``
    below always count against the full, unfiltered ``points``, regardless of
    ``pos_only``, so a cell with broken measurements still reports that fact
    even when its ranked endpoint excludes them. Pass ``pos_only=False`` to
    inspect a cell's full, unfiltered picture instead.

    ``endpoint_losses`` is the loss curve of whichever run supplied the
    endpoint. When supplied, a negative endpoint is diagnosed instead of
    merely flagged. When it is not supplied but ``sweep_name`` and ``output_dir``
    are, it is fetched (and cached) via :func:`fetch_endpoint_losses` instead
    -- an explicit ``endpoint_losses`` always wins, so a caller that already
    has the curve never triggers a Comet lookup it didn't ask for.
    """
    inference_tokens = target_inference_tokens_for(sweep_facts, inference_seconds)
    summary = summarize_frontier(points, inference_tokens=inference_tokens, pos_only=pos_only)

    smoothed_points = get_smooth_code_points(
        sweep_facts["sweep_name"],
        inference_tokens
    )
    smoothed_summary = summarize_frontier(
        smoothed_points,
        inference_tokens=inference_tokens
    )

    resolved_losses = endpoint_losses
    resolved_online_code_bits = online_code_bits
    if resolved_losses is None and output_dir is not None:
        if summary.endpoint_run_index is not None:
            fetched = fetch_endpoint_losses(
                sweep_name, summary.endpoint_run_index, output_dir=output_dir, fetch_metric=fetch_metric
            )
            if fetched is not None:
                resolved_losses, resolved_online_code_bits = fetched

    diagnosis = None
    if resolved_losses and summary.endpoint_model_bits is not None:
        diagnosis = diagnose(
            resolved_losses,
            model_bits=summary.endpoint_model_bits,
            vocab_size=sweep_facts["vocab_size"],
            online_code_bits=resolved_online_code_bits,
            train_tokens=sweep_facts["token_budget"],
        )

    negative_point_fraction = (
        None
        if summary.point_count == 0
        else summary.negative_point_count / summary.point_count
    )
    row_structural_fraction = (
        None
        if summary.endpoint_model_bits is None or summary.endpoint_data_bits is None
        else structural_fraction(summary.endpoint_model_bits, summary.endpoint_data_bits)
    )
    entropy_bits_per_second = (
        None
        if summary.endpoint_data_bits is None or inference_seconds <= 0
        else summary.endpoint_data_bits / inference_seconds
    )

    return {
        "sweep_name": sweep_facts["sweep_name"],
        "dataset": sweep_facts["dataset"],
        "tokenizer": sweep_facts["tokenizer"],
        "inference_seconds": inference_seconds,
        "inference_hours": inference_seconds / 3600.0,
        "inference_tokens": summary.inference_tokens,
        "train_audio_hours": audio_hours_seen(sweep_facts),
        "repeat_factor": repeat_factor_for(sweep_facts),
        "prequential_assumption_holds": prequential_assumption_holds(sweep_facts),
        "frontier_points": summary.point_count,
        "negative_points": summary.negative_point_count,
        "negative_point_fraction": negative_point_fraction,
        "endpoint_compute_flops": summary.endpoint_compute_flops,
        "epiplexity_bits": summary.endpoint_model_bits,
        "entropy_bits": summary.endpoint_data_bits,
        "entropy_bits_per_second": entropy_bits_per_second,
        "smoothed_compute_flops": smoothed_summary.endpoint_compute_flops,
        "smoothed_epiplexity_bits": smoothed_summary.endpoint_model_bits,
        "smoothed_entropy_bits": smoothed_summary.endpoint_data_bits,
        "structural_fraction": row_structural_fraction,
        "headroom_bits_per_decade": summary.headroom_bits_per_decade,
        "is_saturated": summary.is_saturated,
        "endpoint_valid": (
            None if summary.endpoint_model_bits is None else summary.endpoint_model_bits >= 0.0
        ),
        "negativity_mechanism": None if diagnosis is None else str(diagnosis.mechanism),
        "loss_curve_shape": None if diagnosis is None else str(diagnosis.shape),
    }


def results_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Tidy one row per cell, sorted for reading rather than for machines."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    return frame.sort_values(["tokenizer", "dataset"]).reset_index(drop=True)


def comparability_warnings(frame: pd.DataFrame) -> list[str]:
    """Conditions that make a cross-cell comparison unsound, stated plainly.

    Returned as text so a report can print them next to its own table. The
    intent is that no figure in the paper is produced without this list being
    empty or its contents quoted in the caption. Operates directly on the
    table :func:`results_frame` returns -- every field a warning needs is
    already a column, so nothing here reaches back into a sweep's facts.

    Deliberately does not warn about facts that are true of every multi-
    tokenizer table by construction rather than about this particular set of
    measurements: a fixed token budget buys a different amount of audio under
    every tokenizer (documented in :func:`audio_hours_seen`'s own docstring,
    and visible directly in the ``train_audio_hours``/``tokenizer`` columns),
    tokenizers' uniform-code bitrates always differ from each other, and --
    since the token budget, not duration, is the fixed comparison axis (team
    decision, 2026-08-25) -- different datasets naturally have different
    native inference-set sizes, which is not a defect either. A warning that
    fires on every cross-tokenizer or cross-dataset table regardless of
    whether anything is actually wrong with it stops distinguishing a real
    problem from the ordinary case, which defeats the "empty list or
    captioned" contract this function exists to support (ethanalwaise-del,
    PR #121: "it seems like either this warning or the next one will always
    show up" / "this will show up whenever we look at measurements from two
    different tokenizers").

    What *is* still worth flagging on the inference-set axis: the same
    dataset's tokenizer variants reporting different inference-set sizes.
    ``prepare_audio.py`` tokenizes one shared, already-split ``AudioDataset``
    once per dataset and loops tokenizers over it (verified directly:
    ``ds = validate_audio(ds_cfg["loader"]())`` runs once, outside the
    ``for tokenizer_name in tokenizer_names`` loop), so every tokenizer's
    ``test.bin`` for a given dataset encodes the same held-out audio -- their
    ``inference_seconds`` should agree once converted back through each
    tokenizer's own rate, and a mismatch means the split or a ``test.bin``
    itself is inconsistent, not that duration needs restandardizing.

    On the training-exposure axis: training exposure that varies for cells
    sharing the same tokenizer. Most sweeps share the standard token budget,
    but not all -- ``sweeps/fsd50k.yaml`` deliberately sets ``T`` to 16x the
    standard value, a real, checked-in exception, not a hypothetical -- so
    that warning states the mismatch and its consequence (epiplexity is
    extensive in dataset size, so ranking across it conflates structure with
    exposure) without asserting a cause it cannot verify from the frame alone.
    """
    warnings: list[str] = []
    if frame.empty:
        return warnings

    inference_seconds = frame[["dataset", "inference_seconds"]].copy()
    inference_seconds["inference_seconds"] = inference_seconds["inference_seconds"].round(6)
    uneven_datasets = sorted(
        str(dataset)
        for dataset, group in inference_seconds.groupby("dataset")["inference_seconds"]
        if group.nunique() > 1
    )
    if uneven_datasets:
        warnings.append(
            f"{len(uneven_datasets)} dataset(s) have cells with different "
            f"inference-set sizes across tokenizer variants ({', '.join(uneven_datasets)}); "
            "every tokenizer's test.bin for one dataset should encode the "
            "same held-out audio, so this means the split or a test.bin "
            "itself is inconsistent, not that duration needs restandardizing"
        )

    hours = frame[["tokenizer", "train_audio_hours"]].copy()
    hours["train_audio_hours"] = hours["train_audio_hours"].round(3)
    uneven_tokenizers = sorted(
        str(tokenizer)
        for tokenizer, group in hours.groupby("tokenizer")["train_audio_hours"]
        if group.nunique() > 1
    )
    if uneven_tokenizers:
        warnings.append(
            f"{len(uneven_tokenizers)} tokenizer(s) have cells with different "
            f"training audio despite sharing a tokenizer ({', '.join(uneven_tokenizers)}); "
            "not explained by tokenizer choice, so check whether the token "
            "budget differed by design (as fsd50k's does) or by accident -- "
            "either way, epiplexity is extensive in dataset size, so ranking "
            "these cells against each other conflates structure with exposure"
        )

    repeating_frame = frame.loc[
        frame["prequential_assumption_holds"].eq(False), ["sweep_name", "repeat_factor"]
    ].sort_values("sweep_name")
    if not repeating_frame.empty:
        # Expected, not a defect: whatever token budget a given sweep used
        # (most share the standard value, some -- fsd50k.yaml -- deliberately
        # don't), a corpus smaller than that budget requires replaying it to
        # reach it. Team decision (2026-08-25) is to acknowledge this in the
        # paper rather than avoid it, which means the exact factor has to be
        # quotable, not just a yes/no flag -- so it's named per cell here,
        # not just counted.
        detail = ", ".join(
            f"{row.sweep_name} ({row.repeat_factor:.1f}x)" for row in repeating_frame.itertuples()
        )
        warnings.append(
            f"{len(repeating_frame)} cell(s) replay training tokens ({detail}); "
            "the prequential estimator codes already-seen tokens there, so its "
            "loss curve reflects memorization -- state the factor in the caption"
        )

    unknown_repeats = sorted(frame.loc[frame["repeat_factor"].isna(), "sweep_name"])
    if unknown_repeats:
        warnings.append(
            f"{len(unknown_repeats)} cell(s) have no recorded clip count, so "
            "whether they replay training tokens is unknown "
            f"({', '.join(unknown_repeats)})"
        )

    unsaturated = sorted(frame.loc[frame["is_saturated"].eq(False), "sweep_name"])
    if unsaturated:
        warnings.append(
            f"{len(unsaturated)} cell(s) had not saturated at the compute "
            f"ceiling ({', '.join(unsaturated)}); their endpoints are lower "
            "bounds, not estimates"
        )

    invalid = sorted(frame.loc[frame["epiplexity_bits"] < 0, "sweep_name"])
    if invalid:
        warnings.append(
            f"{len(invalid)} cell(s) have a negative endpoint epiplexity "
            f"({', '.join(invalid)}); a negative description length is a "
            "measurement failure and must be diagnosed before the cell is ranked"
        )

    return warnings


# --------------------------------------------------------------------------
# CLI: fetch sweeps, build the table, print and optionally save it.
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m epiaudio.analysis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory holding the sweep output bundles",
    )
    parser.add_argument(
        "--sweep-prefix",
        default="",
        help="only analyse sweeps whose name starts with this prefix",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "directory holding tokenized datasets, used to measure a sweep's "
            "original inference-set size from <dataset>/test.bin instead of "
            "supplying it explicitly"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="write the cell table here instead of only printing it",
    )
    parser.add_argument(
        "--keep-negative-points",
        action="store_true",
        help=(
            "include model_bits < 0 points in each cell's frontier instead of "
            "the default of excluding them from it. negative_points/"
            "negative_point_fraction are unaffected either way -- they always "
            "count against the full, unfiltered point cloud"
        ),
    )
    return parser.parse_args(argv)


def _resolve_ambiguous_cells(
    resolved: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[list[str], set[str]]:
    """Sweeps whose (dataset, tokenizer) is claimed by more than one sweep_name.

    ``--sweep-prefix`` matches by sweep_name, and sweep_name is not guaranteed
    unique per (dataset, tokenizer): a sweep re-run under an unchanged name --
    ``birdset_hsn.yaml`` and ``fsd50k.yaml`` carry no run id, unlike the
    date-stamped ``syntheory_intervals_4tokenizer_..._20260807`` -- leaves old
    and new outputs sitting side by side in --output-dir, and both match the
    same prefix. Picking one arbitrarily, or including both as if they were
    different cells, would make the table look complete when it silently used
    a stale or duplicated run. Named here so the caller skips every sweep_name
    in the collision rather than guessing which one is current
    (ethanalwaise-del, PR #121: "this forces you to supply a 1-1 mapping of
    sweep-prefixes to sweeps... I don't think the current sweep nomenclature
    allows for this without pulling from all sweeps using a particular
    dataset [including old ones we don't care about]").
    """
    sweep_names_by_cell: dict[tuple[str, str], list[str]] = {}
    for _, sweep_facts in resolved:
        cell = (str(sweep_facts["dataset"]), str(sweep_facts["tokenizer"]))
        sweep_names_by_cell.setdefault(cell, []).append(str(sweep_facts["sweep_name"]))

    messages: list[str] = []
    ambiguous_names: set[str] = set()
    for (dataset, tokenizer), sweep_names in sorted(sweep_names_by_cell.items()):
        if len(sweep_names) <= 1:
            continue
        ambiguous_names.update(sweep_names)
        messages.append(
            f"{dataset}/{tokenizer}: {len(sweep_names)} sweeps under this "
            f"--sweep-prefix all resolve to this cell ({', '.join(sorted(sweep_names))}); "
            "picking one would silently choose a possibly-stale or duplicated "
            "run, so all of them are skipped -- narrow --sweep-prefix (or "
            "--output-dir) to select a single run per cell"
        )
    return messages, ambiguous_names


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    sweep_config_paths = _discover_sweep_configs(args.output_dir, args.sweep_prefix)
    if not sweep_config_paths:
        print(f"[analysis] no sweep outputs found under {args.output_dir}")
        return 1

    provenance_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    resolved: list[tuple[str, dict[str, Any]]] = []
    for sweep_config_path in sweep_config_paths:
        sweep_name = _infer_sweep_name(sweep_config_path)
        try:
            sweep_facts = load_sweep_facts(sweep_config_path, data_root=args.data_root)
        except (MissingProvenanceError, ValueError, FileNotFoundError) as exc:
            skipped.append(f"{sweep_name}: {exc}")
            continue
        resolved.append((sweep_name, sweep_facts))

    ambiguous_messages, ambiguous_names = _resolve_ambiguous_cells(resolved)
    skipped.extend(ambiguous_messages)

    for sweep_name, sweep_facts in resolved:
        if sweep_facts["sweep_name"] in ambiguous_names:
            continue

        # Each cell reported at its own recorded size: the token budget is
        # the fixed comparison axis (team decision, 2026-08-25), not duration,
        # so no target is ever passed to restandardize a cell's data term.
        inference_seconds = units.tokens_to_seconds(sweep_facts["inference_tokens"], sweep_facts["tokenizer"])
        try:
            points = load_points(sweep_name, sweep_facts)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(f"{sweep_name}: {exc}")
            continue

        provenance_rows.append(build_provenance_row(sweep_facts))
        result_rows.append(
            build_row(
                sweep_facts,
                points,
                inference_seconds=inference_seconds,
                pos_only=not args.keep_negative_points,
                sweep_name=sweep_name,
                output_dir=args.output_dir,
            )
        )

    print(
        f"\n[analysis] cells reported at each cell's own inference-set size "
        f"({len(result_rows)} analysed, {len(skipped)} skipped)"
    )

    frame = results_frame(result_rows)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        if provenance_rows:
            print("\n=== cell provenance ===")
            print(pd.DataFrame(provenance_rows).to_string(index=False))
        if not frame.empty:
            print("\n=== native results ===")
            print(frame.to_string(index=False))

    warnings = comparability_warnings(frame)
    if warnings:
        print("\n=== comparability warnings ===")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n[analysis] no comparability warnings")

    if skipped:
        print("\n=== skipped sweeps ===")
        for note in skipped:
            print(f"  - {note}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\n[analysis] wrote {args.csv}")

    return 0

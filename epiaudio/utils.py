import bisect
import os
from typing import Any, cast

import comet_ml
import numpy as np
import pandas as pd
from comet_ml.query import Parameter
from dotenv import load_dotenv


RUN_INDEX_COLUMN = "run_idx"
COMPUTE_COLUMN = "Compute"
MODEL_BITS_COLUMN = "Model Description Length: K(M)"
DATA_BITS_COLUMN = "Data Given Model: K(X|M)"


def _get_experiments(
        sweep_name: str, 
        comet_workspace: str="epi-audio"
) -> list[comet_ml.APIExperiment]:
    load_dotenv()
    api_key = os.environ.get("COMET_ML_API")
    if not api_key:
        raise RuntimeError("COMET_ML_API is not set.")
    api = comet_ml.API(api_key=api_key)
    query = Parameter("sweep_name") == sweep_name
    experiments = cast(list, api.query(comet_workspace, "epiaudio", query))

    return [comet_ml.APIExperiment(
        previous_experiment=experiment.key,
        api_key=api_key,
    ) for experiment in experiments]


def nearest_value(
    sorted_ts_values: list[tuple[int, float]], ts: int, tolerance_ms: int = 10
) -> float | None:
    """Value whose timestamp is closest to `ts`, or None if none is within `tolerance_ms`.

    Comet's client stamps each key within a single `log_metrics(dict, ...)`
    call with a few ms of independent per-key jitter, so different metric
    names logged in the very same call can end up 1-2ms apart. An exact
    timestamp match silently drops ~10-15% of real eval points as a
    result; real eval events are hundreds+ ms apart, so a small tolerance
    recovers those without risking pairing across two different events.
    """
    if not sorted_ts_values:
        return None
    times = [t for t, _ in sorted_ts_values]
    idx = bisect.bisect_left(times, ts)
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(times)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(times[i] - ts))
    if abs(times[best] - ts) > tolerance_ms:
        return None
    return sorted_ts_values[best][1]


def timestamped_values(records: list[dict], scale: float = 1.0) -> list[tuple[int, float]]:
    """(timestamp, metricValue * scale) pairs from Comet records, sorted by timestamp."""
    return sorted(
        (int(r["timestamp"]), float(r["metricValue"]) * scale)
        for r in records
        if r.get("timestamp") is not None
    )


def get_ts_metrics(experiment, metric, scale: float = 1.0) -> list[tuple[int, float]]:
    records = experiment.get_metrics(metric)
    records_by_ts = timestamped_values(records, scale=scale)

    return records_by_ts


def get_parameter(experiment, name: str, must_exist: bool=True) -> str | None:
    """Current value of a Comet-logged parameter, or None if never logged."""
    summary = experiment.get_parameters_summary(name)
    if summary is None:
        raise RuntimeError(f"({experiment.key}) has no parameter '{name}'")
    return summary["valueCurrent"] if summary else None


def load_run_comet_provenance(sweep_name: str) -> dict[int, dict[str, Any]] | None:
    """Per-run Comet identity, keyed by run_idx."""
    experiments = _get_experiments(sweep_name)
    data = dict()

    for run_idx, experiment in enumerate(experiments):
        experiment_key = experiment.key
        train_student = get_parameter(experiment, "train_student") == "True"
        teacher_ema_active = get_parameter(experiment, "teacher_ema_active") == "True"
        student_ema_active = get_parameter(experiment, "student_ema_active") == "True"

        data[run_idx] = {
            "experiment_key": str(experiment_key),
            "train_student": train_student,
            "teacher_ema_active": teacher_ema_active,
            "student_ema_active": student_ema_active
        }

    return data


def fetch_sweep_records(
    sweep_name: str,
    *,
    test_tokens: int,
    num_params_param: str="num_params",
    train_tokens_metric: str="teacher_tokens",
    km_metric: str="K(M)",
    ema_loss_metric: str="ema_teacher_eval_loss"
) -> pd.DataFrame:
    """Pulls the compute, model description length and entropy code length
    records from CometML for runs from a given sweep. Returns a data frame
    giving these metrics and an associated run id at every evaluation step.
    """
    experiments = _get_experiments(sweep_name)
    data = []

    for run_idx, experiment in enumerate(experiments):
        num_params_raw = get_parameter(experiment, num_params_param)
        num_params = float(cast(str, num_params_raw))
        train_tokens_by_ts = get_ts_metrics(experiment, train_tokens_metric) or []
        compute_by_ts = get_ts_metrics(experiment, "compute") or []
        km_by_ts = get_ts_metrics(experiment, km_metric, scale=1e6) or []
        ema_loss_records = experiment.get_metrics(ema_loss_metric) or []

        # Align by `timestamp`, not `step`: step is `student_step`, which never
        # advances when train_student is False, so every eval event in the run
        # would otherwise collide on the same step (see GridSweeper._fetch_epiplexity).
        for r in ema_loss_records:
            ts = r.get("timestamp")
            if ts is None:
                continue
            ts = int(ts)
            train_tokens = nearest_value(train_tokens_by_ts, ts)
            train_compute = nearest_value(compute_by_ts, ts)
            model_bits = nearest_value(km_by_ts, ts)
            if any(val is None for val in (num_params, train_tokens, train_compute, model_bits)):
                continue

            compute: float = cast(float, train_compute) + 2*num_params*test_tokens
            data_bits = float(r["metricValue"]) * test_tokens / np.log(2)
            data.append((
                run_idx,
                compute,
                model_bits,
                data_bits
            ))

    return pd.DataFrame(data, columns= [
        RUN_INDEX_COLUMN,
        COMPUTE_COLUMN,
        MODEL_BITS_COLUMN,
        DATA_BITS_COLUMN
    ])

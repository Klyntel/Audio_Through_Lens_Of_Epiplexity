import os
from typing import cast

from dotenv import load_dotenv

import numpy as np
from scipy.stats import linregress
import comet_ml
from comet_ml.query import Parameter
from comet_ml.api import APIExperiment

from epiaudio.utils import nearest_value, get_ts_metrics, get_parameter
from epiaudio.analysis.frontier import CodeLengthPoint


COMET_WORKSPACE = "epi-audio"
TRAIN_TOKENS_METRIC = "teacher_tokens"
COMPUTE_METRIC = "compute"
RAW_LOSS_METRIC = "train_loss"
EMA_LOSS_METRIC = "ema_teacher_eval_loss"


def _model_bits_from_loss_curve(loss_curve) -> list[float]:
    loss_curve = sorted(loss_curve, key=lambda p: p[0])
    tokens, losses = zip(*loss_curve)
    n_points = len(tokens)
    model_bits = []
    for n in range(1, n_points):
        area_under_curve = np.trapezoid(losses[:n + 1], x=tokens[:n + 1]) 
        area_under_final_loss = (tokens[n] - tokens[0])*losses[n]
        bits = (area_under_curve - area_under_final_loss) / np.log(2)
        model_bits.append(bits)

    return model_bits


def _smooth_loss_curve(loss_curve) -> list[tuple[float, float]]:
    loss_curve = sorted(loss_curve, key=lambda p: p[0])
    shift = min(np.log(p[0]) for p in loss_curve) - 1
    tokens = np.array([np.log(np.log(p[0]) - shift) for p in loss_curve])
    losses = np.array([np.log(p[1]) for p in loss_curve])

    fit = cast(tuple[float, float, float, float, float], linregress(tokens, losses))
    slope, intercept, *_ = fit

    def smooth_loss(tokens: float, intercept: float, slope: float) -> float:
        x = np.log(np.log(tokens) - shift)
        return np.exp(intercept + slope * x)

    return [
        (tokens, smooth_loss(tokens, intercept, slope))
        for tokens, _ in loss_curve
    ]


def _smooth_run_points(
    experiment: APIExperiment,
    idx: int,
    test_tokens: int
) -> list[CodeLengthPoint]:
    num_params_raw = get_parameter(experiment, "num_params")
    num_params = float(cast(str, num_params_raw))

    train_tokens_records = get_ts_metrics(experiment, TRAIN_TOKENS_METRIC)
    compute_records = get_ts_metrics(experiment, COMPUTE_METRIC)
    raw_loss_records = get_ts_metrics(experiment, RAW_LOSS_METRIC)
    ema_loss_records = experiment.get_metrics(EMA_LOSS_METRIC)

    loss_curve = []
    entropy_curve = []
    for r in ema_loss_records:
        ts = r.get("timestamp")
        if ts is None:
            continue
        ts = int(ts)
        train_tokens = nearest_value(train_tokens_records, ts)
        train_compute = nearest_value(compute_records, ts)
        compute: float = cast(float, train_compute) + 2*num_params*test_tokens
        raw_loss = nearest_value(raw_loss_records, ts)
        ema_loss = float(r["metricValue"])
        if any(val is None for val in (compute, raw_loss, ema_loss)):
            continue
        entropy_bits = ema_loss * test_tokens / np.log(2)
        entropy_curve.append((compute, entropy_bits))
        loss_curve.append((train_tokens, raw_loss))

    smoothed_loss_curve = _smooth_loss_curve(loss_curve)
    model_bit_counts = _model_bits_from_loss_curve(smoothed_loss_curve)

    return [
        CodeLengthPoint(
            compute_flops=compute,
            model_bits=model_bits,
            data_bits=entropy_bits,
            run_index=idx
        )
        for (compute, entropy_bits), model_bits in zip(entropy_curve[1:], model_bit_counts)
    ]


def get_smooth_code_points(
    sweep_name: str,
    test_tokens: int
) -> list[CodeLengthPoint]:
    load_dotenv()
    
    if not os.environ.get("COMET_ML_API"):
        raise RuntimeError("Comet ML API key missing")

    api_key = os.environ["COMET_ML_API"]
    api = comet_ml.API(api_key=api_key)
    query = Parameter("sweep_name") == sweep_name
    experiments = cast(list, api.query(COMET_WORKSPACE, "epiaudio", query))

    all_points = []
    for run_idx, experiment in enumerate(experiments):
        api_experiment = comet_ml.APIExperiment(
            previous_experiment=experiment.key,
            api_key=api_key,
        )
        code_points = _smooth_run_points(api_experiment, run_idx, test_tokens)
        all_points.extend(code_points)

    return all_points

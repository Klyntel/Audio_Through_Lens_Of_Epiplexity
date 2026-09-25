"""Fit L(N,D) from a completed prerequential sweep and derive paper-style S_T."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comet_ml import APIExperiment
from comet_ml.query import Parameter
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epiaudio.sweep import COMET_WORKSPACE
from epiaudio.utils import nearest_value, timestamped_values


_LEGACY_TOKENIZERS = {
    "ascii96": (96, 512),
    "puzzles": (64, 512),
    "grayscale": (256, 1024),
}
_TRAIN_LOSS_METRIC = "train_loss"


def _parameters(experiment: object) -> dict[str, str]:
    return {
        str(x["name"]): str(x["valueCurrent"])
        for x in experiment.get_parameters_summary()
    }  # type: ignore[attr-defined]


def _num_params(params: dict[str, str]) -> float | None:
    """Use logged counts, or reconstruct historical runs from their model grid."""
    if "num_params" in params:
        return float(params["num_params"])
    try:
        depth = int(params["model.N"])
        budget = float(params["model.P"])
        tokenizer = params.get("tokenizer", "")
        vocab, sequence = _LEGACY_TOKENIZERS[tokenizer]
        width = round(((budget * 1e6 / depth / 12) ** 0.5) / 64) * 64
        return float(12 * depth * width**2 + (2 * vocab + sequence) * width)
    except (KeyError, ValueError):
        return None


def _sweep_experiments(api: object, name: str) -> list[object]:
    """Fetch exactly the runs that logged this sweep name, including old sweeps."""
    return (
        api.query(  # type: ignore[attr-defined]
            COMET_WORKSPACE, "epiaudio", Parameter("sweep_name") == name
        )
        or []
    )


def _samples(
    key: str, num_params: float, targets: list[float]
) -> list[tuple[float, float, float]]:
    """Sample the logged training-loss curve at token targets."""
    experiment = APIExperiment(
        previous_experiment=key, api_key=os.environ["COMET_ML_API"]
    )
    tokens = timestamped_values(experiment.get_metrics("teacher_tokens") or [])
    curve = [
        (float(token), float(row["metricValue"]))
        for row in experiment.get_metrics(_TRAIN_LOSS_METRIC) or []
        if row.get("timestamp") is not None
        if (token := nearest_value(tokens, int(row["timestamp"]))) is not None
    ]
    return (
        [
            (num_params, *min(curve, key=lambda x: abs(np.log(x[0] / target))))
            for target in targets
        ]
        if curve
        else []
    )


def _fit(rows: np.ndarray, restarts: int) -> tuple[np.ndarray, float]:
    n, d, loss = np.log(rows[:, 0]), np.log(rows[:, 1]), np.log(rows[:, 2])

    def objective(x: np.ndarray) -> float:
        a, log_alpha, b, logit_beta, e = x
        alpha, beta = np.exp(log_alpha), expit(logit_beta)
        residual = (
            np.logaddexp.reduce([a - alpha * n, b - beta * d, np.full_like(n, e)])
            - loss
        )
        delta = 1e-3
        return float(
            np.where(
                abs(residual) <= delta,
                residual**2 / 2,
                delta * (abs(residual) - delta / 2),
            ).sum()
        )

    rng = np.random.default_rng(0)
    start = np.array([0.0, np.log(0.5), 0.0, 0.0, np.log(rows[:, 2].min() / 2)])
    starts = [start]
    starts.extend(rng.normal(start, 2.0, size=(restarts - 1, 5)))
    result = min(
        (
            minimize(
                objective,
                x,
                method="L-BFGS-B",
                options={"maxiter": 10_000},
            )
            for x in starts
        ),
        key=lambda x: x.fun,
    )
    if not result.success:
        raise RuntimeError(f"scaling-law fit failed: {result.message}")
    a, log_alpha, b, logit_beta, e = result.x
    return np.array([a, np.exp(log_alpha), b, expit(logit_beta), e]), float(result.fun)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit L=E+A/N^alpha+B/D^beta from Comet."
    )
    parser.add_argument("sweep_name")
    parser.add_argument("--tokens", default="1000000,3000000,10000000,31250000")
    parser.add_argument("--restarts", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scaling_laws"))
    args = parser.parse_args(argv)
    targets = [float(x) for x in args.tokens.split(",")]
    load_dotenv(Path.cwd() / ".env")
    if not os.environ.get("COMET_ML_API"):
        raise RuntimeError(
            "COMET_ML_API is required to read completed sweep histories."
        )
    import comet_ml

    # These rows are the complete empirical input to the scaling-law model.
    rows: list[tuple[float, float, float]] = []
    for summary in _sweep_experiments(
        comet_ml.API(api_key=os.environ["COMET_ML_API"]), args.sweep_name
    ):
        params = _parameters(summary)
        num_params = _num_params(params)
        if params.get("train_student", "false").lower() == "false" and num_params:
            rows += _samples(summary.key, num_params, targets)
    data = np.asarray(rows)
    if len(data) < 6 or min(len(np.unique(data[:, i])) for i in (0, 1)) < 2:
        raise RuntimeError(
            "need two model sizes, two token budgets, and six loss observations"
        )
    x, huber = _fit(data, args.restarts)
    log_a, alpha, log_b, beta, log_e = x
    a, b, e = np.exp([log_a, log_b, log_e])
    n0, d0 = a ** (1 / alpha), b ** (1 / beta)
    st = {
        str(int(d)): float(beta / (1 - beta) * d0**beta * d ** (1 - beta) / np.log(2))
        for d in targets
    }
    predicted = np.exp(
        np.logaddexp.reduce(
            [
                log_a - alpha * np.log(data[:, 0]),
                log_b - beta * np.log(data[:, 1]),
                np.full(len(data), log_e),
            ]
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "sweep_name": args.sweep_name,
        "loss_metric": _TRAIN_LOSS_METRIC,
        "observations": len(data),
        "huber_loss": huber,
        "E": float(e),
        "A": float(a),
        "B": float(b),
        "alpha": float(alpha),
        "beta": float(beta),
        "N0": float(n0),
        "D0": float(d0),
        "S_T_bits": st,
    }
    (args.output_dir / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.loglog(data[:, 2], predicted, "o", alpha=0.7)
    bounds = [
        min(data[:, 2].min(), predicted.min()),
        max(data[:, 2].max(), predicted.max()),
    ]
    plt.loglog(bounds, bounds, "k--")
    plt.xlabel("training loss")
    plt.ylabel("fitted loss")
    plt.tight_layout()
    plt.savefig(args.output_dir / "fit.png", dpi=180)
    plt.close()
    print(json.dumps({"D0": d0, "beta": beta, "S_T_bits": st}, indent=2))


if __name__ == "__main__":
    main()

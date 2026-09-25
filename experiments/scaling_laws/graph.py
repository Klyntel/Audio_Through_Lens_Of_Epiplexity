"""Plot paper-style scaling frontiers from completed scaling-law fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
def structural_bits(tokens: float, beta: float, d0: float) -> float:
    return beta / (1 - beta) * d0**beta * tokens ** (1 - beta) / np.log(2)


def optimal_tokens(compute: float, alpha: float, beta: float, n0: float, d0: float, data_tokens: float) -> float:
    def derivative(tokens: float) -> float:
        left = d0**beta * beta * (data_tokens - tokens) * tokens ** (-beta - 1)
        right = 3 * alpha * data_tokens * n0**alpha * 2**alpha * compute**(-alpha) * (3 * tokens + data_tokens) ** (alpha - 1)
        return left - right

    high = data_tokens * 0.999999
    return high if derivative(high) > 0 else brentq(derivative, 1.0, high, maxiter=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style S_T and optimal-token scaling curves.")
    parser.add_argument("fits", nargs="+", type=Path, help="fit.json files from fit.py")
    parser.add_argument("--data-tokens", type=float, default=1e12, help="Coded dataset size D in the paper's graph.")
    parser.add_argument("--min-compute", type=float, default=1e15)
    parser.add_argument("--max-compute", type=float, default=1e30)
    parser.add_argument("--output", type=Path, default=Path("outputs/scaling_laws/fsd50k_frontiers.png"))
    args = parser.parse_args()
    compute = np.geomspace(args.min_compute, args.max_compute, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for fit_path in args.fits:
        fit = json.loads(fit_path.read_text())
        name = fit_path.parent.name
        label = "OpenWebText" if "openwebtext" in name else name.removeprefix("fsd50k_").replace("_", " ").title()
        points = []
        for value in compute:
            try:
                tokens = optimal_tokens(value, fit["alpha"], fit["beta"], fit["N0"], fit["D0"], args.data_tokens)
                points.append((value, tokens))
            except ValueError:
                continue
        values, tokens = np.asarray(points).T
        line = axes[0].plot(values, [structural_bits(t, fit["beta"], fit["D0"]) for t in tokens], label=label)[0]
        axes[0].axhline(structural_bits(args.data_tokens, fit["beta"], fit["D0"]), color=line.get_color(), ls="--", alpha=0.5)
        axes[1].plot(values, tokens, color=line.get_color())
        axes[1].axhline(args.data_tokens, color=line.get_color(), ls="--", alpha=0.5)
    axes[0].set(ylabel=r"$S_T(X)$ (bits)", xlabel="Compute (FLOPs)")
    axes[1].set(ylabel="Optimal train tokens", xlabel="Compute (FLOPs)")
    for axis in axes:
        axis.set(xscale="log", yscale="log")
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
"""Run the one-GPU fallback sweep, then fit its completed loss surface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epiaudio.sweep import REPO_ROOT, _build_sweeper, load_sweep_configs
from epiaudio.sweep_preflight import print_preflight, validate_sweep
from experiments.scaling_laws import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="One-GPU fallback scaling-law sweep.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("fsd50k_encodec.yaml")
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for cfg in load_sweep_configs(str(args.config)):
        method = str(cfg.get("method", "grid"))
        check = validate_sweep(
            cfg,
            str(args.config),
            method=method,
            override_args=args.override,
            repo_root=REPO_ROOT,
        )
        print_preflight(check)
        if not check.ready:
            raise SystemExit(1)
        if args.preflight_only:
            continue
        sweep = _build_sweeper(
            cfg, method, args.override, backend="torch", sweep_seed=None
        )
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}; sweep={sweep.sweep_name}")
        sweep.sweep()
        fit.main(
            [
                sweep.sweep_name,
                "--output-dir",
                f"outputs/scaling_laws/{sweep.sweep_name}",
            ]
        )


if __name__ == "__main__":
    main()

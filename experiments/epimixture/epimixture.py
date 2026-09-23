"""Command-line entry point for the modular EpiMixture experiment package.

The implementation lives in focused modules: ``config`` validates designs,
``data`` prepares token data and mixtures, ``sweeps`` runs/evaluates models,
and ``runner`` coordinates the resumable lifecycle.
"""

from __future__ import annotations

import argparse
import json

from .config import ExperimentSpec, MixtureSpec, load_spec
from .data import (
    _write_mixture_split,
    allocate_samples,
    build_mixtures,
    dataset_path as _dataset_path,
    list_registry,
    plan,
    prepare_sources,
)
from .runner import run_experiment
from .sweeps import (
    _path_sha256,
    analyze,
    generate_sweeps,
    reuse_comet_sweeps,
    run_fixed_generalization,
    run_sweeps,
    run_transfer,
    selections_complete as _selections_complete,
    transfer_complete as _transfer_complete,
)

__all__ = [
    "ExperimentSpec", "MixtureSpec", "_dataset_path", "_path_sha256", "_selections_complete",
    "_transfer_complete", "_write_mixture_split", "allocate_samples", "analyze", "build_mixtures",
    "generate_sweeps", "list_registry", "load_spec", "plan", "prepare_sources", "reuse_comet_sweeps", "run_experiment",
    "run_fixed_generalization", "run_sweeps", "run_transfer",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the EpiMixture experiment.")
    parser.add_argument("config", help="Path to an EpiMixture YAML design.")
    parser.add_argument(
        "command",
        choices=["list-registry", "run", "plan", "prepare", "build-mixtures", "generate-sweeps", "run-sweeps", "transfer", "generalize", "analyze"],
        help="Use 'run' for the complete experiment; individual stages are for recovery.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace generated data/configs only after explicit opt-in.")
    parser.add_argument(
        "--reuse-comet-sweeps",
        action="store_true",
        help="Read complete historical PF-grid results from Comet instead of starting sweep runs.",
    )
    args = parser.parse_args()
    if args.command == "list-registry":
        print(json.dumps(list_registry(), indent=2))
        return
    spec = load_spec(args.config)
    if args.command == "run":
        print(
            run_experiment(
                spec,
                overwrite=args.overwrite,
                reuse_comet_sweeps=args.reuse_comet_sweeps,
            )
        )
    elif args.command == "plan":
        print(plan(spec))
    elif args.command == "prepare":
        prepare_sources(spec, overwrite=args.overwrite)
    elif args.command == "build-mixtures":
        build_mixtures(spec, overwrite=args.overwrite)
    elif args.command == "generate-sweeps":
        print("\n".join(str(path) for path in generate_sweeps(spec, overwrite=args.overwrite)))
    elif args.command == "run-sweeps":
        if args.reuse_comet_sweeps:
            reuse_comet_sweeps(spec)
        else:
            run_sweeps(spec)
    elif args.command in {"transfer", "generalize"}:
        run_fixed_generalization(spec)
    else:
        print(analyze(spec))


if __name__ == "__main__":
    main()

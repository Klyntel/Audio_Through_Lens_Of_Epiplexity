"""The resumable, single-command EpiMixture lifecycle."""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentSpec
from .data import build_mixtures, plan, prepare_sources
from .sweeps import (
    analyze,
    generate_sweeps,
    reuse_comet_sweeps as load_comet_sweeps,
    run_sweeps,
    run_transfer,
    selections_complete,
    transfer_complete,
)


def run_experiment(
    spec: ExperimentSpec,
    *,
    overwrite: bool = False,
    reuse_comet_sweeps: bool = False,
) -> Path:
    """Run the complete EpiMixture lifecycle and return its report."""
    plan(spec)
    prepare_sources(spec, overwrite=overwrite)
    build_mixtures(spec, overwrite=overwrite)
    generate_sweeps(spec, overwrite=overwrite)
    if overwrite or not selections_complete(spec):
        if reuse_comet_sweeps:
            load_comet_sweeps(spec)
        else:
            run_sweeps(spec)
    else:
        print("[epimixture] keeping completed sweep selections")
    if overwrite or not transfer_complete(spec):
        run_transfer(spec)
    else:
        print("[epimixture] keeping completed transfer evaluations")
    return analyze(spec)

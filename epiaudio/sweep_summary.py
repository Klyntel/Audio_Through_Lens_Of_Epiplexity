"""Persist and publish aggregate Pareto-front sweep results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from epiaudio.utils import (
    COMPUTE_COLUMN,
    MODEL_BITS_COLUMN,
    DATA_BITS_COLUMN
)


TOTAL_BITS_COLUMN = "Two-Part Code Length: K(M,X) = K(M) + K(X|M)"
PARETO_SUFFIX = "__pareto_front_data.csv"
ALL_RUN_SUFFIX = "__all_run_data.csv"
PLOT_SUFFIX = "_pareto_front.png"
CONFIG_SUFFIX = "__sweep_config.yaml"
SUMMARY_SUFFIX = "__summary.json"
SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweepOutputBundle:
    sweep_name: str
    pareto_csv: Path
    all_run_csv: Path | None = None
    plot: Path | None = None
    sweep_config: Path | None = None
    summary_json: Path | None = None

    def existing_assets(self) -> list[Path]:
        return [
            path
            for path in (
                self.all_run_csv,
                self.pareto_csv,
                self.sweep_config,
                self.summary_json,
            )
            if path is not None and path.is_file()
        ]


def bundle_for_sweep(output_dir: str | Path, sweep_name: str) -> SweepOutputBundle:
    output_path = Path(output_dir)

    def optional_path(suffix: str) -> Path | None:
        path = output_path / f"{sweep_name}{suffix}"
        return path if path.is_file() else None

    pareto_csv = output_path / f"{sweep_name}{PARETO_SUFFIX}"
    return SweepOutputBundle(
        sweep_name=sweep_name,
        pareto_csv=pareto_csv,
        all_run_csv=optional_path(ALL_RUN_SUFFIX),
        plot=optional_path(PLOT_SUFFIX),
        sweep_config=optional_path(CONFIG_SUFFIX),
        summary_json=optional_path(SUMMARY_SUFFIX),
    )


def discover_sweep_outputs(
    output_dir: str | Path,
    sweep_prefix: str,
) -> list[SweepOutputBundle]:
    """Return every Pareto output bundle whose sweep name starts with prefix."""
    output_path = Path(output_dir)
    bundles = []
    for pareto_csv in sorted(output_path.glob(f"{sweep_prefix}*{PARETO_SUFFIX}")):
        sweep_name = pareto_csv.name.removesuffix(PARETO_SUFFIX)
        bundles.append(bundle_for_sweep(output_path, sweep_name))
    return bundles


def summarize_pareto_csv(
    pareto_csv: str | Path,
    *,
    sweep_name: str,
) -> dict[str, Any]:
    """Read the maximum-compute Pareto point, matching PFGridSweeper._report."""
    path = Path(pareto_csv)
    frame = pd.read_csv(path)
    required = {
        COMPUTE_COLUMN,
        MODEL_BITS_COLUMN,
        DATA_BITS_COLUMN,
        TOTAL_BITS_COLUMN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no Pareto-front rows")

    endpoint = frame.sort_values(COMPUTE_COLUMN).iloc[-1]
    model_bits = float(endpoint[MODEL_BITS_COLUMN])
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "sweep_name": sweep_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "pareto_epiplexity_bits": model_bits,
        "pareto_compute": float(endpoint[COMPUTE_COLUMN]),
        "pareto_data_given_model_bits": float(endpoint[DATA_BITS_COLUMN]),
        "pareto_two_part_code_bits": float(endpoint[TOTAL_BITS_COLUMN]),
        "pareto_frontier_points": int(len(frame)),
        # A negative description length is an estimator/configuration warning,
        # not meaningful negative complexity.
        "epiplexity_valid": model_bits >= 0.0,
    }


def write_summary_json(
    summary: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def upload_sweep_summary(
    bundle: SweepOutputBundle,
    *,
    project_name: str,
    workspace: str,
    api_key: str | None = None,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    experiment_factory: Callable[..., Any] | None = None,
) -> str | None:
    """Upload one aggregate sweep result as a dedicated Comet experiment."""
    resolved_api_key = api_key or os.environ.get("COMET_ML_API")
    if not resolved_api_key:
        raise RuntimeError("COMET_ML_API is not set")
    if not bundle.pareto_csv.is_file():
        raise FileNotFoundError(bundle.pareto_csv)

    summary = summarize_pareto_csv(
        bundle.pareto_csv,
        sweep_name=bundle.sweep_name,
    )
    summary_path = bundle.summary_json or (
        bundle.pareto_csv.parent / f"{bundle.sweep_name}{SUMMARY_SUFFIX}"
    )
    write_summary_json(summary, summary_path)
    bundle = SweepOutputBundle(
        sweep_name=bundle.sweep_name,
        pareto_csv=bundle.pareto_csv,
        all_run_csv=bundle.all_run_csv,
        plot=bundle.plot,
        sweep_config=bundle.sweep_config,
        summary_json=summary_path,
    )

    if experiment_factory is None:
        from comet_ml import Experiment

        experiment_factory = Experiment

    experiment = experiment_factory(
        api_key=resolved_api_key,
        project_name=project_name,
        workspace=workspace,
    )
    try:
        experiment.set_name(f"{bundle.sweep_name}__summary")
        experiment.add_tag("sweep-summary")
        experiment.add_tag(bundle.sweep_name)
        for tag in tags or []:
            experiment.add_tag(tag)

        logged_parameters = {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "source_sweep_name": bundle.sweep_name,
            **(parameters or {}),
        }
        experiment.log_parameters(logged_parameters)
        experiment.log_metrics(
            {
                key: value
                for key, value in summary.items()
                if key.startswith("pareto_") and isinstance(value, (int, float))
            }
            | {"epiplexity_valid": int(summary["epiplexity_valid"])}
        )

        for asset in bundle.existing_assets():
            experiment.log_asset(
                file_data=str(asset),
                file_name=asset.name,
            )
        if bundle.plot is not None and bundle.plot.is_file():
            experiment.log_image(
                image_data=str(bundle.plot),
                name=bundle.plot.stem,
            )

        get_key = getattr(experiment, "get_key", None)
        return str(get_key()) if callable(get_key) else None
    finally:
        experiment.end()

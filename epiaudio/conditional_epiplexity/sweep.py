"""Deterministic grid sweeps for conditional epiplexity experiments."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig, OmegaConf

import comet_ml
from comet_ml.exceptions import NotFound
from comet_ml.query import Parameter

from epiaudio.conditional_epiplexity.curve import extract_reported_points
from epiaudio.downstream.teacher_experiment import (
    COMET_API_KEY_VAR,
    COMET_PROJECT,
    EPI_AUDIO_WORKSPACE,
    dataset_and_tokenizer,
    log_experiment,
    log_sweep_summary,
    run_experiment,
    teacher_experiment_parameters,
)
from epiaudio.downstream.teacher_training import TeacherTrainingConfig


MetricValue = float | int
TrainFunction = Callable[[DictConfig], dict[str, MetricValue]]
_MANAGED_PARAMETERS = {"checkpoint_path", "name", "resume_from"}
_PARAMETER_BUDGET = "parameter_budget_millions"
_EVAL_TOKEN_BUDGET = "eval_tokens_per_batch"
_VALID_PARAMETERS = {
    *(field.name for field in fields(TeacherTrainingConfig)),
    _PARAMETER_BUDGET,
    _EVAL_TOKEN_BUDGET,
}
_COMET_PARAM_EXCLUDE = {"checkpoint_path", "name", "resume_from", "sweep_name"}
SUMMARY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ConditionalSweepRun:
    """The configuration and metrics produced by one sweep point."""

    name: str
    parameters: dict[str, Any]
    checkpoint_path: str
    metrics: dict[str, MetricValue]
    source: str
    experiment_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "checkpoint_path": self.checkpoint_path,
            "metrics": self.metrics,
            "source": self.source,
            "experiment_key": self.experiment_key,
        }


@dataclass(frozen=True)
class ConditionalSweepPoint:
    """One reported checkpoint from a conditional sweep run."""

    run_index: int
    run_name: str
    report_index: int
    metrics: dict[str, MetricValue]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "run_name": self.run_name,
            "report_index": self.report_index,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ConditionalSweepResult:
    """All sweep runs plus the run selected by the configured metric."""

    sweep_name: str
    metric_name: str
    goal: str
    runs: tuple[ConditionalSweepRun, ...]
    best_run: ConditionalSweepRun
    best_point: ConditionalSweepPoint
    pareto_front: tuple[ConditionalSweepPoint, ...]
    summary_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "sweep_name": self.sweep_name,
            "metric": {"name": self.metric_name, "goal": self.goal},
            "best_run": self.best_run.as_dict(),
            "best_point": self.best_point.as_dict(),
            "pareto_front": [point.as_dict() for point in self.pareto_front],
            "runs": [run.as_dict() for run in self.runs],
        }


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a YAML mapping.")
    return {str(key): item for key, item in value.items()}


def _parameter_values(name: str, specification: Any) -> tuple[Any, ...]:
    spec = _as_mapping(specification, field_name=f"parameters.{name}")
    has_value = "value" in spec
    has_values = "values" in spec
    if has_value == has_values:
        raise ValueError(
            f"parameters.{name} must define exactly one of 'value' or 'values'."
        )
    if has_value:
        return (spec["value"],)

    values = spec["values"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"parameters.{name}.values must be a non-empty list.")
    return tuple(values)


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", text).strip("-_") or "value"


def load_sweep_configs(
    path: str | Path,
    *,
    sweep_name: str | None = None,
) -> tuple[Path, tuple[DictConfig, ...]]:
    """Split a sweep YAML into one configuration per input dataset."""
    config_path = Path(path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError("Sweep config must be a YAML mapping.")

    parameters = _as_mapping(config.get("parameters"), field_name="parameters")
    if "ds_path" not in parameters:
        raise ValueError("parameters.ds_path must be provided.")
    ds_paths = _parameter_values("ds_path", parameters["ds_path"])
    if any(not isinstance(ds_path, str) or not ds_path.strip() for ds_path in ds_paths):
        raise TypeError("Every parameters.ds_path value must be a non-empty path.")

    base_name = str(
        sweep_name if sweep_name is not None else config.get("name", "")
    ).strip()
    if not base_name:
        raise ValueError("Sweep name must be provided.")
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty path.")

    split_configs = []
    suffixes = set()
    for ds_path in ds_paths:
        representation = dataset_and_tokenizer(ds_path)
        suffix = (
            representation[1]
            if representation is not None
            else _slug(Path(ds_path).name)
        )
        if suffix in suffixes:
            raise ValueError(f"Dataset sweep suffix '{suffix}' is not unique.")
        suffixes.add(suffix)

        split_config = copy.deepcopy(config)
        OmegaConf.update(split_config, "name", f"{base_name}_{suffix}", merge=False)
        OmegaConf.update(
            split_config,
            "output_dir",
            str(Path(output_dir) / suffix),
            merge=False,
        )
        OmegaConf.update(
            split_config,
            "parameters.ds_path",
            {"value": ds_path},
            merge=False,
        )
        split_configs.append(split_config)
    return config_path, tuple(split_configs)


class ConditionalGridSweep:
    """Run a Cartesian grid over the existing conditional teacher runtime.

    The sweep does not implement a second estimator or training loop. Every point
    calls ``teacher_experiment.run_experiment``, which returns the conditional
    prequential, evaluation, and two-part-code metrics produced by the trainer.
    """

    def __init__(
        self,
        config: DictConfig,
        *,
        config_path: Path,
        train_fn: TrainFunction | None = None,
        log_to_comet: bool | None = None,
        resume: bool | None = None,
        resume_sweep_name: str | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.sweep_name = str(config.get("name", "")).strip()
        if not self.sweep_name:
            raise ValueError("Sweep name must be provided.")
        if resume_sweep_name is not None:
            if not resume_sweep_name.strip():
                raise ValueError("resume_sweep_name must not be empty.")
            self.sweep_name = resume_sweep_name.strip()

        base_config = config.get("base_config")
        if not isinstance(base_config, str) or not base_config.strip():
            raise ValueError("base_config must be a non-empty path.")
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = self.config_path.parent / base_path
        self.base_config_path = base_path.resolve()
        if not self.base_config_path.is_file():
            raise FileNotFoundError(
                f"Base teacher config not found: {self.base_config_path}"
            )

        output_dir = config.get("output_dir")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty path.")
        self.output_dir = Path(output_dir).resolve()

        metric = _as_mapping(config.get("metric"), field_name="metric")
        self.metric_name = str(metric.get("name", "")).strip()
        if not self.metric_name:
            raise ValueError("metric.name must be provided.")
        self.goal = str(metric.get("goal", "minimize")).strip().lower()
        if self.goal not in {"minimize", "maximize"}:
            raise ValueError("metric.goal must be 'minimize' or 'maximize'.")

        configured_log_to_comet = config.get("log_to_comet", True)
        if not isinstance(configured_log_to_comet, bool):
            raise TypeError("log_to_comet must be true or false.")
        if log_to_comet is not None and not isinstance(log_to_comet, bool):
            raise TypeError("log_to_comet override must be true or false.")
        self.log_to_comet = (
            configured_log_to_comet if log_to_comet is None else log_to_comet
        )

        configured_resume = config.get("resume", True)
        if not isinstance(configured_resume, bool):
            raise TypeError("resume must be true or false.")
        if resume is not None and not isinstance(resume, bool):
            raise TypeError("resume override must be true or false.")
        self.resume = configured_resume if resume is None else resume
        if resume_sweep_name is not None and not self.log_to_comet:
            raise ValueError("Comet logging must be enabled for --resume-sweep.")
        self.resume_sweep_name = self.sweep_name
        self._require_existing_comet_sweep = resume_sweep_name is not None
        self._resume_index: dict[str, dict[str, str]] = {}
        self._comet_api_key: str | None = None
        self._train_fn = train_fn

        retain_checkpoints = config.get("retain_checkpoints", True)
        if not isinstance(retain_checkpoints, bool):
            raise TypeError("retain_checkpoints must be true or false.")
        self.retain_checkpoints = retain_checkpoints
        if not self.retain_checkpoints and not self.log_to_comet:
            raise ValueError(
                "Comet logging is required when retain_checkpoints is false."
            )

        parameters = _as_mapping(config.get("parameters"), field_name="parameters")
        self._parameter_names = tuple(parameters)
        self._parameter_values: dict[str, tuple[Any, ...]] = {}
        for name, specification in parameters.items():
            if name in _MANAGED_PARAMETERS:
                raise ValueError(f"'{name}' is managed by the sweep.")
            if name not in _VALID_PARAMETERS:
                raise ValueError(f"Unknown teacher parameter: {name}")
            self._parameter_values[name] = _parameter_values(name, specification)

        yaml_num_repeats = config.get("num_repeats")
        if yaml_num_repeats is not None:
            if isinstance(yaml_num_repeats, bool) or not isinstance(
                yaml_num_repeats, int
            ):
                raise TypeError("num_repeats must be a positive integer.")
            if yaml_num_repeats <= 0:
                raise ValueError("num_repeats must be a positive integer.")
            sweep_seed = config.get("seed", 0)
            if isinstance(sweep_seed, bool) or not isinstance(sweep_seed, int):
                raise TypeError("seed must be an integer.")
            repeat_seeds = tuple(
                int(child.generate_state(1)[0])
                for child in np.random.SeedSequence(sweep_seed).spawn(
                    yaml_num_repeats
                )
            )
            self._parameter_values["seed"] = repeat_seeds
            if "seed" not in self._parameter_names:
                self._parameter_names = (*self._parameter_names, "seed")
        self._varying_parameters = tuple(
            name for name, values in self._parameter_values.items() if len(values) > 1
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        train_fn: TrainFunction | None = None,
        log_to_comet: bool | None = None,
        resume: bool | None = None,
        resume_sweep_name: str | None = None,
    ) -> ConditionalGridSweep:
        config_path = Path(path)
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise TypeError("Sweep config must be a YAML mapping.")
        return cls(
            config,
            config_path=config_path,
            train_fn=train_fn,
            log_to_comet=log_to_comet,
            resume=resume,
            resume_sweep_name=resume_sweep_name,
        )

    def iter_points(self) -> Iterator[dict[str, Any]]:
        """Yield grid points in stable YAML declaration order."""
        value_lists = [self._parameter_values[name] for name in self._parameter_names]
        for values in itertools.product(*value_lists):
            yield dict(zip(self._parameter_names, values, strict=True))

    def _run_name(self, index: int, point: Mapping[str, Any]) -> str:
        suffix = "__".join(
            f"{_slug(name)}-{_slug(point[name])}" for name in self._varying_parameters
        )
        base = f"{_slug(self.sweep_name)}__{index:03d}"
        return f"{base}__{suffix}" if suffix else base

    @staticmethod
    def _merge_dataset_metadata(config: DictConfig) -> None:
        """Use prepare_audio's model and class metadata for each dataset axis."""
        ds_path = Path(str(config.ds_path))
        metadata_path = ds_path / "metadata.yaml"
        if not metadata_path.is_file():
            return
        metadata = OmegaConf.load(metadata_path)
        vocab_size = OmegaConf.select(metadata, "model.V", default=None)
        seq_length = OmegaConf.select(metadata, "model.L", default=None)
        if vocab_size is None or seq_length is None:
            raise ValueError(
                f"{metadata_path} must define model.V and model.L for a sweep."
            )
        config.vocab_size = int(vocab_size)
        config.seq_length = int(seq_length)

        manifest_path = ds_path / "metadata.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Could not read dataset manifest '{manifest_path}': {exc}"
                ) from exc
            num_classes = manifest.get("num_classes")
            if (
                isinstance(num_classes, bool)
                or not isinstance(num_classes, int)
                or num_classes <= 0
            ):
                raise ValueError(
                    f"{manifest_path} must define a positive integer num_classes."
                )
            config.num_classes = num_classes

    @staticmethod
    def _derive_embed_dim(config: DictConfig) -> bool:
        """Fit the complete classifier within the requested parameter budget."""
        budget = config.get(_PARAMETER_BUDGET)
        if budget is None:
            return True
        if isinstance(budget, bool) or not isinstance(budget, Real) or budget <= 0:
            raise ValueError(f"{_PARAMETER_BUDGET} must be a positive number.")
        num_layers = int(config.num_layers)
        per_head_dim = int(config.per_head_dim)
        num_classes = config.get("num_classes")
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError(
                f"{Path(str(config.ds_path)) / 'metadata.json'} must define a "
                "positive integer num_classes when using parameter budgets."
            )

        budget_parameters = float(budget) * 1e6
        linear_parameters = (
            int(config.vocab_size) + int(config.seq_length) + num_classes
        )
        # The classifier has 12*N*D^2 transformer parameters and
        # (V + L + C)*D token, position, and class-readout parameters.
        max_embed_dim = (2.0 * budget_parameters) / (
            math.sqrt(
                linear_parameters**2
                + 48 * num_layers * budget_parameters
            )
            + linear_parameters
        )
        embed_dim = math.floor(max_embed_dim / per_head_dim) * per_head_dim
        if embed_dim < per_head_dim:
            return False
        config.embed_dim = embed_dim
        return True

    @staticmethod
    def _derive_eval_batch_size(config: DictConfig) -> None:
        """Keep evaluation memory comparable across tokenizer sequence lengths."""
        token_budget = config.get(_EVAL_TOKEN_BUDGET)
        if token_budget is None:
            return
        if isinstance(token_budget, bool) or not isinstance(token_budget, int):
            raise TypeError(f"{_EVAL_TOKEN_BUDGET} must be a positive integer.")
        if token_budget <= 0:
            raise ValueError(f"{_EVAL_TOKEN_BUDGET} must be a positive integer.")
        config.eval_batch_size = max(1, token_budget // int(config.seq_length))

    def iter_run_configs(
        self,
    ) -> Iterator[tuple[str, dict[str, Any], DictConfig]]:
        """Build validated per-run configs without starting training."""
        for index, point in enumerate(self.iter_points(), start=1):
            config = OmegaConf.load(self.base_config_path)
            if not isinstance(config, DictConfig):
                raise TypeError("Base teacher config must be a YAML mapping.")
            for name, value in point.items():
                OmegaConf.update(config, name, value, merge=False, force_add=True)

            run_name = self._run_name(index, point)
            checkpoint_path = self.output_dir / "checkpoints" / f"{run_name}.pt"
            OmegaConf.update(config, "name", run_name, merge=False, force_add=True)
            OmegaConf.update(
                config,
                "checkpoint_path",
                str(checkpoint_path),
                merge=False,
                force_add=True,
            )
            OmegaConf.update(config, "resume_from", None, merge=False, force_add=True)
            OmegaConf.update(
                config,
                "sweep_name",
                self.sweep_name,
                merge=False,
                force_add=True,
            )

            self._merge_dataset_metadata(config)
            if not self._derive_embed_dim(config):
                print(
                    f"[conditional-sweep] skipping {run_name}: "
                    f"{_PARAMETER_BUDGET}={config.get(_PARAMETER_BUDGET)} cannot "
                    f"fit embed_dim={config.per_head_dim} for {config.ds_path}"
                )
                continue
            self._derive_eval_batch_size(config)
            TeacherTrainingConfig.from_cfg(config)
            yield run_name, point, config

    @staticmethod
    def _comet_str(value: Any) -> str:
        """Match Comet's parameter-summary string representation."""
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, type(None), int, float)):
            return json.dumps(value)
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), separators=(",", ":"))
        return str(value)

    @classmethod
    def _expected_comet_parameters(cls, config: DictConfig) -> dict[str, str]:
        return {
            str(key): cls._comet_str(value)
            for key, value in teacher_experiment_parameters(config).items()
            if str(key) not in _COMET_PARAM_EXCLUDE
        }

    def _initialize_comet_resume(self) -> None:
        if not self.log_to_comet:
            return
        load_dotenv()
        api_key = os.environ.get(COMET_API_KEY_VAR)
        if not api_key or api_key == "<INSERT_API_KEY>":
            raise RuntimeError(
                f"{COMET_API_KEY_VAR} is required for a Comet-enabled sweep."
            )
        self._comet_api_key = api_key
        if not self.resume:
            return
        try:
            api = comet_ml.API(api_key=api_key)
            query = Parameter("sweep_name") == self.resume_sweep_name
            experiments = api.query(EPI_AUDIO_WORKSPACE, COMET_PROJECT, query) or []
        except NotFound:
            # The project is created by the first logged run, so a fresh sweep
            # legitimately has nothing to resume yet.
            experiments = []
        except Exception as exc:
            raise RuntimeError("Conditional sweep Comet lookup failed.") from exc

        for experiment in experiments:
            if experiment is None:
                continue
            experiment_key = getattr(experiment, "key", None)
            if not isinstance(experiment_key, str) or not experiment_key:
                continue
            try:
                summaries = experiment.get_parameters_summary() or []
                parameters = {
                    str(summary["name"]): str(summary["valueCurrent"])
                    for summary in summaries
                    if "name" in summary and "valueCurrent" in summary
                }
            except Exception as exc:
                print(
                    "[conditional-sweep] warning: could not read parameters for "
                    f"Comet experiment {experiment_key}: {exc!r}"
                )
                continue
            self._resume_index[experiment_key] = parameters

        if self._require_existing_comet_sweep and not self._resume_index:
            raise RuntimeError(
                f"No Comet experiments found for sweep '{self.resume_sweep_name}'."
            )
        if self._resume_index:
            print(
                f"[conditional-sweep] found {len(self._resume_index)} prior Comet "
                f"experiment(s) for '{self.resume_sweep_name}'"
            )

    @staticmethod
    def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, MetricValue]:
        numeric: dict[str, MetricValue] = {}
        for name, value in metrics.items():
            if isinstance(value, bool):
                numeric[str(name)] = int(value)
            elif isinstance(value, Real):
                numeric[str(name)] = float(value)
        return numeric

    def _comet_metrics(self, experiment_key: str) -> dict[str, MetricValue] | None:
        try:
            experiment = comet_ml.APIExperiment(
                previous_experiment=experiment_key,
                api_key=self._comet_api_key,
            )
            summaries = experiment.get_metrics_summary() or []
        except Exception as exc:
            print(
                "[conditional-sweep] warning: could not read metrics for Comet "
                f"experiment {experiment_key}: {exc!r}"
            )
            return None
        if isinstance(summaries, Mapping):
            summaries = [summaries]
        values: dict[str, float] = {}
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            name = summary.get("name")
            current = summary.get("valueCurrent")
            if not isinstance(name, str) or current is None:
                continue
            try:
                values[name] = float(current)
            except (TypeError, ValueError):
                continue
        if "optimizer_steps" not in values:
            return None
        try:
            self._selection_value(values)
        except ValueError:
            return None
        return values

    def _find_comet_resume(
        self,
        config: DictConfig,
    ) -> tuple[str, dict[str, MetricValue]] | None:
        if not self._resume_index:
            return None
        expected = self._expected_comet_parameters(config)
        for experiment_key, logged in self._resume_index.items():
            if not expected.keys() <= logged.keys():
                continue
            if not all(logged[name] == value for name, value in expected.items()):
                continue
            metrics = self._comet_metrics(experiment_key)
            if metrics is not None:
                return experiment_key, metrics
        return None

    @staticmethod
    def _checkpoint_matches_config(
        checkpoint: Mapping[str, Any],
        config: DictConfig,
    ) -> bool:
        saved = checkpoint.get("config")
        if not isinstance(saved, Mapping):
            return False
        expected = asdict(TeacherTrainingConfig.from_cfg(config))
        return all(
            name == "resume_from" or saved.get(name) == value
            for name, value in expected.items()
        )

    def _local_checkpoint(
        self,
        config: DictConfig,
    ) -> tuple[dict[str, MetricValue] | None, bool]:
        """Return ``(completed metrics, partial checkpoint is resumable)``."""
        checkpoint_path = Path(str(config.checkpoint_path))
        if not self.resume or not checkpoint_path.is_file():
            return None, False
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            print(
                f"[conditional-sweep] warning: could not read {checkpoint_path}: "
                f"{exc!r}; starting this point from scratch"
            )
            return None, False
        if not isinstance(checkpoint, Mapping) or not self._checkpoint_matches_config(
            checkpoint, config
        ):
            print(
                f"[conditional-sweep] warning: {checkpoint_path} belongs to a "
                "different configuration; starting this point from scratch"
            )
            return None, False

        stored_metrics = checkpoint.get("metrics")
        if isinstance(stored_metrics, Mapping):
            metrics = self._numeric_metrics(stored_metrics)
            try:
                self._selection_value(metrics)
            except ValueError:
                pass
            else:
                return metrics, False
        return None, True

    def _train(self, config: DictConfig) -> dict[str, MetricValue]:
        if self._train_fn is not None:
            return self._train_fn(config)
        return run_experiment(config, log_to_comet=False)

    def _selection_value(self, metrics: Mapping[str, MetricValue]) -> float:
        value = metrics.get(self.metric_name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"Run did not return numeric selection metric '{self.metric_name}'."
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"Selection metric '{self.metric_name}' must be finite, got {numeric}."
            )
        return numeric

    @staticmethod
    def _point_compute(point: ConditionalSweepPoint) -> float:
        compute = point.metrics.get("training_flops_estimate")
        if isinstance(compute, bool) or not isinstance(compute, Real):
            raise ValueError("Reported point is missing numeric training compute.")
        numeric = float(compute)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("Reported point training compute must be finite and non-negative.")
        return numeric

    def _reported_points(
        self,
        runs: list[ConditionalSweepRun],
    ) -> list[ConditionalSweepPoint]:
        points = []
        for run_index, run in enumerate(runs):
            reports = extract_reported_points(run.metrics) or (run.metrics,)
            for report_index, metrics in enumerate(reports):
                if metrics.get("conditional_epiplexity_valid") != 1:
                    continue
                point = ConditionalSweepPoint(
                    run_index=run_index,
                    run_name=run.name,
                    report_index=report_index,
                    metrics=dict(metrics),
                )
                self._point_compute(point)
                self._selection_value(point.metrics)
                points.append(point)
        return points

    def _lower_convex_hull(
        self,
        points: list[ConditionalSweepPoint],
    ) -> list[ConditionalSweepPoint]:
        """Build the pooled lower hull and retain one median point per run."""
        ordered = sorted(
            points,
            key=lambda point: (
                self._point_compute(point),
                self._selection_value(point.metrics),
            ),
        )
        hull: list[ConditionalSweepPoint] = []
        for point in ordered:
            point_x = self._point_compute(point)
            point_y = self._selection_value(point.metrics)
            while len(hull) >= 2:
                first, second = hull[-2], hull[-1]
                first_x = self._point_compute(first)
                first_y = self._selection_value(first.metrics)
                second_x = self._point_compute(second)
                second_y = self._selection_value(second.metrics)
                cross = (second_x - first_x) * (point_y - first_y) - (
                    second_y - first_y
                ) * (point_x - first_x)
                if cross > 0:
                    break
                hull.pop()
            if hull and point_y > min(
                self._selection_value(candidate.metrics) for candidate in hull
            ):
                continue
            hull.append(point)

        by_run: dict[int, list[ConditionalSweepPoint]] = {}
        for point in hull:
            by_run.setdefault(point.run_index, []).append(point)
        reduced = []
        for run_points in by_run.values():
            median_compute = float(
                np.median([self._point_compute(point) for point in run_points])
            )
            reduced.append(
                min(
                    run_points,
                    key=lambda point: abs(
                        self._point_compute(point) - median_compute
                    ),
                )
            )
        return sorted(reduced, key=self._point_compute)

    def run(self) -> ConditionalSweepResult:
        """Train every point, persist a JSON summary, and return the best run."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_comet_resume()
        completed: list[ConditionalSweepRun] = []
        summary_config: DictConfig | None = None
        summary_tags: set[str] = set()
        for run_name, point, config in self.iter_run_configs():
            if summary_config is None:
                summary_config = config
            representation = dataset_and_tokenizer(str(config.ds_path))
            if representation is not None:
                summary_tags.update(representation)
            local_metrics, local_resumable = self._local_checkpoint(config)
            comet_resume = self._find_comet_resume(config)
            experiment_key: str | None = None

            if comet_resume is not None:
                experiment_key, metrics = comet_resume
                source = "comet"
                print(
                    f"[conditional-sweep] reusing {run_name} from Comet "
                    f"experiment {experiment_key}"
                )
            elif local_metrics is not None:
                metrics = local_metrics
                source = "local-checkpoint"
                print(f"[conditional-sweep] reusing completed {run_name} checkpoint")
                if self.log_to_comet:
                    experiment_key = log_experiment(config, metrics)
            else:
                if local_resumable:
                    OmegaConf.update(
                        config,
                        "resume_from",
                        str(config.checkpoint_path),
                        merge=False,
                        force_add=True,
                    )
                    source = "local-resume"
                    print(f"[conditional-sweep] resuming partial {run_name} checkpoint")
                else:
                    source = "trained"
                    print(f"[conditional-sweep] running {run_name}")
                metrics = self._train(config)
                if self.log_to_comet:
                    experiment_key = log_experiment(config, metrics)

            self._selection_value(metrics)
            if (
                not self.retain_checkpoints
                and experiment_key is not None
                and Path(str(config.checkpoint_path)).is_file()
            ):
                Path(str(config.checkpoint_path)).unlink()
                print(
                    f"[conditional-sweep] removed completed checkpoint for "
                    f"{run_name} after Comet confirmed experiment {experiment_key}"
                )
            completed.append(
                ConditionalSweepRun(
                    name=run_name,
                    parameters=point,
                    checkpoint_path=str(config.checkpoint_path),
                    metrics=metrics,
                    source=source,
                    experiment_key=experiment_key,
                )
            )

        if not completed:
            raise ValueError("Conditional sweep contains no runs.")
        reported_points = self._reported_points(completed)
        if not reported_points:
            raise RuntimeError(
                "No reported point produced a valid conditional epiplexity estimate."
            )
        pareto_front = self._lower_convex_hull(reported_points)
        chooser = min if self.goal == "minimize" else max
        best_point = chooser(
            pareto_front,
            key=lambda point: self._selection_value(point.metrics),
        )
        best_run = completed[best_point.run_index]
        summary_path = self.output_dir / "summary.json"
        result = ConditionalSweepResult(
            sweep_name=self.sweep_name,
            metric_name=self.metric_name,
            goal=self.goal,
            runs=tuple(completed),
            best_run=best_run,
            best_point=best_point,
            pareto_front=tuple(pareto_front),
            summary_path=summary_path,
        )
        temporary_path = summary_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(summary_path)
        summary_experiment_key = None
        if self.log_to_comet:
            assert summary_config is not None
            summary_metrics: dict[str, MetricValue] = {
                "completed_runs": len(completed),
                "resumed_runs": sum(run.source != "trained" for run in completed),
                "best_selection_value": self._selection_value(best_point.metrics),
            }
            for name in (
                "conditional_epiplexity_bits",
                "two_part_code_bits",
                "ema_accuracy",
                "ema_macro_f1",
                "ema_weighted_f1",
            ):
                value = best_point.metrics.get(name)
                if isinstance(value, Real) and not isinstance(value, bool):
                    summary_metrics[f"best_{name}"] = float(value)
            try:
                summary_experiment_key = log_sweep_summary(
                    summary_config,
                    sweep_name=self.sweep_name,
                    parameters={
                        "base_config": str(self.base_config_path),
                        "selection_metric": self.metric_name,
                        "selection_goal": self.goal,
                        "run_count": len(completed),
                        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                    },
                    metrics=summary_metrics,
                    summary_path=summary_path,
                    extra_tags=sorted(summary_tags),
                )
            except Exception as exc:
                print(
                    "[conditional-sweep] warning: summary upload failed: "
                    f"{exc!r}; local summary remains at {summary_path}"
                )
        print(
            f"[conditional-sweep] best point: {best_run.name} "
            f"step={int(best_point.metrics['optimizer_steps'])} "
            f"({self.metric_name}={self._selection_value(best_point.metrics):.6g})"
        )
        print(f"[conditional-sweep] summary: {summary_path}")
        if summary_experiment_key is not None:
            print(
                "[conditional-sweep] Comet summary experiment: "
                f"{summary_experiment_key}"
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a conditional epiplexity teacher grid sweep."
    )
    parser.add_argument("sweep_config", help="Path to a conditional sweep YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the run configs without training.",
    )
    parser.add_argument(
        "--resume-sweep",
        metavar="NAME",
        help=(
            "Resume completed points using this Comet sweep-name prefix. The "
            "tokenizer suffix is appended automatically; by default, the name "
            "in the sweep YAML is used."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching Comet runs and local checkpoints.",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Run locally without Comet logging or Comet resume lookup.",
    )
    args = parser.parse_args()
    if args.no_resume and args.resume_sweep:
        parser.error("--no-resume cannot be combined with --resume-sweep.")
    if args.no_comet and args.resume_sweep:
        parser.error("--no-comet cannot be combined with --resume-sweep.")

    config_path, configs = load_sweep_configs(
        args.sweep_config,
        sweep_name=args.resume_sweep,
    )
    for config in configs:
        resume_sweep_name = str(config.name) if args.resume_sweep else None
        sweep = ConditionalGridSweep(
            config,
            config_path=config_path,
            log_to_comet=False if args.no_comet else None,
            resume=False if args.no_resume else None,
            resume_sweep_name=resume_sweep_name,
        )
        if args.dry_run:
            print(f"\n# Sweep: {sweep.sweep_name}")
            for run_name, _, run_config in sweep.iter_run_configs():
                print(
                    f"\n# {run_name}\n"
                    f"{OmegaConf.to_yaml(run_config, resolve=True)}"
                )
            continue
        sweep.run()


if __name__ == "__main__":
    main()

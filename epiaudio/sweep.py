"""Hyperparameter sweeps for epiplexity audio experiments.

`epiaudio/main.py` is a thin wrapper that builds a single config from the CLI
(via Hydra) and calls `train.train_and_evaluate(cfg)`. This module adds the
layer above that: a `Sweeper` runs `train_and_evaluate` many times over a grid
of hyperparameters, names and logs each run on Comet ML, and reports the
configuration that minimizes epiplexity (the prequential K(M) estimate).

A sweep is described by a wandb-style sweep YAML, e.g.
`epiplexity/picodo/sweeps/chess.yaml`, which contains:

    program: main.py
    method: grid
    name: <sweep name>
    command: [ ..., -cn, <base_config>, key=value, ... ]
    parameters:
      "model.N": {values: [3, 6, 12]}
      "model.P": {values: [1, 2, 5]}
      "seed":    {value: 0}

The `command` tells us which Hydra base config to start from (`-cn <name>`,
resolved against `epiplexity/picodo/configs`) plus any fixed overrides; the
`parameters` block defines the grid.

Working directory:
    Runs call ``train.train_and_evaluate`` directly, so (unlike a Hydra app)
    the working directory is NOT changed per run. Any relative paths in the
    sweep config (e.g. ``ds_path=data/fsd50k``) are resolved against the
    current directory, so launch the sweep from the repository root. Use
    absolute paths (or ``--override ds_path=/abs/...``) to run from elsewhere.

Usage (from the repository root):
    uv run python -m epiaudio.sweep epiplexity/picodo/sweeps/chess.yaml
    uv run python -m epiaudio.sweep <sweep.yaml> --override ds_path=/abs/data/fsd50k --override T=2048000
"""

from __future__ import annotations
from typing import cast

import comet_ml
from comet_ml.query import Parameter
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import copy
from datetime import datetime
import abc
import argparse
import itertools
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import os
import traceback

import numpy as np
import pandas as pd

from epiaudio.utils import nearest_value, timestamped_values, get_parameter
from epiaudio.sweep_preflight import print_preflight, validate_sweep
from epiaudio.sweep_summary import (
    CONFIG_SUFFIX,
    SweepOutputBundle,
    upload_sweep_summary,
)
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS

REPO_ROOT = Path(__file__).resolve().parent.parent
PICODO_DIR = REPO_ROOT / "epiplexity" / "picodo"
PICODO_CONFIG_DIR = PICODO_DIR / "configs"
COMET_WORKSPACE = "epi-audio"
SWEEP_OUTPUT_DIR = REPO_ROOT / "sweep_outputs"

# Mirror epiaudio/main.py so `picodo.*` imports inside train.py resolve.
sys.path.insert(0, str(PICODO_DIR))

def load_sweep_configs(path: str) -> list[DictConfig]:
    """Load a sweep YAML file into a DictConfig."""
    result = OmegaConf.load(path)
    assert isinstance(result, DictConfig), f"Expected DictConfig from {path}, got {type(result)}"

    ds_paths = OmegaConf.select(result, "parameters.ds_path.values", default=None)
    if ds_paths is None:
        raise ValueError("At least one dataset path must be provided")
    ds_paths = OmegaConf.to_container(ds_paths)
    if not isinstance(ds_paths, list):
        raise ValueError(f"ds_paths must be a list, got {type(ds_paths)}")

    cfgs = []
    name = OmegaConf.select(result, "name", default="sweep")

    for path in ds_paths:
        if not isinstance(path, str):
            raise ValueError(f"Each dataset path must be a string, got {type(path)}")

        cfg = copy.deepcopy(result)
        OmegaConf.update(cfg, "parameters.ds_path.values", [path])

        tokenizer_name = path.split("_")[-1]
        OmegaConf.update(cfg, "name", f"{name}_{tokenizer_name}")

        cfgs.append(cfg)

    return cfgs


def _fmt_override_value(value: Any) -> str:
    """Format a Python value the way Hydra expects on the override line."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


class Sweeper(abc.ABC):
    """Abstract base class for a hyperparameter sweep.

    Parameters
    ----------
    cfg:
        The parsed sweep YAML (see module docstring). Provides the base Hydra
        config name, fixed command overrides, and the `parameters` grid.
    extra_overrides:
        Optional Hydra-style overrides (e.g. ``"ds_path=/abs/data/fsd50k"``)
        applied to every run, on top of the sweep file. Handy for pointing a
        sweep at a different dataset or shrinking it for a CPU smoke test.
    backend:
        Training backend to use: ``"torch"`` (default, and the only supported
        backend) uses the PyTorch DDP trainer (train_torch.py). ``"jax"`` is
        recognized but raises NotImplementedError; the original JAX/Flax
        trainer (train.py) hasn't been updated for this Sweeper's return
        contract. The torch backend requires PyTorch + CUDA to be installed.
    """

    def __init__(
        self,
        cfg: DictConfig,
        extra_overrides: list[str] | None = None,
        backend: str = "torch",
        sweep_seed: int | None = None,
        sweep_name: str | None = None,
    ):
        self.cfg = cfg
        self.extra_overrides = list(extra_overrides or [])
        self.base_config_name, self.command_overrides = self._parse_command(cfg)

        name = str(OmegaConf.select(cfg, "name", default="sweep"))
        self.start_time = datetime.now()
        time_created = self.start_time.strftime("%Y-%m-%d_%H:%M:%S")

        # experiment_key -> {param_name: value} for every experiment already logged
        # under `sweep_name`, if resuming one. See _find_resumed_experiment().
        self._resume_index: dict[str, dict[str, str]] = {}

        if sweep_name is not None:
            load_dotenv()  # make COMET_ML_API available
            project_name = self._override_value(
                "wandb_project", self.command_overrides, self.extra_overrides
            )
            if not os.environ.get("COMET_ML_API"):
                raise RuntimeError(
                    f"Cannot resume sweep {sweep_name!r}: COMET_ML_API is not set."
                )
            if project_name is None:
                raise RuntimeError(
                    f"Cannot resume sweep {sweep_name!r}: could not determine "
                    "wandb_project from the sweep config."
                )

            try:
                api = comet_ml.API(api_key=os.environ["COMET_ML_API"])
                query = Parameter("sweep_name") == sweep_name
                old_experiments = api.query(COMET_WORKSPACE, project_name, query) or []
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot resume sweep {sweep_name!r}: Comet lookup failed."
                ) from exc

            if len(old_experiments) > 0:
                for exp in old_experiments:
                    if exp is None:
                        continue
                    experiment_key = exp.key
                    if not isinstance(experiment_key, str) or not experiment_key:
                        continue
                    try:
                        params = {
                            p["name"]: p["valueCurrent"]
                            for p in (exp.get_parameters_summary() or [])
                        }
                    except Exception as exc:
                        print(
                            f"[sweep] [WARNING]: failed to read parameters for "
                            f"experiment {experiment_key}: {exc!r}; skipping."
                        )
                        continue
                    self._resume_index[experiment_key] = params
                print(
                    f"[sweep] resuming sweep '{sweep_name}': found "
                    f"{len(self._resume_index)} previous experiment(s) in Comet"
                )
            else:
                raise RuntimeError(
                    f"Cannot resume sweep {sweep_name!r}: no matching Comet experiments found."
                )

            if not self._resume_index:
                raise RuntimeError(
                    f"Cannot resume sweep {sweep_name!r}: no previous experiments "
                    "had readable parameters."
                )

        if sweep_name is None:
            sweep_name = f"{name}__{time_created}"

        self.sweep_name = sweep_name
        self.sweep_seed = sweep_seed  # None means defer to YAML seed: field
        self.results: list[dict[str, Any]] = []

        backend = backend.lower()
        if backend == "torch":
            try:
                import epiaudio.train_torch as train_torch
            except ImportError as exc:
                raise ImportError(
                    "PyTorch backend requested but epiaudio.train_torch could not be "
                    "imported. Install torch (see pyproject.toml pytorch extras)."
                ) from exc
            self._train_fn = train_torch.train_and_evaluate
            self._train_torch = train_torch
        elif backend == "jax":
            raise NotImplementedError(
                "The JAX backend is not supported for sweeps: epiaudio/train.py "
                "still returns a bare epiplexity float rather than the "
                "(experiment_key, test_tokens) pair this Sweeper reads back from "
                "Comet. Use backend='torch'."
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}; choose 'torch' or 'jax'.")
        self.backend = backend

    @staticmethod
    def _parse_command(cfg: DictConfig) -> tuple[str, list[str]]:
        """Extract the base config name (``-cn X``) and fixed overrides from `command`."""
        base_config = "default"
        overrides: list[str] = []
        command_node = OmegaConf.select(cfg, "command", default=None)
        # resolve=False keeps wandb templating tokens like ${env} as literal
        # strings instead of trying to resolve them as interpolations.
        command_raw = OmegaConf.to_container(command_node, resolve=False) if command_node is not None else []
        command: list[Any] = command_raw if isinstance(command_raw, list) else []
        i = 0
        while i < len(command):
            token = str(command[i])
            if token in ("-cn", "--config-name") and i + 1 < len(command):
                base_config = str(command[i + 1])
                i += 2
                continue
            # Skip launcher/templating tokens and the program name.
            if token in ("${env}", "${args_no_hyphens}", "python", "python3") or token.endswith(".py"):
                i += 1
                continue
            if "=" in token:
                overrides.append(token)
            i += 1
        return base_config, overrides

    @staticmethod
    def _override_value(key: str, *override_lists: list[str]) -> str | None:
        """Look up `key=value` from override strings. Later lists win on a
        collision, matching build_run_config's Hydra composition order
        (command_overrides applied first, extra_overrides layered on top)."""
        value = None
        for overrides in override_lists:
            for ov in overrides:
                k, sep, v = ov.partition("=")
                if sep and k == key:
                    value = v
        return value

    @abc.abstractmethod
    def iter_points(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(fixed, point)`` pairs to run.

        ``fixed`` holds parameters shared by every run; ``point`` holds the
        per-run parameter combination layered on top. The base class only
        commits to *iterating* over points, not enumerating them up front:
        grid search can compute every combination in advance, but other
        strategies (e.g. Bayesian search) generate points lazily as earlier
        results arrive.
        """
        raise NotImplementedError

    def run_name_for(self, point: dict[str, Any]) -> str:
        """Build an explicit, human-readable run name for a grid point."""
        if not point:
            return self.sweep_name
        suffix = "_".join(f"{key.split('.')[-1]}{value}" for key, value in point.items())
        return f"{self.sweep_name}__{suffix}__{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}"

    def build_run_config(self, fixed: dict[str, Any], point: dict[str, Any], run_name: str) -> DictConfig:
        """Compose the full per-run config from the base Hydra config + overrides."""
        overrides = list(self.command_overrides)
        for key, value in {**fixed, **point}.items():
            overrides.append(f"{key}={_fmt_override_value(value)}")
        overrides.extend(self.extra_overrides)
        # run_name / sweep_name are not part of the base schema, so append them.
        overrides.append(f"+run_name={run_name}")
        overrides.append(f"+sweep_name={self.sweep_name}")

        with initialize_config_dir(version_base=None, config_dir=str(PICODO_CONFIG_DIR)):
            cfg = compose(config_name=self.base_config_name, overrides=overrides)

        ds_path = OmegaConf.select(cfg, "ds_path", default=None)
        if ds_path is not None:
            metadata_path = Path(str(ds_path)) / "metadata.yaml"
            if metadata_path.exists():
                metadata_cfg = OmegaConf.load(metadata_path)
                merged = OmegaConf.merge(cfg, metadata_cfg)
                assert isinstance(merged, DictConfig), f"Expected DictConfig, got {type(merged)}"
                cfg = merged
            else:
                print(
                    f"[sweep] [WARNING]: no metadata.yaml found at '{metadata_path}' "
                    f"for run '{run_name}'; using sweep-config values as-is."
                )

        return cfg

    @staticmethod
    def _ema_active(cfg: DictConfig, key: str) -> bool:
        """Mirrors train_torch.py's `_ema_decay`: an EMA model only gets built
        (and only then does e.g. "ema_student_eval_loss" get logged, instead
        of the plain "student_eval_loss") if `cfg[key]` is set and > 0."""
        val = OmegaConf.select(cfg, key, default=None)
        return val is not None and float(val) > 0.0

    def _fetch_epiplexity(
        self,
        experiment_key: str | None,
        test_tokens: int,
        train_student: bool,
        teacher_ema_active: bool,
        student_ema_active: bool,
    ) -> tuple[float | None, float | None]:
        """Estimate epiplexity for one run from its logged Comet metrics.

        Epiplexity is K(M) at the training step minimizing the two-part code
        K(M,X) = K(M) + K(X|M). Returns (None, None) if Comet logging was
        disabled or no eval step was logged.

        Mirrors epiplexity/notebooks/chess_order.ipynb's `set_test_tokens`:
        requential (`train_student`) uses K(M)_req + the student's eval loss;
        prequential (not `train_student`) uses K(M) + the teacher's eval loss.
        In both cases, if that model's EMA isn't configured (so the "ema_*"
        variant was never logged), fall back to the plain eval loss instead.
        """
        if experiment_key is None:
            return None, None

        import comet_ml

        api_experiment = comet_ml.APIExperiment(
            previous_experiment=experiment_key,
            api_key=os.environ.get("COMET_ML_API"),
        )

        if train_student:
            km_metric = "K(M)_req"
            loss_metric = "ema_student_eval_loss" if student_ema_active else "student_eval_loss"
        else:
            km_metric = "K(M)"
            loss_metric = "ema_teacher_eval_loss" if teacher_ema_active else "teacher_eval_loss"

        km_records = api_experiment.get_metrics(km_metric) or []
        loss_records = api_experiment.get_metrics(loss_metric) or []

        # Both are written in the same `log_metrics(pending_eval_metrics,
        # step=student_step)` call in train_torch.py's eval block, so align
        # them by `timestamp` rather than `step`: when train_student is False,
        # student_step never advances, so every eval event in the run shares
        # the same (useless) step and only `timestamp` still distinguishes them.
        km_by_ts = timestamped_values(km_records, scale=1e6)

        km_values: list[float] = []
        code_lengths: list[float] = []
        for r in loss_records:
            ts = r.get("timestamp")
            if ts is None:
                continue
            k_m = nearest_value(km_by_ts, int(ts))
            if k_m is None:
                continue

            k_x_given_m = float(r["metricValue"]) * test_tokens / np.log(2)
            km_values.append(k_m)
            code_lengths.append(k_m + k_x_given_m)

        if not code_lengths:
            return None, None

        min_idx = int(np.argmin(code_lengths))
        return km_values[min_idx], code_lengths[min_idx]

    # Keys excluded from the resume config match below:
    _COMET_PARAM_EXCLUDE = {"run_name", "sweep_name", "D"}

    @staticmethod
    def _comet_str(value: Any) -> str:
        """Stringify paramter like cometml to cleanly match with current config
        """
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, type(None), int, float)):
            return json.dumps(value)
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), separators=(",", ":"))
        return str(value)

    def _expected_comet_params(self, run_cfg: DictConfig) -> dict[str, str]:
        """Formats comet_ml params from old experiments to look like
        
        Our config for easier comparing.
        """
        container = OmegaConf.to_container(run_cfg, resolve=True)
        assert isinstance(container, dict)
        flat = self._train_torch._flatten_dict(container)
        flat = {k.split("/")[-1]: v for k, v in flat.items()}
        return {
            k: self._comet_str(v)
            for k, v in flat.items()
            if k not in self._COMET_PARAM_EXCLUDE
        }

    @staticmethod
    def _reached_final_teacher_step(
        teacher_step_records: list[dict[str, Any]], num_train_steps: int
    ) -> bool:
        """Return whether a Comet run reached the trainer's final step."""
        logged_steps = [
            int(float(record["metricValue"]))
            for record in teacher_step_records
            if record.get("metricValue") is not None
        ]
        return bool(logged_steps) and max(logged_steps) >= num_train_steps

    def _find_resumed_experiment(
        self,
        run_cfg: DictConfig,
        train_student: bool,
        teacher_ema_active: bool,
        student_ema_active: bool,
    ) -> tuple[str, int, float, float] | None:
        """If `--sweep-name` pointed at a previous sweep and one of its
        experiments' logged hyperparameters exactly match this run_cfg,
        return ``(experiment_key, test_tokens, epiplexity, code_length)`` so
        the caller can skip training entirely.

        A config match is reusable only if its logged ``teacher_step`` reached
        the expected final training step. A run that crashed after an earlier
        evaluation must be rerun rather than mistaken for a completed run.
        """
        if not self._resume_index:
            return None

        ds_path = OmegaConf.select(run_cfg, "ds_path", default=None)
        vocab = OmegaConf.select(run_cfg, "model.V", default=None)
        if ds_path is None or vocab is None:
            return None
        test_path = Path(str(ds_path)) / "test.bin"
        if not test_path.exists():
            return None
        # Mirrors train_torch.py's `_run_training`: vocabs above uint16's range
        # (e.g. SQCodec's 117,649) are written as int32 by prepare_audio.py.
        ds_dtype = np.uint16 if int(vocab) <= 65536 else np.uint32
        test_tokens = int(len(np.memmap(test_path, dtype=ds_dtype, mode="r")))

        configured_train_tokens = OmegaConf.select(run_cfg, "T", default=None)
        if configured_train_tokens:
            train_tokens = int(configured_train_tokens)
        else:
            train_path = Path(str(ds_path)) / "train.bin"
            if not train_path.exists():
                return None
            train_tokens = int(len(np.memmap(train_path, dtype=ds_dtype, mode="r")))
        num_train_steps = train_tokens // (
            int(OmegaConf.select(run_cfg, "B"))
            * int(OmegaConf.select(run_cfg, "model.L"))
        )

        expected = self._expected_comet_params(run_cfg)
        for exp_key, params in self._resume_index.items():
            if not (expected.keys() <= params.keys()):
                continue

            # Only compare paramters that are not random
            if not all(expected[k] == params[k] for k in expected):
                continue
            try:
                api_experiment = comet_ml.APIExperiment(
                    previous_experiment=exp_key,
                    api_key=os.environ.get("COMET_ML_API"),
                )
                teacher_steps = api_experiment.get_metrics("teacher_step") or []
                if not self._reached_final_teacher_step(
                    teacher_steps, num_train_steps
                ):
                    continue

                epiplexity, code_length = self._fetch_epiplexity(
                    exp_key, test_tokens, train_student,
                    teacher_ema_active, student_ema_active,
                )
            except Exception as exc:
                print(
                    f"[sweep] [WARNING]: resume candidate {exp_key} matched config but "
                    f"failed to fetch epiplexity: {exc!r}; treating as not resumable."
                )
                continue
            if epiplexity is None or code_length is None:
                continue  # completed training but no usable evaluation — rerun
            return exp_key, test_tokens, epiplexity, code_length
        return None

    @abc.abstractmethod
    def sweep(self) -> dict[str, Any]:
        """Run the sweep and return the best (minimum-epiplexity) result."""
        raise NotImplementedError


class GridSweeper(Sweeper):
    """Grid search that minimizes epiplexity over the sweep's `parameters`.

    Accepts the same constructor arguments as ``Sweeper`` (including
    ``backend='torch'`` / ``backend='jax'``).
    """

    def iter_points(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        yield from self.iter_grid()

    def iter_grid(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(fixed, point)`` dicts: fixed params and one grid combination each.

        Seed handling:
        - ``seed:`` at the YAML top level (or ``--sweep-seed`` CLI) sets the global
          sweep seed used to derive per-repeat training seeds via SeedSequence.
        - ``num_repeats:`` at the YAML top level controls how many independent repeats
          to run per grid point; the seed axis is injected automatically.
        - Backward compat: ``seed: values: [...]`` in ``parameters`` still works —
          the list length becomes the repeat count and the values are ignored.
        """
        params = OmegaConf.select(self.cfg, "parameters", default={}) or {}

        # --- resolve sweep seed: CLI > YAML top-level seed: > 0 ---
        yaml_seed = OmegaConf.select(self.cfg, "seed", default=None)
        sweep_seed = (
            self.sweep_seed if self.sweep_seed is not None
            else int(yaml_seed) if yaml_seed is not None
            else 0
        )

        # --- resolve repeat count: YAML num_repeats: > seed: values: length > 1 ---
        yaml_num_repeats = OmegaConf.select(self.cfg, "num_repeats", default=None)
        legacy_seed_count: int | None = None
        seed_spec = (params or {}).get("seed")
        if seed_spec is not None:
            _sv = OmegaConf.select(seed_spec, "values", default=None)
            if _sv is not None:
                legacy_seed_count = len(list(_sv))

        num_repeats = (
            int(yaml_num_repeats) if yaml_num_repeats is not None
            else legacy_seed_count if legacy_seed_count is not None
            else 1
        )

        # derive num_repeats independent seeds from the global sweep seed
        repeat_seeds = [
            int(c.generate_state(1)[0])
            for c in np.random.SeedSequence(sweep_seed).spawn(num_repeats)
        ]

        # --- build grid, skipping seed (injected via repeat_seeds) ---
        fixed: dict[str, Any] = {}
        grid_keys: list[str] = []
        grid_values: list[list[Any]] = []

        for key, spec in params.items():
            if key == "seed":
                continue
            if spec is None:
                continue
            values = OmegaConf.select(spec, "values", default=None)
            if values is not None:
                values = list(values)
                if len(values) == 1:
                    fixed[key] = values[0]
                else:
                    grid_keys.append(key)
                    grid_values.append(values)
                continue
            if OmegaConf.select(spec, "value", default=None) is not None or "value" in spec:
                fixed[key] = spec["value"]

        combinations = itertools.product(*grid_values) if grid_values else [()]
        for combo in combinations:
            base_point = dict(zip(grid_keys, combo))
            for seed_val in repeat_seeds:
                yield fixed, {**base_point, "seed": seed_val}

    def sweep(self) -> dict[str, Any]:
        load_dotenv()  # make COMET_ML_API available for train.py's Comet init

        if (
            "COMET_ML_API" not in os.environ or
            os.environ["COMET_ML_API"] == "<INSERT_API_KEY>"
        ):
            print("[WARNING]: CometML API not set, continue anyway?")
            input()

        best: dict[str, Any] | None = None
        points = list(self.iter_points())
        print(f"[sweep] '{self.sweep_name}': {len(points)} run(s) over base config "
              f"'{self.base_config_name}'")

        for idx, (fixed, point) in enumerate(points, start=1):
            run_name = self.run_name_for(point)
            run_cfg = self.build_run_config(fixed, point, run_name)

            train_student = bool(OmegaConf.select(run_cfg, "train_student", default=True))
            teacher_ema_active = self._ema_active(run_cfg, "teacher_ema")
            student_ema_active = self._ema_active(run_cfg, "student_ema")

            resumed = self._find_resumed_experiment(
                run_cfg, train_student, teacher_ema_active, student_ema_active
            )
            if resumed is not None:
                experiment_key, test_tokens, epiplexity, code_length_estimate = resumed
                print(
                    f"\n[sweep] ({idx}/{len(points)}) resumed run '{run_name}' from Comet "
                    f"experiment {experiment_key}  point={point}"
                )
            else:
                print(f"\n[sweep] ({idx}/{len(points)}) starting run '{run_name}'  point={point}")

                try:
                    experiment_key, test_tokens = self._train_fn(run_cfg)
                except SystemExit:
                    # train.py calls exit() for invalid configs (e.g. model.D < 64).
                    print(f"[sweep] run '{run_name}' skipped (config rejected by train.py)")
                    traceback.print_exc()
                    continue
                except Exception as exc:  # keep the sweep going if one point fails
                    print(f"[sweep] run '{run_name}' failed: {exc!r}")
                    traceback.print_exc()
                    continue

                try:
                    epiplexity, code_length_estimate = self._fetch_epiplexity(
                        experiment_key, test_tokens, train_student,
                        teacher_ema_active, student_ema_active,
                    )
                except Exception as exc:  # Comet read API can be flaky; don't kill the sweep over it
                    print(f"[sweep] run '{run_name}' -> failed to fetch epiplexity from Comet: {exc!r}")
                    traceback.print_exc()
                    epiplexity, code_length_estimate = None, None
            if epiplexity is None:
                print(f"[sweep] run '{run_name}' -> no epiplexity estimate "
                      "(Comet logging disabled or no eval step logged)")
                self.results.append(
                    {"run_name": run_name, "point": point, "code_length": None, "epiplexity": None}
                )
                continue

            result = {"run_name": run_name, "point": point, "code_length": code_length_estimate, "epiplexity": epiplexity}
            self.results.append(result)
            print(f"[sweep] run '{run_name}' -> epiplexity K(M) = {epiplexity}  code_length K(M,X) = {code_length_estimate}")

            # minimize the two-part code K(M,X) over the sweep; report K(M) at that point.
            if best is None or epiplexity is None or code_length_estimate < best["code_length"]:
                best = result

        total_runtime = (datetime.now() - self.start_time).total_seconds() / 3600
        print(f"Sweep runtime: {total_runtime:.2f} hours")
        self._report(best)
        return best or {}

    def _group_key(self, point: dict[str, Any]) -> str:
        """Point identity ignoring seed, for bootstrap aggregation."""
        return "_".join(
            f"{k}={v}" for k, v in sorted(point.items()) if k != "seed"
        )

    def _report(self, best: dict[str, Any] | None) -> None:
        print("\n[sweep] ===== results (sorted by epiplexity) =====")
        ranked = sorted(
            self.results,
            key=lambda r: (r["code_length"] is None, r["code_length"]),
        )
        for result in ranked:
            print(f"  K(M)={result['epiplexity']}  <-  {result['run_name']}")

        # Bootstrap summary: mean ± std per grid point (ignoring seed axis).
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        group_names: dict[str, str] = {}
        for r in self.results:
            if r["epiplexity"] is None:
                continue
            key = self._group_key(r["point"])
            groups[key].append(r["epiplexity"])
            group_names[key] = self.run_name_for(
                {k: v for k, v in r["point"].items() if k != "seed"}
            )

        if any(len(v) > 1 for v in groups.values()):
            print("\n[sweep] ===== bootstrap summary (mean ± std per grid point) =====")
            summaries = []
            for key, vals in groups.items():
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                summaries.append((mean, std, key, vals))
            for mean, std, key, vals in sorted(summaries):
                print(f"  K(M) = {mean:.4f} ± {std:.4f}  (n={len(vals)})  <-  {group_names[key]}")

        if best is None:
            print("[sweep] no run produced an epiplexity measurement")
        else:
            print(f"\n[sweep] BEST (minimum epiplexity): '{best['run_name']}' "
                  f"with K(M) = {best['epiplexity']}")


class PFGridSweeper(GridSweeper):
    """GridSweeper that estimates epiplexity from a Pareto front, not just
    each run's own best step.

    The original notebooks (e.g. epiplexity/notebooks/chess_order.ipynb,
    `compute_lower_convex_hull`) pool (compute, K(M,X)) points across *every*
    run in the sweep and read epiplexity off the lower convex hull of that
    pooled set, rather than off any single run's own argmin — see the paper's
    "Estimating the Pareto Frontier" (Appendix B.1, ~pages 48-49) and Figure 2b
    /Figure 9. This subclass reproduces that: `_fetch_epiplexity` additionally
    caches each run's full per-step curve, and `_report` builds the pooled
    hull from those curves. `sweep()` itself is untouched — it only ever calls
    `self._fetch_epiplexity(...)` and `self._report(...)`, so overriding those
    two methods is enough.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # One entry per run with a usable eval step: (compute, K(M), K(M,X)).
        self._curves: list[list[tuple[float, float, float]]] = []
        # Parallel to _curves, one entry per run: identity needed to re-query
        # this run's Comet history later (epiaudio.analysis.pipeline.fetch_endpoint_losses).
        self._run_provenance: list[dict[str, Any]] = []

    def _fetch_epiplexity(
        self,
        experiment_key: str | None,
        test_tokens: int,
        train_student: bool,
        teacher_ema_active: bool,
        student_ema_active: bool,
    ) -> tuple[float | None, float | None]:
        """Same metric fetch as GridSweeper, plus caching the full curve.

        Pulls the same K(M)/loss pair GridSweeper._fetch_epiplexity does, but
        additionally pulls the "compute" metric train_torch.py already logs
        per step (`compute_spent = 6 * teacher_tokens_seen * num_params + 2 * test_tokens * num_params`),
        logged alongside K(M)/K(M)_req at eval time, so it shares their
        `step`), and caches every matched (compute, K(M), K(M,X)) point for
        this run in `self._curves` for `_report` to pool into a Pareto front,
        alongside this call's own arguments in `self._run_provenance`. Still
        returns this run's own best point, exactly like the base class, so
        the unmodified `sweep()` loop keeps working as before.
        """
        if experiment_key is None:
            return None, None

        import comet_ml

        api_experiment = comet_ml.APIExperiment(
            previous_experiment=experiment_key,
            api_key=os.environ.get("COMET_ML_API"),
        )

        if train_student:
            km_metric = "K(M)_req"
            loss_metric = "ema_student_eval_loss" if student_ema_active else "student_eval_loss"
        else:
            km_metric = "K(M)"
            loss_metric = "ema_teacher_eval_loss" if teacher_ema_active else "teacher_eval_loss"

        km_records = api_experiment.get_metrics(km_metric) or []
        loss_records = api_experiment.get_metrics(loss_metric) or []
        compute_records = api_experiment.get_metrics("compute") or []
        num_params_raw = get_parameter(api_experiment, "num_params")
        num_params = float(cast(str, num_params_raw))

        # Align by `timestamp`, not `step`: step is `student_step`, which never
        # advances when train_student is False, so every eval event in the run
        # would otherwise collide on the same step (see GridSweeper._fetch_epiplexity).
        km_by_ts = timestamped_values(km_records, scale=1e6)
        compute_by_ts = timestamped_values(compute_records)

        curve: list[tuple[float, float, float]] = []
        for r in loss_records:
            ts = r.get("timestamp")
            if ts is None:
                continue
            ts = int(ts)
            k_m = nearest_value(km_by_ts, ts)
            train_compute = nearest_value(compute_by_ts, ts)
            compute: float = cast(float, train_compute) + 2*num_params*test_tokens
            if k_m is None or compute is None:
                continue

            k_x_given_m = float(r["metricValue"]) * test_tokens / np.log(2)
            curve.append((compute, k_m, k_m + k_x_given_m))

        if not curve:
            return None, None

        self._curves.append(curve)
        self._run_provenance.append(
            {
                "experiment_key": experiment_key,
                "train_student": train_student,
                "teacher_ema_active": teacher_ema_active,
                "student_ema_active": student_ema_active,
            }
        )

        k_m_values = [k_m for _, k_m, _ in curve]
        code_lengths = [k_mx for _, _, k_mx in curve]
        min_idx = int(np.argmin(code_lengths))
        return k_m_values[min_idx], code_lengths[min_idx]

    @staticmethod
    def _lower_convex_hull(
        points: list[tuple[float, float, int]],
        tol: float = 0.0,
        reduce: str = "median",
    ) -> list[tuple[float, float, int]]:
        """Lower convex hull of (x, y, run_idx) points, minimizing y as x grows.

        Direct translation of chess_order.ipynb's `compute_lower_convex_hull`
        (monotone-chain lower hull, y non-increasing along the frontier within
        `tol` relative tolerance). `reduce` collapses hull points that belong
        to the same run down to one (its median-x point by default), so that
        one run's own trajectory can't supply multiple points on the frontier
        (paper Appendix B.1: "we still often observe that multiple checkpoints
        from a single training run appear on the Pareto frontier... we retain
        only the median point... per training run").
        """
        if len(points) < 2:
            return sorted(points, key=lambda p: p[0])

        sorted_points = sorted(points, key=lambda p: p[0])

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        hull: list[tuple[float, float, int]] = []
        for point in sorted_points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            if hull:
                min_y_so_far = min(p[1] for p in hull)
                if point[1] - min_y_so_far > tol * max(min_y_so_far, point[1]):
                    continue
            hull.append(point)

        by_run: dict[int, list[tuple[float, float, int]]] = {}
        for p in hull:
            by_run.setdefault(p[2], []).append(p)

        reduced: list[tuple[float, float, int]] = []
        for run_points in by_run.values():
            xs = [p[0] for p in run_points]
            if reduce == "median":
                idx = int(np.argmin([abs(x - np.median(xs)) for x in xs]))
            elif reduce == "min":
                idx = int(np.argmin(xs))
            elif reduce == "max":
                idx = int(np.argmax(xs))
            else:
                raise ValueError(f"Unknown reduce method: {reduce!r}")
            reduced.append(run_points[idx])
        return sorted(reduced, key=lambda p: p[0])

    def _report(self, best: dict[str, Any] | None) -> None:
        super()._report(best)

        pooled = [
            (compute, k_mx, run_idx)
            for run_idx, curve in enumerate(self._curves)
            for compute, _k_m, k_mx in curve
        ]
        if not pooled:
            print("\n[sweep] no data to build a Pareto front from")
            return

        hull = self._lower_convex_hull(pooled)
        km_lookup = {
            (compute, k_mx): k_m
            for curve in self._curves
            for compute, k_m, k_mx in curve
        }

        all_run_path, pareto_path = self._save_csvs(hull, km_lookup)

        print("\n[sweep] ===== Pareto front (compute vs K(M,X)) =====")
        for compute, k_mx, _run_idx in hull:
            k_m = km_lookup[(compute, k_mx)]
            print(f"  compute={compute:.4g}  K(M)={k_m:.6g}  K(M,X)={k_mx:.6g}")

        compute, k_mx, _run_idx = hull[-1]
        k_m = km_lookup[(compute, k_mx)]
        print(
            f"\n[sweep] Pareto-front epiplexity at max evaluated compute "
            f"({compute:.4g}): K(M) = {k_m}"
        )

        plot_path = self._plot_pareto_front(pooled, hull)
        config_path = self._save_sweep_config()
        bundle = SweepOutputBundle(
            sweep_name=self.sweep_name,
            all_run_csv=all_run_path,
            pareto_csv=pareto_path,
            plot=plot_path,
            sweep_config=config_path,
        )
        self._upload_summary_to_comet(bundle)

    def _build_df_from_data(
        self,
        data: list[tuple[int, float, float, float, float]]
    ) -> pd.DataFrame:
        columns = [
            "run_idx",
            "Compute",
            "Model Description Length: K(M)",
            "Data Given Model: K(X|M)",
            "Two-Part Code Length: K(M,X) = K(M) + K(X|M)"
        ]
        df = pd.DataFrame(data, columns=columns)

        return df

    def _build_all_run_df(self) -> pd.DataFrame:
        """Every pooled point, with the Comet provenance to re-query its run.

        Separate from `_build_df_from_data`: the pareto-front CSV built from
        that one is keyed by hull membership, not by run, so it has no
        column to attach per-run provenance to.
        """
        columns = [
            "run_idx",
            "Compute",
            "Model Description Length: K(M)",
            "Data Given Model: K(X|M)",
            "Two-Part Code Length: K(M,X) = K(M) + K(X|M)",
            "experiment_key",
            "train_student",
            "teacher_ema_active",
            "student_ema_active",
        ]
        rows = [
            (
                run_idx, compute, k_m, k_mx - k_m, k_mx,
                provenance["experiment_key"], provenance["train_student"],
                provenance["teacher_ema_active"], provenance["student_ema_active"],
            )
            for run_idx, (curve, provenance) in enumerate(zip(self._curves, self._run_provenance))
            for compute, k_m, k_mx in curve
        ]
        return pd.DataFrame(rows, columns=columns)

    def _save_csvs(
        self,
        hull: list[tuple[float, float, int]],
        km_lookup: dict[tuple[float, float], float]
    ) -> tuple[Path, Path]:
        pf_data = []
        for compute, k_mx, run_idx in hull:
            k_m = km_lookup[(compute, k_mx, )]
            pf_data.append((run_idx, compute, k_m, k_mx - k_m, k_mx))

        all_run_df = self._build_all_run_df()
        pf_df = self._build_df_from_data(pf_data)

        SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_run_out_path = SWEEP_OUTPUT_DIR / f"{self.sweep_name}__all_run_data.csv"
        pf_out_path = SWEEP_OUTPUT_DIR / f"{self.sweep_name}__pareto_front_data.csv"

        all_run_df.to_csv(all_run_out_path, index=False)
        pf_df.to_csv(pf_out_path, index=False)
        return all_run_out_path, pf_out_path

    def _save_sweep_config(self) -> Path:
        SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SWEEP_OUTPUT_DIR / f"{self.sweep_name}{CONFIG_SUFFIX}"
        OmegaConf.save(config=self.cfg, f=output_path)
        return output_path

    @staticmethod
    def _summary_parameter(cfg: DictConfig, key: str) -> Any:
        node = OmegaConf.select(cfg, f"parameters.{key}", default=None)
        if node is None:
            return None
        if "value" in node:
            return OmegaConf.select(node, "value", default=None)
        values = OmegaConf.select(node, "values", default=None)
        if values is None:
            return None
        return OmegaConf.to_container(values, resolve=True)

    def _summary_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "method": str(OmegaConf.select(self.cfg, "method", default="pf_grid")),
            "metric": str(OmegaConf.select(self.cfg, "metric.name", default="K(M)")),
            "backend": self.backend,
            "base_config": self.base_config_name,
            "sweep_seed": (
                self.sweep_seed
                if self.sweep_seed is not None
                else OmegaConf.select(self.cfg, "seed", default=0)
            ),
            "num_repeats": int(
                OmegaConf.select(self.cfg, "num_repeats", default=1)
            ),
        }
        for key in ("ds_path", "T", "T_eval", "B", "A", "model.N", "model.P"):
            value = self._summary_parameter(self.cfg, key)
            if value is not None:
                parameters[key] = value

        dataset_and_tokenizer = self._summary_dataset_and_tokenizer()
        if dataset_and_tokenizer is not None:
            parameters["dataset"], parameters["tokenizer"] = dataset_and_tokenizer
        return parameters
    
    def _summary_dataset_and_tokenizer(self) -> tuple[str, str] | None:
        ds_paths = self._summary_parameter(self.cfg, "ds_path")
        if isinstance(ds_paths, str):
            ds_path = ds_paths
        elif isinstance(ds_paths, list) and len(ds_paths) == 1:
            ds_path = str(ds_paths[0])
        else:
            return None

        path_name = Path(ds_path).name
        matches = [
            tokenizer
            for tokenizer in TOKENIZER_SPECS
            if path_name.endswith(f"_{tokenizer}")
        ]
        if not matches:
            return None
        tokenizer = max(matches, key=len)
        dataset = path_name.removesuffix(f"_{tokenizer}")
        return dataset, tokenizer

    def _upload_summary_to_comet(self, bundle: SweepOutputBundle) -> None:
        project_name = self._override_value(
            "wandb_project",
            self.command_overrides,
            self.extra_overrides,
        )
        if project_name is None:
            print(
                "[sweep] [WARNING]: aggregate outputs were saved locally but "
                "wandb_project could not be determined; skipping Comet summary upload."
            )
            return
        if not os.environ.get("COMET_ML_API"):
            print(
                "[sweep] [WARNING]: aggregate outputs were saved locally but "
                "COMET_ML_API is not set; skipping Comet summary upload."
            )
            return

        try:
            dataset_and_tokenizer = self._summary_dataset_and_tokenizer()
            tags = (
                list(dataset_and_tokenizer)
                if dataset_and_tokenizer is not None
                else []
            )
            experiment_key = upload_sweep_summary(
                bundle,
                project_name=project_name,
                workspace=COMET_WORKSPACE,
                parameters=self._summary_parameters(),
                tags=tags,
            )
            suffix = f" ({experiment_key})" if experiment_key else ""
            print(
                f"[sweep] uploaded aggregate outputs to Comet summary experiment"
                f"{suffix}"
            )
        except Exception as exc:
            # A reporting outage must not invalidate a completed GPU sweep; all
            # outputs remain available locally for the post-hoc uploader.
            print(
                "[sweep] [WARNING]: failed to upload aggregate outputs to Comet: "
                f"{exc!r}. Local files are intact; retry with "
                "`python -m epiaudio.upload_sweep_outputs`."
            )


    def _plot_pareto_front(
        self,
        pooled: list[tuple[float, float, int]],
        hull: list[tuple[float, float, int]],
    ) -> Path | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[sweep] skipping Pareto front plot (matplotlib unavailable: {exc!r})")
            return None

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            [p[0] for p in pooled], [p[1] for p in pooled],
            s=8, alpha=0.25, color="gray", label="all eval steps",
        )
        ax.plot(
            [p[0] for p in hull], [p[1] for p in hull],
            "o-", color="C0", label="Pareto front",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("compute (approx. 6ND + 2ND')")
        ax.set_ylabel("K(M,X) (bits)")
        ax.set_title(f"{self.sweep_name}: epiplexity Pareto front")
        ax.legend()
        fig.tight_layout()

        SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SWEEP_OUTPUT_DIR / f"{self.sweep_name}_pareto_front.png"
        try:
            fig.savefig(out_path, dpi=150)
            print(f"[sweep] saved Pareto front plot to {out_path}")
            return out_path
        except Exception as exc:
            print(f"[sweep] failed to save Pareto front plot: {exc!r}")
            return None
        finally:
            plt.close(fig)


def _build_sweeper(
    cfg: DictConfig,
    method: str,
    extra_overrides: list[str],
    backend: str = "torch",
    sweep_seed: int | None = None,
    sweep_name: str | None = None,
) -> Sweeper:
    method = (method or "grid").lower()
    if method == "grid":
        return GridSweeper(cfg, extra_overrides=extra_overrides, backend=backend, sweep_seed=sweep_seed, sweep_name=sweep_name)
    if method == "pf_grid":
        return PFGridSweeper(cfg, extra_overrides=extra_overrides, backend=backend, sweep_seed=sweep_seed, sweep_name=sweep_name)
    raise ValueError(f"Unsupported sweep method: {method!r} (choose 'grid' or 'pf_grid')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an epiplexity hyperparameter sweep.")
    parser.add_argument(
        "sweep_config",
        help="Path to a sweep YAML, e.g. epiplexity/picodo/sweeps/chess.yaml",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Sweep method override (default: the 'method' field in the YAML, else 'grid').",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override applied to every run (repeatable).",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "jax"],
        help=(
            "Training backend: 'torch' uses the PyTorch DDP trainer (default); "
            "'jax' uses the original JAX/Flax trainer."
        ),
    )
    parser.add_argument(
        "--sweep-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override the global sweep seed (overrides 'seed:' in the YAML). "
            "Used to derive independent per-repeat training seeds via SeedSequence. "
            "If omitted, the YAML 'seed:' field is used (default 0 if absent)."
        ),
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Resume a previous sweep by name: if experiments tagged with this sweep_name "
            "already exist in Comet, points whose hyperparameters match one exactly are "
            "skipped and their results pulled from Comet instead of retraining."
        ),
    )
    parser.add_argument(
        "--ds-path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Run only the resolved sweep for this dataset path. Repeat to select "
            "multiple paths; useful for assigning independent sweeps to GPU groups."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the sweep configuration and data without starting any runs.",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    all_cfgs = load_sweep_configs(args.sweep_config)
    if args.ds_path:
        selected_paths = set(args.ds_path)
        available_paths = {
            str(OmegaConf.select(cfg, "parameters.ds_path.values.0")) for cfg in all_cfgs
        }
        unknown_paths = selected_paths - available_paths
        if unknown_paths:
            parser.error(
                "--ds-path is not present in the sweep config: "
                + ", ".join(sorted(unknown_paths))
            )
        all_cfgs = [
            cfg
            for cfg in all_cfgs
            if str(OmegaConf.select(cfg, "parameters.ds_path.values.0"))
            in selected_paths
        ]
    for cfg in all_cfgs:
        try:
            method = args.method or str(OmegaConf.select(cfg, "method", default="grid"))
            preflight = validate_sweep(
                cfg,
                args.sweep_config,
                method=method,
                override_args=args.override,
                repo_root=REPO_ROOT,
            )
            print_preflight(preflight)
            if not preflight.ready or args.preflight_only:
                continue
            sweeper = _build_sweeper(cfg, method, args.override, backend=args.backend, sweep_seed=args.sweep_seed)
            sweeper.sweep()
        except Exception as exc:
            sweep_name = OmegaConf.select(cfg, "name", default="")
            print(f"Sweep '{sweep_name}' failed: {exc!r}")
            traceback.print_exc()
    end_time = datetime.now()
    total_runtime = (end_time - start_time).total_seconds() / 3600
    print(f"TOTAL RUNTIME: {total_runtime:.2f} hours")


if __name__ == "__main__":
    main()

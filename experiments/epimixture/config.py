"""Configuration models and durable JSON helpers for EpiMixture."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MixtureSpec:
    """One deterministic source-mixture recipe."""

    name: str
    weights: dict[str, float]
    train_samples: int
    test_samples: int


@dataclass(frozen=True)
class FixedGeneralizationSpec:
    """A pre-declared transformer and training regime for transfer tests.

    These fields are intentionally independent of the PF-grid result.  Dataset
    metadata is still used only to set the tokenizer's vocabulary and context
    length.
    """

    depth: int = 8
    width: int = 512
    head_dim: int = 64
    seed: int = 20_260_825
    train_tokens: int = 31_250_000
    batch_size: int = 256
    accumulation_steps: int = 8
    num_evals: int = 20
    eval_tokens: int = 131_072
    learning_rate: float = 2.0
    schedule: str = "const"
    warmup_tokens: int = 16_384
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    teacher_ema: float = 50.0
    compile: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated, path-resolved EpiMixture configuration."""

    config_path: Path
    name: str
    sources: tuple[str, ...]
    heldout: str
    tokenizer: str
    data_root: Path
    output_root: Path
    sweep_template: Path
    seed: int
    mixtures: tuple[MixtureSpec, ...]
    preparation: dict[str, int]
    preparation_sample_limit: int | None
    raw_roots: dict[str, Path]
    transfer_eval_tokens: int
    fixed_generalization: FixedGeneralizationSpec
    comet_sweeps: dict[str, str]

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return (*self.sources, *(mixture.name for mixture in self.mixtures))


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _as_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _as_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label} must be a non-negative number.")
    return float(value)


def _as_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _resolve_path(config_path: Path, value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    if not path.exists() and label == "sweep_template":
        repo_candidate = Path(__file__).resolve().parents[2] / value
        if repo_candidate.exists():
            path = repo_candidate
    return path.resolve()


def load_spec(config_path: str | Path) -> ExperimentSpec:
    """Load and validate a YAML experiment design without touching data."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ValueError(f"Experiment config does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config = _as_mapping(raw, "experiment config")

    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")
    sources_raw = config.get("sources")
    if not isinstance(sources_raw, list) or not 2 <= len(sources_raw) <= 5:
        raise ValueError("sources must contain two to five registered dataset names.")
    if any(not isinstance(source, str) or not source for source in sources_raw):
        raise ValueError("Every source must be a non-empty string.")
    sources = tuple(sources_raw)
    if len(set(sources)) != len(sources):
        raise ValueError("sources must not contain duplicates.")

    heldout = config.get("heldout")
    if not isinstance(heldout, str) or not heldout:
        raise ValueError("heldout must be one registered dataset name.")
    if heldout in sources:
        raise ValueError("heldout must not also be a training source.")

    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer:
        raise ValueError("tokenizer must be a registered discrete tokenizer name.")

    data_root_raw = config.get("data_root", "data")
    output_root_raw = config.get("output_root", "outputs")
    sweep_template_raw = config.get("sweep_template")
    paths = (data_root_raw, output_root_raw, sweep_template_raw)
    if not all(isinstance(value, str) and value for value in paths):
        raise ValueError("data_root, output_root, and sweep_template must be non-empty paths.")
    sweep_template = _resolve_path(path, str(sweep_template_raw), "sweep_template")
    if not sweep_template.is_file():
        raise ValueError(f"sweep_template does not exist: {sweep_template}")

    seed = config.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    mixtures_raw = config.get("mixtures")
    if not isinstance(mixtures_raw, list) or not mixtures_raw:
        raise ValueError("mixtures must contain at least one recipe.")
    mixtures: list[MixtureSpec] = []
    mixture_names: set[str] = set()
    for index, value in enumerate(mixtures_raw):
        mixture = _as_mapping(value, f"mixtures[{index}]")
        mixture_name = mixture.get("name")
        if not isinstance(mixture_name, str) or not mixture_name:
            raise ValueError(f"mixtures[{index}].name must be a non-empty string.")
        if mixture_name in mixture_names or mixture_name in sources:
            raise ValueError(f"Mixture name {mixture_name!r} is not unique.")
        mixture_names.add(mixture_name)
        weights_raw = _as_mapping(mixture.get("weights"), f"mixtures[{index}].weights")
        if set(weights_raw) != set(sources):
            raise ValueError(
                f"mixtures[{index}].weights must contain exactly the configured sources "
                f"({', '.join(sources)})."
            )
        weights: dict[str, float] = {}
        for source, weight in weights_raw.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError(f"Weight for {source!r} must be a positive number.")
            weights[str(source)] = float(weight)
        mixtures.append(
            MixtureSpec(
                name=mixture_name,
                weights=weights,
                train_samples=_as_positive_int(mixture.get("train_samples"), f"mixtures[{index}].train_samples"),
                test_samples=_as_positive_int(mixture.get("test_samples"), f"mixtures[{index}].test_samples"),
            )
        )

    preparation_raw = _as_mapping(config.get("preparation", {}), "preparation")
    preparation = {
        key: _as_positive_int(preparation_raw.get(key, default), f"preparation.{key}")
        for key, default in {"num_workers": 1, "batch_size": 8, "threads_per_worker": 4}.items()
    }
    sample_limit_raw = preparation_raw.get("max_samples_per_split")
    preparation_sample_limit = None if sample_limit_raw is None else _as_positive_int(
        sample_limit_raw, "preparation.max_samples_per_split"
    )
    raw_roots_raw = _as_mapping(config.get("raw_roots", {}), "raw_roots")
    allowed_root_names = set((*sources, heldout))
    unknown_roots = sorted(set(raw_roots_raw) - allowed_root_names)
    if unknown_roots:
        raise ValueError(f"raw_roots has unknown datasets: {', '.join(unknown_roots)}")
    raw_roots: dict[str, Path] = {}
    for dataset_name, root in raw_roots_raw.items():
        if not isinstance(root, str) or not root:
            raise ValueError(f"raw_roots.{dataset_name} must be a non-empty path.")
        raw_roots[str(dataset_name)] = _resolve_path(path, root, f"raw_roots.{dataset_name}")
    transfer_raw = _as_mapping(config.get("transfer", {}), "transfer")
    transfer_eval_tokens = _as_positive_int(transfer_raw.get("eval_tokens", 131_072), "transfer.eval_tokens")

    generalization_raw = _as_mapping(config.get("generalization", {}), "generalization")
    generalization_model = _as_mapping(generalization_raw.get("model", {}), "generalization.model")
    generalization_training = _as_mapping(generalization_raw.get("training", {}), "generalization.training")
    fixed_generalization = FixedGeneralizationSpec(
        depth=_as_positive_int(generalization_model.get("N", 8), "generalization.model.N"),
        width=_as_positive_int(generalization_model.get("D", 512), "generalization.model.D"),
        head_dim=_as_positive_int(generalization_model.get("dh", 64), "generalization.model.dh"),
        seed=seed if generalization_training.get("seed") is None else _as_nonnegative_int(
            generalization_training["seed"], "generalization.training.seed"
        ),
        train_tokens=_as_positive_int(generalization_training.get("T", 31_250_000), "generalization.training.T"),
        batch_size=_as_positive_int(generalization_training.get("B", 256), "generalization.training.B"),
        accumulation_steps=_as_positive_int(generalization_training.get("A", 8), "generalization.training.A"),
        num_evals=_as_positive_int(generalization_training.get("num_evals", 20), "generalization.training.num_evals"),
        eval_tokens=_as_positive_int(generalization_training.get("T_eval", transfer_eval_tokens), "generalization.training.T_eval"),
        learning_rate=_as_nonnegative_float(generalization_training.get("lr", 2.0), "generalization.training.lr"),
        schedule=str(generalization_training.get("schedule", "const")),
        warmup_tokens=_as_positive_int(generalization_training.get("warmup_tokens", 16_384), "generalization.training.warmup_tokens"),
        beta1=_as_nonnegative_float(generalization_training.get("b1", 0.9), "generalization.training.b1"),
        beta2=_as_nonnegative_float(generalization_training.get("b2", 0.95), "generalization.training.b2"),
        weight_decay=_as_nonnegative_float(generalization_training.get("weight_decay", 0.0), "generalization.training.weight_decay"),
        teacher_ema=_as_nonnegative_float(generalization_training.get("teacher_ema", 50.0), "generalization.training.teacher_ema"),
        compile=bool(generalization_training.get("compile", False)),
    )
    if fixed_generalization.width % fixed_generalization.head_dim:
        raise ValueError("generalization.model.D must be divisible by generalization.model.dh.")

    comet_raw = _as_mapping(config.get("comet_sweeps", {}), "comet_sweeps")
    unknown_comet_sweeps = sorted(set(comet_raw) - set((*sources, *(mixture.name for mixture in mixtures))))
    if unknown_comet_sweeps:
        raise ValueError(f"comet_sweeps has unknown training datasets: {', '.join(unknown_comet_sweeps)}")
    comet_sweeps: dict[str, str] = {}
    for dataset_id, sweep_name in comet_raw.items():
        if not isinstance(sweep_name, str) or not sweep_name:
            raise ValueError(f"comet_sweeps.{dataset_id} must be a non-empty sweep name.")
        comet_sweeps[str(dataset_id)] = sweep_name
    return ExperimentSpec(
        config_path=path,
        name=name,
        sources=sources,
        heldout=heldout,
        tokenizer=tokenizer,
        data_root=_resolve_path(path, str(data_root_raw), "data_root"),
        output_root=_resolve_path(path, str(output_root_raw), "output_root"),
        sweep_template=sweep_template,
        seed=seed,
        mixtures=tuple(mixtures),
        preparation=preparation,
        preparation_sample_limit=preparation_sample_limit,
        raw_roots=raw_roots,
        transfer_eval_tokens=transfer_eval_tokens,
        fixed_generalization=fixed_generalization,
        comet_sweeps=comet_sweeps,
    )


def write_json(path: Path, content: Mapping[str, Any]) -> None:
    """Atomically persist an experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(content, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise

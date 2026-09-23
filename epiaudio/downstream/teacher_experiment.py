import os
import argparse
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS
from epiaudio.downstream.teacher_train_eval import train_and_evaluate
from epiaudio.downstream.teacher_training import TeacherTrainingConfig

try:
    from comet_ml import Experiment

    HAS_COMET = True
except ImportError:
    HAS_COMET = False
    Experiment = None

EPI_AUDIO_WORKSPACE = "epi-audio"
COMET_PROJECT = "conditional-epiplexity"
COMET_API_KEY_VAR = "COMET_ML_API"


def name_experiment(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = f"{base_name}__{timestamp}"

    return name


def dataset_and_tokenizer(ds_path: str) -> tuple[str, str] | None:
    """Infer prepare_audio's dataset/tokenizer pair from its output path."""
    path_name = Path(ds_path).name
    representation_name = path_name.split("_classify_", maxsplit=1)[0]
    for tokenizer in sorted(TOKENIZER_SPECS, key=len, reverse=True):
        suffix = f"_{tokenizer}"
        if representation_name.endswith(suffix):
            return representation_name.removesuffix(suffix), tokenizer
    return None


def conditional_experiment_tags(
    cfg: DictConfig,
    *,
    run_kind: str,
    extra_tags: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return stable, searchable tags for Experiment 2 Comet runs."""
    tags = ["experiment-2", "conditional-epiplexity", run_kind]
    sweep_name = OmegaConf.select(cfg, "sweep_name", default=None)
    if sweep_name:
        tags.append(str(sweep_name))
    configured_tag = OmegaConf.select(cfg, "tag", default=None)
    if configured_tag:
        tags.append(str(configured_tag))

    ds_path = OmegaConf.select(cfg, "ds_path", default=None)
    if ds_path:
        representation = dataset_and_tokenizer(str(ds_path))
        if representation is not None:
            tags.extend(representation)
    tags.extend(str(tag) for tag in extra_tags if str(tag))
    return tuple(dict.fromkeys(tags))


def _comet_api_key() -> str:
    load_dotenv()
    comet_api_key = os.environ.get(COMET_API_KEY_VAR)
    if not comet_api_key or comet_api_key == "<INSERT_API_KEY>":
        raise RuntimeError(f"{COMET_API_KEY_VAR} is not set.")
    return comet_api_key


def _experiment_key(experiment: Any) -> str | None:
    get_key = getattr(experiment, "get_key", None)
    if not callable(get_key):
        return None
    key = get_key()
    return str(key) if key else None


def teacher_experiment_parameters(cfg: DictConfig) -> dict[str, Any]:
    """Return the explicit config plus all resolved teacher-runtime defaults."""
    raw_parameters = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw_parameters, dict):
        raise TypeError("Teacher config must be a mapping.")
    return {
        **{str(key): value for key, value in raw_parameters.items()},
        **asdict(TeacherTrainingConfig.from_cfg(cfg)),
    }


def log_experiment(
    cfg: DictConfig,
    metrics: dict[str, float | int],
    *,
    extra_tags: Iterable[str] = (),
) -> str | None:
    """Log one completed teacher run and return its Comet experiment key."""
    ds_path = OmegaConf.select(cfg, "ds_path", default=None)
    base_name = OmegaConf.select(cfg, "name", default="")

    assert ds_path is not None, "Data path must be provided."
    parameters = teacher_experiment_parameters(cfg)
    if not HAS_COMET or Experiment is None:
        raise RuntimeError("Comet ML is required when logging is enabled.")

    experiment = Experiment(
        api_key=_comet_api_key(),
        project_name=COMET_PROJECT,
        workspace=EPI_AUDIO_WORKSPACE,
    )
    try:
        sweep_name = OmegaConf.select(cfg, "sweep_name", default=None)
        experiment.set_name(
            str(base_name) if sweep_name else name_experiment(base_name)
        )
        for tag in conditional_experiment_tags(
            cfg,
            run_kind="sweep-run" if sweep_name else "teacher-run",
            extra_tags=extra_tags,
        ):
            experiment.add_tag(tag)
        experiment.log_parameters(parameters)
        experiment.log_metrics(metrics)
        return _experiment_key(experiment)
    finally:
        experiment.end()


def log_sweep_summary(
    cfg: DictConfig,
    *,
    sweep_name: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float | int],
    summary_path: Path,
    extra_tags: Iterable[str] = (),
) -> str | None:
    """Upload one aggregate conditional-sweep summary to Comet."""
    if not HAS_COMET or Experiment is None:
        raise RuntimeError("Comet ML is required when logging is enabled.")
    experiment = Experiment(
        api_key=_comet_api_key(),
        project_name=COMET_PROJECT,
        workspace=EPI_AUDIO_WORKSPACE,
    )
    try:
        experiment.set_name(f"{sweep_name}__summary")
        for tag in conditional_experiment_tags(
            cfg,
            run_kind="sweep-summary",
            extra_tags=extra_tags,
        ):
            experiment.add_tag(tag)
        experiment.log_parameters(
            {"sweep_name": sweep_name, "source_sweep_name": sweep_name, **parameters}
        )
        experiment.log_metrics(dict(metrics))
        experiment.log_asset(file_data=str(summary_path), file_name=summary_path.name)
        return _experiment_key(experiment)
    finally:
        experiment.end()


def run_experiment(
    cfg: DictConfig,
    *,
    log_to_comet: bool = True,
) -> dict[str, float | int]:
    """Train one teacher, optionally log it, and return its metrics."""
    ds_path = OmegaConf.select(cfg, "ds_path", default=None)
    assert ds_path is not None, "Data path must be provided."

    metrics = train_and_evaluate(cfg)
    if log_to_comet and HAS_COMET:
        log_experiment(cfg, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional-label teacher.")
    parser.add_argument(
        "teacher_config",
        help=(
            "Path to a conditional-label teacher YAML, e.g. "
            "epiaudio/conditional_epiplexity/configs/teacher/"
            "syntheory_chords_encodec_midi_program_name.yaml"
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.teacher_config)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Teacher config must be a YAML mapping.")
    run_experiment(cfg)


if __name__ == "__main__":
    main()

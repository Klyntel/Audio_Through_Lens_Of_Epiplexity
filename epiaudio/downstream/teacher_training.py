"""Shared state and utilities for conditional-label teacher training."""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from epiaudio.conditional_epiplexity.prequential import (
    ConditionalPrequentialEstimator,
)
from epiaudio.downstream.audio_classification_dataset import (
    AudioClassificationDataset,
)
from epiaudio.downstream.audio_classifier import AudioClassifier


DEFAULT_SEED = 42
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TeacherTrainingConfig:
    """Validated, process-safe configuration for one teacher run."""

    ds_path: str
    checkpoint_path: str
    resume_from: str | None
    seed: int
    n_epochs: int
    batch_size: int
    accumulation_steps: int
    max_steps: int | None
    num_workers: int
    world_size: int
    num_layers: int
    embed_dim: int
    per_head_dim: int
    vocab_size: int
    seq_length: int
    embed_init_std: float
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    embed_lr_mult: float
    schedule: str
    decay_frac: float
    warmup_steps: int
    warmup_tokens: int
    teacher_ema: float
    amp_dtype: str
    save_every_steps: int
    num_evals: int
    eval_split: str
    eval_batch_size: int

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        *,
        seed: int = DEFAULT_SEED,
    ) -> TeacherTrainingConfig:
        legacy_keys = sorted({"A", "B", "model", "opt"}.intersection(cfg.keys()))
        if legacy_keys:
            raise ValueError(
                "Conditional teacher configs use spelled-out parameter names; "
                f"replace legacy keys: {', '.join(legacy_keys)}."
            )
        if "allow_data_repetition" in cfg:
            raise ValueError(
                "Conditional teacher training is strictly one-pass; remove "
                "allow_data_repetition. Use the classification training pipeline "
                "for multi-epoch performance experiments."
            )

        ds_path = str(cfg.get("ds_path", ""))
        checkpoint = cfg.get("checkpoint_path")
        if checkpoint is None:
            checkpoint = str(Path(ds_path) / "model_checkpoint.pt")
        resume = cfg.get("resume_from")
        if resume is True or resume == "default":
            resume = checkpoint
        max_steps = cfg.get("max_steps")

        config = cls(
            ds_path=ds_path,
            checkpoint_path=str(checkpoint),
            resume_from=str(resume) if resume else None,
            seed=int(cfg.get("seed", seed)),
            n_epochs=int(cfg.get("n_epochs", 1)),
            batch_size=int(cfg.get("batch_size", 256)),
            accumulation_steps=int(cfg.get("accumulation_steps", 1)),
            max_steps=int(max_steps) if max_steps is not None else None,
            num_workers=int(cfg.get("num_workers", 0)),
            world_size=int(cfg.get("world_size", 0)),
            num_layers=int(cfg.get("num_layers", 3)),
            embed_dim=int(cfg.get("embed_dim", 192)),
            per_head_dim=int(cfg.get("per_head_dim", 64)),
            vocab_size=int(cfg.get("vocab_size", 1024)),
            seq_length=int(cfg.get("seq_length", 512)),
            embed_init_std=float(cfg.get("embed_init_std", 0.1)),
            learning_rate=float(cfg.get("learning_rate", 2.0)),
            beta1=float(cfg.get("beta1", 0.9)),
            beta2=float(cfg.get("beta2", 0.95)),
            epsilon=float(cfg.get("epsilon", 1e-20)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            embed_lr_mult=float(cfg.get("embed_lr_mult", 1.0)),
            schedule=str(cfg.get("schedule", "const")),
            decay_frac=float(cfg.get("decay_frac", 0.0)),
            warmup_steps=int(cfg.get("warmup_steps", 0)),
            warmup_tokens=int(cfg.get("warmup_tokens", 0)),
            teacher_ema=float(cfg.get("teacher_ema", 50.0)),
            amp_dtype=str(cfg.get("amp_dtype", "bfloat16")),
            save_every_steps=int(cfg.get("save_every_steps", 0)),
            num_evals=int(cfg.get("num_evals", 1)),
            eval_split=str(cfg.get("eval_split", "test")),
            eval_batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 256))),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "per_head_dim": self.per_head_dim,
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "eval_batch_size": self.eval_batch_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.ds_path:
            raise ValueError("ds_path must be provided.")
        if self.n_epochs != 1:
            raise ValueError(
                "n_epochs must be exactly 1 for one-pass conditional teacher "
                "training. Use the classification training pipeline for "
                "multi-epoch performance experiments."
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set.")
        if self.num_workers < 0 or self.world_size < 0:
            raise ValueError("num_workers and world_size must be non-negative.")
        if self.embed_dim % self.per_head_dim:
            raise ValueError("embed_dim must be divisible by per_head_dim.")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1).")
        if self.weight_decay < 0 or self.embed_lr_mult <= 0:
            raise ValueError(
                "weight_decay must be non-negative and embed_lr_mult positive."
            )
        if not 0 <= self.decay_frac <= 1:
            raise ValueError("decay_frac must be in [0, 1].")
        if self.warmup_steps < 0 or self.warmup_tokens < 0:
            raise ValueError("Warmup budgets must be non-negative.")
        if self.teacher_ema < 0:
            raise ValueError("teacher_ema must be non-negative.")
        if self.amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float32, float16, or bfloat16.")
        if self.save_every_steps < 0:
            raise ValueError("save_every_steps must be non-negative.")
        if self.num_evals <= 0:
            raise ValueError("num_evals must be positive.")


@dataclass
class TrainingCounters:
    """Exact one-pass label and compute accounting."""

    examples_seen: int = 0
    conditioning_tokens_seen: int = 0
    label_tokens_seen: int = 0
    compute_tokens_seen: int = 0

    def update(self, num_examples: int, data: AudioClassificationDataset) -> None:
        self.examples_seen += num_examples
        self.conditioning_tokens_seen += (
            num_examples * data.conditioning_tokens_per_example
        )
        self.label_tokens_seen += num_examples * data.label_tokens_per_example
        self.compute_tokens_seen += num_examples * data.compute_tokens_per_example


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def resolve_ema_decay(value: float) -> float | None:
    if value == 0:
        return None
    if value < 1:
        return value
    # Match the ECA trainer's interpretation of values >= 1 as an EMA window.
    return math.exp(-1.0 / value)


def create_ema_model(model: nn.Module, decay: float | None) -> nn.Module | None:
    if decay is None:
        return None
    return copy.deepcopy(model).requires_grad_(False)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    source = unwrap_model(model)
    for ema_parameter, parameter in zip(
        ema_model.parameters(), source.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), source.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def create_teacher(
    config: TeacherTrainingConfig,
    num_classes: int,
) -> AudioClassifier:
    return AudioClassifier(
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        per_head_dim=config.per_head_dim,
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        num_classes=num_classes,
        embed_init_std=config.embed_init_std,
    )


def amp_settings(device: torch.device, name: str) -> tuple[bool, torch.dtype]:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    return device.type == "cuda" and dtype != torch.float32, dtype


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def _atomic_save(state: dict[str, Any], checkpoint_path: str) -> None:
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    *,
    checkpoint_path: str,
    config: TeacherTrainingConfig,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    micro_batches_in_epoch: int,
    global_step: int,
    counters: TrainingCounters,
    estimator: ConditionalPrequentialEstimator,
    metrics: dict[str, float | int] | None = None,
    reported_points: list[dict[str, float | int]] | None = None,
) -> None:
    estimator_state = estimator.state_dict()
    state = {
        "version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "model": unwrap_model(model).state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "trainer": {
            "epoch": epoch,
            "micro_batches_in_epoch": micro_batches_in_epoch,
            "global_step": global_step,
            "prequential_estimator": estimator_state,
            # Retain the original sufficient-statistic fields so checkpoints
            # remain readable by the teacher runtime introduced in PR #102.
            "label_nll_sum": estimator.online_label_nll_nats,
            "label_count": estimator.label_count,
        },
        "counters": asdict(counters),
        "rng": _rng_state(),
        "metrics": metrics,
        "reported_points": reported_points or [],
    }
    _atomic_save(state, checkpoint_path)


def load_checkpoint(
    path: str,
    *,
    config: TeacherTrainingConfig,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[
    int,
    int,
    int,
    TrainingCounters,
    ConditionalPrequentialEstimator,
    list[dict[str, float | int]],
]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported teacher checkpoint version in '{path}'.")
    saved_config = checkpoint["config"]
    for field in (
        "ds_path",
        "seed",
        "n_epochs",
        "batch_size",
        "accumulation_steps",
        "num_layers",
        "embed_dim",
        "per_head_dim",
        "vocab_size",
        "seq_length",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "embed_lr_mult",
        "schedule",
        "decay_frac",
        "warmup_steps",
        "warmup_tokens",
        "teacher_ema",
        "amp_dtype",
        "num_evals",
        "eval_split",
        "eval_batch_size",
    ):
        if saved_config.get(field) != getattr(config, field):
            raise ValueError(f"Checkpoint field '{field}' does not match this run.")
    model.load_state_dict(checkpoint["model"])
    if ema_model is not None:
        if checkpoint["ema_model"] is None:
            raise ValueError("Checkpoint does not contain the requested EMA teacher.")
        ema_model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint["rng"])
    trainer = checkpoint["trainer"]
    estimator_state = trainer.get("prequential_estimator")
    if estimator_state is None:
        estimator = ConditionalPrequentialEstimator(
            online_label_nll_nats=trainer["label_nll_sum"],
            label_count=trainer["label_count"],
        )
    else:
        if not isinstance(estimator_state, Mapping):
            raise ValueError(
                "Checkpoint prequential estimator state must be a mapping."
            )
        estimator = ConditionalPrequentialEstimator.from_state_dict(estimator_state)
    raw_points = checkpoint.get("reported_points", [])
    if not isinstance(raw_points, list) or any(
        not isinstance(point, Mapping) for point in raw_points
    ):
        raise ValueError("Checkpoint reported_points must be a list of mappings.")
    reported_points: list[dict[str, float | int]] = []
    for point in raw_points:
        numeric_point: dict[str, float | int] = {}
        for name, value in point.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("Checkpoint reported point metrics must be numeric.")
            numeric_point[str(name)] = value if isinstance(value, int) else float(value)
        reported_points.append(numeric_point)
    return (
        int(trainer["epoch"]),
        int(trainer["micro_batches_in_epoch"]),
        int(trainer["global_step"]),
        TrainingCounters(**checkpoint["counters"]),
        estimator,
        reported_points,
    )

"""Runtime for teacher-only conditional epiplexity training.

The teacher estimates ``p(Y | X)`` with ordinary multiclass cross-entropy.
Only the label contributes to NLL; the audio prefix is tracked as compute.
"""

from __future__ import annotations

import contextlib
import math
import os
import socket
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from epiaudio.conditional_epiplexity.curve import flatten_reported_points
from epiaudio.conditional_epiplexity.evaluation import (
    ConditionalClassificationMetrics,
    summarize_classification,
    summarize_label_prior,
)
from epiaudio.conditional_epiplexity.prequential import (
    ConditionalPrequentialEstimator,
)
from epiaudio.downstream.audio_classification_dataset import (
    AudioClassificationDataset,
    create_audio_classification_dataloader,
)
from epiaudio.downstream.teacher_training import (
    DEFAULT_SEED,
    TeacherTrainingConfig,
    TrainingCounters,
    amp_settings,
    create_ema_model,
    create_teacher,
    load_checkpoint,
    resolve_ema_decay,
    save_checkpoint,
    seed_everything,
    unwrap_model,
    update_ema,
)
from epiaudio.train_torch import _build_param_groups, get_scheduler


def _global_loss_stats(
    loss_sum: float,
    count: int,
    device: torch.device,
) -> tuple[float, int]:
    values = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())


@torch.inference_mode()
def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    num_classes: int,
    *,
    distributed: bool = False,
) -> ConditionalClassificationMetrics:
    model.eval()
    loss_sum = 0.0
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )
    for audio_tokens, labels in loader:
        audio_tokens = audio_tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(audio_tokens)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        loss_sum += float(loss.item())
        predictions = logits.argmax(dim=-1)
        indices = labels * num_classes + predictions
        confusion += torch.bincount(
            indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
    if distributed:
        global_loss = torch.tensor(loss_sum, dtype=torch.float64, device=device)
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        loss_sum = float(global_loss.item())
    return summarize_classification(
        loss_sum,
        tuple(tuple(int(value) for value in row) for row in confusion.cpu().tolist()),
    )


def _build_metrics(
    *,
    estimator: ConditionalPrequentialEstimator,
    raw_evaluation: ConditionalClassificationMetrics,
    ema_evaluation: ConditionalClassificationMetrics,
    eval_data: AudioClassificationDataset,
    model: nn.Module,
    counters: TrainingCounters,
    global_step: int,
    ema_enabled: bool,
) -> dict[str, float | int]:
    """Assemble the paper-aligned code-length and evaluation metrics."""
    estimate = estimator.estimate(raw_evaluation.label_nll_nats_per_label)
    label_prior = summarize_label_prior(eval_data.class_counts)
    if raw_evaluation.label_count != eval_data.num_examples:
        raise RuntimeError("Evaluation accounting does not match the dataset manifest.")
    if estimator.label_count != counters.label_tokens_seen:
        raise RuntimeError("Prequential label accounting does not match training counters.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    eval_compute_tokens = (
        eval_data.num_examples * eval_data.compute_tokens_per_example
    )
    training_flops = 6 * parameter_count * counters.compute_tokens_seen
    conditional_decode_flops = 2 * parameter_count * eval_compute_tokens
    selected_evaluation = ema_evaluation if ema_enabled else raw_evaluation
    two_part_code_nats = (
        estimate.conditional_epiplexity_nats + selected_evaluation.label_code_nats
    )
    metrics: dict[str, float | int] = {
        # Backward-compatible names. Cross-entropy is measured in nats.
        "train_label_nll": estimate.online_label_nll_nats_per_label,
        "raw_label_nll": raw_evaluation.label_nll_nats_per_label,
        "ema_label_nll": ema_evaluation.label_nll_nats_per_label,
        "optimizer_steps": global_step,
        **asdict(counters),
        # Prequential model code: Equation 8 with Appendix B.1's IID
        # held-out approximation for the final raw teacher's label code.
        "online_label_code_nats": estimate.online_label_code_nats,
        "online_label_code_bits": estimate.online_label_code_bits,
        "online_label_nll_nats_per_label": (
            estimate.online_label_nll_nats_per_label
        ),
        "online_label_nll_bits_per_label": (
            estimate.online_label_nll_bits_per_label
        ),
        "reference_label_code_nats": estimate.reference_label_code_nats,
        "reference_label_code_bits": estimate.reference_label_code_bits,
        "reference_label_nll_nats_per_label": (
            estimate.reference_label_nll_nats_per_label
        ),
        "reference_label_nll_bits_per_label": (
            estimate.reference_label_nll_bits_per_label
        ),
        "conditional_epiplexity_nats": estimate.conditional_epiplexity_nats,
        "conditional_epiplexity_bits": estimate.conditional_epiplexity_bits,
        "conditional_epiplexity_bits_per_label": (
            estimate.conditional_epiplexity_bits_per_label
        ),
        "conditional_epiplexity_valid": int(
            estimate.conditional_epiplexity_nats >= 0.0
        ),
        # Use the EMA teacher for the entropy and data-code terms when enabled,
        # matching the existing sweep convention. The raw teacher remains the
        # reference model for the prequential estimate above.
        "conditional_entropy_estimate_nats_per_label": (
            selected_evaluation.label_nll_nats_per_label
        ),
        "conditional_entropy_estimate_bits_per_label": (
            selected_evaluation.label_nll_bits_per_label
        ),
        "conditional_entropy_estimate_code_nats": selected_evaluation.label_code_nats,
        "conditional_entropy_estimate_code_bits": selected_evaluation.label_code_bits,
        "two_part_code_nats": two_part_code_nats,
        "two_part_code_bits": two_part_code_nats / math.log(2.0),
        # Marginal baselines clarify how much label uncertainty exists before
        # conditioning on audio. They are not conditional-entropy estimates.
        "label_prior_entropy_nats_per_label": (
            label_prior.entropy_nats_per_label
        ),
        "label_prior_entropy_bits_per_label": (
            label_prior.entropy_bits_per_label
        ),
        "label_prior_code_nats": label_prior.entropy_code_nats,
        "label_prior_code_bits": label_prior.entropy_code_bits,
        "uniform_label_nll_nats_per_label": (
            label_prior.uniform_nll_nats_per_label
        ),
        "uniform_label_nll_bits_per_label": (
            label_prior.uniform_nll_bits_per_label
        ),
        "uniform_label_code_nats": label_prior.uniform_code_nats,
        "uniform_label_code_bits": label_prior.uniform_code_bits,
        # Section 4's 6ND training and 2ND conditional decoding estimates. D
        # includes both conditioning audio tokens and the one label token.
        "model_parameters": parameter_count,
        "eval_compute_tokens": eval_compute_tokens,
        "training_flops_estimate": training_flops,
        "conditional_decode_flops_estimate": conditional_decode_flops,
        "two_part_compute_flops_estimate": (
            training_flops + conditional_decode_flops
        ),
        "ema_enabled": int(ema_enabled),
    }
    metrics.update(raw_evaluation.as_dict("raw"))
    metrics.update(ema_evaluation.as_dict("ema"))
    return metrics


def _evaluate_report(
    *,
    model: nn.Module,
    ema_model: nn.Module | None,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_data: AudioClassificationDataset,
    estimator: ConditionalPrequentialEstimator,
    counters: TrainingCounters,
    global_step: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    distributed: bool,
) -> dict[str, float | int]:
    raw_model = unwrap_model(model)
    raw_evaluation = _evaluate_classifier(
        raw_model,
        loader,
        device,
        amp_enabled,
        amp_dtype,
        eval_data.num_classes,
        distributed=distributed,
    )
    ema_evaluation = (
        _evaluate_classifier(
            ema_model,
            loader,
            device,
            amp_enabled,
            amp_dtype,
            eval_data.num_classes,
            distributed=distributed,
        )
        if ema_model is not None
        else raw_evaluation
    )
    metrics = _build_metrics(
        estimator=estimator,
        raw_evaluation=raw_evaluation,
        ema_evaluation=ema_evaluation,
        eval_data=eval_data,
        model=raw_model,
        counters=counters,
        global_step=global_step,
        ema_enabled=ema_model is not None,
    )
    model.train()
    return metrics


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _local_examples_per_epoch(
    data: AudioClassificationDataset,
    world_size: int,
) -> int:
    remainder = data.num_examples % world_size
    if remainder:
        raise ValueError(
            f"Train split size ({data.num_examples}) must be divisible by "
            f"world_size ({world_size}); otherwise DDP would omit {remainder} "
            "examples and make the one-pass result hardware-dependent. This "
            "trainer never drops samples; preprocess to a fixed multiple or "
            "choose a compatible world_size."
        )
    return data.num_examples // world_size


def _is_checkpoint_step(save_every_steps: int, global_step: int) -> bool:
    return bool(save_every_steps and global_step % save_every_steps == 0)


def _evaluation_steps(
    target_steps: int,
    warmup_steps: int,
    num_evals: int,
) -> frozenset[int]:
    """Match Experiment 1's reported-step schedule for multi-point curves."""
    if num_evals == 1:
        return frozenset({target_steps})
    linear_start = min(warmup_steps, target_steps - 1)
    linear = np.linspace(linear_start, target_steps - 1, num_evals)
    geometric = np.geomspace(1, target_steps, 50)
    return frozenset(
        int(step)
        for step in (*linear, *geometric, target_steps)
        if 1 <= int(step) <= target_steps
    )


def _run_training(
    rank: int,
    world_size: int,
    config: TeacherTrainingConfig,
) -> dict[str, float | int] | None:
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if is_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    seed_everything(config.seed)

    try:
        train_data = AudioClassificationDataset(
            "train",
            Path(config.ds_path) / "train.bin",
            Path(config.ds_path) / "metadata.json",
        )
        eval_data = AudioClassificationDataset(
            config.eval_split,
            Path(config.ds_path) / f"{config.eval_split}.bin",
            Path(config.ds_path) / "metadata.json",
        )
        if train_data.conditioning_tokens_per_example != config.seq_length:
            raise ValueError(
                "Configured seq_length does not match the classification manifest."
            )
        local_examples_per_epoch = _local_examples_per_epoch(train_data, world_size)
        denominator = world_size * config.accumulation_steps
        if config.batch_size % denominator:
            raise ValueError(
                f"Global batch_size ({config.batch_size}) must be divisible by "
                f"world_size * accumulation_steps ({denominator})."
            )
        microbatch_size = config.batch_size // denominator
        loader = create_audio_classification_dataloader(
            train_data,
            batch_size=microbatch_size,
            shuffle=True,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        eval_dataset = (
            Subset(eval_data, range(rank, eval_data.num_examples, world_size))
            if is_distributed
            else eval_data
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        steps_per_epoch = math.ceil(len(loader) / config.accumulation_steps)
        available_steps = steps_per_epoch
        target_steps = min(config.max_steps or available_steps, available_steps)
        if target_steps == 0:
            raise ValueError("Training configuration produces zero optimizer steps.")

        raw_model = create_teacher(config, train_data.num_classes).to(device)
        decay = resolve_ema_decay(config.teacher_ema)
        ema_model = create_ema_model(raw_model, decay)
        param_groups = _build_param_groups(
            raw_model,
            config.learning_rate,
            config.embed_dim,
            config.embed_lr_mult,
            config.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            fused=device.type == "cuda",
        )
        amp_enabled, amp_dtype = amp_settings(device, config.amp_dtype)
        scaler = GradScaler(
            device.type,
            enabled=amp_enabled and amp_dtype == torch.float16,
        )
        warmup_steps = config.warmup_steps
        if not warmup_steps and config.warmup_tokens:
            tokens_per_step = config.batch_size * train_data.compute_tokens_per_example
            warmup_steps = math.ceil(config.warmup_tokens / tokens_per_step)
        schedule_fn = get_scheduler(
            config.schedule,
            config.decay_frac,
            warmup_steps,
            available_steps,
        )
        evaluation_steps = _evaluation_steps(
            target_steps,
            warmup_steps,
            config.num_evals,
        )

        epoch = micro_batches_in_epoch = global_step = 0
        counters = TrainingCounters()
        estimator = ConditionalPrequentialEstimator()
        reported_points: list[dict[str, float | int]] = []
        if config.resume_from is not None:
            (
                epoch,
                micro_batches_in_epoch,
                global_step,
                counters,
                estimator,
                reported_points,
            ) = load_checkpoint(
                config.resume_from,
                config=config,
                model=raw_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
        reported_points = [
            point
            for point in reported_points
            if int(point.get("optimizer_steps", -1)) in evaluation_steps
        ]
        reported_steps = {
            int(point["optimizer_steps"])
            for point in reported_points
            if "optimizer_steps" in point
        }

        model: nn.Module = raw_model
        if is_distributed:
            model = DDP(raw_model, device_ids=[rank])
        model.train()

        state_epoch = epoch
        state_microbatch = micro_batches_in_epoch
        # A completed checkpoint stores epoch=1, making this range empty. Fresh
        # and partial checkpoints store epoch=0 and execute the sole data pass.
        for current_epoch in range(epoch, 1):
            sampler = getattr(loader, "sampler", None)
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(current_epoch)
            elif loader.generator is not None:
                loader.generator.manual_seed(config.seed + current_epoch)
            skipped = micro_batches_in_epoch if current_epoch == epoch else 0
            if skipped % config.accumulation_steps:
                raise ValueError(
                    "Checkpoint resumes inside a gradient-accumulation group."
                )
            local_loss_sum = 0.0
            local_count = 0
            for microbatch_index, (audio_tokens, labels) in enumerate(loader):
                if microbatch_index < skipped:
                    continue
                if global_step >= target_steps:
                    break

                position = microbatch_index % config.accumulation_steps
                group_start = microbatch_index - position
                group_size = min(
                    config.accumulation_steps,
                    len(loader) - group_start,
                )
                group_example_count = min(
                    group_size * microbatch_size,
                    local_examples_per_epoch - group_start * microbatch_size,
                )
                group_end = position + 1 == group_size
                if position == 0:
                    optimizer.zero_grad(set_to_none=True)
                    scale = schedule_fn(global_step)
                    for group in optimizer.param_groups:
                        group["lr"] = group["_base_lr"] * scale
                    local_loss_sum = 0.0
                    local_count = 0

                audio_tokens = audio_tokens.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                sync_context = (
                    model.no_sync()  # type: ignore[union-attr]
                    if is_distributed and not group_end
                    else contextlib.nullcontext()
                )
                with (
                    sync_context,
                    torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ),
                ):
                    logits = model(audio_tokens)
                    loss_sum = F.cross_entropy(
                        logits.float(),
                        labels,
                        reduction="sum",
                    )
                # Each microbatch contributes a loss sum divided by the total
                # examples in its accumulation group. Summing these gradients
                # therefore gives the group mean; dividing by accumulation_steps
                # again would under-scale it. Count-based normalization also
                # handles a short final group correctly.
                scaler.scale(loss_sum / group_example_count).backward()
                local_loss_sum += float(loss_sum.detach().item())
                local_count += labels.numel()

                if not group_end:
                    continue
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    assert decay is not None
                    update_ema(ema_model, model, decay)
                global_loss_sum, global_count = _global_loss_stats(
                    local_loss_sum,
                    local_count,
                    device,
                )
                estimator.update(global_loss_sum, global_count)
                counters.update(global_count, train_data)
                global_step += 1

                state_epoch = current_epoch
                state_microbatch = microbatch_index + 1
                if state_microbatch == len(loader):
                    state_epoch += 1
                    state_microbatch = 0
                if (
                    global_step in evaluation_steps
                    and global_step not in reported_steps
                ):
                    reported_points.append(
                        _evaluate_report(
                            model=model,
                            ema_model=ema_model,
                            loader=eval_loader,
                            eval_data=eval_data,
                            estimator=estimator,
                            counters=counters,
                            global_step=global_step,
                            device=device,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            distributed=is_distributed,
                        )
                    )
                    reported_steps.add(global_step)
                checkpoint_step = _is_checkpoint_step(
                    config.save_every_steps,
                    global_step,
                )
                if rank == 0 and checkpoint_step:
                    save_checkpoint(
                        checkpoint_path=config.checkpoint_path,
                        config=config,
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=state_epoch,
                        micro_batches_in_epoch=state_microbatch,
                        global_step=global_step,
                        counters=counters,
                        estimator=estimator,
                        reported_points=reported_points,
                    )
                if is_distributed and checkpoint_step:
                    dist.barrier()
            micro_batches_in_epoch = 0
            if global_step >= target_steps:
                break

        if global_step not in reported_steps:
            reported_points.append(
                _evaluate_report(
                    model=model,
                    ema_model=ema_model,
                    loader=eval_loader,
                    eval_data=eval_data,
                    estimator=estimator,
                    counters=counters,
                    global_step=global_step,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    distributed=is_distributed,
                )
            )
        if is_distributed:
            dist.barrier()
            model = unwrap_model(model)
            dist.destroy_process_group()
        metrics: dict[str, float | int] | None = None
        if rank == 0:
            metrics = dict(reported_points[-1])
            metrics.update(flatten_reported_points(reported_points))
            save_checkpoint(
                checkpoint_path=config.checkpoint_path,
                config=config,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=state_epoch,
                micro_batches_in_epoch=state_microbatch,
                global_step=global_step,
                counters=counters,
                estimator=estimator,
                metrics=metrics,
                reported_points=reported_points,
            )
        return metrics
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def train_and_evaluate(
    cfg: DictConfig,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int]:
    """Train one conditional-label teacher and return NLL/accounting metrics."""
    config = TeacherTrainingConfig.from_cfg(cfg, seed=seed)
    available_gpus = torch.cuda.device_count()
    world_size = config.world_size or max(available_gpus, 1)
    if world_size > 1 and available_gpus < world_size:
        raise ValueError(
            f"Requested world_size={world_size}, but only {available_gpus} CUDA "
            "devices are available."
        )

    if world_size == 1:
        metrics = _run_training(0, 1, config)
        assert metrics is not None
        return metrics

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_find_free_port())
    spawn(
        _run_training,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )
    checkpoint = torch.load(
        config.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Training checkpoint does not contain final metrics.")
    return metrics

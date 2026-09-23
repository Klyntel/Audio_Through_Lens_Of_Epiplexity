"""PyTorch training script for epiaudio — drop-in replacement for train.py.

Replicates the exact training behaviour of the JAX/Flax train.py:
  - Teacher-only or teacher + student (KL-distillation) training
  - Gradient accumulation, EMA models, KL-gating
  - Identical epiplexity metrics: K(X), K(X|M), K(M), K(M)_req
  - Same Comet-ML logging keys
  - Checkpoint save/load (local or GCS)

Performance upgrades over the JAX version (following MosaicML Composer
recommendations — https://github.com/mosaicml/composer):
  - Flash Attention 2 via torch.nn.functional.scaled_dot_product_attention
  - BF16 mixed precision via torch.amp.autocast
  - Fused AdamW optimizer (torch.optim.AdamW fused=True)
  - torch.compile for teacher and student forward passes
  - Multi-GPU data parallelism via PyTorch DDP (spawned automatically)

The public entry point is ``train_and_evaluate(cfg)`` with the same signature
and return value as the JAX version, so ``sweep.py`` can call it unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import pickle
import socket
import subprocess
import tempfile
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

try:
    from comet_ml import Experiment

    HAS_COMET = True
except ImportError:
    HAS_COMET = False
    Experiment = None  # type: ignore[assignment, misc]  # HAS_COMET guards all uses

import epiaudio.model_torch as model_lib

# ---------------------------------------------------------------------------
# Pure-Python / numpy utilities (no JAX dependency)
# ---------------------------------------------------------------------------

## HUMAN NOTE: Rewriting this done simply because we wanted to pull
## It out of jax based  files? Seems like a waste.

## HN stands for HUMAN NOTE.


def _flatten_dict(d: Any, prefix: str | None = None) -> dict:
    from collections.abc import Mapping

    if isinstance(d, Mapping):
        out: dict = {}
        for k, v in d.items():
            nested = k if prefix is None else f"{prefix}/{k}"
            out |= _flatten_dict(v, nested)
        return out
    return {prefix: d}

# It was noted by Ethan that warmup in og code is illtyped
# At times a float is used, others its a int
# impacts lr schedule. 
def get_scheduler(schedule: str, decay_frac: float, warmup: float, total_steps: int):
    """Pure-Python version of picodo's get_scheduler — same logic, no JAX.

    HN [Sean] Verified.
    """
    decay_frac = float(decay_frac)

    def base_fn(t: float) -> float:
        if schedule == "const" or decay_frac == 0.0:
            return 1.0
        if schedule == "linear":
            t_norm = max(0.0, (t - (1.0 - decay_frac)) / decay_frac)
            return max(0.0, 1.0 - t_norm)
        raise ValueError(f"Unknown schedule: {schedule!r}")

    if warmup > 0:
        denom = max(total_steps - warmup, 1)

        ## HN: replaces jnp.where function
        def schedule_fn(t: int) -> float:
            tf = float(t)
            if tf < warmup:
                return tf / warmup
            return base_fn((tf - warmup) / denom)
    else:

        def schedule_fn(t: int) -> float:
            return base_fn(float(t) / total_steps)

    return schedule_fn


def make_ds_loader(
    ds_path: str,
    split: str,
    seq_len: int,
    batch_size: int,
    bos_id: int = 0,
    rank: int = 0,
    world_size: int = 1,
    dtype: DTypeLike = np.int16,
):
    """Rank-aware numpy data loader.  Matches picodo/data.py: make_ds_loader.

    Each rank loads ``batch_size // world_size`` samples starting at the
    appropriate offset within the global batch, so all ranks together cover
    the same token stream as the single-process JAX version.

    ``dtype`` must match how prepare_audio.py wrote the .bin file (see
    TOKENIZERS in that module) — e.g. SQCodec's vocab of 117,649 exceeds
    int16's range, so it's written as int32 and must be read back as such,
    unlike encodec/DAC which fit in int16.

    HN:  Assumes data is already shuffled. TODO review our data for shuffling.
    Otherwise equal to og, watch typing with tokens..

    NEW ADDITION: Mutliprocess batching
    """
    data_map = np.memmap(f"{ds_path}/{split}.bin", dtype=dtype, mode="r")
    n_tokens = len(data_map)

    assert batch_size % world_size == 0, (
        f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
    )
    local_bs = batch_size // world_size

    def get_batch(idx: int):
        d = np.memmap(f"{ds_path}/{split}.bin", dtype=dtype, mode="r")
        max_idx = n_tokens // (batch_size * seq_len)
        idx = idx % max_idx

        global_start = batch_size * seq_len * idx
        rank_offset = rank * local_bs * seq_len
        start_idx = global_start + rank_offset + seq_len * np.arange(local_bs)
        token_idx = start_idx[:, None] + np.arange(seq_len)[None, :]
        tokens = d[token_idx].astype(np.int64)

        bos_col = np.full((local_bs, 1), bos_id, dtype=tokens.dtype)
        tokens = np.concatenate([bos_col, tokens[:, :-1]], axis=1)
        mask = np.ones_like(tokens, dtype=bool)
        return tokens, mask

    return get_batch, n_tokens


class _SplitTooSmallError(ValueError):
    """A dataset split has fewer tokens than one full sequence, so no batch of any
    size can be built from it. Kept distinct from plain ValueError so callers can
    treat this specific, expected condition (fall back to another split) without
    also swallowing unrelated errors, e.g. a corrupted .bin file raising numpy's
    own ValueError for a byte count that isn't a multiple of the dtype's itemsize.
    """


def _clamped_eval_batch_size(
    ds_path: str,
    split: str,
    seq_len: int,
    requested_batch_size: int,
    dtype: DTypeLike,
) -> int:
    """Shrink an eval batch size to fit a split too small for `requested_batch_size`.

    make_ds_loader's get_batch computes `max_idx = n_tokens // (batch_size * seq_len)`;
    a split with fewer than one full batch (max_idx == 0) crashes with ZeroDivisionError
    the first time a batch is fetched. Evaluating with a smaller batch just makes the
    estimate noisier, an acceptable tradeoff for eval that training batch size shouldn't
    silently take on, so this is applied only at eval call sites, not inside
    make_ds_loader itself.
    """
    n_tokens = len(np.memmap(f"{ds_path}/{split}.bin", dtype=dtype, mode="r"))
    max_samples = n_tokens // seq_len
    if max_samples < 1:
        raise _SplitTooSmallError(
            f"'{split}' split at {ds_path} has only {n_tokens} tokens, fewer than one "
            f"sequence of length {seq_len}; cannot evaluate on it."
        )
    return min(requested_batch_size, max_samples)


def make_downstream_ds_loader(
    ds_path: str,
    split: str,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    dtype: DTypeLike = np.int16,
):
    """Downstream dataset loader with optional mask file.

    ``dtype`` must match how prepare_audio.py wrote the .bin file — see
    make_ds_loader's docstring.

    HN: Verified with batching
    """
    data_map = np.memmap(f"{ds_path}/{split}.bin", dtype=dtype, mode="r")
    n_tokens = len(data_map)
    has_mask = os.path.exists(f"{ds_path}/{split}_mask.bin")
    if not has_mask:
        print(f"No mask found for {ds_path}/{split}, using ones")

    assert batch_size % world_size == 0  # HN: Where does this assert come from?
    local_bs = batch_size // world_size

    def get_batch(idx: int):
        d = np.memmap(f"{ds_path}/{split}.bin", dtype=dtype, mode="r")
        msk = (
            np.memmap(f"{ds_path}/{split}_mask.bin", dtype=np.bool_, mode="r")
            if has_mask
            else np.ones(n_tokens, dtype=bool)
        )
        max_idx = n_tokens // (batch_size * seq_len)
        idx = idx % max_idx

        global_start = batch_size * seq_len * idx
        rank_offset = rank * local_bs * seq_len
        start_idx = global_start + rank_offset + seq_len * np.arange(local_bs)
        token_idx = start_idx[:, None] + np.arange(seq_len)[None, :]
        tokens = d[token_idx].astype(np.int64)
        masks = msk[token_idx]
        return tokens, masks

    return get_batch, n_tokens


def get_in_out(batch: tuple | np.ndarray, device: torch.device, pad_id: int = 0):
    """PyTorch version of picodo/data.py: get_in_out.

    HN: Verified
    """
    if isinstance(batch, tuple):
        x_np, mask_np = batch
    else:
        x_np, mask_np = batch, None

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    pad_col = torch.full((x.shape[0], 1), pad_id, dtype=x.dtype, device=device)
    y = torch.cat([x[:, 1:], pad_col], dim=1)

    if mask_np is not None:
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        false_col = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=device)
        weights = torch.cat([mask[:, 1:], false_col], dim=1).float()
    else:
        weights = (y != pad_id).float()

    return x, y, weights


# ---------------------------------------------------------------------------
# GCS utilities (copied from picodo/train.py — no JAX dependency)
# HN: Basically taken as is with small changes made to code style.
# We could remove this since we don't use GCS.
# ---------------------------------------------------------------------------


def _is_gcs_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("gs://")


def _run_gsutil_cmd(args: list[str]):
    cmd = ["gsutil"] + args
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("gsutil not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gsutil failed: {' '.join(cmd)}\n{exc.stderr}") from exc


def _gcs_path_exists(path: str) -> bool:
    try:
        result = subprocess.run(
            ["gsutil", "-q", "stat", path], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError as exc:
        raise RuntimeError("gsutil not found on PATH.") from exc


def _download_gcs_to_temp(path: str) -> str:
    suffix = os.path.splitext(path)[-1] or ".pkl"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    _run_gsutil_cmd(["cp", path, tmp.name])
    return tmp.name


def _upload_temp_to_gcs(local_path: str, gcs_path: str):
    _run_gsutil_cmd(["cp", local_path, gcs_path])


# ---------------------------------------------------------------------------
# MuP parameter groups (mirrors _compute_mup_scales from picodo)
# ---------------------------------------------------------------------------


def _build_param_groups(
    model: nn.Module,
    base_lr: float,
    D: int,
    embed_lr_mult: float,
    weight_decay: float,
) -> list[dict]:
    """Build AdamW parameter groups with MuP learning-rate scaling.

    Mirrors picodo's _compute_mup_scales logic:
      - ndim < 2 or 'embed' in name  →  base_lr * embed_lr_mult
      - all other weight matrices     →  base_lr / D
    Stores '_base_lr' for runtime schedule scaling.

    HN: Also looks good. Seems equlivent.
    """
    embed_params, weight_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embed" in name:
            embed_params.append(param)
        else:
            weight_params.append(param)

    groups = []
    if embed_params:
        lr = base_lr * embed_lr_mult
        groups.append(
            {
                "params": embed_params,
                "lr": lr,
                "_base_lr": lr,
                "weight_decay": weight_decay,
            }
        )
    if weight_params:
        lr = base_lr / D
        groups.append(
            {
                "params": weight_params,
                "lr": lr,
                "_base_lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return groups


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _ce_loss(
    logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Sequence cross-entropy matching picodo's loss_fn.

    HN: Looks equal to _student_ce_loss... but name was changed...
    """
    logits = logits.float()
    losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none"
    )
    losses = losses.view(y.shape)  # [B, L]
    per_seq = (losses * weights).sum(-1) / (weights.sum(-1) + 1e-6)
    return per_seq.mean()


def _kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """KL divergence matching picodo's distill_loss_and_grads.

    Computed on positions [:, :-1] (shift by 1, like the JAX version).
    Teacher logits are stop-gradiented; student logits are detached here
    so KL does NOT contribute to student gradients (matching JAX behaviour
    where only CE is differentiated).
    """
    teacher_logits = teacher_logits.float().detach()[:, :-1, :]
    student_logits = student_logits.float().detach()[:, :-1, :]
    weights = weights[:, :-1]

    t_log_p = F.log_softmax(teacher_logits, dim=-1)
    s_log_p = F.log_softmax(student_logits, dim=-1)
    per_tok_kl = (t_log_p.exp() * (t_log_p - s_log_p)).sum(-1)  # [B, L-1]
    per_seq_kl = (per_tok_kl * weights).sum(-1) / (weights.sum(-1) + 1e-6)
    return per_seq_kl.mean()


# ---------------------------------------------------------------------------
# EMA update (in-place on the EMA model; no grad tracking needed)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _update_ema(ema_model: nn.Module, src_model: nn.Module, decay: float) -> None:
    src = src_model.module if isinstance(src_model, DDP) else src_model
    for ema_p, src_p in zip(ema_model.parameters(), src.parameters()):
        ema_p.data.mul_(decay).add_(src_p.data, alpha=1.0 - decay)
    for ema_b, src_b in zip(ema_model.buffers(), src.buffers()):
        ema_b.copy_(src_b)


# ---------------------------------------------------------------------------
# Eval helpers
# HN: Verified Section
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_step(
    model: nn.Module,
    dataset: list,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict:
    mdl = model.module if isinstance(model, DDP) else model
    mdl.eval()
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
        x, _, _ = get_in_out(dataset[0], device=device)
         # custom model method, not on nn.Module stubs  # [B, L, D]
        features = mdl.get_features(x).float()  # type: ignore[attr-defined]

        total_loss = 0.0
        for batch in dataset:
            x, y, w = get_in_out(batch, device=device)
            logits = mdl(x)
            total_loss += _ce_loss(logits, y, w).item()

    mdl.train()
    return {
        "eval_loss": total_loss / len(dataset),
        "features": features,
    }


def _eval_features(
    features: torch.Tensor,
    prev_features: torch.Tensor,
) -> dict:
    with torch.no_grad():
        delta = features - prev_features
        return {
            "h": math.sqrt(float(features.pow(2).mean())),
            "dh": math.sqrt(float(delta.pow(2).mean())),
        }


# ---------------------------------------------------------------------------
# Downstream fine-tune + eval (full port of picodo's downstream_ft_eval)
# HN: LGTM verified.
# ---------------------------------------------------------------------------


def _downstream_eval_step(
    model: nn.Module, dataset: list, device: torch.device, amp_dtype: torch.dtype
) -> dict:
    mdl = model.module if isinstance(model, DDP) else model
    mdl.eval()
    total_loss = total_tok_acc = total_acc = total_cp = 0.0
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype):
        for batch in dataset:
            x, y, w = get_in_out(batch, device=device)
            logits = mdl(x).float()
            losses = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none"
            ).view(y.shape)
            per_seq = (losses * w).sum(-1) / (w.sum(-1) + 1e-6)
            total_loss += per_seq.mean().item()

            preds = logits.argmax(-1)
            tok_correct = (preds == y) | (w == 0)
            seq_correct = tok_correct.all(dim=-1)
            total_acc += seq_correct.float().mean().item()
            tacc = ((preds == y).float() * w).sum(-1) / (w.sum(-1) + 1e-6)
            total_tok_acc += tacc.mean().item()

            probs = F.softmax(logits, dim=-1)
            tok_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            masked = torch.where(w > 0, tok_probs, torch.ones_like(tok_probs))
            seq_p = masked.prod(dim=-1)
            total_cp += seq_p.mean().item()

    n = len(dataset)
    mdl.train()
    return {
        "down_loss": total_loss / n,
        "down_token_acc": total_tok_acc / n,
        "down_acc": total_acc / n,
        "down_correct_prob": total_cp / n,
    }


def _downstream_ft_eval(
    model: nn.Module,
    ds_train_downstream: list,
    ds_test_downstream: list,
    device: torch.device,
    mup_scales_base_lr: float,
    D: int,
    cfg: DictConfig,
    amp_dtype: torch.dtype,
) -> dict:
    """Fine-tune a copy of model on downstream task and report metrics.
    Matches picodo's downstream_ft_eval exactly.
    """

    # -----------------------------------------------------------
    # HN: 1) clone + optional re-initialisation
    # -----------------------------------------------------------

    src = model.module if isinstance(model, DDP) else model
    ft_model = copy.deepcopy(src).to(device)

    # Optional re-initialisation — use a fixed seed matching the original's
    # jax.random.PRNGKey(cfg.seed) so reinit is identical at every downstream
    # eval checkpoint, not dependent on how far the global RNG has advanced.
    embed_std = float(cfg.model.embed_init_std)
    if cfg.reinit_embed:
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(int(cfg.seed))
        # nn.Module.__getattr__ stubs return Tensor|Module; cast to Tensor for init
        nn.init.normal_(ft_model.embed.weight, std=embed_std)  # type: ignore[arg-type]
        nn.init.normal_(ft_model.pos_embed.weight, std=embed_std)  # type: ignore[arg-type]
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
    if cfg.reinit_readout:
        nn.init.zeros_(ft_model.readout.weight)  # type: ignore[arg-type]

    # -----------------------------------------------------------
    # HN: 2) scheduler + optimiser
    # -----------------------------------------------------------
    batch_size_ft = (
        int(cfg.B_ft) if getattr(cfg, "B_ft", None) is not None else int(cfg.B)
    )
    tokens_per_step_ft = batch_size_ft * cfg.model.L
    num_train_steps = max(1, cfg.T_downstream // tokens_per_step_ft)
    ft_sched = get_scheduler("linear", 1.0, num_train_steps * 0.1, num_train_steps)
    ft_lr = float(cfg.opt.lr) * float(cfg.opt.ft_lr_mult)

    # HN: TODO missing base transformation stack (used for the trainable subset)
    # HN: its possible that JAX treats optimizer like tranformations?
    # HN: This part needs extra review

    if cfg.linear_probing:
        probe_params = list(ft_model.readout.parameters())  # type: ignore[attr-defined]
        ft_opt = torch.optim.AdamW(
            [{"params": probe_params, "lr": ft_lr / D, "_base_lr": ft_lr / D}],
            betas=(cfg.opt.b1, cfg.opt.b2),
            eps=1e-20,
            fused=True,
        )
    else:
        groups = _build_param_groups(ft_model, ft_lr, D, cfg.opt.embed_lr_mult, 0.0)
        ft_opt = torch.optim.AdamW(
            groups, betas=(cfg.opt.b1, cfg.opt.b2), eps=1e-20, fused=True
        )

    ft_model.train()
    report_every = max(1, len(ds_train_downstream) // 10)
    running = 0.0

    for step_i, batch in enumerate(ds_train_downstream):
        scale = ft_sched(step_i)
        for g in ft_opt.param_groups:
            g["lr"] = g["_base_lr"] * scale

        ft_opt.zero_grad()
        with autocast(device_type="cuda", dtype=amp_dtype):
            x, y, w = get_in_out(batch, device=device)
            logits = ft_model(x)
            loss = _ce_loss(logits, y, w)  # HN: train loss metric in og
        loss.backward()
        ft_opt.step()
        running += loss.item()

        if (step_i + 1) % report_every == 0:
            pct = (step_i + 1) / len(ds_train_downstream) * 100
            avg = running / report_every
            print(
                f"FT step {step_i + 1}/{len(ds_train_downstream)} ({pct:.1f}%): loss={avg:.4f}"
            )
            running = 0.0

    metrics = _downstream_eval_step(ft_model, ds_test_downstream, device, amp_dtype)
    return {f"{k}_ft": v for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Distributed helpers
# HN: No mirror in og, required for pyorch
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _all_reduce_mean(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


# ---------------------------------------------------------------------------
# Core training logic (runs inside each DDP worker)
# ---------------------------------------------------------------------------


def _run_training(rank: int, world_size: int, cfg: DictConfig) -> tuple[str | None, int]:
    """Actual training loop; called in each DDP process (or directly for single-GPU)."""

    # ---- device FIRST, then distributed init (required for NCCL) -----------
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    is_main = rank == 0

    # ---- global RNG seeding (mirrors JAX: model uses cfg.seed, gen uses cfg.seed+1) --
    # All ranks use the same base seed so model weights are identical before DDP sync.
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Per-rank generation seed: spawn world_size independent seeds from seed+1
    # (mirrors JAX's distill_rng = PRNGKey(cfg.seed+1)). Each rank gets a
    # statistically uncorrelated seed so generate() produces distinct token
    # sequences across GPUs rather than identical copies.
    _gen_ss = np.random.SeedSequence(seed + 1)
    rank_gen_seed = int(_gen_ss.spawn(world_size)[rank].generate_state(1)[0])

    # ---- CUDA determinism settings ----------------------------------------
    # cuBLAS: fix workspace size so algorithm selection is stable across runs.
    # Must be set before any GEMM; env var takes effect process-wide.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    # Note: F.scaled_dot_product_attention (flash attention) backward uses
    # non-deterministic atomics that cudnn.deterministic does NOT cover.
    # Residual K(M) variance of ~1e-4 from SDPA atomics is expected on H100.
    # Set NCCL_DETERMINISTIC=1 in the environment to also fix all-reduce order
    # (significant throughput cost; not set here by default).

    # ---- amp / compile config ---------------------------------------------
    _dtype_map = {
        None: None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    explicit_dtype = _dtype_map.get(cfg.model.dtype, None)
    # Use AMP unless the model is explicitly cast to a non-float32 dtype
    amp_enabled = explicit_dtype is None or explicit_dtype == torch.float32
    amp_dtype = torch.bfloat16  # H100 native, recommended by Composer

    # ---- save / resume paths ----------------------------------------------
    # HN: Actually mirror train_and_evaluate here
    save_target = cfg.save
    save_path = None
    if isinstance(save_target, str):
        save_path = save_target
    elif save_target:
        save_path = (
            f"gs://wandb_model_checkpoints_us_central2/"
            f"{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T}/trained_torch.pt"
        )
        if is_main:
            print(f"cfg.save=True; defaulting to {save_path}")

    resume_path = cfg.resume_from
    if not resume_path:
        resume_path = None
    elif resume_path == "default":
        if cfg.T_ckpt is not None:
            if cfg.T <= 0:
                cfg.T = cfg.T_ckpt
            resume_path = (
                f"gs://wandb_model_checkpoints_us_central2/"
                f"{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T_ckpt}/trained_torch.pt"
            )
        else:
            resume_path = (
                f"gs://wandb_model_checkpoints_us_central2/"
                f"{cfg.ds_path}_N{cfg.model.N}_P{cfg.model.P}_T{cfg.T}/trained_torch.pt"
            )

    # HN: Looks good at this point.

    # ---- training mode flags ----------------------------------------------
    train_teacher_base = cfg.train_teacher
    train_student_base = cfg.train_student
    assert train_teacher_base or train_student_base, (
        "At least one of train_teacher or train_student must be True"
    )

    enforce_max_kl = train_teacher_base and train_student_base
    max_kl = float(cfg.max_kl) if enforce_max_kl else float("inf")
    if enforce_max_kl and max_kl <= 0:
        raise ValueError("max_kl must be a positive number")

    # HN: Looks good at this point.

    # ---- gradient accumulation --------------------------------------------
    accumulation_steps = int(cfg.A) if cfg.A is not None else 1
    if accumulation_steps < 1:
        raise ValueError("cfg.A must be >= 1")
    if int(cfg.B) % accumulation_steps != 0:
        raise ValueError(
            f"cfg.B ({cfg.B}) must be divisible by cfg.A ({accumulation_steps})"
        )

    teacher_batch_size = int(cfg.B)
    teacher_microbatch_size = teacher_batch_size // accumulation_steps
    ft_batch_size = (
        int(cfg.B_ft) if getattr(cfg, "B_ft", None) is not None else teacher_batch_size
    )

    # HN: Looks good at this point.

    # Validate DDP divisibility
    # HN: pytorch spefific here
    if is_ddp:
        assert teacher_microbatch_size % world_size == 0, (
            f"teacher_microbatch_size ({teacher_microbatch_size}) must be "
            f"divisible by world_size ({world_size})"
        )
        if ft_batch_size % world_size != 0:
            ft_batch_size = (ft_batch_size // world_size) * world_size

    # ---- data loaders -----------------------------------------------------
    # Matches prepare_audio.py's TOKENIZERS table, which writes signed int16
    # for vocabs that fit (max token id <= 32767) and int32 for anything
    # bigger (e.g. SQCodec's 117,649) — must read back with the same dtype.
    ds_dtype = np.int16 if cfg.model.V <= 32768 else np.int32
    get_batch_train_teacher, ds_train_size = make_ds_loader(
        cfg.ds_path,
        "train",
        cfg.model.L,
        teacher_microbatch_size,
        rank=rank,
        world_size=world_size,
        dtype=ds_dtype,
    )
    # Test set is only evaluated on rank 0, so load up to a cfg.B-sized batch there.
    # Using rank=0, world_size=1 gives each test batch shape [eval_batch_size, L],
    # matching the original JAX version which evaluated cfg.T_eval tokens total.
    # eval_batch_size may be smaller than cfg.B on a small split (see
    # _clamped_eval_batch_size) -- training batch size (teacher_microbatch_size,
    # above) is never adjusted this way.
    try:
        eval_batch_size = _clamped_eval_batch_size(cfg.ds_path, "test", cfg.model.L, cfg.B, dtype=ds_dtype)
        get_batch_test, ds_test_size = make_ds_loader(
            cfg.ds_path, "test", cfg.model.L, eval_batch_size, rank=0, world_size=1, dtype=ds_dtype,
        )
    except (FileNotFoundError, KeyError, _SplitTooSmallError) as e:
        if is_main:
            print(
                f"WARNING: no usable 'test' split at {cfg.ds_path} ({e!r}); "
                "falling back to 'val' split for test-set evaluation."
            )
        try:
            eval_batch_size = _clamped_eval_batch_size(cfg.ds_path, "val", cfg.model.L, cfg.B, dtype=ds_dtype)
            get_batch_test, ds_test_size = make_ds_loader(
                cfg.ds_path, "val", cfg.model.L, eval_batch_size, rank=0, world_size=1, dtype=ds_dtype,
            )
        except (FileNotFoundError, KeyError, _SplitTooSmallError) as e2:
            raise RuntimeError(
                f"No usable 'test' or 'val' split at {cfg.ds_path}. "
                f"test error: {e!r}; val error: {e2!r}"
            ) from e2

    if is_main:
        print(f"Train: {ds_train_size:.2g} tokens")
        print(f"Test:  {ds_test_size:.2g} tokens")

    # HN: Looks good at this point.
    # HN: After this point we have a divergence wtih model loading
    # HN: and no handle for CFG.T being none.
    # HN: TODO MORE REVIEW HERE

    # ---- token limits -----------------------------------------------------
    cfg.T = cfg.T or ds_train_size  # Matchs 140 in epiaudio/train.py
    cfg.T_eval = cfg.T_eval or ds_test_size

    # HN: no handle for cfg.model.p here
    # HN: no mesh loading [helper func takes care of it?]

    # HN: TODO missing 143 - 180 in epiaudio/train.py

    # ---- token counts per step --------------------------------------------
    train_tokens_per_step = teacher_batch_size * cfg.model.L
    eval_tokens_per_step = eval_batch_size * cfg.model.L
    num_train_steps = cfg.T // train_tokens_per_step
    num_test_steps = cfg.T_eval // eval_tokens_per_step
    assert num_test_steps > 0, "num_test_steps must be nonzero. Increase batch size or decrease batch size/sequence length."

    # HN: Above looks good

    # ---- pre-load test set ------------------------------------------------
    ds_test = [get_batch_test(i) for i in range(num_test_steps)]

    # ---- downstream dataset -----------------------------------------------
    ds_test_downstream = None
    ds_train_downstream = None
    num_test_steps_downstream = 0
    if cfg.downstream_ds_path is not None:
        get_batch_train_dn, ds_train_size_dn = make_downstream_ds_loader(
            cfg.downstream_ds_path,
            "train",
            cfg.model.L,
            ft_batch_size,
            rank=0,
            world_size=1,
            dtype=ds_dtype,
        )
        get_batch_test_dn, ds_test_size_dn = make_downstream_ds_loader(
            cfg.downstream_ds_path, "test", cfg.model.L, ft_batch_size, rank=0, world_size=1, dtype=ds_dtype,
        )
        cfg.T_eval_downstream = cfg.T_eval_downstream or ds_test_size_dn
        dn_tps = ft_batch_size * cfg.model.L
        num_test_steps_downstream = cfg.T_eval_downstream // dn_tps
        ds_test_downstream = [
            get_batch_test_dn(i) for i in range(num_test_steps_downstream)
        ]
        cfg.T_downstream = cfg.T_downstream or ds_train_size_dn
        num_train_steps_dn = cfg.T_downstream // dn_tps
        ds_train_downstream = [get_batch_train_dn(i) for i in range(num_train_steps_dn)]

        with open(f"{cfg.downstream_ds_path}/meta.pkl", "rb") as f:
            meta_dn = pickle.load(f)
        assert meta_dn["vocab_size"] <= cfg.model.V, (
            f"Downstream vocab {meta_dn['vocab_size']} > pre-trained vocab {cfg.model.V}"
        )
        if is_main:
            print(f"Downstream train: {ds_train_size_dn:.2g} tokens")
            print(f"Downstream test:  {ds_test_size_dn:.2g} tokens")
        if cfg.downstream_ds_path == "fen2cp":
            cfg.reinit_readout = True
            if is_main:
                print("Reinitializing readout for fen2cp")

    # HN: Above section addresses mesh distributed training
    # HN: diff because pytorch but overall looks equal in logic

    # ---- schedule ---------------------------------------------------------
    warmup_steps = cfg.opt.warmup_tokens // train_tokens_per_step
    schedule_fn = get_scheduler(
        cfg.opt.schedule, cfg.opt.decay_frac, warmup_steps, num_train_steps
    )

    # HN: Looks good

    # ---- teacher model ----------------------------------------------------
    teacher_model = model_lib.create_model(cfg.model).to(device)
    if explicit_dtype is not None and explicit_dtype != torch.float32:
        teacher_model = teacher_model.to(explicit_dtype)

    # Optional checkpoint resume
    if resume_path is not None:
        local_ckpt = resume_path
        tmp_ckpt = None
        if _is_gcs_path(resume_path):
            if not _gcs_path_exists(resume_path):
                raise FileNotFoundError(f"Checkpoint {resume_path} not on GCS")
            tmp_ckpt = _download_gcs_to_temp(resume_path)
            local_ckpt = tmp_ckpt
        if not os.path.exists(local_ckpt):
            raise FileNotFoundError(f"Checkpoint {local_ckpt} not found")
        try:
            state = torch.load(local_ckpt, map_location=device, weights_only=True)
            teacher_model.load_state_dict(state)
        finally:
            if tmp_ckpt and os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)
        if is_main:
            print(f"Loaded checkpoint from {resume_path}")

    num_params = sum(p.numel() for p in teacher_model.parameters())
    if is_main:
        print(f"Number of parameters: {num_params:.2g}")
        for k, v in cfg.items():
            print(f"{k}: {v}")

    # HN: loading model and counting parameters look fine.

    # Wrap with DDP first, then optionally torch.compile (PyTorch 2.x
    # recommended order). Some remote environments intentionally lack a C++
    # compiler required by Inductor; ``compile=false`` keeps those sweeps
    # reproducible in eager mode.
    compile_enabled = bool(OmegaConf.select(cfg, "compile", default=True))
    if is_ddp:
        teacher_model = DDP(teacher_model, device_ids=[rank])

    # torch.compile() stubs return Callable; cast back to nn.Module so downstream
    # type-checks still see it as a Module (DDP/unwrap/param-group helpers, etc.)
    if compile_enabled:
        teacher_model = cast(nn.Module, torch.compile(teacher_model))

    # ---- MuP optimizer for teacher (schedule baked in via manual lr) ------
    # Looks to try to mirror 207 - 232 + load in student model.
    # HN: TODO The order of operations here feels werid.
    D = int(cfg.model.D)
    teacher_lr = float(cfg.opt.lr)
    teacher_b2 = float(cfg.opt.b2)
    weight_decay = float(getattr(cfg.opt, "wd", 0.0) or 0.0)

    teacher_param_groups = _build_param_groups(
        teacher_model,
        teacher_lr,
        D,
        float(cfg.opt.embed_lr_mult),
        weight_decay,
    )
    teacher_optimizer = torch.optim.AdamW(
        teacher_param_groups,
        betas=(float(cfg.opt.b1), teacher_b2),
        eps=1e-20,
        fused=torch.cuda.is_available(),
    )

    # ---- EMA decay --------------------------------------------------------
    def _ema_decay(val) -> float | None:
        v = float(val) if val is not None else 0.0
        if v <= 0.0:
            return None
        window = int(round(v)) if v > 1.0 else max(1, int(round(v * num_train_steps)))
        return 1.0 - 1.0 / max(1, window)

    teacher_ema_decay = _ema_decay(cfg.teacher_ema)

    # Unwrap to raw model for EMA copy (before compile/DDP wrapping)
    def _unwrap(m: nn.Module) -> nn.Module:
        """Peel off torch.compile and DDP wrappers to reach the raw nn.Module."""
        while isinstance(m, DDP) or hasattr(m, "_orig_mod"):
            if isinstance(m, DDP):
                m = m.module
            elif hasattr(m, "_orig_mod"):
                # torch.compile sets _orig_mod; not in nn.Module stubs
                m = m._orig_mod  # type: ignore[assignment]
        return m

    teacher_ema_model = (
        copy.deepcopy(_unwrap(teacher_model)).to(device)
        if teacher_ema_decay is not None
        else None
    )

    # ---- student model (optional) -----------------------------------------
    student_model = None
    student_optimizer = None
    student_ema_model = None
    student_cfg = None
    student_ema_decay_val = None

    if train_student_base:
        student_cfg = copy.deepcopy(cfg.model)
        if cfg.model.P_student is not None:
            if cfg.model.P_student >= cfg.model.P:
                if is_main:
                    print(
                        f"Skipping: P_student={cfg.model.P_student} >= P={cfg.model.P}"
                    )
                if is_ddp:
                    dist.destroy_process_group()
                return None, 0
            student_cfg.P = cfg.model.P_student
            student_cfg.D = (
                round(((student_cfg.P * 1e6 / student_cfg.N / 12) ** 0.5) / 64) * 64
            )
            if student_cfg.D < 64:
                raise ValueError("Student model D must be >= 64")

        student_model = model_lib.create_model(student_cfg).to(device)
        if explicit_dtype is not None and explicit_dtype != torch.float32:
            student_model = student_model.to(explicit_dtype)

        if is_ddp:
            student_model = DDP(student_model, device_ids=[rank])
        if compile_enabled:
            student_model = cast(nn.Module, torch.compile(student_model))

        student_D = int(student_cfg.D)
        student_pg = _build_param_groups(
            student_model,
            teacher_lr,
            student_D,
            float(cfg.opt.embed_lr_mult),
            weight_decay,
        )
        student_optimizer = torch.optim.AdamW(
            student_pg,
            betas=(float(cfg.opt.b1), float(cfg.opt.b2)),
            eps=1e-20,
            fused=torch.cuda.is_available(),
        )

        student_ema_decay_val = _ema_decay(cfg.student_ema)
        student_ema_model = (
            copy.deepcopy(_unwrap(student_model)).to(device)
            if student_ema_decay_val is not None
            else None
        )

    # ---- student batch size -----------------------------------------------
    student_batch_size = student_microbatch_size = distill_tokens_per_step = 0
    if train_student_base:
        student_batch_size = (
            int(cfg.B_student) if cfg.B_student is not None else int(cfg.B)
        )
        if is_ddp:
            assert student_batch_size % world_size == 0
        if student_batch_size % accumulation_steps != 0:
            raise ValueError(
                f"B_student ({student_batch_size}) must be divisible by A ({accumulation_steps})"
            )
        student_microbatch_size = student_batch_size // accumulation_steps
        if is_ddp:
            assert student_microbatch_size % world_size == 0
        distill_tokens_per_step = student_batch_size * cfg.model.L

    # ---- Comet ML (rank 0 only) -------------------------------------------
    experiment = None
    if is_main and HAS_COMET and cfg.wandb_project is not None:
        comet_api = os.environ.get("COMET_ML_API")
        if comet_api:
            config = _flatten_dict(cfg)
            config["num_params"] = num_params
            config = {k.split("/")[-1]: v for k, v in config.items()}

            # None branch unreachable here (HAS_COMET guard above)
            experiment = Experiment(  # type: ignore[misc]
                api_key=comet_api,
                project_name=cfg.wandb_project,
                workspace="epi-audio",
            )

            # When launched from a sweep, each run is given an explicit name so it can
            # be identified in the Comet UI; the sweep name is attached as a tag so all
            # runs in the same sweep can be grouped/compared together.
            run_name = OmegaConf.select(cfg, "run_name", default=None)
            sweep_name = OmegaConf.select(cfg, "sweep_name", default=None)
            if run_name:
                experiment.set_name(run_name)
            if sweep_name:
                experiment.add_tag(sweep_name)
            if OmegaConf.select(cfg, "tag", default=None):
                experiment.add_tag(cfg.tag)
            experiment.log_parameters(config)

    # HN: Looks good

    # ---- eval / downstream eval step sets ---------------------------------
    eval_steps = set(
        [int(i) for i in np.linspace(warmup_steps, num_train_steps - 1, cfg.num_evals)]
        + [int(i) for i in np.geomspace(1, num_train_steps, 50)]
    )
    num_evals_dn = getattr(cfg, "num_evals_downstream", cfg.num_evals)
    downstream_eval_steps = set(
        [int(i) for i in np.linspace(warmup_steps, num_train_steps - 1, cfg.num_evals)][
            :: max(1, cfg.num_evals // num_evals_dn)
        ][1:]
    )

    # HN: Looks good

    # ---- training state accumulators --------------------------------------
    tau = 0.0
    KX = 0.0
    train_loss_sum = 0.0
    elapsed = 0
    prev_features = None
    pending_eval_metrics: dict | None = None

    teacher_tokens_seen = 0
    student_tokens_seen = 0
    last_distill_kl = 0.0
    cumulative_distill_kl = 0.0
    current_km_req: float | None = None
    if train_student_base:
        current_km_req = 0.0

    teacher_model.train()
    if student_model is not None:
        student_model.train()

    step = teacher_step = student_step = 0
    train_teacher = train_teacher_base
    train_student = train_student_base

    pbar = tqdm(total=num_train_steps, disable=not is_main)

    # ======================================================================
    # Main training loop
    # ======================================================================
    while teacher_step < num_train_steps:
        schedule_scale = schedule_fn(teacher_step)
        lr_t = teacher_lr * schedule_scale
        teacher_train_loss: float | None = None
        train_teacher = train_teacher_base
        train_student = train_student_base

        if enforce_max_kl and last_distill_kl > max_kl:
            train_teacher = False

        step += 1

        # ------------------------------------------------------------------
        # Teacher training step
        # ------------------------------------------------------------------
        if not train_teacher_base:
            teacher_step += 1
            pbar.update(1)
        elif train_teacher:
            # Set schedule-scaled lr for this step
            for g in teacher_optimizer.param_groups:
                g["lr"] = g["_base_lr"] * schedule_scale

            teacher_optimizer.zero_grad()
            accum_loss = torch.tensor(0.0, device=device)

            for accum_idx in range(accumulation_steps):
                micro_idx = teacher_step * accumulation_steps + accum_idx
                batch_raw = get_batch_train_teacher(micro_idx)

                sync_ctx = (
                    # DDP method, not on nn.Module stubs
                    teacher_model.no_sync()  # type: ignore[attr-defined]
                    if is_ddp and accum_idx < accumulation_steps - 1
                    else contextlib.nullcontext()
                )
                with (
                    sync_ctx,
                    autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled),
                ):
                    x, y, w = get_in_out(batch_raw, device=device)
                    logits = teacher_model(x)
                    micro_loss = _ce_loss(logits, y, w)

                (micro_loss / accumulation_steps).backward()
                accum_loss += micro_loss.detach()

            mean_loss_local = (accum_loss / accumulation_steps).item()

            # All-reduce loss for correct global epiplexity accounting
            if is_ddp:
                mean_loss_global = _all_reduce_mean(mean_loss_local, device)
            else:
                mean_loss_global = mean_loss_local

            teacher_optimizer.step()
            teacher_train_loss = mean_loss_global
            teacher_tokens_seen += train_tokens_per_step
            teacher_step += 1
            pbar.update(1)
            tau += lr_t

            if teacher_ema_model is not None:
                assert teacher_ema_decay is not None  # ema_model is only created when decay is set
                _update_ema(teacher_ema_model, teacher_model, teacher_ema_decay)

        # HN: TODO
        compute_spent = 6 * teacher_tokens_seen * num_params
        # HN: Note 6 is in the old code but what is that hyperparamter
        # HN: and does it still make sense here?

        if teacher_train_loss is not None:
            KX += teacher_train_loss * train_tokens_per_step / 1e6 / math.log(2)
            train_loss_sum += teacher_train_loss
            elapsed += 1

        step_log: dict = {"step": step, "teacher_step": teacher_step}

        # ------------------------------------------------------------------
        # Student distillation step
        # ------------------------------------------------------------------
        if train_student and student_model is not None:
            assert student_optimizer is not None  # set in the train_student_base init block above
            # Choose teacher for generation: EMA if available
            gen_model = (
                teacher_ema_model
                if teacher_ema_model is not None
                else _unwrap(teacher_model)
            )

            torch.manual_seed(rank_gen_seed + teacher_step)
            torch.cuda.manual_seed(rank_gen_seed + teacher_step)
            with torch.no_grad():
                # custom model method, not on nn.Module stubs
                synth_tokens, teacher_logits = gen_model.generate(  # type: ignore[attr-defined]
                    batch_size=student_batch_size // world_size
                    if is_ddp
                    else student_batch_size,
                    seq_len=cfg.model.L,
                    device=device,
                    bos_token=0,
                    temperature=1.0,
                )

            synth_mask = torch.ones_like(synth_tokens, dtype=torch.bool)

            # Student gradient accumulation
            for g in student_optimizer.param_groups:
                g["lr"] = g["_base_lr"] * schedule_scale
            student_optimizer.zero_grad()

            accum_ce = torch.tensor(0.0, device=device)
            accum_kl = torch.tensor(0.0, device=device)

            local_smbs = (
                student_microbatch_size // world_size
                if is_ddp
                else student_microbatch_size
            )

            for accum_idx in range(accumulation_steps):
                s = accum_idx * local_smbs
                e = s + local_smbs
                micro_tokens = synth_tokens[s:e]
                micro_mask = synth_mask[s:e]
                micro_t_logits = teacher_logits[s:e]

                sync_ctx = (
                    # DDP method, not on nn.Module stubs
                    student_model.no_sync()  # type: ignore[attr-defined]
                    if is_ddp and accum_idx < accumulation_steps - 1
                    else contextlib.nullcontext()
                )
                with (
                    sync_ctx,
                    autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled),
                ):
                    x_s, y_s, w_s = get_in_out(
                        (micro_tokens, micro_mask), device=device
                    )
                    s_logits = student_model(x_s)
                    ce_loss = _ce_loss(s_logits, y_s, w_s)

                (ce_loss / accumulation_steps).backward()
                accum_ce += ce_loss.detach()

                # KL for monitoring only — gradients are CE-only (matches JAX)
                kl = _kl_loss(micro_t_logits, s_logits.detach(), w_s)
                accum_kl += kl

            mean_ce_local = (accum_ce / accumulation_steps).item()
            mean_kl_local = (accum_kl / accumulation_steps).item()

            if is_ddp:
                mean_ce_global = _all_reduce_mean(mean_ce_local, device)
                mean_kl_global = _all_reduce_mean(mean_kl_local, device)
            else:
                mean_ce_global = mean_ce_local
                mean_kl_global = mean_kl_local

            student_optimizer.step()
            student_step += 1
            last_distill_kl = mean_kl_global
            cumulative_distill_kl += mean_kl_global
            student_tokens_seen += distill_tokens_per_step
            current_km_req = (
                cumulative_distill_kl * distill_tokens_per_step / 1e6 / math.log(2)
            )

            if student_ema_model is not None:
                assert student_ema_decay_val is not None  # ema_model is only created when decay is set
                _update_ema(student_ema_model, student_model, student_ema_decay_val)

            step_log.update(
                {
                    "student_step": student_step,
                    "distill_ce": mean_ce_global,
                    "distill_kl": mean_kl_global,
                    "K(M)_req": current_km_req,
                }
            )

        # HN: Looks Good so far

        # ------------------------------------------------------------------
        # Token accounting
        # ------------------------------------------------------------------
        step_log["teacher_tokens"] = int(teacher_tokens_seen)
        if train_student_base:
            step_log["student_tokens"] = int(student_tokens_seen)
        tokens = (
            int(teacher_tokens_seen)
            if train_teacher_base and train_student_base
            else max(teacher_tokens_seen, student_tokens_seen)
        )
        step_log["tokens"] = tokens

        if is_main and experiment is not None:
            experiment.log_metrics(step_log, step=student_step)

        # Log pending eval metrics from previous step
        if is_main and pending_eval_metrics is not None:
            if experiment is not None:
                experiment.log_metrics(pending_eval_metrics, step=student_step)
            pending_eval_metrics = None

        # ------------------------------------------------------------------
        # Evaluation
        # HN: Looks good here. This is the big one, so lets
        # HN: TODO: Get one last review here0
        # ------------------------------------------------------------------
        if (teacher_step in eval_steps) or (teacher_step == num_train_steps):
            eval_src = (
                teacher_ema_model
                if teacher_ema_model is not None
                else _unwrap(teacher_model)
            )

            if is_main:
                pending_eval_metrics = _eval_step(eval_src, ds_test, device, amp_dtype)
                features = pending_eval_metrics.pop("features")

                if teacher_ema_model is not None:
                    ema_loss = pending_eval_metrics["eval_loss"]
                    raw_metrics = _eval_step(
                        _unwrap(teacher_model), ds_test, device, amp_dtype
                    )
                    raw_metrics.pop("features")
                    pending_eval_metrics["ema_teacher_eval_loss"] = ema_loss
                    pending_eval_metrics["teacher_eval_loss"] = raw_metrics["eval_loss"]
                else:
                    pending_eval_metrics["teacher_eval_loss"] = pending_eval_metrics[
                        "eval_loss"
                    ]

                if prev_features is not None:
                    pending_eval_metrics |= _eval_features(features, prev_features)
                else:
                    pending_eval_metrics["h"] = math.sqrt(float(features.pow(2).mean()))
                prev_features = features

                pending_eval_metrics |= {
                    "tokens": tokens,
                    "compute": compute_spent,
                    "tau": tau,
                    "lr": lr_t,
                    "student_step": student_step,
                    "teacher_step": teacher_step,
                    "step": step,
                    "student_tokens": student_tokens_seen,
                    "teacher_tokens": teacher_tokens_seen,
                }

                if elapsed > 0:
                    L_avg = train_loss_sum / elapsed
                    train_loss_sum = 0
                    elapsed = 0
                    K_XM = L_avg * (teacher_tokens_seen / 1e6) / math.log(2)
                    K_M = KX - K_XM
                    pending_eval_metrics["train_loss"] = L_avg
                    pending_eval_metrics["K(X|M)"] = K_XM
                    pending_eval_metrics["K(M)"] = K_M
                else:
                    train_loss_sum = 0
                    elapsed = 0

                pending_eval_metrics["K(X)"] = KX
                if current_km_req is not None:
                    pending_eval_metrics["K(M)_req"] = current_km_req

                # Student eval
                if train_student_base and student_model is not None:
                    s_src = (
                        student_ema_model
                        if student_ema_model is not None
                        else _unwrap(student_model)
                    )

                    if student_ema_model is not None:
                        sm_raw = _eval_step(
                            _unwrap(student_model), ds_test, device, amp_dtype
                        )
                        sm_raw.pop("features")
                        sm_ema = _eval_step(s_src, ds_test, device, amp_dtype)
                        sm_ema.pop("features")
                        pending_eval_metrics["student_eval_loss"] = sm_raw["eval_loss"]
                        pending_eval_metrics["ema_student_eval_loss"] = sm_ema[
                            "eval_loss"
                        ]
                    else:
                        sm = _eval_step(s_src, ds_test, device, amp_dtype)
                        sm.pop("features")
                        pending_eval_metrics["student_eval_loss"] = sm["eval_loss"]

                    pending_eval_metrics["train_mode"] = (
                        "both"
                        if (train_teacher and train_student)
                        else "teacher_only"
                        if train_teacher
                        else "student_only"
                    )
                    if last_distill_kl is not None:
                        pending_eval_metrics["kl_current"] = last_distill_kl
                        pending_eval_metrics["max_kl"] = float(max_kl)
                        pending_eval_metrics["max_kl_enforced"] = bool(enforce_max_kl)

                # Downstream eval
                if (
                    cfg.downstream_ds_path is not None
                    and teacher_step in downstream_eval_steps
                ):
                    dn_metrics: dict = {}
                    # These are set whenever downstream_ds_path is not None (see above)
                    assert ds_test_downstream is not None
                    assert ds_train_downstream is not None
                    dn_eval_src = (
                        teacher_ema_model
                        if teacher_ema_model is not None
                        else _unwrap(teacher_model)
                    )
                    dn_metrics |= _downstream_eval_step(
                        dn_eval_src, ds_test_downstream, device, amp_dtype
                    )
                    dn_metrics |= _downstream_ft_eval(
                        dn_eval_src,
                        ds_train_downstream,
                        ds_test_downstream,
                        device,
                        teacher_lr,
                        D,
                        cfg,
                        amp_dtype,
                    )
                    if train_student_base and student_model is not None:
                        assert student_cfg is not None  # set in the train_student_base init block above
                        s_dn = (
                            student_ema_model
                            if student_ema_model is not None
                            else _unwrap(student_model)
                        )
                        s_dn_m = _downstream_eval_step(
                            s_dn, ds_test_downstream, device, amp_dtype
                        )
                        dn_metrics |= {f"student_{k}": v for k, v in s_dn_m.items()}
                        s_ft_m = _downstream_ft_eval(
                            s_dn,
                            ds_train_downstream,
                            ds_test_downstream,
                            device,
                            teacher_lr,
                            int(student_cfg.D),
                            cfg,
                            amp_dtype,
                        )
                        dn_metrics |= {f"student_{k}": v for k, v in s_ft_m.items()}
                    pending_eval_metrics |= dn_metrics

            if is_ddp:
                dist.barrier()

    # ---- final flush -------------------------------------------------------
    experiment_key = experiment.get_key() if experiment is not None else None
    if is_main and experiment is not None:
        if pending_eval_metrics:
            experiment.log_metrics(pending_eval_metrics, step=student_step)
        experiment.end()

    # ---- checkpoint --------------------------------------------------------
    if is_main and save_path is not None:
        state_dict = _unwrap(teacher_model).state_dict()

        is_gcs = _is_gcs_path(save_path)
        if is_gcs:
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            tmp.close()
            local_target = tmp.name
        else:
            local_target = save_path
            d = os.path.dirname(local_target)
            if d:
                os.makedirs(d, exist_ok=True)
        try:
            torch.save(state_dict, local_target)
            if is_gcs:
                _upload_temp_to_gcs(local_target, save_path)
        finally:
            if is_gcs and os.path.exists(local_target):
                os.remove(local_target)
        print(f"Saved final checkpoint to {save_path}")

    pbar.close()
    if is_ddp:
        dist.destroy_process_group()

    return experiment_key, ds_test_size


# HN: All looks good


# ---------------------------------------------------------------------------
# DDP worker entry points
# ---------------------------------------------------------------------------


def _ddp_worker(rank: int, world_size: int, cfg: DictConfig, result_queue) -> None:
    result = _run_training(rank, world_size, cfg)
    if rank == 0:  # HN: Why does only rank 0 put the result up
        # HN: shouldn't all the ranks do that?
        result_queue.put(result)


# ---------------------------------------------------------------------------
# Public entry point (same signature as JAX train.py)
# ---------------------------------------------------------------------------


def train_and_evaluate(cfg: DictConfig) -> tuple[str | None, int]:
    """Train (and optionally distil) a transformer on audio tokens.

    Replicates the exact behaviour of epiaudio/train.py using PyTorch.
    Automatically uses all available CUDA GPUs via DDP.

    Returns ``(experiment_key, test_tokens)``: the Comet experiment key used
    to read back step metrics for the epiplexity computation (or None if
    Comet logging is disabled or the config was rejected), and the number of
    tokens in the test split.
    """
    # ---- pre-process cfg (same order as the JAX version) ------------------
    if os.path.exists(f"{cfg.ds_path}/meta.pkl"):
        with open(f"{cfg.ds_path}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
            cfg.model.V = int(math.ceil(meta["vocab_size"] / 32)) * 32
    elif cfg.ds_path == "open":
        cfg.model.V = 96
    elif cfg.ds_path == "cifar5m":
        cfg.model.V = 256

    assert cfg.T is not None or (
        getattr(cfg, "scale", None) is not None
        and getattr(cfg, "exponent", None) is not None
    ), "Either T or fitted scale/exponent must be provided."

    if cfg.model.P is not None:
        cfg.model.D = round(((cfg.model.P * 1e6 / cfg.model.N / 12) ** 0.5) / 64) * 64
        if cfg.model.D < 64:
            print(f"Skipping run: model.D={cfg.model.D} < 64 (too small).")
            return None, 0

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size <= 1:
        return _run_training(rank=0, world_size=1, cfg=cfg)

    # Spawn one process per GPU
    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # NVLS (NVLink multicast) is unsupported in most container environments
    # and causes "Cuda failure 1 'invalid argument'" in DDP init.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    # spawn exists at runtime; pyright stubs are incomplete
    torch.multiprocessing.spawn(  # type: ignore[attr-defined]
        _ddp_worker,
        args=(world_size, cfg, result_queue),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )

    return result_queue.get() if not result_queue.empty() else (None, 0)

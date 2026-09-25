"""
Next token prediction experiments for measuring next token prediction performance.
Decoder-only transformers are trained to perform next token prediction. Evaluation is based on
auto-regressively predicting the contiguous sequence of length num_pred_tokens which follows
the first num_prefix_tokens in each tokenized audio clip.

Am experiment is described by a wandb-style YAML, e.g.
`epiplexity/downstream/next_token/configs/birdset_hsn_encodec.yaml`, which contains:

    name: <experiment_name>
    ds_path: <dataset_path>
    ood_ds_paths: list[<dataset_path>]
    parameters:
        "n_pretrain_epochs": <int>
        "n_finetune_epochs": <int>
        "batch_size": <int>
        "num_layers": <int>
        "embed_dim": <int>
        "per_head_dim": <int>
        "vocab_size": <int>
        "seq_length": <int>
        "num_prefix_tokens": <int>
        "num_pred_tokens": <int>
        "zero_shot": <bool>

The ds_path should be a path to the primary dataset, which will be used to pretrain a model M.
Each element in ood_ds_paths should be a path to an OOD dataset. On each of these datasets,
a fresh model will be pretrained and a copy of the original model M will be finetuned.
Performance metrics are reported on the test split of each dataset, allowing comparison
between pretraining on an OOD dataset vs. pretraining on the primary dataset and finetuning on
the OOD dataset.

Set zero_shot to True if intending to use pretrained checkpoints. In this case, the checkpointed
model will be used instead of pretraining a fresh one, and no finetuning will be done on the
out-of-distribution datasets.

Working directory:
    Any relative paths in the config are resolved against the current directory, so launch
    from the repository root.

Usage (from the repository root):
    uv run python -m epiaudio.downstream.next_token.experiment epiaudio/downstream/next_token/configs/<config_file_name>
"""

import os
import argparse
from typing import Any
from dotenv import load_dotenv
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from epiaudio.downstream.next_token.train_eval import train_and_evaluate

try:
    from comet_ml import Experiment
    HAS_COMET = True
except ImportError:
    HAS_COMET = False
    Experiment = None

EPI_AUDIO_WORKSPACE = "epi-audio"
COMET_PROJECT = "next_token_prediction"
ZERO_SHOT_COMET_PROJECT = "zero_shot_next_token_prediction"
COMET_API_KEY_VAR = "COMET_ML_API"

def name_experiment(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = f"{base_name}_next_token__{timestamp}"

    return name

def create_checkpoint_path(ds_path: str, ddp: bool=False) -> str:
    prefix = "ddp_model" if ddp else "pretrained_model"
    path = os.path.join(ds_path, f"{prefix}_checkpoint.pt")

    return path

def update_config(cfg: DictConfig, updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value)

def create_params(cfg: DictConfig) -> tuple[dict[str, str | int | None], DictConfig]:
    ds_path = OmegaConf.select(cfg, "ds_path", default=None)
    if ds_path is None:
        raise ValueError("Data path must be provided.")

    default_pretrained_checkpoint_path = create_checkpoint_path(ds_path)
    pretrained_checkpoint_path = OmegaConf.select(
        cfg,
        "pretrained_checkpoint_path",
        default=default_pretrained_checkpoint_path
    )

    default_ddp_checkpoint_path = create_checkpoint_path(ds_path, ddp=True)
    ddp_checkpoint_path = OmegaConf.select(
        cfg,
        "ddp_checkpoint_path",
        default=default_ddp_checkpoint_path
    )

    ood_ds_paths = OmegaConf.select(cfg, "ood_ds_paths", default=None)
    ood_ds_paths = ood_ds_paths if ood_ds_paths else []

    n_pretrain_epochs = OmegaConf.select(cfg, "n_pretrain_epochs", default=50)
    n_finetune_epochs = OmegaConf.select(cfg, "n_finetune_epochs", default=5)
    batch_size = OmegaConf.select(cfg, "batch_size", default=256)
    num_layers = OmegaConf.select(cfg, "num_layers", default=3)
    embed_dim = OmegaConf.select(cfg, "embed_dim", default=192)
    per_head_dim = OmegaConf.select(cfg, "per_head_dim", default=64)
    vocab_size = OmegaConf.select(cfg, "vocab_size", default=1024)
    seq_length = OmegaConf.select(cfg, "seq_length", default=512)
    num_prefix_tokens = OmegaConf.select(cfg, "num_prefix_tokens", default=300)
    num_pred_tokens = OmegaConf.select(cfg, "num_pred_tokens", default=75)
    tokenizer_name = OmegaConf.select(cfg, "tokenizer_name", default=None)
    zero_shot = OmegaConf.select(cfg, "zero_shot", default=True)

    if tokenizer_name == "dac" and num_prefix_tokens % 12 + num_pred_tokens % 12 != 0:
        msg = (
            f"num_prefix_tokens ({num_prefix_tokens}) and "
            f"num_pred_tokens ({num_pred_tokens}) must be multiples of 12"
        )
        raise ValueError(msg)

    parameters = {
        "ds_path": ds_path,
        "ood_ds_paths": ood_ds_paths,
        "pretrained_checkpoint_path": pretrained_checkpoint_path,
        "ddp_checkpoint_path": ddp_checkpoint_path,
        "n_pretrain_epochs": n_pretrain_epochs,
        "n_finetune_epochs": n_finetune_epochs,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "per_head_dim": per_head_dim,
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "num_prefix_tokens": num_prefix_tokens,
        "num_pred_tokens": num_pred_tokens,
        "tokenizer_name": tokenizer_name,
        "zero_shot": zero_shot
    }
    model_cfg = OmegaConf.create(parameters)
    for key in [
        "ds_path",
        "ood_ds_paths",
        "pretrained_checkpoint_path",
        "ddp_checkpoint_path",
        "tokenizer_name"
    ]:
        parameters.pop(key)

    return parameters, model_cfg

def run_experiment(cfg: DictConfig, pretrain_ood: bool) -> None:
    parameters, model_cfg = create_params(cfg)

    ds_path = model_cfg.ds_path
    num_prefix_tokens = model_cfg.num_prefix_tokens
    num_pred_tokens = model_cfg.num_pred_tokens
    og_pretrained_checkpoint_path = model_cfg.pretrained_checkpoint_path

    all_metrics = dict()

    if model_cfg.zero_shot:
        print(f"Evaluating on {ds_path}:")
    else:
        print(f"Training and evaluating on {ds_path}:")
    metrics = train_and_evaluate(model_cfg, num_prefix_tokens, num_pred_tokens)
    all_metrics[ds_path] = metrics

    for path in model_cfg.ood_ds_paths:
        if model_cfg.zero_shot:
            print(f"Evaluating on {path}")
        elif pretrain_ood:
            print(f"Pretraining/finetuning and evaluating on {path}:")
        else:
            print(f"Finetuning and evaluating on {path}:")
        report = dict()
        updates = {
            "ds_path": path,
            "pretrained_checkpoint_path": create_checkpoint_path(path),
            "ddp_checkpoint_path": create_checkpoint_path(path, ddp=True)
        }
        update_config(model_cfg, updates)

        if pretrain_ood:
            pretrain_metrics = train_and_evaluate(model_cfg, num_prefix_tokens, num_pred_tokens)
            report["pretrain"] = pretrain_metrics

        update_config(
            model_cfg,
            {"pretrained_checkpoint_path": og_pretrained_checkpoint_path}
        )
        finetune_metrics = train_and_evaluate(
            model_cfg,
            num_prefix_tokens,
            num_pred_tokens,
            finetune=True
        )

        report["finetune"] = finetune_metrics
        all_metrics[path] = report

    if HAS_COMET:
        assert Experiment is not None

        load_dotenv()

        if COMET_API_KEY_VAR not in os.environ or os.environ[COMET_API_KEY_VAR] == "<INSERT_API_KEY>":
            print("[WARNING]: CometML API not set, continue anyway?")
            input()
        comet_api_key = os.environ.get(COMET_API_KEY_VAR)

        project_name = ZERO_SHOT_COMET_PROJECT if model_cfg.zero_shot else COMET_PROJECT
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=project_name,
            workspace=EPI_AUDIO_WORKSPACE
        )
        base_name = OmegaConf.select(cfg, "name", default="")
        experiment.set_name(name_experiment(base_name))
        experiment.log_parameters(parameters)
        experiment.log_metrics(all_metrics)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an audio next token prediction experiment.")
    parser.add_argument(
        "config",
        help="Path to a next token prediction YAML, e.g. epiaudio/next_token/demand_encodec.yaml",
    )
    parser.add_argument(
        "--pretrain-ood",
        type=bool,
        default=False,
        help="If true, a fresh model will be pretrained on each ood dataset.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    assert isinstance(cfg, DictConfig), "Invalid config file"

    run_experiment(cfg, args.pretrain_ood)

if __name__ == "__main__":
    main()

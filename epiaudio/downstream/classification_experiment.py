import os
import argparse
from dotenv import load_dotenv
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from epiaudio.sweep import load_sweep_configs
from epiaudio.downstream.train_eval import train_and_evaluate

try:
    from comet_ml import Experiment
    HAS_COMET = True
except ImportError:
    HAS_COMET = False
    Experiment = None

EPI_AUDIO_WORKSPACE = "epi-audio"
COMET_PROJECT = "classification"
COMET_API_KEY_VAR = "COMET_ML_API"

def name_experiment(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = f"{base_name}__{timestamp}"

    return name

def run_experiment(cfg: DictConfig) -> None:
    ds_path = OmegaConf.select(cfg, "ds_path", default=None)
    base_name = OmegaConf.select(cfg, "name", default="")

    assert ds_path is not None, "Data path must be provided."

    n_epochs = OmegaConf.select(cfg, "n_epochs", default=50)
    batch_size = OmegaConf.select(cfg, "batch_size", default=256)
    num_layers = OmegaConf.select(cfg, "num_layers", default=3)
    embed_dim = OmegaConf.select(cfg, "embed_dim", default=192)
    per_head_dim = OmegaConf.select(cfg, "per_head_dim", default=64)
    vocab_size = OmegaConf.select(cfg, "vocab_size", default=1024)
    seq_length = OmegaConf.select(cfg, "seq_length", default=512)

    parameters = {
        "ds_path": ds_path,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "embed_dim": embed_dim,
        "per_head_dim": per_head_dim,
        "vocab_size": vocab_size,
        "seq_length": seq_length
    }
    model_cfg = OmegaConf.create(parameters)
    parameters.pop("ds_path")
    metrics = train_and_evaluate(model_cfg)

    if HAS_COMET:
        assert Experiment is not None

        load_dotenv()

        if COMET_API_KEY_VAR not in os.environ or os.environ[COMET_API_KEY_VAR] == "<INSERT_API_KEY>":
            print("[WARNING]: CometML API not set, continue anyway?")
            input()
        comet_api_key = os.environ.get(COMET_API_KEY_VAR)

        experiment = Experiment(
            api_key=comet_api_key,
            project_name=COMET_PROJECT,
            workspace=EPI_AUDIO_WORKSPACE
        )
        base_name = OmegaConf.select(cfg, "name", default="")
        experiment.set_name(name_experiment(base_name))
        experiment.log_parameters(parameters)
        experiment.log_metrics(metrics)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an audio classification experiment.")
    parser.add_argument(
        "classification_config",
        help="Path to a classification YAML, e.g. epiaudio/classification/demand_encodec.yaml",
    )
    args = parser.parse_args()

    cfg = load_sweep_configs(args.classification_config)[0]
    run_experiment(cfg)

if __name__ == "__main__":
    main()
"""Generate next-token-prediction config YAMLs for every (dataset, tokenizer) pair.

Each generated file mirrors the structure of
`epiaudio/downstream/next_token/configs/syntheory_intervals_encodec.yaml`.
"""

import argparse
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS

# TOKENIZER_SPECS shapes describe a tokenizer's output for a five-second clip
# (see tokenizer_specs.py), so token counts scale linearly off that duration.
CLIP_SECONDS = 5.0

# Where syntheory_intervals_encodec.yaml (and its siblings) live.
DEFAULT_TOKENIZER_NAMES = "dac,encodec,sqcodec,xcodec"
DEFAULT_OUTPUT_ROOT = Path("epiaudio") / "downstream" / "next_token" / "configs"
DEFAULT_INPUT_ROOT = "data"


class _IndentedListDumper(yaml.Dumper):
    """Indents `- ` list items under their parent key, matching the hand-written configs."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


def generate_configs(
    dataset_names: list[str],
    *,
    tokenizer_names: tuple[str, ...] = tuple(DEFAULT_TOKENIZER_NAMES.split(",")),
    input_root: str = DEFAULT_INPUT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    prefix_seconds: float = 1.0,
    pred_seconds: float = 1.0,
    n_pretrain_epochs: int = 50,
    n_finetune_epochs: int = 5,
    batch_size: int = 64,
    num_layers: int = 3,
    embed_dim: int = 192,
    per_head_dim: int = 64,
) -> None:
    """Write one next-token-prediction config YAML per (dataset, tokenizer) pair into `output_root`.

    For each `dataset_name` in `dataset_names` and `tokenizer_name` in `tokenizer_names`,
    writes `{output_root}/{dataset_name}_{tokenizer_name}.yaml`, with `ds_path`/`ood_ds_paths`
    pointing at `{input_root}/{dataset_name}_{tokenizer_name}`. `vocab_size`, `seq_length`,
    `num_prefix_tokens`, and `num_pred_tokens` are derived from `TOKENIZER_SPECS` rather than
    taken as arguments: the first two come straight from the tokenizer's spec, and the latter
    two are `prefix_seconds`/`pred_seconds` converted to token counts using that tokenizer's
    tokens-per-second rate.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        for tokenizer_name in tokenizer_names:
            spec = TOKENIZER_SPECS[tokenizer_name]
            tokens_per_second = spec.sequence_length / CLIP_SECONDS

            name = f"{dataset_name}_{tokenizer_name}"
            ood_ds_paths = [
                f"{input_root}/{other_dataset_name}_{tokenizer_name}"
                for other_dataset_name in dataset_names
                if other_dataset_name != dataset_name
            ]

            config = OmegaConf.create({
                "name": name,
                "ds_path": f"{input_root}/{name}",
                "ood_ds_paths": ood_ds_paths,
                "n_pretrain_epochs": n_pretrain_epochs,
                "n_finetune_epochs": n_finetune_epochs,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "embed_dim": embed_dim,
                "per_head_dim": per_head_dim,
                "vocab_size": spec.vocab_size,
                "seq_length": spec.sequence_length,
                "num_prefix_tokens": round(prefix_seconds * tokens_per_second),
                "num_pred_tokens": round(pred_seconds * tokens_per_second),
                "tokenizer_name": tokenizer_name,
            })

            text = yaml.dump(
                OmegaConf.to_container(config),
                Dumper=_IndentedListDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            (output_root / f"{name}.yaml").write_text(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate next-token-prediction config YAMLs for one or more datasets."
    )
    parser.add_argument(
        "--dataset_names",
        required=True,
        help="Comma-separated dataset names, e.g. syntheory_intervals,syntheory_chords",
    )
    parser.add_argument(
        "--tokenizer_names",
        default=DEFAULT_TOKENIZER_NAMES,
        help=f"Comma-separated tokenizer names. Defaults to {','.join(DEFAULT_TOKENIZER_NAMES)}.",
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory to write config YAMLs into. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--input_root",
        default=DEFAULT_INPUT_ROOT,
        help=f"Path prefix used for ood ds_paths. Defaults to {DEFAULT_INPUT_ROOT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    kwargs = {}
    if args.tokenizer_names is not None:
        kwargs["tokenizer_names"] = tuple(args.tokenizer_names.split(","))
    if args.output_root is not None:
        kwargs["output_root"] = args.output_root
    if args.input_root is not None:
        kwargs["input_root"] = args.input_root

    generate_configs(args.dataset_names.split(","), **kwargs)


if __name__ == "__main__":
    main()

"""Backfill missing ``__sweep_config.yaml`` files for sweep outputs.

For each sweep YAML in ``sweep_yaml_dir`` (e.g. ``epiaudio/sweeps/``), match
every ``ds_path`` entry to the run it produced under ``sweep_output_dir``
(matched by the ``{name}_{tokenizer}`` prefix, picking the most recent
timestamp), then write that run's ``__sweep_config.yaml`` -- the schema
``epiaudio.analysis.pipeline.load_sweep_facts`` expects but older sweep runs
never saved.

    uv run python -m epiaudio.analysis.build_sweep_configs
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from epiaudio.analysis.pipeline import infer_tokenizer
from epiaudio.sweep_summary import CONFIG_SUFFIX

DEFAULT_SWEEP_YAML_DIR = Path("epiaudio/sweeps")
DEFAULT_SWEEP_OUTPUT_DIR = Path("sweep_outputs")

_TIMESTAMP_PATTERN = r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"


def _ds_paths(parameters: dict[str, Any]) -> list[str]:
    """A sweep parameter block's ``ds_path`` entries, whether ``value`` or ``values``."""
    node = parameters.get("ds_path")
    if not isinstance(node, dict):
        return []
    if "values" in node:
        return list(node["values"])
    if "value" in node:
        return [node["value"]]
    return []


def _latest_sweep_name(sweep_output_dir: Path, sweep_base_name: str) -> str | None:
    """``{sweep_base_name}__<timestamp>`` of the most recent matching run, if any."""
    pattern = re.compile(rf"^{re.escape(sweep_base_name)}__({_TIMESTAMP_PATTERN})__")
    latest_timestamp: str | None = None
    for csv_path in sweep_output_dir.glob(f"{sweep_base_name}__*.csv"):
        match = pattern.match(csv_path.name)
        if match is None:
            continue
        timestamp = match.group(1)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
    if latest_timestamp is None:
        return None
    return f"{sweep_base_name}__{latest_timestamp}"


def create_configs(sweep_yaml_dir: str | Path, sweep_output_dir: str | Path) -> None:
    """Write a ``__sweep_config.yaml`` for every sweep/``ds_path`` run found."""
    sweep_yaml_dir = Path(sweep_yaml_dir)
    sweep_output_dir = Path(sweep_output_dir)

    for yaml_path in sorted(sweep_yaml_dir.glob("*.yaml")):
        container = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=False)
        if not isinstance(container, dict):
            print(f"[build_sweep_configs] {yaml_path} is not a mapping, skipping")
            continue

        name = container.get("name")
        parameters = container.get("parameters")
        if not name or not isinstance(parameters, dict):
            print(f"[build_sweep_configs] {yaml_path} has no name/parameters, skipping")
            continue

        for ds_path in _ds_paths(parameters):
            tokenizer_name = infer_tokenizer(ds_path)
            if tokenizer_name is None:
                print(f"[build_sweep_configs] {ds_path!r} has no known tokenizer suffix, skipping")
                continue

            sweep_base_name = f"{name}_{tokenizer_name}"
            sweep_name = _latest_sweep_name(sweep_output_dir, sweep_base_name)
            if sweep_name is None:
                print(
                    f"[build_sweep_configs] no run under {sweep_output_dir} matches "
                    f"{sweep_base_name!r}, skipping"
                )
                continue

            cell_container = copy.deepcopy(container)
            cell_container["name"] = sweep_base_name
            cell_container["parameters"]["ds_path"] = {"values": [ds_path]}

            out_path = sweep_output_dir / f"{sweep_name}{CONFIG_SUFFIX}"
            OmegaConf.save(OmegaConf.create(cell_container), out_path)
            print(f"[build_sweep_configs] wrote {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-yaml-dir",
        type=Path,
        default=DEFAULT_SWEEP_YAML_DIR,
        help="directory holding the sweep definition YAMLs",
    )
    parser.add_argument(
        "--sweep-output-dir",
        type=Path,
        default=DEFAULT_SWEEP_OUTPUT_DIR,
        help="directory holding the sweep output bundles",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    create_configs(args.sweep_yaml_dir, args.sweep_output_dir)


if __name__ == "__main__":
    main()

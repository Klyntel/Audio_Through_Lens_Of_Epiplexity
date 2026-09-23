"""Dataset preparation, deterministic mixture construction, and planning."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import SCHEMA_VERSION, ExperimentSpec, write_json


def source_dataset_path(spec: ExperimentSpec, dataset_name: str) -> Path:
    return spec.data_root / f"{dataset_name}_{spec.tokenizer}"


def mixture_dataset_path(spec: ExperimentSpec, mixture_name: str) -> Path:
    return spec.data_root / f"epimixture_{spec.name}_{mixture_name}_{spec.tokenizer}"


def dataset_path(spec: ExperimentSpec, dataset_id: str) -> Path:
    mixture_names = {mixture.name for mixture in spec.mixtures}
    return mixture_dataset_path(spec, dataset_id) if dataset_id in mixture_names else source_dataset_path(spec, dataset_id)


def plan(spec: ExperimentSpec) -> Path:
    """Write the exact source, mixture, sweep, and result paths for a design."""
    plan_path = spec.output_root / "plan.json"
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "config": str(spec.config_path),
        "tokenizer": spec.tokenizer,
        "seed": spec.seed,
        "raw_roots": {name: str(path) for name, path in spec.raw_roots.items()},
        "sources": {name: str(source_dataset_path(spec, name)) for name in spec.sources},
        "heldout": {spec.heldout: str(source_dataset_path(spec, spec.heldout))},
        "mixtures": [
            {
                "name": mixture.name,
                "path": str(mixture_dataset_path(spec, mixture.name)),
                "weights": mixture.weights,
                "train_samples": mixture.train_samples,
                "test_samples": mixture.test_samples,
            }
            for mixture in spec.mixtures
        ],
        "sweeps": {dataset_id: str(spec.output_root / "sweeps" / f"{dataset_id}.yaml") for dataset_id in spec.dataset_ids},
        "selections": {dataset_id: str(spec.output_root / "selections" / f"{dataset_id}.json") for dataset_id in spec.dataset_ids},
    }
    write_json(plan_path, content)
    return plan_path


def _registry() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load existing preparation registries only when preparation is requested."""
    from epiaudio.dataset.load_dataloaders import AUDIO_PREPROCESSING_DATASETS
    from epiaudio.dataset.prepare_audio import DATASETS, TOKENIZERS

    return DATASETS, TOKENIZERS, AUDIO_PREPROCESSING_DATASETS


def list_registry() -> dict[str, list[str]]:
    """Return explicit registry names for a user to choose from."""
    datasets, tokenizers, audio_preprocessing = _registry()
    return {
        "audio_preprocessing_datasets": sorted(audio_preprocessing),
        "all_epiaudio_datasets": sorted(datasets),
        "tokenizers": sorted(tokenizers),
    }


def _validate_registry(spec: ExperimentSpec) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    datasets, tokenizers, _ = _registry()
    missing = sorted(set((*spec.sources, spec.heldout)) - set(datasets))
    if missing:
        raise ValueError(f"Unknown dataset registry entries: {', '.join(missing)}. Run list-registry first.")
    if spec.tokenizer not in tokenizers:
        raise ValueError(f"Unknown tokenizer {spec.tokenizer!r}. Run list-registry first.")
    if spec.tokenizer == "whisper":
        raise ValueError("whisper is continuous features, not supported by unconditional token sweeps.")
    return datasets, tokenizers


def prepare_sources(spec: ExperimentSpec, *, overwrite: bool = False) -> None:
    """Prepare each source and unseen target through EpiAudio's token pipeline."""
    datasets, tokenizers = _validate_registry(spec)
    from epiaudio.dataset.prepare_audio import prepare_ds, set_all_seeds
    from epiaudio.dataset.validate_audio import validate_audio

    tokenizer = tokenizers[spec.tokenizer]
    set_all_seeds(spec.seed)
    for dataset_name in (*spec.sources, spec.heldout):
        output = source_dataset_path(spec, dataset_name)
        required = [output / "train.bin", output / "test.bin", output / "metadata.yaml"]
        if not overwrite and all(path.is_file() for path in required):
            print(f"[epimixture] keeping prepared {dataset_name}: {output}")
            continue
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError(f"Prepared dataset is incomplete: {output}. Use --overwrite after inspection.")
        print(f"[epimixture] preparing {dataset_name} with {spec.tokenizer}: {output}")
        source = datasets[dataset_name]
        raw_root = spec.raw_roots.get(dataset_name)
        if raw_root is not None and not source.get("supports_root", False):
            raise ValueError(f"Dataset {dataset_name!r} does not support a configured raw root.")
        ds = source["loader"](raw_root) if raw_root is not None else source["loader"]()
        if source["decode"] is not None:
            ds = validate_audio(ds)
        if spec.preparation_sample_limit is not None:
            from datasets import DatasetDict

            ds.data = DatasetDict(
                {
                    split: rows.shuffle(seed=spec.seed).select(range(min(len(rows), spec.preparation_sample_limit)))
                    for split, rows in ds.data.items()
                }
            )
        prepare_ds(
            ds=ds,
            tokenizer_cls=tokenizer["cls"],
            expected_token_shape=tokenizer["shape"],
            dtype=tokenizer["dtype"],
            class_column=None,
            train_path=str(output / "train.bin"),
            test_path=str(output / "test.bin"),
            metadata_path=str(output / "metadata.json"),
            decode_fn=source["decode"],
            num_workers=spec.preparation["num_workers"],
            batch_size=spec.preparation["batch_size"],
            threads_per_worker=spec.preparation["threads_per_worker"],
            seed=spec.seed,
        )


def allocate_samples(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate an exact count proportionally, using stable largest remainders."""
    if total <= 0 or not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("total and every mixture weight must be positive.")
    weight_sum = sum(weights.values())
    raw = {name: total * weight / weight_sum for name, weight in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    for name, _ in sorted(raw.items(), key=lambda item: (-(item[1] - counts[item[0]]), item[0]))[:remainder]:
        counts[name] += 1
    return counts


def _read_token_layout(dataset_path: Path) -> tuple[np.dtype[Any], int, dict[str, Any]]:
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing token metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dtype = np.dtype(metadata["tokenizer"]["dtype"])
        sequence_length = int(metadata["tokenizer"]["sequence_length"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid token metadata: {metadata_path}") from exc
    if sequence_length <= 0:
        raise ValueError(f"Invalid tokenizer sequence length in {metadata_path}")
    return dtype, sequence_length, metadata


def _write_mixture_split(
    source_paths: Mapping[str, Path], *, split: str, weights: Mapping[str, float], requested_samples: int,
    destination: Path, seed: int, overwrite: bool,
) -> dict[str, int]:
    """Sample whole tokenized clips without replacement and interleave sources."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Mixture output exists: {destination}. Use --overwrite after inspection.")
    layouts = {name: _read_token_layout(path) for name, path in source_paths.items()}
    dtypes = {layout[0] for layout in layouts.values()}
    lengths = {layout[1] for layout in layouts.values()}
    if len(dtypes) != 1 or len(lengths) != 1:
        raise ValueError("Every mixture source must share the same tokenizer dtype and sequence length.")
    dtype, sequence_length = next(iter(dtypes)), next(iter(lengths))
    counts = allocate_samples(requested_samples, weights)
    arrays: dict[str, Any] = {}
    for name, path in source_paths.items():
        input_path = path / f"{split}.bin"
        if not input_path.is_file():
            raise ValueError(f"Missing {split} split for mixture source {name!r}: {input_path}")
        raw = np.memmap(input_path, dtype=dtype, mode="r")
        if raw.size % sequence_length:
            raise ValueError(f"{input_path} is not a whole number of tokenizer clips.")
        rows = raw.reshape((-1, sequence_length))
        if counts[name] > len(rows):
            raise ValueError(f"Mixture requests {counts[name]} {split} clips from {name!r}, but only {len(rows)} are available.")
        arrays[name] = rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = np.memmap(temporary, dtype=dtype, mode="w+", shape=(requested_samples, sequence_length))
    rng = np.random.default_rng(seed)
    schedule = np.repeat(np.arange(len(source_paths), dtype=np.int16), [counts[name] for name in source_paths])
    rng.shuffle(schedule)
    for source_index, name in enumerate(source_paths):
        positions = np.flatnonzero(schedule == source_index)
        output[positions] = arrays[name][rng.choice(len(arrays[name]), size=len(positions), replace=False)]
    output.flush()
    del output
    os.replace(temporary, destination)
    return counts


def build_mixtures(spec: ExperimentSpec, *, overwrite: bool = False) -> None:
    """Create configured mixtures from prepared, tokenizer-compatible sources."""
    source_paths = {name: source_dataset_path(spec, name) for name in spec.sources}
    for name, path in source_paths.items():
        if not path.is_dir():
            raise ValueError(f"Source {name!r} is not prepared: {path}")
    for mixture_index, mixture in enumerate(spec.mixtures):
        output = mixture_dataset_path(spec, mixture.name)
        if output.exists() and any(output.iterdir()) and not overwrite:
            required = [output / name for name in ("train.bin", "test.bin", "metadata.yaml", "mixture_manifest.json")]
            if all(path.is_file() for path in required):
                print(f"[epimixture] keeping mixture {mixture.name}: {output}")
                continue
            raise FileExistsError(f"Mixture output is incomplete: {output}. Use --overwrite after inspection.")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(next(iter(source_paths.values())) / "metadata.yaml", output / "metadata.yaml")
        train_counts = _write_mixture_split(source_paths, split="train", weights=mixture.weights,
            requested_samples=mixture.train_samples, destination=output / "train.bin", seed=spec.seed + mixture_index * 2, overwrite=overwrite)
        test_counts = _write_mixture_split(source_paths, split="test", weights=mixture.weights,
            requested_samples=mixture.test_samples, destination=output / "test.bin", seed=spec.seed + mixture_index * 2 + 1, overwrite=overwrite)
        write_json(output / "mixture_manifest.json", {
            "schema_version": SCHEMA_VERSION, "experiment": spec.name, "mixture": mixture.name,
            "tokenizer": spec.tokenizer, "seed": spec.seed,
            "sources": {name: str(path) for name, path in source_paths.items()},
            "weights": mixture.weights, "splits": {"train": train_counts, "test": test_counts},
        })
        print(f"[epimixture] wrote mixture {mixture.name}: {output}")

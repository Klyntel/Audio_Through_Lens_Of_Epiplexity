"""Validated runtime data loading for conditional classification experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


METADATA_SCHEMA_VERSION = 1
CLASSIFICATION_TASK_TYPE = "single_label_multiclass"


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Manifest field '{field}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Manifest field '{field}' must be at least {minimum}.")
    return value


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field '{field}' must be an object.")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset manifest '{path}': {exc}") from exc
    return _require_dict(manifest, "root")


class AudioClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped ``(audio_tokens, class_index)`` examples.

    The manifest produced by :mod:`epiaudio.dataset.prepare_audio` is the source
    of truth for the class vocabulary and storage layout. The final stored value
    in each row is exposed only as the target; it is never included in the model
    input.
    """

    def __init__(
        self,
        split: str,
        data_path: str | os.PathLike[str],
        metadata_path: str | os.PathLike[str],
    ) -> None:
        self.split = split
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        manifest = _load_manifest(self.metadata_path)

        self._validate_task(manifest)
        split_metadata = _require_dict(manifest.get(split), split)
        tokenizer_metadata = _require_dict(manifest.get("tokenizer"), "tokenizer")

        self.num_examples = _require_int(
            split_metadata.get("num_samples"),
            f"{split}.num_samples",
            minimum=1,
        )
        self.conditioning_tokens_per_example = _require_int(
            split_metadata.get("tokens_per_sample"),
            f"{split}.tokens_per_sample",
            minimum=1,
        )
        self.label_tokens_per_example = 1
        # Conditional decoding time accounts for both X and Y, while code length
        # is evaluated only on Y (Appendix B.1 of the epiplexity paper). As in the
        # paper's autoregressive factorization, the transformer processes X and
        # predicts Y from X's final position; this accounting does not imply an
        # additional query token in the model input.
        self.compute_tokens_per_example = (
            self.conditioning_tokens_per_example + self.label_tokens_per_example
        )

        # Compatibility aliases used by the current downstream training path.
        self.num_samples = self.num_examples
        self.tokens_per_sample = self.conditioning_tokens_per_example

        self.dtype = self._validate_storage_metadata(
            split_metadata,
            tokenizer_metadata,
        )
        self.num_classes = _require_int(
            manifest.get("num_classes"),
            "num_classes",
            minimum=1,
        )
        if self.num_classes - 1 > np.iinfo(self.dtype).max:
            raise ValueError(
                f"Storage dtype {self.dtype.name} cannot represent all class indices."
            )
        self.label_type = manifest.get("label_type")
        self.class_names, self.class_to_index = self._validate_class_vocabulary(
            manifest
        )
        self.class_counts = self._validate_class_counts(split_metadata)
        self.class_weights = (
            [self.num_examples / count for count in self.class_counts]
            if split == "train"
            else []
        )

        self._data: np.memmap | None = None
        self._data_process_id: int | None = None
        self._validate_binary_file()

    @staticmethod
    def _validate_task(manifest: dict[str, Any]) -> None:
        schema_version = _require_int(manifest.get("schema_version"), "schema_version")
        if schema_version != METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported metadata schema version {schema_version}; "
                f"expected {METADATA_SCHEMA_VERSION}."
            )
        if manifest.get("task_type") != CLASSIFICATION_TASK_TYPE:
            raise ValueError(
                "AudioClassificationDataset requires task_type "
                f"'{CLASSIFICATION_TASK_TYPE}'."
            )
        if manifest.get("label_position") != -1:
            raise ValueError(
                "Classification labels must occupy the final row position (-1)."
            )
        if not isinstance(manifest.get("label_column"), str):
            raise ValueError("Manifest field 'label_column' must be a string.")

    def _validate_storage_metadata(
        self,
        split_metadata: dict[str, Any],
        tokenizer_metadata: dict[str, Any],
    ) -> np.dtype[Any]:
        split_dtype = split_metadata.get("dtype")
        tokenizer_dtype = tokenizer_metadata.get("dtype")
        if split_dtype != tokenizer_dtype:
            raise ValueError(
                f"Manifest dtype mismatch: {self.split}.dtype is {split_dtype!r}, "
                f"but tokenizer.dtype is {tokenizer_dtype!r}."
            )
        try:
            dtype = np.dtype(split_dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported dataset dtype {split_dtype!r}.") from exc
        if not np.issubdtype(dtype, np.integer):
            raise ValueError("Classification storage dtype must be an integer dtype.")

        sequence_length = _require_int(
            tokenizer_metadata.get("sequence_length"),
            "tokenizer.sequence_length",
            minimum=1,
        )
        if sequence_length != self.conditioning_tokens_per_example:
            raise ValueError(
                f"Manifest sequence-length mismatch: {self.split}.tokens_per_sample "
                f"is {self.conditioning_tokens_per_example}, but "
                f"tokenizer.sequence_length is {sequence_length}."
            )

        token_shape = tokenizer_metadata.get("token_shape")
        if not isinstance(token_shape, list) or not token_shape:
            raise ValueError(
                "Manifest field 'tokenizer.token_shape' must be a non-empty list."
            )
        dimensions = [
            _require_int(value, f"tokenizer.token_shape[{index}]", minimum=1)
            for index, value in enumerate(token_shape)
        ]
        if int(np.prod(dimensions)) != sequence_length:
            raise ValueError(
                "The tokenizer token shape does not match its sequence length."
            )
        return dtype

    def _validate_class_vocabulary(
        self,
        manifest: dict[str, Any],
    ) -> tuple[tuple[str | int, ...], dict[str, int]]:
        if self.label_type not in {"string", "integer"}:
            raise ValueError(
                "Manifest field 'label_type' must be 'string' or 'integer'."
            )
        class_names = manifest.get("class_names")
        if not isinstance(class_names, list) or len(class_names) != self.num_classes:
            raise ValueError("Manifest class_names length must equal num_classes.")
        expected_type = str if self.label_type == "string" else int
        if any(
            isinstance(name, bool) or not isinstance(name, expected_type)
            for name in class_names
        ):
            raise ValueError(
                f"Every class name must have label type {self.label_type}."
            )
        if len({str(name) for name in class_names}) != self.num_classes:
            raise ValueError("Manifest class names must be unique.")

        class_to_index = _require_dict(manifest.get("class_to_index"), "class_to_index")
        if any(
            not isinstance(name, str)
            or isinstance(index, bool)
            or not isinstance(index, int)
            for name, index in class_to_index.items()
        ):
            raise ValueError("Manifest class_to_index must map strings to integers.")
        expected_mapping = {str(name): index for index, name in enumerate(class_names)}
        if class_to_index != expected_mapping:
            raise ValueError(
                "Manifest class_to_index must map class_names to contiguous indices "
                "in the same order."
            )
        return tuple(class_names), dict(expected_mapping)

    def _validate_class_counts(self, split_metadata: dict[str, Any]) -> tuple[int, ...]:
        counts = split_metadata.get("class_counts")
        if not isinstance(counts, list) or len(counts) != self.num_classes:
            raise ValueError(
                f"Manifest field '{self.split}.class_counts' must contain one "
                "entry per class."
            )
        validated = tuple(
            _require_int(count, f"{self.split}.class_counts[{index}]", minimum=0)
            for index, count in enumerate(counts)
        )
        if sum(validated) != self.num_examples:
            raise ValueError(
                f"Manifest field '{self.split}.class_counts' must sum to num_samples."
            )
        if self.split == "train" and any(count == 0 for count in validated):
            raise ValueError("Every train-derived class must occur in the train split.")
        return validated

    def _validate_binary_file(self) -> None:
        if not self.data_path.is_file():
            raise ValueError(f"Dataset binary file does not exist: '{self.data_path}'.")
        row_width = self.conditioning_tokens_per_example + 1
        expected_bytes = self.num_examples * row_width * self.dtype.itemsize
        actual_bytes = self.data_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Dataset binary size is {actual_bytes} bytes; manifest shape "
                f"({self.num_examples}, {row_width}) with dtype {self.dtype.name} "
                f"requires {expected_bytes} bytes."
            )

        data = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode="r",
            shape=(self.num_examples, row_width),
        )
        labels = np.asarray(data[:, -1], dtype=np.int64)
        if np.any(labels < 0) or np.any(labels >= self.num_classes):
            invalid = int(labels[(labels < 0) | (labels >= self.num_classes)][0])
            raise ValueError(
                f"Split '{self.split}' contains class index {invalid}; expected an "
                f"index in [0, {self.num_classes})."
            )
        observed_counts = tuple(
            int(value) for value in np.bincount(labels, minlength=self.num_classes)
        )
        if observed_counts != self.class_counts:
            raise ValueError(
                f"Stored labels for split '{self.split}' do not match manifest "
                "class_counts."
            )
        del data

    def label_to_index(self, label: str | int) -> int:
        """Resolve a raw class label using the train-derived manifest mapping."""
        expected_type = str if self.label_type == "string" else int
        if isinstance(label, bool) or not isinstance(label, expected_type):
            raise KeyError(f"Label {label!r} does not have type {self.label_type}.")
        try:
            return self.class_to_index[str(label)]
        except KeyError as exc:
            raise KeyError(f"Unknown class label {label!r}.") from exc

    def index_to_label(self, index: int) -> str | int:
        """Return the raw class label for a contiguous class index."""
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.num_classes
        ):
            raise KeyError(f"Unknown class index {index!r}.")
        return self.class_names[index]

    def _open(self) -> None:
        process_id = os.getpid()
        if self._data is not None and self._data_process_id != process_id:
            self.close()
        if self._data is None:
            self._data = np.memmap(
                self.data_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.num_examples, self.conditioning_tokens_per_example + 1),
            )
            self._data_process_id = process_id

    def close(self) -> None:
        """Close this process's lazy memmap, if it has been opened."""
        if self._data is not None:
            mmap = getattr(self._data, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._data = None
        self._data_process_id = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_data"] = None
        state["_data_process_id"] = None
        return state

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        assert self._data is not None
        row = self._data[index]
        audio_tokens = torch.from_numpy(
            row[: self.conditioning_tokens_per_example].copy()
        ).long()
        class_index = torch.tensor(int(row[-1]), dtype=torch.long)
        return audio_tokens, class_index


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_audio_classification_dataloader(
    dataset: AudioClassificationDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a deterministic loader, automatically sharded when using DDP.

    Distributed training uses equal, non-overlapping shards by dropping at most
    ``world_size - 1`` examples before partitioning.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers to be positive.")

    distributed = dist.is_available() and dist.is_initialized()
    resolved_world_size = world_size
    if resolved_world_size is None:
        resolved_world_size = dist.get_world_size() if distributed else 1
    resolved_rank = rank
    if resolved_rank is None:
        resolved_rank = dist.get_rank() if distributed else 0
    if resolved_world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not 0 <= resolved_rank < resolved_world_size:
        raise ValueError("rank must be in [0, world_size).")

    sampler: Sampler[int] | None = None
    if resolved_world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=resolved_world_size,
            rank=resolved_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
    )

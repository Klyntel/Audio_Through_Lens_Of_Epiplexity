"""Shared token-stream preparation for the bootstrap-modalities experiment.

This is deliberately experiment-local.  It shares the on-disk EpiAudio
contract between the text, chess, and image adapters without altering the
audio-specific ``prepare_ds`` worker pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import DTypeLike
from omegaconf import OmegaConf


METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TokenDatasetContract:
    """Static representation and provenance information for one dataset."""

    dataset: str
    tokenizer: str
    vocab_size: int
    sequence_length: int
    source: dict[str, Any]
    dtype: np.dtype = np.dtype(np.int16)


@dataclass
class TokenDatasetWriter:
    """Write token streams, optional aligned sidecars, and EpiAudio metadata."""

    output_dir: Path
    contract: TokenDatasetContract
    _splits: dict[str, dict[str, int | str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        protected = [self.output_dir / name for name in ("train.bin", "test.bin", "metadata.yaml")]
        if any(path.exists() for path in protected):
            raise FileExistsError(
                f"Refusing to overwrite prepared data in {self.output_dir}. Remove it explicitly "
                "or choose another --output path."
            )

    def write_split(
        self,
        name: str,
        chunks: Iterable[np.ndarray | tuple[np.ndarray, Mapping[str, np.ndarray]]],
        *,
        num_tokens: int,
        num_samples: int,
        sidecars: Mapping[str, DTypeLike] | None = None,
    ) -> None:
        """Write one flattened token split and any same-length auxiliary streams."""
        if name in self._splits:
            raise ValueError(f"Split '{name}' was already written.")
        if num_tokens <= 0 or num_samples <= 0:
            raise ValueError(f"Split '{name}' must contain tokens and samples.")

        token_stream = np.memmap(
            self.output_dir / f"{name}.bin",
            dtype=self.contract.dtype,
            mode="w+",
            shape=(num_tokens,),
        )
        sidecar_streams = {
            sidecar_name: np.memmap(
                self.output_dir / f"{name}_{sidecar_name}.bin",
                dtype=np.dtype(dtype),
                mode="w+",
                shape=(num_tokens,),
            )
            for sidecar_name, dtype in (sidecars or {}).items()
        }
        cursor = 0
        for chunk in chunks:
            if isinstance(chunk, tuple):
                token_chunk, sidecar_chunks = chunk
            else:
                token_chunk, sidecar_chunks = chunk, {}
            flat_tokens = np.asarray(token_chunk, dtype=self.contract.dtype).reshape(-1)
            end = cursor + len(flat_tokens)
            if end > num_tokens:
                raise RuntimeError(f"Split '{name}' produced more than {num_tokens} tokens.")
            token_stream[cursor:end] = flat_tokens
            if set(sidecar_chunks) != set(sidecar_streams):
                raise ValueError(
                    f"Split '{name}' sidecars must be {sorted(sidecar_streams)}, "
                    f"got {sorted(sidecar_chunks)}."
                )
            for sidecar_name, sidecar_chunk in sidecar_chunks.items():
                flat_sidecar = np.asarray(sidecar_chunk, dtype=sidecar_streams[sidecar_name].dtype).reshape(-1)
                if len(flat_sidecar) != len(flat_tokens):
                    raise ValueError(f"Split '{name}' sidecar '{sidecar_name}' is not token-aligned.")
                sidecar_streams[sidecar_name][cursor:end] = flat_sidecar
            cursor = end
        token_stream.flush()
        for stream in sidecar_streams.values():
            stream.flush()
        if cursor != num_tokens:
            raise RuntimeError(f"Split '{name}' wrote {cursor} tokens, expected {num_tokens}.")
        self._splits[name] = {"num_samples": num_samples, "num_tokens": num_tokens}

    def write_splits(
        self,
        chunks: Iterable[Mapping[str, np.ndarray]],
        *,
        splits: Mapping[str, tuple[int, int]],
    ) -> None:
        """Write several token-only splits from one source pass.

        This supports split assignment that is decided per source batch, as in
        CIFAR-5M, without accumulating batches in memory or re-tokenizing the
        source once per split.
        """
        names = set(splits)
        if not names or names & set(self._splits):
            raise ValueError("Each non-empty split set may be written only once.")
        streams = {
            name: np.memmap(
                self.output_dir / f"{name}.bin",
                dtype=self.contract.dtype,
                mode="w+",
                shape=(num_tokens,),
            )
            for name, (num_tokens, _) in splits.items()
        }
        cursors = dict.fromkeys(splits, 0)
        for batch in chunks:
            if set(batch) != names:
                raise ValueError(f"Expected split batch keys {sorted(names)}, got {sorted(batch)}.")
            for name, token_chunk in batch.items():
                flat_tokens = np.asarray(token_chunk, dtype=self.contract.dtype).reshape(-1)
                end = cursors[name] + len(flat_tokens)
                if end > len(streams[name]):
                    raise RuntimeError(f"Split '{name}' produced too many tokens.")
                streams[name][cursors[name] : end] = flat_tokens
                cursors[name] = end
        for name, stream in streams.items():
            stream.flush()
            expected_tokens, num_samples = splits[name]
            if cursors[name] != expected_tokens:
                raise RuntimeError(f"Split '{name}' wrote {cursors[name]} tokens, expected {expected_tokens}.")
            self._splits[name] = {"num_samples": num_samples, "num_tokens": expected_tokens}

    def finalize(self) -> None:
        """Publish metadata only after the required streams have completed."""
        if {"train", "test"} - set(self._splits):
            raise ValueError("Both train and test must be written before publishing metadata.")
        metadata = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "dataset": self.contract.dataset,
            "tokenizer": {
                "name": self.contract.tokenizer,
                "dtype": self.contract.dtype.name,
                "sequence_length": self.contract.sequence_length,
                "token_shape": [self.contract.sequence_length],
            },
            "source": self.contract.source,
        }
        for name, split in self._splits.items():
            metadata[name] = {**split, "dtype": self.contract.dtype.name}
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        OmegaConf.save(
            OmegaConf.create(
                {"model": {"V": self.contract.vocab_size, "L": self.contract.sequence_length}}
            ),
            self.output_dir / "metadata.yaml",
        )

"""Stream AVSpeech into a token dataset without materializing source rows in RAM.

The standard audio preparer relies on a sized Hugging Face Dataset so it can
preallocate a memmap.  AVSpeech has millions of source clips, so this small
experiment helper keeps only one tokenizer batch in memory and appends tokens
directly to a temporary binary file before atomically publishing each split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import load_dataset
from tqdm import tqdm

from epiaudio.dataset.audio_tokenizers import EnCodecTokenizer
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS


DATASET_NAME = "ProgramComputer/avspeech-visual-audio"
SPLIT_ROWS = {"train": 2_621_845, "test": 183_273}
SECONDS = 5
METADATA_SCHEMA_VERSION = 1
AUDIO_TOKEN_TASK_TYPE = "audio_token_sequence"


def _audio_clip(
    row: dict[str, Any], *, generator: np.random.Generator
) -> tuple[torch.Tensor, int]:
    samples = row["audio"].get_all_samples()
    waveform, sample_rate = samples.data, samples.sample_rate
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    expected_samples = SECONDS * sample_rate
    if waveform.shape[-1] < expected_samples:
        waveform = functional.pad(waveform, (0, expected_samples - waveform.shape[-1]))
    else:
        start = generator.integers(0, waveform.shape[-1] - expected_samples + 1)
        waveform = waveform[:, start : start + expected_samples]
    return waveform, sample_rate


def _write_split(
    *,
    split: str,
    limit: int,
    destination: Path,
    tokenizer: Any,
    tokens_per_sample: int,
    dtype: np.dtype[Any],
    batch_size: int,
    seed: int,
) -> tuple[int, int]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing {destination}.")
    stream = load_dataset(
        DATASET_NAME,
        split=split,
        columns=["clip_id", "audio"],
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10_000)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Remove unfinished output before retrying: {temporary}")
    random = np.random.default_rng(seed)
    count = 0
    skipped = 0
    total = SPLIT_ROWS[split] if limit < 0 else min(limit, SPLIT_ROWS[split])
    try:
        with (
            temporary.open("xb") as output,
            tqdm(total=total, desc=split, unit="clip") as progress,
        ):
            batch: list[dict[str, Any]] = []
            for row in stream:
                if row.get("audio") is None:
                    skipped += 1
                    continue
                batch.append(row)
                if len(batch) < batch_size and (
                    limit < 0 or count + len(batch) < limit
                ):
                    continue
                clips = [_audio_clip(item, generator=random) for item in batch]
                waveforms, sample_rates = zip(*clips)
                token_rows = np.asarray(
                    tokenizer.tokenize_batch(
                        list(waveforms),
                        list(sample_rates),
                        tokens_per_sample,
                        dtype=dtype,
                    ),
                    dtype=dtype,
                ).reshape(len(batch), tokens_per_sample)
                token_rows.tofile(output)
                count += len(batch)
                progress.update(len(batch))
                batch = []
                if limit >= 0 and count == limit:
                    break
            if batch and (limit < 0 or count < limit):
                clips = [_audio_clip(item, generator=random) for item in batch]
                waveforms, sample_rates = zip(*clips)
                token_rows = np.asarray(
                    tokenizer.tokenize_batch(
                        list(waveforms),
                        list(sample_rates),
                        tokens_per_sample,
                        dtype=dtype,
                    ),
                    dtype=dtype,
                ).reshape(len(batch), tokens_per_sample)
                token_rows.tofile(output)
                count += len(batch)
                progress.update(len(batch))
            if limit >= 0 and count != limit:
                raise RuntimeError(
                    f"AVSpeech {split} ended after {count} valid clips; expected {limit}."
                )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream AVSpeech into fixed-length audio tokens."
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-limit",
        type=int,
        default=-1,
        help="-1 streams the complete train split.",
    )
    parser.add_argument(
        "--test-limit", type=int, default=-1, help="-1 streams the complete test split."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    token_shape = TOKENIZER_SPECS["encodec"].shape
    tokens_per_sample = int(np.prod(token_shape))
    dtype = np.dtype(np.int16)
    output_dir = args.output_root / "avspeech_encodec"
    output_dir.mkdir(parents=True, exist_ok=True)

    EnCodecTokenizer.download_models_local()
    tokenizer = EnCodecTokenizer(device=args.device, local_files_only=True)
    split_results = {
        split: _write_split(
            split=split,
            limit=limit,
            destination=output_dir / f"{split}.bin",
            tokenizer=tokenizer,
            tokens_per_sample=tokens_per_sample,
            dtype=dtype,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        for split, limit in (("train", args.train_limit), ("test", args.test_limit))
    }
    EnCodecTokenizer.write_metadata(str(output_dir))
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "task_type": AUDIO_TOKEN_TASK_TYPE,
        "source_dataset": DATASET_NAME,
        "streaming": True,
        "seed": args.seed,
        "tokenizer": {
            "dtype": dtype.name,
            "sequence_length": tokens_per_sample,
            "token_shape": list(token_shape),
        },
        **{
            split: {
                "num_samples": count,
                "null_audio_rows_skipped": skipped,
                "tokens_per_sample": tokens_per_sample,
                "dtype": dtype.name,
            }
            for split, (count, skipped) in split_results.items()
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **split_results}, indent=2))


if __name__ == "__main__":
    main()

"""Drop rows whose audio file is corrupted from an AudioDataset.

Every audio file is opened and decoded window by window -- never holding more than one
chunk of decoded audio in memory at a time, no matter how long the source file is -- to
confirm it isn't corrupted. The work is sharded across a bounded pool of worker processes,
each running a small thread pool, so a multi-core box validates a large dataset in parallel
instead of file by file.

Note: torchcodec's AudioDecoder decodes audio on CPU via ffmpeg; it has no GPU decode path
(unlike its VideoDecoder, which does), so worker count is sized off CPU count regardless of
how many GPUs are visible.
"""

import sys
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from typing import Any, cast

from datasets import DatasetDict
from torchcodec.decoders import AudioDecoder  # pyright: ignore[reportPrivateImportUsage]

from audio_preprocessing.datasets import AudioDataset
from audio_preprocessing.realize import open_source
from epiaudio.dataset.load_dataloaders import AUDIO_PREPROCESSING_DATASETS


# Bounds how much decoded audio one validation pass holds in memory at once, regardless of
# how long the source file is.
_CHUNK_SECONDS = 30.0
# I/O-bound decode calls, so a handful of threads share each worker process's CPU budget.
_THREADS_PER_WORKER = 4
# More processes plateau once the nested thread pools fill the machine (see mds.py).
_MAX_CPU_WORKERS = 16


def validate_audio(dataset: AudioDataset) -> AudioDataset:
    """Return a copy of path-backed datasets with corrupted files removed.

    Args:
        dataset: A dataset whose splits may provide an ``audio_path`` column.
            Datasets with embedded Hugging Face ``audio`` values have no stable
            filesystem path to validate, so they are returned unchanged.

    Returns:
        AudioDataset: A new dataset, identical to ``dataset`` except that any split with
        corrupted audio has those rows removed.
    """
    data = dataset.data
    cleaned = dict()
    for split, rows in data.items():
        if "audio_path" not in rows.column_names:
            warnings.warn(
                f"Skipping path validation for '{split}': no audio_path column.",
                stacklevel=2,
            )
            cleaned[split] = rows
            continue
        ok = _validate_paths(rows["audio_path"])
        cleaned[split] = rows.select([index for index, valid in enumerate(ok) if valid])

    cleaned_data = DatasetDict(cleaned)
    cleaned_dataset = AudioDataset(
        data=cleaned_data,
        label_source=dataset.label_source
    )

    for split in data:
        n_orig = len(data[split])
        n_corrupt = n_orig - len(cleaned_data[split])
        print(f"count: orig: {n_orig}, corrupt: {n_corrupt}")
        if n_corrupt > 0:
            warnings.warn(f"{n_corrupt} out of {n_orig} total audio files were corrupt in {split} split.")

    return cleaned_dataset


def _validate_paths(paths: list[str]) -> list[bool]:
    """Return one bool per path: True if that file decodes cleanly end to end."""
    paths = list(paths)
    n = len(paths)

    workers = _resolve_workers(n)

    if workers == 1:
        return _validate_chunk(paths)

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(_validate_chunk, paths[start:end])
            for start, end in _partition_bounds(n, workers)
        ]
        chunks = [future.result() for future in futures]

    return [ok for chunk in chunks for ok in chunk]


def _validate_chunk(paths: list[str]) -> list[bool]:
    """Validate one shard of paths in its own process."""
    with ThreadPoolExecutor(max_workers=_THREADS_PER_WORKER) as pool:
        return list(pool.map(_is_decodable, paths))


def _is_decodable(path: str) -> bool:
    """Decode the full file in bounded chunks, discarding samples as soon as they're read."""
    try:
        decoder = AudioDecoder(cast(Any, open_source(path)))
        duration = decoder.metadata.duration_seconds
        if duration is None:
            return False

        cursor = 0.0
        while cursor < duration:
            end = min(cursor + _CHUNK_SECONDS, duration)
            decoder.get_samples_played_in_range(cursor, end)
            cursor = end
        return True
    except Exception:
        warnings.warn(f"Could not decode {path}.")
        return False


def _resolve_workers(n: int) -> int:
    return min(_MAX_CPU_WORKERS, os.cpu_count() or 1, n)


def _partition_bounds(n: int, workers: int) -> list[tuple[int, int]]:
    size = math.ceil(n / workers)
    return [(start, min(start + size, n)) for start in range(0, n, size)]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError("Dataset name must be provided.")

    if len(sys.argv) > 2:
        raise ValueError(f"Extraneous arguments found: {[sys.argv[i] for i in range(2, len(sys.argv))]}")

    dataset_name = sys.argv[1]
    if dataset_name not in AUDIO_PREPROCESSING_DATASETS:
        raise ValueError(f"Dataset name must be in {list(AUDIO_PREPROCESSING_DATASETS.keys())}, got {dataset_name}.")

    ds_cfg  = AUDIO_PREPROCESSING_DATASETS[dataset_name]
    ds = ds_cfg["loader"]()

    validate_audio(ds)

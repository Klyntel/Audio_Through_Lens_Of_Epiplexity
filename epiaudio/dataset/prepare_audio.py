""" Convert AudioDatasets to tokens for classification tasks.

Usage: uv run python epiaudio/dataset/prepre_audio.py <dataset> <comma-separated tokenizers> <class_column> <num_workers[int]> <batch_size[int]> <threads_per_worker[int]>
e.g.   uv run python epiaudio/dataset/prepare_audio.py syntheory_intervals dac,encodec midi_program_name 8 16 8

Defaults set to 16 32 8
"""

import json
import math
import multiprocessing
import os
import queue
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast
import warnings

from audio_preprocessing.datasets import AudioDataset
from epiaudio.dataset.audio_tokenizers import (
    WhisperTokenizer,
    DACTokenizer,
    EnCodecTokenizer,
    SQCodecTokenizer,
    XCodecTokenizer
)
from epiaudio.dataset.wavtokenizer import WavTokenizerTokenizer
from epiaudio.dataset.tokenizer_specs import TOKENIZER_SPECS
from epiaudio.dataset.load_birdset import load_birdset, birdset_decode
from epiaudio.dataset.load_locata import load_locata_traintest, locata_decode
from epiaudio.dataset.load_starss23 import load_starss23_traintest, starss23_decode
from epiaudio.dataset.validate_audio import validate_audio
from epiaudio.dataset.load_dataloaders import AUDIO_PREPROCESSING_DATASETS

from transformers.utils import logging as hf_logging
import numpy as np
import tqdm
import torch.nn.functional as F
import random
import torch


METADATA_SCHEMA_VERSION = 1
CLASSIFICATION_TASK_TYPE = "single_label_multiclass"
AUDIO_TOKEN_TASK_TYPE = "audio_token_sequence"
ClassLabel = str | int


def set_all_seeds(seed: int = 0) -> None:
    """
    Allow datasets to be somewhat reproduceable to create
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clip(args, padding=True, seconds_fix=5) -> tuple[torch.Tensor, int]:
    """
    Loads a single sample of audio to be processed

    Current behavior will sample a random seconds_fix 
    chunk of a recording for estimating epiplexity.

    If tokenizer handles padding, padding can be disabled.
    """
    split, idx, decode_fn = args
    if decode_fn is not None:
        y, sr = decode_fn(split[idx])
    else:
        samples = split[idx]["audio"].get_all_samples()
        y, sr = samples.data, samples.sample_rate

    ## padding and tuncation
    expected_size = seconds_fix * sr

    # Padding false implies that padding is handled by tokenizer
    if y.shape[-1] < expected_size and padding:
        y = F.pad(y, pad=(0, expected_size - y.shape[-1]), mode='constant', value=0)
    else:
        start = np.random.randint(0, y.shape[-1] - expected_size + 1) #nonexclusive
        y = y[:, start : start + expected_size]
    return y, sr


def worker_ranges(num_samples: int, num_workers: int) -> list[tuple[int, int]]:
    """Split [0, num_samples) into up to num_workers contiguous, non-overlapping ranges.
    Each worker will be assigned deterministically a sample to process
    """
    per_worker = math.ceil(num_samples / num_workers)
    ranges = []
    for i in range(num_workers):
        start = i * per_worker
        if start >= num_samples:
            break
        ranges.append((start, min(start + per_worker, num_samples)))
    return ranges


def _normalize_class_label(
    label: object,
    *,
    class_column: str,
    split: str,
    row_index: int,
) -> ClassLabel:
    """Return a JSON-safe scalar class label or raise a useful error."""
    if isinstance(label, (bool, np.bool_)):
        raise ValueError(
            f"Column '{class_column}' in split '{split}' must contain string or "
            f"integer class labels; row {row_index} contains a boolean."
        )
    if isinstance(label, (int, np.integer)):
        return int(cast(Any, label))
    if isinstance(label, (str, np.str_)):
        return str(label)
    if isinstance(label, list) and len(label) == 1:
        item = label[0]
        if isinstance(item, int) or isinstance(item, str):
            return item
    raise ValueError(
        f"Column '{class_column}' in split '{split}' must contain one scalar "
        f"string or integer per example; row {row_index} contains "
        f"{type(label).__name__}."
    )


def _normalized_split_labels(
    ds_split: Any,
    *,
    class_column: str,
    split: str,
) -> list[ClassLabel]:
    column_names = getattr(ds_split, "column_names", None)
    if column_names is not None and class_column not in column_names:
        raise ValueError(
            f"Classification column '{class_column}' is missing from split "
            f"'{split}'. Available columns: {column_names}."
        )

    try:
        labels = ds_split[class_column]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Classification column '{class_column}' is missing from split "
            f"'{split}'."
        ) from exc

    return [
        _normalize_class_label(
            label,
            class_column=class_column,
            split=split,
            row_index=row_index,
        )
        for row_index, label in enumerate(labels)
    ]


def _build_class_index(
    train_labels: list[ClassLabel],
    *,
    class_column: str,
) -> tuple[list[ClassLabel], dict[ClassLabel, int]]:
    if not train_labels:
        raise ValueError(
            f"Training split has no labels in classification column "
            f"'{class_column}'."
        )

    label_types = {type(label) for label in train_labels}
    if len(label_types) != 1:
        type_names = ", ".join(sorted(label_type.__name__ for label_type in label_types))
        raise ValueError(
            f"Column '{class_column}' in split 'train' mixes label types "
            f"({type_names}); use one categorical label type."
        )

    class_names = sorted(set(train_labels))
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return class_names, class_to_index


def _encode_class_labels(
    labels: list[ClassLabel],
    *,
    class_to_index: dict[ClassLabel, int],
    class_column: str,
    split: str,
) -> np.ndarray:
    expected_type = type(next(iter(class_to_index)))
    encoded = np.empty(len(labels), dtype=np.int64)

    for row_index, label in enumerate(labels):
        if type(label) is not expected_type:
            raise ValueError(
                f"Column '{class_column}' in split '{split}' must use "
                f"{expected_type.__name__} labels like the training split; row "
                f"{row_index} contains {type(label).__name__}."
            )
        try:
            encoded[row_index] = class_to_index[label]
        except KeyError as exc:
            raise ValueError(
                f"Unknown class label {label!r} in column '{class_column}', "
                f"split '{split}', row {row_index}. All evaluation labels must "
                "occur in the training split."
            ) from exc

    return encoded


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """Atomically publish JSON so readers never observe a partial manifest."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def tokenize_worker(
    worker_id,
    tokenizer_cls,
    device,
    mem_path,
    dtype,
    tokens_per_sample,
    num_samples,
    ds_split,
    class_indices,
    decode_fn,
    start_index,
    end_index,
    batch_size,
    threads_per_worker,
    seed,
    progress_queue
) -> tuple[int, int]:

    """
    Tokenize a chunk of the dataset 
    Owned by a worker

    This code runs on each child process
    """

    set_all_seeds(seed + worker_id)

    classification = class_indices is not None

    # Without this, each worker's torch/BLAS backend defaults to using all visible cores —
    # with num_workers processes running concurrently that oversubscribes the machine badly
    # (num_workers * all_cores threads competing for num_workers * threads_per_worker's
    # worth of actual CPU). Cap this process's intra-op parallelism instead.
    torch.set_num_threads(threads_per_worker)

    # Model files were pre-downloaded into the local HF cache by the main process
    # (see prepare_ds) — force this worker to read from that cache instead of each
    # worker independently hitting the HuggingFace Hub API and risking rate limits.
    hf_logging.disable_progress_bar()
    tokenizer = tokenizer_cls(device=device, local_files_only=True)
    shape = (
        (num_samples, tokens_per_sample + 1)
        if classification
        else (num_samples * tokens_per_sample,)
    )
    memmap = np.memmap(mem_path, dtype=dtype, mode="r+", shape=shape)
    for batch_start in range(start_index, end_index, batch_size):
        batch_end = min(batch_start + batch_size, end_index)
        indices = range(batch_start, batch_end)

        clips = [load_clip((ds_split, idx, decode_fn)) for idx in indices]
        ys, srs = zip(*clips)

        tokens_batch = tokenizer.tokenize_batch(
            list(ys),
            list(srs),
            tokens_per_sample,
            dtype=dtype,
        )
        batch_length = batch_end - batch_start
        tokens_batch = np.asarray(tokens_batch)
        expected_tokens = batch_length * tokens_per_sample
        if tokens_batch.size != expected_tokens:
            raise ValueError(
                f"Tokenizer returned {tokens_batch.size} values for a batch that "
                f"requires {expected_tokens} ({batch_length} samples x "
                f"{tokens_per_sample} tokens)."
            )
        token_rows = tokens_batch.reshape(batch_length, tokens_per_sample)

        if classification:
            memmap[batch_start:batch_end, :tokens_per_sample] = token_rows
            memmap[batch_start:batch_end, tokens_per_sample] = class_indices[
                batch_start:batch_end
            ]
        else:
            token_start = batch_start * tokens_per_sample
            token_end = batch_end * tokens_per_sample
            memmap[token_start:token_end] = token_rows.reshape(-1)
        progress_queue.put((worker_id, len(indices)))
    memmap.flush()
    return start_index, end_index


def prepare_ds(
    *,
    ds: AudioDataset,
    tokenizer_cls,
    expected_token_shape: tuple,
    dtype: np.dtype,
    class_column: str | None,
    train_path: str,
    test_path: str,
    val_path: str = "",
    metadata_path: str = "",
    decode_fn=None,
    num_workers=8,
    batch_size=8,
    threads_per_worker=8,
    seed=0,
) -> None:
    """Tokenize an audio dataset and write binary splits plus one manifest."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if threads_per_worker <= 0:
        raise ValueError("threads_per_worker must be positive.")

    token_shape = tuple(int(dimension) for dimension in expected_token_shape)
    if not token_shape or any(dimension <= 0 for dimension in token_shape):
        raise ValueError("expected_token_shape must contain positive dimensions.")

    token_dtype = np.dtype(dtype)
    tokens_per_sample = int(np.prod(token_shape))
    split_paths = [("train", train_path), ("test", test_path)]
    if val_path:
        split_paths.append(("val", val_path))
    for split, _ in split_paths:
        if split not in ds.data:
            raise ValueError(f"Dataset is missing required split '{split}'.")

    metadata_path = metadata_path or os.path.join(
        os.path.dirname(train_path),
        "metadata.json",
    )
    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "task_type": (
            CLASSIFICATION_TASK_TYPE if class_column else AUDIO_TOKEN_TASK_TYPE
        ),
        "tokenizer": {
            "dtype": token_dtype.name,
            "sequence_length": tokens_per_sample,
            "token_shape": list(token_shape),
        },
    }

    # Shuffle and remove unresolved audio before deriving the class vocabulary or
    # sizing output files. This keeps invalid rows out of both the representation
    # and the train-derived classification schema.
    shuffled_splits: list[tuple[str, str, Any]] = []
    for split, path in split_paths:
        # Hugging Face Dataset.shuffle returns a new dataset rather than mutating.
        ds_split = ds.data[split].shuffle(seed=seed)
        num_samples = len(ds_split)
        if num_samples == 0:
            raise ValueError(f"Dataset split '{split}' is empty.")
        shuffled_splits.append((split, path, ds_split))

    class_to_index: dict[ClassLabel, int] | None = None
    num_classes = 0
    if class_column:
        train_split = next(
            ds_split
            for split, _, ds_split in shuffled_splits
            if split == "train"
        )
        train_labels = _normalized_split_labels(
            train_split,
            class_column=class_column,
            split="train",
        )
        class_names, class_to_index = _build_class_index(
            train_labels,
            class_column=class_column,
        )
        num_classes = len(class_names)
        if not np.issubdtype(token_dtype, np.integer):
            raise ValueError(
                "Classification datasets require an integer storage dtype because "
                "the final value in each row is a class index."
            )
        if num_classes - 1 > np.iinfo(token_dtype).max:
            raise ValueError(
                f"dtype {token_dtype.name} cannot represent {num_classes} class "
                "indices. Choose a wider integer dtype."
            )

        metadata.update(
            {
                "label_column": class_column,
                "label_position": -1,
                "label_type": (
                    "string" if isinstance(class_names[0], str) else "integer"
                ),
                "class_names": class_names,
                "class_to_index": {
                    str(class_name): class_to_index[class_name]
                    for class_name in class_names
                },
                "num_classes": num_classes,
            }
        )

    # Validate and encode every split before creating any binary output. The class
    # vocabulary is always derived from the filtered train split and reused unchanged.
    prepared_splits: list[tuple[str, str, Any, np.ndarray | None]] = []
    for split, path, ds_split in shuffled_splits:
        class_indices = None
        if class_column and class_to_index is not None:
            labels = _normalized_split_labels(
                ds_split,
                class_column=class_column,
                split=split,
            )
            class_indices = _encode_class_labels(
                labels,
                class_to_index=class_to_index,
                class_column=class_column,
                split=split,
            )
        prepared_splits.append((split, path, ds_split, class_indices))

    # Download once in the parent so workers only read the local model cache.
    tokenizer_cls.download_models_local()

    # metadata.yaml remains the source of model.V/model.L for sweep configuration.
    ds_path = os.path.dirname(train_path) or "."
    os.makedirs(ds_path, exist_ok=True)
    try:
        tokenizer_cls.write_metadata(ds_path)
    except NotImplementedError as exc:
        print(f"[prepare_audio] [WARNING]: no metadata.yaml written for '{ds_path}': {exc}")

    # Cap each child process's BLAS/OpenMP thread pools to avoid oversubscription.
    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for variable in thread_variables:
        os.environ[variable] = str(threads_per_worker)

    n_gpus = torch.cuda.device_count()
    ctx = multiprocessing.get_context("spawn")  # CUDA contexts do not survive fork.
    with ctx.Manager() as manager:
        for split, path, ds_split, class_indices in prepared_splits:
            num_samples = len(ds_split)
            num_tokens = num_samples * tokens_per_sample
            shape = (
                (num_samples, tokens_per_sample + 1)
                if class_indices is not None
                else (num_tokens,)
            )
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.memmap(path, dtype=token_dtype, mode="w+", shape=shape).flush()

            ranges = worker_ranges(num_samples, num_workers)
            progress_queue = manager.Queue()
            pbars = [
                tqdm.tqdm(
                    total=end_index - start_index,
                    position=worker_id,
                    desc=f"worker {worker_id}",
                )
                for worker_id, (start_index, end_index) in enumerate(ranges)
            ]
            try:
                with ProcessPoolExecutor(
                    max_workers=len(ranges),
                    mp_context=ctx,
                ) as executor:
                    futures = [
                        executor.submit(
                            tokenize_worker,
                            worker_id,
                            tokenizer_cls,
                            (
                                f"cuda:{worker_id % n_gpus}"
                                if n_gpus > 0
                                else "cpu"
                            ),
                            path,
                            token_dtype,
                            tokens_per_sample,
                            num_samples,
                            ds_split,
                            class_indices,
                            decode_fn,
                            start_index,
                            end_index,
                            batch_size,
                            threads_per_worker,
                            seed,
                            progress_queue,
                        )
                        for worker_id, (start_index, end_index) in enumerate(ranges)
                    ]
                    while not all(future.done() for future in futures):
                        try:
                            worker_id, completed = progress_queue.get(timeout=0.2)
                            pbars[worker_id].update(completed)
                        except queue.Empty:
                            pass
                    while not progress_queue.empty():
                        worker_id, completed = progress_queue.get()
                        pbars[worker_id].update(completed)
                    for future in futures:
                        future.result()  # Re-raise any worker exception.
            finally:
                for pbar in pbars:
                    pbar.close()

            split_metadata: dict[str, Any] = {
                "num_samples": num_samples,
                "tokens_per_sample": tokens_per_sample,
                "dtype": token_dtype.name,
            }
            if class_indices is not None:
                split_metadata["class_counts"] = np.bincount(
                    class_indices,
                    minlength=num_classes,
                ).tolist()
            metadata[split] = split_metadata
            print(f"done: {path}  |  {num_tokens}")

    # Publish only after all split files have completed successfully.
    _atomic_write_json(metadata_path, metadata)


DATASETS = {
    "birdset_hsn": {"loader": lambda: load_birdset("HSN"), "decode": birdset_decode},
    "birdset_xcl": {"loader": lambda: load_birdset("XCL"), "decode": birdset_decode},
    "locata": {"loader": load_locata_traintest, "decode": locata_decode},
    "starss23": {"loader": load_starss23_traintest, "decode": starss23_decode}
}

DATASETS |= AUDIO_PREPROCESSING_DATASETS

TOKENIZERS = {
    # Whisper: float32 mel spectrogram features (80 mel bins × 500 frames for 5s)
    # Note: continuous features — use for audio-model preprocessing, not epiplexity training directly
    "whisper": {"cls": WhisperTokenizer, "shape": TOKENIZER_SPECS["whisper"].shape, "dtype": np.int16},

    # EnCodec: discrete integer tokens (0–1023), 75 frames/sec → 375 tokens per 5s clip
    # Compatible with epiplexity trainer (model.V=1024)
    "encodec": {"cls": EnCodecTokenizer, "shape": TOKENIZER_SPECS["encodec"].shape, "dtype": np.int16},

    # DAC: discrete integer tokens, 16kHz, 50 frames/sec → 12 codebooks × 250 frames per 5s clip
    "dac":     {"cls": DACTokenizer,     "shape": TOKENIZER_SPECS["dac"].shape, "dtype": np.int16},

    # SQCodec: discrete integer tokens (0–117648), 16kHz, 355.56 frames/sec → 1778 tokens per 5s
    # clip (encode_audio() returns 1800 frames, but the trailing ~22 are encoded padding from
    # EnCodec.preprocess()'s right-pad-to-fill_length step, not real content — tokenize() drops them)
    # Single quantizer (FSQ, levels=[7,7,7,7,7,7]); use model.V=117649 in epiplexity config
    # int32 required: codebook size 117,649 exceeds int16 max (32,767)
    "sqcodec": {"cls": SQCodecTokenizer, "shape": TOKENIZER_SPECS["sqcodec"].shape, "dtype": np.int32},

    "xcodec": {"cls": XCodecTokenizer, "shape":  TOKENIZER_SPECS["xcodec"].shape, "dtype": np.int16},

    # WavTokenizer: discrete 24 kHz tokens, 40 frames/sec → 200 tokens per 5s clip.
    "wavtokenizer": {"cls": WavTokenizerTokenizer, "shape": TOKENIZER_SPECS["wavtokenizer"].shape, "dtype": np.int16},
}

if __name__ == "__main__":
    import sys
    SEED = 0


    set_all_seeds(SEED)

    # Usage: uv run python epiaudio/dataset/prepare_audio.py <dataset> <comma-separated tokenizers> <class_column> <world_size> <num_workers[int]> <batch_size[int]> <threads_per_worker[int]> 
    # e.g.   uv run python epiaudio/dataset/prepare_audio.py syntheory_intervals dac,encodec midi_program_name 8 1 8 16 8
    dataset_name       = sys.argv[1] if len(sys.argv) > 1 else "birdset_hsn"
    tokenizer_names     = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dac", "encodec", "sqcodec", "xcodec"]
    class_column       = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "null" else None # use a class column argument of "null" if you only want audio but want to specify the numerical parameters
    world_size         = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    num_workers        = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    batch_size         = int(sys.argv[6]) if len(sys.argv) > 6 else 32
    threads_per_worker = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    val_bin_fn = "valid.bin" if len(sys.argv) > 8 else ""

    if dataset_name not in DATASETS:
        print(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS)}")
        sys.exit(1)

    ds_cfg  = DATASETS[dataset_name]
    ds = validate_audio(ds_cfg["loader"]())
    orig_len = len(ds.data["train"])
    if class_column:
        trim_len = world_size * (len(ds.data["train"]) // world_size)
        num_removed = len(ds.data["train"]) - trim_len
        if num_removed > 0:
            warnings.warn(
                f"Number of training samples must be a multiple of {world_size}. "
                f"{num_removed} samples were removed. "
                f"{len(ds.data["train"]) - num_removed} samples remain."
            )
        ds.data["train"] = ds.data["train"].select(range(trim_len))
        train_labels = set(ds.data["train"][class_column])
        for split in ds.data:
            if split == "train":
                continue
            trim_split = ds.data[split].filter(
                lambda example: example[class_column] in train_labels
            )
            if len(trim_split) == 0:
                raise RuntimeError(f"{split} split is empty")
            num_removed = len(ds.data[split]) - len(trim_split)
            if num_removed > 0:
                warnings.warn(
                    f"{split} split contains {num_removed} samples "
                    "with class label not seen in the training set. "
                    "These samples have been removed. "
                    f"{len(ds.data[split]) - num_removed} samples remain."
                )
            ds.data[split] = trim_split
    
    new_len = len(ds.data["train"])
    if new_len == 0:
        raise RuntimeError("Training split is empty")

    for tokenizer_name in tokenizer_names:
        if tokenizer_name not in TOKENIZERS:
            warnings.warn(f"Unknown tokenizer '{tokenizer_name}', skipping. Available: {list(TOKENIZERS)}")
            continue

        out_parent_dir = f"{dataset_name}_{tokenizer_name}"
        if class_column:
            out_parent_dir += f"_classify_{class_column}"
        out_dir    = os.path.join("data", out_parent_dir)
        train_path = os.path.join(out_dir, 'train.bin')
        test_path  = os.path.join(out_dir, 'test.bin')
        if len(val_bin_fn) > 0:
            val_path = os.path.join(out_dir, val_bin_fn)
        else:
            val_path = ""
        metadata_path = os.path.join(out_dir, "metadata.json")
        os.makedirs(out_dir, exist_ok=True)

        tok_cfg = TOKENIZERS[tokenizer_name]

        try:
            prepare_ds(
                ds=ds,
                tokenizer_cls=tok_cfg["cls"],
                expected_token_shape=tok_cfg["shape"],
                dtype=tok_cfg["dtype"],
                class_column=class_column,
                train_path=train_path,
                test_path=test_path,
                val_path=val_path,
                metadata_path=metadata_path,
                decode_fn=ds_cfg["decode"],
                num_workers=num_workers,
                batch_size=batch_size,
                threads_per_worker=threads_per_worker,
                seed=SEED
            )
            print(orig_len, class_column, new_len)
        except Exception as e:
            warnings.warn(f"Error in tokenizing dataset with {tokenizer_name}: {e}, skipping.", stacklevel=2)
            continue

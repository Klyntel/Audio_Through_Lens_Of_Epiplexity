"""Load BirdSet from a pre-downloaded Arrow snapshot.

## Environment compatibility

This module runs in the same environment as the rest of the project (datasets>=5.0.0).
It does NOT call load_dataset and therefore never touches BirdSet's custom builder,
which is incompatible with datasets>=4.0.0. Instead it reads the Arrow files written
by download_birdset.py via load_from_disk, which is version-agnostic.

If the snapshot doesn't exist yet, run the download script first:

    uv run epiaudio/dataset/download_birdset.py [HSN XCL ...]
"""

from typing import cast

from datasets import Dataset, DatasetDict, load_from_disk
from audio_preprocessing.datasets import AudioDataset

DEFAULT_SUBSET = "HSN"
DATA_DIR = "data"
DEFAULT_VAL_RATIO = 0.1


def birdset_decode(sample):
    from torchcodec.decoders._audio_decoder import AudioDecoder
    decoder = AudioDecoder(sample["filepath"])
    start = sample.get("start_time")
    end = sample.get("end_time")

    # Train samples have start_time=None — use the first detected event as anchor
    if start is None or end is None:
        events = sample.get("detected_events")
        if events and len(events) > 0:
            start = float(events[0][0])
            end = start + 5.0   # fixed 5s window from event start
        # else: full file (test_5s clips are already 5s)

    if start is not None and end is not None:
        samples = decoder.get_samples_played_in_range(
            start_seconds=float(start),
            stop_seconds=float(end),
        )
    else:
        samples = decoder.get_all_samples()
    return samples.data, int(samples.sample_rate)


def load_birdset(subset: str = DEFAULT_SUBSET, val_ratio: float = DEFAULT_VAL_RATIO, seed: int = 42) -> AudioDataset:
    path = f"{DATA_DIR}/birdset_{subset.lower()}_raw"
    if not __import__("os").path.exists(path):
        raise FileNotFoundError(
            f"BirdSet/{subset} not found at '{path}'. "
            f"Run: uv run epiaudio/dataset/download_birdset.py {subset}"
        )
    # load_from_disk returns Dataset | DatasetDict; we always save a DatasetDict
    ds = cast(DatasetDict, load_from_disk(path))

    # test_5s is pre-chunked into 5-second segments; plain "test" is full soundscapes
    splits: dict[str, Dataset] = {"test": ds["test_5s"]}

    if val_ratio > 0:
        train_val = ds["train"].train_test_split(test_size=val_ratio, seed=seed)
        splits["train"] = cast(Dataset, train_val["train"])
        splits["val"] = cast(Dataset, train_val["test"])
    else:
        splits["train"] = ds["train"]

    return AudioDataset(data=DatasetDict(splits.items()))

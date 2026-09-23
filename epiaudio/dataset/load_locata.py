"""Load the LOCATA corpus (Zenodo record 3630471) into an ``AudioDataset``.

Unlike HF-Hub datasets (e.g. ``load_birdset``) where ``load_dataset`` handles
everything, LOCATA ships as raw ``dev.zip``/``eval.zip`` archives on Zenodo. The
shared ``zenodo_downloader.download_zenodo`` helper fetches and extracts those
archives (each ``foo.zip`` -> ``root/foo/``); this loader then walks the
``task/recording/array`` tree and wraps the clips in an ``AudioDataset``. The
``audio`` column holds a file path, decoded lazily downstream by
``data.realize.read_window``.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, Sequence

from datasets import Audio, Dataset, DatasetDict

from audio_preprocessing.datasets import AudioDataset
from epiaudio.dataset.zenodo_downloader import download_zenodo

ZENODO_RECORD = "3630471"
SPLIT_ARCHIVES = {"train": "dev.zip", "eval": "eval.zip"}  # dev -> train, eval -> eval
VALID_ARRAYS = ("benchmark2", "eigenmike", "dicit", "dummy")
DEFAULT_ROOT = os.path.join("data", "locata_raw")


def _split_dir(root: str, split: str) -> str:
    """Directory ``download_zenodo`` extracts ``split``'s archive into (``root/<stem>``)."""
    return os.path.join(root, os.path.splitext(SPLIT_ARCHIVES[split])[0])


def _scan(split_dir: str, split: str, kinds, arrays, tasks) -> list[dict]:
    """One row per relevant wav, parsing task/recording/array from the path."""
    rows = []
    for path in sorted(glob.glob(os.path.join(split_dir, "**", "audio_*.wav"), recursive=True)):
        task_name, rec_name, array, fname = path.split(os.sep)[-4:]
        if not (task_name.startswith("task") and rec_name.startswith("recording")):
            continue
        kind = "array" if fname.startswith("audio_array") else "source" if fname.startswith("audio_source") else None
        if kind not in kinds or array not in arrays:
            continue
        task = int(task_name[len("task"):])
        if tasks is not None and task not in tasks:
            continue
        rows.append({
            "audio": os.path.abspath(path),
            "split": split,
            "task": task,
            "recording": int(rec_name[len("recording"):]),
            "array": array,
            "kind": kind,
        })
    return rows


def load_locata(
    root: str = DEFAULT_ROOT,
    *,
    download: bool = True,
    splits: Sequence[str] = ("train", "eval"),
    kinds: Sequence[str] = ("array",),
    arrays: Sequence[str] = VALID_ARRAYS,
    tasks: Iterable[int] | None = None,
) -> AudioDataset:
    """Build an ``AudioDataset`` over LOCATA (``train`` <- dev, ``eval`` <- eval)."""
    if download:
        # skip archives for splits we didn't ask for (matched by stem, e.g. "eval")
        skip = [os.path.splitext(a)[0] for s, a in SPLIT_ARCHIVES.items() if s not in splits]
        download_zenodo(ZENODO_RECORD, output_dir=root, do_not_download=skip)

    data = {}
    for split in splits:
        split_dir = _split_dir(root, split)
        rows = _scan(split_dir, split, kinds, arrays, tasks)
        if not rows:
            raise RuntimeError(f"No LOCATA clips found for split '{split}' under {split_dir}")
        data[split] = Dataset.from_list(rows).cast_column("audio", Audio())
    return AudioDataset(data=DatasetDict(data))


def locata_decode(sample: dict):
    """Decode one LOCATA row into ``(waveform, sample_rate)``.

    The ``audio`` column holds a file path (see ``_scan``), so unlike HF-Audio
    datasets we decode it here. Returns the full multi-channel clip; the random
    5-second crop is applied downstream by ``prepare_audio.load_clip``.
    """
    samples = sample["audio"].get_all_samples()
    return samples.data, int(samples.sample_rate)


def load_locata_traintest(root: str = DEFAULT_ROOT, **kwargs) -> AudioDataset:
    """LOCATA wrapped for ``prepare_audio``: dev -> ``train``, eval -> ``test``.

    ``prepare_ds`` indexes ``ds.data["train"]``/``ds.data["test"]``, whereas the
    raw corpus ships ``dev``/``eval`` archives, so remap ``eval`` to ``test``.
    """
    dd = load_locata(root=root, splits=("train", "eval"), **kwargs).data
    return AudioDataset(data=DatasetDict({"train": dd["train"], "test": dd["eval"]}))

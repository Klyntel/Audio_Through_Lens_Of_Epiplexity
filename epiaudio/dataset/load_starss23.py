"""Load the STARSS23 FOA corpus (Zenodo record 7880637) into an ``AudioDataset``.

STARSS23 (Sony-TAu Realistic Spatial Soundscapes 2023) ships as raw archives on
Zenodo rather than on the HF Hub, so - like ``load_locata`` - this loader leans
on the shared ``zenodo_downloader.download_zenodo`` helper to fetch and extract
the archives (each ``foo.zip`` -> ``root/foo/``) and then walks the extracted
tree to wrap the clips in an ``AudioDataset``. The ``audio`` column holds a file
path, decoded lazily downstream by ``data.realize.read_window``.

Only the first-order-Ambisonics (FOA) audio is loaded: ``foa_dev.zip`` (the
``dev-train-*`` / ``dev-test-*`` development split) and ``foa_eval.zip`` (the
unlabeled evaluation clips). The MIC, video, and metadata archives are skipped.

Development clips are named ``fold[F]_room[R]_mix[N].wav`` and live under
``dev-{train,test}-{sony,tau}/`` subfolders; evaluation clips are named
``mix[N].wav`` and carry no room/fold/metadata (see the record README).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Sequence

from datasets import Audio, Dataset, DatasetDict

from audio_preprocessing.datasets import AudioDataset
from epiaudio.dataset.zenodo_downloader import download_zenodo

ZENODO_RECORD = "7880637"

# Which archive each split lives in. dev-train/dev-test share foa_dev.zip; the
# evaluation clips ship separately in foa_eval.zip.
SPLIT_ARCHIVES = {"train": "foa_dev.zip", "test": "foa_dev.zip", "eval": "foa_eval.zip"}

# Recording sites contributing to the development set (see README).
VALID_ORIGINS = ("sony", "tau")

# Everything in the record that this FOA loader never touches (the MIC + video
# audio, the metadata, and the docs). The FOA archives are added to the skip
# list conditionally in ``load_starss23`` based on the requested splits.
NON_FOA_STEMS = ("mic_dev", "mic_eval", "video_dev", "video_eval", "metadata_dev", "LICENSE", "README")

# FOA audio is 4-channel first-order Ambisonics sampled at 24 kHz.
NUM_CHANNELS = 4
SAMPLE_RATE = 24000

DEFAULT_ROOT = os.path.join("data", "starss23_raw")

# dev clips: fold[F]_room[R]_mix[N].wav ; eval clips: mix[N].wav
_DEV_NAME = re.compile(r"fold(\d+)_room(\d+)_mix(\d+)", re.IGNORECASE)
_EVAL_NAME = re.compile(r"^mix(\d+)$", re.IGNORECASE)


def _archive_dir(root: str, split: str) -> str:
    """Directory ``download_zenodo`` extracts ``split``'s archive into (``root/<stem>``)."""
    return os.path.join(root, os.path.splitext(SPLIT_ARCHIVES[split])[0])


def _scan_dev(split_dir: str, split: str, origins: Sequence[str]) -> list[dict]:
    """One row per FOA dev clip whose ``dev-<split>-<origin>`` folder matches.

    Walks recursively so it is robust to whether the zip extracts its clips
    directly under ``foa_dev/`` or nested inside an extra ``foa_dev/foa_dev/``.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True)):
        # Locate the ``dev-<train|test>-<origin>`` component of the path.
        subset = next((p for p in path.split(os.sep) if p.startswith("dev-")), None)
        if subset is None or subset.count("-") != 2:
            continue
        _, kind, origin = subset.split("-")
        if kind != split or origin not in origins:
            continue
        m = _DEV_NAME.search(os.path.basename(path))
        if m is None:
            continue
        fold, room, mix = (int(g) for g in m.groups())
        rows.append({
            "audio": os.path.abspath(path),
            "split": split,
            "origin": origin,
            "fold": fold,
            "room": room,
            "mix": mix,
        })
    return rows


def _scan_eval(split_dir: str) -> list[dict]:
    """One row per FOA evaluation clip (``mix[N].wav``; no room/fold/origin labels)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True)):
        m = _EVAL_NAME.match(os.path.splitext(os.path.basename(path))[0])
        if m is None:
            continue
        rows.append({
            "audio": os.path.abspath(path),
            "split": "eval",
            "origin": None,
            "fold": None,
            "room": None,
            "mix": int(m.group(1)),
        })
    return rows


def load_starss23(
    root: str = DEFAULT_ROOT,
    *,
    download: bool = True,
    splits: Sequence[str] = ("train", "test"),
    origins: Sequence[str] = VALID_ORIGINS,
) -> AudioDataset:
    """Build an ``AudioDataset`` over STARSS23 FOA audio.

    ``train`` <- ``foa_dev/dev-train-*``, ``test`` <- ``foa_dev/dev-test-*``,
    ``eval`` <- ``foa_eval`` (unlabeled). ``origins`` filters the development
    splits by recording site (``sony`` / ``tau``); it does not affect ``eval``.
    """
    unknown = [s for s in splits if s not in SPLIT_ARCHIVES]
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; valid: {list(SPLIT_ARCHIVES)}")

    if download:
        # Only fetch the FOA archives the requested splits need; skip the rest
        # of the record (other FOA archives + all MIC/video/metadata/docs).
        needed = {os.path.splitext(SPLIT_ARCHIVES[s])[0] for s in splits}
        all_foa = {os.path.splitext(a)[0] for a in SPLIT_ARCHIVES.values()}
        skip = list(NON_FOA_STEMS) + sorted(all_foa - needed)
        download_zenodo(ZENODO_RECORD, output_dir=root, do_not_download=skip)

    data = {}
    for split in splits:
        split_dir = _archive_dir(root, split)
        rows = _scan_eval(split_dir) if split == "eval" else _scan_dev(split_dir, split, origins)
        if not rows:
            raise RuntimeError(f"No STARSS23 FOA clips found for split '{split}' under {split_dir}")
        data[split] = Dataset.from_list(rows).cast_column("audio", Audio())
    return AudioDataset(data=DatasetDict(data))


def starss23_decode(sample: dict):
    """Decode one STARSS23 row into ``(waveform, sample_rate)``.

    Like ``locata_decode``, the ``audio`` column holds a file path, so decode it
    here and return the full 4-channel FOA clip; the random 5-second crop is
    applied downstream by ``prepare_audio.load_clip``.
    """
    samples = sample["audio"].get_all_samples()
    return samples.data, int(samples.sample_rate)


def load_starss23_traintest(root: str = DEFAULT_ROOT, **kwargs) -> AudioDataset:
    """STARSS23 FOA wrapped for ``prepare_audio``: dev-train -> ``train``, dev-test -> ``test``.

    ``prepare_ds`` indexes ``ds.data["train"]`` / ``ds.data["test"]``, which the
    STARSS23 development split already provides directly (unlike LOCATA, no
    ``eval`` -> ``test`` remap is needed).
    """
    return load_starss23(root=root, splits=("train", "test"), **kwargs)

"""Adapter for the published WavTokenizer discrete codec.

The upstream project is not distributed as an installable Python package, so
its source checkout is supplied explicitly through ``WAVTOKENIZER_REPO_PATH``.
Weights and the matching configuration are fetched from the Hugging Face model
repository through ``huggingface_hub`` and cached normally.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download

from epiaudio.dataset.audio_tokenizers import Tokenizer


class WavTokenizerTokenizer(Tokenizer):
    """The 24 kHz, 40-token/s, single-codebook WavTokenizer model."""

    MODEL_REPO_ID = "novateur/WavTokenizer"
    MODEL_IDS = [MODEL_REPO_ID]
    CHECKPOINT = "WavTokenizer_small_600_24k_4096.ckpt"
    CONFIG = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    TARGET_SR = 24_000
    VOCAB_SIZE = 4_096
    TOKENS_PER_SECOND = 40
    SEQ_LEN = TOKENS_PER_SECOND * 5
    SOURCE_ENVIRONMENT_VARIABLE = "WAVTOKENIZER_REPO_PATH"

    @classmethod
    def create_cfg(cls) -> dict[str, dict[str, int]]:
        return {"model": {"V": cls.VOCAB_SIZE, "L": cls.SEQ_LEN}}

    @classmethod
    def _source_root(cls) -> Path:
        source = os.environ.get(cls.SOURCE_ENVIRONMENT_VARIABLE)
        if not source:
            raise RuntimeError(
                f"Set {cls.SOURCE_ENVIRONMENT_VARIABLE} to a checkout of "
                "https://github.com/jishengpeng/WavTokenizer before preparing WavTokenizer data."
            )
        root = Path(source).expanduser().resolve()
        if not (root / "decoder" / "pretrained.py").is_file():
            raise RuntimeError(
                f"{cls.SOURCE_ENVIRONMENT_VARIABLE} does not contain decoder/pretrained.py: {root}"
            )
        return root

    @classmethod
    def _model_root(cls, *, local_files_only: bool) -> Path:
        return Path(
            snapshot_download(
                repo_id=cls.MODEL_REPO_ID,
                allow_patterns=[cls.CHECKPOINT, cls.CONFIG],
                local_files_only=local_files_only,
            )
        )

    @classmethod
    def download_models_local(cls) -> None:
        """Warm only the selected 1.59 GB checkpoint and its configuration."""
        cls._source_root()
        cls._model_root(local_files_only=False)

    def __init__(self, device: Any = None, local_files_only: bool = False):
        super().__init__()
        source_root = self._source_root()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        pretrained = importlib.import_module("decoder.pretrained")
        wavtokenizer_cls = getattr(pretrained, "WavTokenizer")
        model_root = self._model_root(local_files_only=local_files_only)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = wavtokenizer_cls.from_pretrained0802(
            str(model_root / self.CONFIG), str(model_root / self.CHECKPOINT)
        ).to(self.device)
        self.model.eval()

    def _prepare_audio(self, y: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)
        audio = y.mean(dim=0)
        max_samples = self.TARGET_SR * 5
        audio = audio[:max_samples]
        if audio.numel() < max_samples:
            audio = torch.nn.functional.pad(audio, (0, max_samples - audio.numel()))
        return audio

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        bandwidth_id = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            _, codes = self.model.encode_infer(audio.to(self.device), bandwidth_id=bandwidth_id)
        if codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError(f"Unexpected WavTokenizer code shape: {tuple(codes.shape)}")
        return cast(torch.Tensor, codes[0])

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        tokens = self._encode(self._prepare_audio(y, sr).unsqueeze(0))[0]
        if tokens.shape != torch.Size(shape):
            raise ValueError(f"WavTokenizer produced {tuple(tokens.shape)}, expected {shape}.")
        return tokens.cpu()

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.int16) -> np.ndarray:
        audio = torch.stack([self._prepare_audio(y, sr) for y, sr in zip(ys, srs)]).to(self.device)
        tokens = self._encode(audio)
        expected = (len(ys), n_tokens)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"WavTokenizer produced {tuple(tokens.shape)}, expected {expected}.")
        return tokens.detach().cpu().numpy().astype(dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode ``[batch, 200]`` WavTokenizer codes to 24 kHz waveforms."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.SEQ_LEN:
            raise ValueError(
                "WavTokenizer decode expects [batch, "
                f"{self.SEQ_LEN}] token IDs, got {tuple(x.shape)}."
            )
        if torch.any(x < 0) or torch.any(x >= self.VOCAB_SIZE):
            raise ValueError(
                f"WavTokenizer token IDs must be in [0, {self.VOCAB_SIZE - 1}]."
            )
        # Upstream codes_to_features expects [codebook, batch, frames].
        codes = x.to(self.device, dtype=torch.long).unsqueeze(0)
        # Upstream AdaLayerNorm multiplies a [batch, frames, channels] tensor;
        # retain a singleton frame axis so the per-example conditioning broadcasts
        # correctly for batches larger than one.
        bandwidth_id = torch.zeros(
            (x.shape[0], 1), dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            features = self.model.codes_to_features(codes)
            waveform = self.model.decode(features, bandwidth_id=bandwidth_id)
        if waveform.ndim != 2:
            raise ValueError(f"Unexpected decoded WavTokenizer shape: {tuple(waveform.shape)}")
        return waveform

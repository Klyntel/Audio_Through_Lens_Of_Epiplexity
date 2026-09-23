"""Lightweight tokenizer invariants shared by preprocessing and sweep checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class TokenizerSpec:
    """Static properties of one tokenizer's five-second output."""

    shape: tuple[int, ...]
    vocab_size: int | None

    @property
    def sequence_length(self) -> int:
        return prod(self.shape)


TOKENIZER_SPECS: dict[str, TokenizerSpec] = {
    "whisper": TokenizerSpec(shape=(80, 500), vocab_size=None),
    "encodec": TokenizerSpec(shape=(375,), vocab_size=1024),
    "dac": TokenizerSpec(shape=(250, 12), vocab_size=1024),
    "sqcodec": TokenizerSpec(shape=(1778,), vocab_size=117_649),
    "xcodec": TokenizerSpec(shape=(250, 8), vocab_size=1024),
    "wavtokenizer": TokenizerSpec(shape=(200,), vocab_size=4096),  
    # Reference non-audio contracts from epiplexity/picodo/dataset.
    "ascii96": TokenizerSpec(shape=(512,), vocab_size=96),
    "puzzles": TokenizerSpec(shape=(512,), vocab_size=64),
    "grayscale": TokenizerSpec(shape=(1024,), vocab_size=256),
}

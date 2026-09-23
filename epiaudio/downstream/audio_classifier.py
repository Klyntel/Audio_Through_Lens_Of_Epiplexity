"""Conditional audio classifier built from the epiplexity transformer blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from epiaudio.model_torch import TransformerBackbone


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class CausalAudioEncoder(TransformerBackbone):
    """The EpiAudio causal transformer without its token-prediction readout.

    Reusing :class:`epiaudio.model_torch.TransformerBackbone` keeps embeddings,
    initialization, QK-normalized attention, depth-scaled residuals, and final
    normalization aligned with the existing epiplexity model. The last hidden
    state is the only position that has observed every audio token.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        per_head_dim: int,
        vocab_size: int,
        seq_length: int,
        embed_init_std: float = 0.1,
    ) -> None:
        validated_num_layers = _positive_int(num_layers, "num_layers")
        validated_embed_dim = _positive_int(embed_dim, "embed_dim")
        validated_per_head_dim = _positive_int(per_head_dim, "per_head_dim")
        validated_vocab_size = _positive_int(vocab_size, "vocab_size")
        validated_seq_length = _positive_int(seq_length, "seq_length")
        if validated_embed_dim % validated_per_head_dim != 0:
            raise ValueError("embed_dim must be divisible by per_head_dim.")
        if (
            isinstance(embed_init_std, bool)
            or not isinstance(embed_init_std, (int, float))
            or not math.isfinite(embed_init_std)
            or embed_init_std <= 0
        ):
            raise ValueError("embed_init_std must be positive.")

        super().__init__(
            D=validated_embed_dim,
            L=validated_seq_length,
            N=validated_num_layers,
            V=validated_vocab_size,
            dh=validated_per_head_dim,
            embed_init_std=float(embed_init_std),
        )
        self.num_layers = validated_num_layers
        self.embed_dim = validated_embed_dim
        self.per_head_dim = validated_per_head_dim
        self.vocab_size = validated_vocab_size
        self.seq_length = validated_seq_length

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 2:
            raise ValueError("audio tokens must have shape (batch, sequence_length).")
        if tokens.shape[0] == 0:
            raise ValueError("audio token batch must contain at least one example.")
        if tokens.shape[1] == 0:
            raise ValueError("audio token sequence must contain at least one token.")
        if tokens.shape[1] > self.seq_length:
            raise ValueError(
                f"Audio sequence length {tokens.shape[1]} exceeds configured "
                f"maximum {self.seq_length}."
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("audio tokens must use torch.int32 or torch.int64.")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return normalized hidden states for every audio-token position."""
        self._validate_tokens(tokens)
        return self.get_normalized_features(tokens)


class AudioClassifier(nn.Module):
    """Predict one categorical class distribution conditioned on audio tokens."""

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        per_head_dim: int,
        vocab_size: int,
        seq_length: int,
        num_classes: int,
        embed_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.encoder = CausalAudioEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            per_head_dim=per_head_dim,
            vocab_size=vocab_size,
            seq_length=seq_length,
            embed_init_std=embed_init_std,
        )
        # This is the same zero-initialized linear readout used by model_torch,
        # with vocabulary logits replaced by class logits.
        self.readout = nn.Linear(embed_dim, self.num_classes, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Return class logits for ``p(Y | X)`` from the final position in X."""
        hidden = self.encoder(audio_tokens)
        return self.readout(hidden[:, -1, :])

    def predict_proba(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Return the teacher distribution over class labels."""
        return F.softmax(self(audio_tokens), dim=-1)

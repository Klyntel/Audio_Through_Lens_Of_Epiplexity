"""PyTorch TransformerDecoder matching the JAX/Flax model in epiplexity/picodo/model.py.

Architecture is identical:
  - Token + positional embeddings (normal init, embed_init_std)
  - N TransformerBlocks with pre-norm (RMSNorm, no learnable scale)
  - QK-norm attention, causal mask via Flash Attention (SDPA)
  - MLP: 4x expansion, GELU, no bias
  - Branch multiplier = 1/N  (depth-scaled residuals)
  - Readout = zero-init linear

Weight init follows fsdp_init from picodo:
  - embeddings  → Normal(0, embed_init_std)
  - attn/mlp in-projections → variance_scaling fan_in
  - readout/attn out-proj   → zeros


Human notes [Sean]
This looks correct for the most part, but the fact it reimplements the functions is
a tad strange, a lot of this could be redone via built in functions.

However this makes sense from what I can tell.
"""

import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Weight initialisation helpers (mirrors fsdp_init from picodo/model.py)
# ---------------------------------------------------------------------------


def _embed_init(weight: torch.Tensor, std: float) -> None:
    nn.init.normal_(weight, mean=0.0, std=std)


def _fan_in_init(weight: torch.Tensor) -> None:
    fan_in = weight.shape[1]
    std = 1.0 / math.sqrt(fan_in)
    nn.init.normal_(weight, mean=0.0, std=std)


def _zero_init(weight: torch.Tensor) -> None:
    nn.init.zeros_(weight)


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D, 4 * D, bias=False)
        self.fc2 = nn.Linear(4 * D, D, bias=False)
        _fan_in_init(self.fc1.weight)
        _zero_init(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, D: int, dh: int) -> None:
        super().__init__()
        self.num_heads = D // dh
        self.head_dim = dh

        self.query_proj = nn.Linear(D, D, bias=False)
        self.key_proj = nn.Linear(D, D, bias=False)
        self.value_proj = nn.Linear(D, D, bias=False)
        self.output_proj = nn.Linear(D, D, bias=False)

        # QK-norm: RMSNorm without learnable scale (matches use_scale=False in JAX)
        self.q_norm = nn.RMSNorm(dh, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(dh, elementwise_affine=False)

        for proj in (self.query_proj, self.key_proj, self.value_proj):
            _fan_in_init(proj.weight)
        _zero_init(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, dh = self.num_heads, self.head_dim

        q = self.query_proj(x).view(B, S, H, dh).transpose(1, 2)  # [B,H,S,dh]
        k = self.key_proj(x).view(B, S, H, dh).transpose(1, 2)
        v = self.value_proj(x).view(B, S, H, dh).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Flash Attention via scaled_dot_product_attention (FA-2 on H100)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.output_proj(out)

    # ------------------------------------------------------------------
    # KV-cache step for autoregressive generation
    # ------------------------------------------------------------------
    def forward_step(
        self,
        x: torch.Tensor,  # [B, D]
        k_cache: torch.Tensor,  # [B, H, max_len, dh]
        v_cache: torch.Tensor,  # [B, H, max_len, dh]
        position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, dh = x.shape[0], self.num_heads, self.head_dim

        q = self.query_proj(x).view(B, H, dh)
        k_new = self.key_proj(x).view(B, H, dh)
        v_new = self.value_proj(x).view(B, H, dh)

        q = self.q_norm(q)
        k_new = self.k_norm(k_new)

        k_cache[:, :, position, :] = k_new
        v_cache[:, :, position, :] = v_new

        # Attend to all valid positions [0, position]
        k_seq = k_cache[:, :, : position + 1, :]  # [B, H, pos+1, dh]
        v_seq = v_cache[:, :, : position + 1, :]

        attn_scores = torch.einsum("bhd,bhtd->bht", q, k_seq) * (dh**-0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.einsum("bht,bhtd->bhd", attn_probs, v_seq)
        context = context.reshape(B, H * dh)
        return self.output_proj(context), k_cache, v_cache


class TransformerBlock(nn.Module):
    def __init__(self, D: int, dh: int, N: int) -> None:
        super().__init__()
        self.ln1 = nn.RMSNorm(D, elementwise_affine=False)
        self.attn = Attention(D, dh)
        self.ln2 = nn.RMSNorm(D, elementwise_affine=False)
        self.mlp = Mlp(D)
        # Depth-scaled residuals — matches JAX's branch_multiplier = 1/N
        self.branch_multiplier = 1.0 / N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x)) * self.branch_multiplier
        x = x + self.mlp(self.ln2(x)) * self.branch_multiplier
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        attn_out, k_cache, v_cache = self.attn.forward_step(
            h, k_cache, v_cache, position
        )
        x = x + attn_out * self.branch_multiplier
        x = x + self.mlp(self.ln2(x)) * self.branch_multiplier
        return x, k_cache, v_cache


# ---------------------------------------------------------------------------
# Shared causal transformer backbone
# ---------------------------------------------------------------------------


class TransformerBackbone(nn.Module):
    """Token embeddings and causal transformer blocks shared by model heads.

    Modules remain direct attributes of subclasses so that extracting this class
    does not add a ``backbone.`` prefix to existing Experiment 1 checkpoints.
    """

    def __init__(
        self,
        *,
        D: int,
        L: int,
        N: int,
        V: int,
        dh: int,
        embed_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self._D, self._L, self._N, self._V, self._dh = D, L, N, V, dh

        self.embed = nn.Embedding(V, D)
        self.pos_embed = nn.Embedding(L, D)
        _embed_init(self.embed.weight, embed_init_std)
        _embed_init(self.pos_embed.weight, embed_init_std)

        self.blocks = nn.ModuleList([TransformerBlock(D, dh, N) for _ in range(N)])
        self.out_ln = nn.RMSNorm(D, elementwise_affine=False)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden states before the final RMSNorm."""
        pos = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        h = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return h

    def get_normalized_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the normalized hidden states consumed by a model head."""
        return self.out_ln(self.get_features(x))


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------


class TransformerDecoder(TransformerBackbone):
    """PyTorch port of picodo's TransformerDecoder.

    Matches the JAX model architecture exactly:
      embed / pos_embed → N blocks → RMSNorm → readout
    """

    def __init__(self, cfg: DictConfig) -> None:
        D = int(cfg.D)
        L = int(cfg.L)
        N = int(cfg.N)
        V = int(cfg.V)
        dh = int(cfg.dh)
        std = float(cfg.embed_init_std) if cfg.embed_init_std is not None else 0.1

        super().__init__(D=D, L=L, N=N, V=V, dh=dh, embed_init_std=std)
        self.readout = nn.Linear(D, V, bias=False)
        _zero_init(self.readout.weight)

    # ------------------------------------------------------------------
    # Forward pass (training / eval)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.get_normalized_features(x))

    # ------------------------------------------------------------------
    # Autoregressive generation with KV cache
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        bos_token: int = 0,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences using a KV cache; matches picodo's generate().

        Returns:
            tokens:         [batch_size, seq_len] long
            teacher_logits: [batch_size, seq_len, V] float
        """
        dtype = self.embed.weight.dtype
        tokens = torch.full(
            (batch_size, seq_len), bos_token, dtype=torch.long, device=device
        )

        H, dh = self._D // self._dh, self._dh
        k_caches = [
            torch.zeros(batch_size, H, seq_len, dh, dtype=dtype, device=device)
            for _ in range(self._N)
        ]
        v_caches = [
            torch.zeros(batch_size, H, seq_len, dh, dtype=dtype, device=device)
            for _ in range(self._N)
        ]

        all_logits: List[torch.Tensor] = []
        pos_idx = torch.zeros(1, dtype=torch.long, device=device)

        for pos in range(seq_len):
            pos_idx.fill_(pos)
            token_ids = tokens[:, pos]  # [B]
            x = self.embed(token_ids) + self.pos_embed(pos_idx)  # [B, D]

            for i, block in enumerate(self.blocks):
                # ModuleList items typed as Module; forward_step is on TransformerBlock
                x, k_caches[i], v_caches[i] = block.forward_step(  # type: ignore[attr-defined]
                    x, k_caches[i], v_caches[i], pos
                )

            logits = self.readout(self.out_ln(x))  # [B, V]
            all_logits.append(logits)

            if pos < seq_len - 1:
                sample_logits = logits / temperature if temperature != 1.0 else logits
                next_tok = torch.multinomial(
                    F.softmax(sample_logits.float(), dim=-1), 1
                ).squeeze(1)
                tokens[:, pos + 1] = next_tok

        teacher_logits = torch.stack(all_logits, dim=1)  # [B, seq_len, V]
        return tokens, teacher_logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_model(cfg: DictConfig) -> TransformerDecoder:
    """Create a TransformerDecoder from a model config dict."""
    return TransformerDecoder(cfg)

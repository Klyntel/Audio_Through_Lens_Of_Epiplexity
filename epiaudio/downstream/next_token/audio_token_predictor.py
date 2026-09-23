import torch
import torch.nn as nn

class AudioTokenPredictor(nn.Module):
    mask: torch.Tensor

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        per_head_dim: int,
        vocab_size: int,
        seq_length: int
    ) -> None:
        super().__init__()

        assert embed_dim % per_head_dim == 0, f"embed_dim ({embed_dim}) is not a multiple of per_head_dim ({per_head_dim})"

        self.seq_length = seq_length
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length)
        self.register_buffer("mask", mask)

        self.embed0 = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_length, embed_dim)
        norm = nn.RMSNorm(embed_dim, elementwise_affine=False)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=embed_dim // per_head_dim,
            dim_feedforward=4*embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=norm)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.seq_length:
            msg = f"number of tokens ({x.shape[1]}) exceeds the context length ({self.seq_length})"
            raise ValueError(msg)

        pos = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        embedding = self.embed0(x) + self.pos_embed(pos)

        return embedding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed(x)
        mask = self.mask[:x.shape[1], :x.shape[1]]
        hidden_states = self.decoder(
            tgt=embedding,
            memory=embedding,
            tgt_mask=mask,
            memory_mask=mask
        )
        logits = self.output_layer(hidden_states)

        return hidden_states, logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        prefix_length: int,
        num_pred_tokens: int,
        device: str="cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        error_msg = (
            f"prefix_length ({prefix_length})"" + num_pred_tokens ({num_pred_tokens}) "
            f"must be at most self.seq_length ({self.seq_length})"
        )
        if prefix_length + num_pred_tokens > self.seq_length:
            raise ValueError(error_msg)

        self.eval()
        all_logits = []
        current_tokens = torch.clone(x).detach()

        for i in range(num_pred_tokens):
            _, logits = self.forward(current_tokens)
            next_token_logits = logits[:, prefix_length + i - 1, :].clone()
            all_logits.append(next_token_logits)

            probs = torch.softmax(next_token_logits, dim=1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            current_tokens[:, prefix_length + i] = next_tokens.squeeze(1)

        hidden_states, _ = self.forward(current_tokens)

        all_logits = torch.stack(all_logits, dim=2)
        tokens_pred = current_tokens[:, prefix_length: prefix_length + num_pred_tokens]

        return hidden_states, all_logits, tokens_pred
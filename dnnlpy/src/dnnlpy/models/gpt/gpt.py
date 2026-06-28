from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

from .utils import sample_next_token

__all__ = [
    'MiniGPTCausalSelfAttention',
    'MiniGPTMLP',
    'MiniGPTBlock',
    'MiniGPT',
]


class MiniGPTCausalSelfAttention(nn.Module):
    """Causal self-attention block for MiniGPT token representations."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = dnn.MultiheadAttention(
            embed_dim, num_heads, bias=bias, dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_output, _ = self.attn(x, x, x, is_causal=True, need_weights=False)
        return attn_output


class MiniGPTMLP(nn.Module):
    """Feed-forward MLP block used inside a MiniGPT decoder block."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            dnn.Linear(embed_dim, hidden_dim, bias=bias),
            dnn.GELU(),
            dnn.Linear(hidden_dim, embed_dim, bias=bias),
            dnn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MiniGPTBlock(nn.Module):
    """Pre-LayerNorm Transformer decoder block for MiniGPT."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = dnn.LayerNorm(embed_dim, bias=bias)
        self.attn = MiniGPTCausalSelfAttention(
            embed_dim, num_heads, bias=bias, dropout=dropout
        )
        self.norm2 = dnn.LayerNorm(embed_dim, bias=bias)
        self.mlp = MiniGPTMLP(embed_dim, hidden_dim, bias=bias, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """A tutorial-sized GPT-style language model with a small number of parameters."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,  # or context window
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
        weight_tying: bool = True,
    ):
        """Create a tiny GPT-style language model.

        Args:
            vocab_size (int): Number of vocabulary entries.
            block_size (int): Maximum context window length.
            embed_dim (int, default: 128): Token and position embedding dimension.
            num_layers (int, default: 4): Number of Transformer decoder blocks.
            num_heads (int, default: 4): Number of attention heads in each block.
            hidden_dim (int, default: 512): Hidden dimension of each feed-forward MLP.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for embeddings, attention,
                and feed-forward layers.
            weight_tying (bool, default: True): Whether to share token embedding weights
                with the language-model output head.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.weight_tying = weight_tying

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.embed_dropout = dnn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                MiniGPTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    bias=bias,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = dnn.LayerNorm(embed_dim, bias=bias)
        self.lm_head = dnn.Linear(embed_dim, vocab_size, bias=bias)

        if weight_tying:
            self.lm_head.weight = cast(nn.Parameter, self.token_embed.weight)
            assert self.lm_head.weight is self.token_embed.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, dnn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, dnn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute the logits for a batch of input sequences."""
        if input_ids.ndim != 2:
            raise AssertionError('`input_ids` must have shape (B, T).')

        T = input_ids.size(1)
        if T > self.block_size:
            raise AssertionError(
                f'Sequence length {T} exceeds block_size {self.block_size}.'
            )

        pos = torch.arange(T, device=input_ids.device)
        x_tok = self.token_embed(input_ids)
        x_pos = self.pos_embed(pos)
        x = self.embed_dropout(x_tok + x_pos)

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def loss(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        """Compute the cross-entropy loss for a batch of input sequences."""
        if targets is not None and targets.ndim != 2:
            raise AssertionError('`targets` must have shape (B, T).')

        logits = self(input_ids)

        if targets is None:
            logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]

        return dF.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

    def generate(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
    ) -> Tensor:
        """Generate new token ids autoregressively.

        Args:
            logits (Tensor): Logits tensor of shape (B, T, V) from the model.
            temperature (float, default: 1.0): Sampling temperature.
            top_k (int | None, default: None): If specified, use top-k sampling.
            top_p (float | None, default: None): If specified, use top-p sampling.
            greedy (bool, default: False): If True, use greedy decoding.
        """
        next_token = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        return next_token

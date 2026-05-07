import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    'SinusoidalPositionalEncoding',
    'SinusoidalTimestepEmbedding',
]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, embed_dim, 2) / embed_dim
        div_term = torch.pow(10000.0, exp_term)

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term[: pe[:, 1::2].size(1)])

        # Add a batch dimension for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]  # type: ignore


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return torch.zeros(
                timesteps.size(0),
                self.embedding_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )

        scale = -math.log(self.max_period) / max(half_dim - 1, 1)
        emb = torch.arange(half_dim, device=timesteps.device) * scale
        emb = timesteps.unsqueeze(1) * emb.exp().unsqueeze(0)
        emb = torch.concat([emb.sin(), emb.cos()], dim=-1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['SinusoidalPositionalEncoding']


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
        pe[:, 1::2] = torch.cos(position / div_term)

        # Add a batch dimension for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]  # type: ignore

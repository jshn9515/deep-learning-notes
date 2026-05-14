import torch.nn as nn
from torch import Tensor

import dnnl.nn as dnn

__all__ = [
    'ViTMLP',
    'ViTEncoderLayer',
    'ViTEncoder',
]


class ViTMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ViTEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = dnn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)

        mlp_input = self.norm2(x)
        x = x + self.mlp(mlp_input)
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ViTEncoderLayer(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

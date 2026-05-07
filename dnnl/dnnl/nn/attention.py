import torch
import torch.nn as nn
from torch import Tensor

from . import functional as F

__all__ = ['MultiheadAttention']

type AttentionOutput = tuple[Tensor, Tensor | None]


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int | None = None,
        vdim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.dropout = dropout
        self.bias = bias

        if embed_dim % num_heads != 0:
            raise AssertionError('`embed_dim` must be divisible by `num_heads`.')
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tensor | AttentionOutput:
        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            attn_mask = (
                padding_mask
                if attn_mask is None
                else attn_mask.logical_or(padding_mask)
                if attn_mask.dtype == torch.bool
                else attn_mask.masked_fill(padding_mask, -torch.inf)
            )

        attn_output, attn_weights = F.multi_head_attention(
            query,
            key,
            value,
            self.num_heads,
            self.q_proj.weight.T,
            self.k_proj.weight.T,
            self.v_proj.weight.T,
            self.out_proj.weight.T,
            q_proj_bias=self.q_proj.bias,
            k_proj_bias=self.k_proj.bias,
            v_proj_bias=self.v_proj.bias,
            out_proj_bias=self.out_proj.bias,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout=self.dropout,
            training=self.training,
            need_weights=need_weights,
        )

        if need_weights:
            if average_attn_weights and attn_weights is not None:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights

        return attn_output

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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise AssertionError('`embed_dim` must be divisible by `num_heads`.')
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

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
        batch_size, target_len, _ = query.size()
        source_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, target_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, source_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, source_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :]
            keep_mask = ~padding_mask
            attn_mask = (
                keep_mask
                if attn_mask is None
                else attn_mask.logical_and(keep_mask)
                if attn_mask.dtype == torch.bool
                else attn_mask.masked_fill(padding_mask, -torch.inf)
            )

        attn_output, attn_weights = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout=self.dropout,
            training=self.training,
            need_weights=True,
            is_causal=is_causal,
        )  # fmt: skip

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            if average_attn_weights and attn_weights is not None:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights

        return attn_output

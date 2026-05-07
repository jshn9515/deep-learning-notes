import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    'attention',
    'scaled_dot_product_attention',
    'multi_head_attention',
]

type AttentionOutput = tuple[Tensor, Tensor | None]


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_weights: bool = False,
) -> AttentionOutput:
    scores = query @ key.transpose(-2, -1)
    attn_weights = scores.softmax(dim=-1)
    output = attn_weights @ value

    if need_weights:
        return output, attn_weights

    return output, None


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    dropout: float = 0.0,
    training: bool = True,
    need_weights: bool = False,
    scale: float | None = None,
) -> AttentionOutput:
    """Compute scaled dot-product attention.

    Boolean masks follow ``torch.nn.functional.scaled_dot_product_attention``:
    ``True`` means the element participates in attention and ``False`` masks it.
    Float masks are additive attention biases.
    """
    if query.size(-1) != key.size(-1):
        raise AssertionError(
            '`query` and `key` must have the same embedding dimension.'
        )
    if key.size(-2) != value.size(-2):
        raise AssertionError('`key` and `value` must have the same sequence length.')
    if is_causal and attn_mask is not None:
        raise AssertionError('`attn_mask` and `is_causal` cannot both be set.')
    if not 0.0 <= dropout <= 1.0:
        raise AssertionError('dropout must be between 0 and 1.')

    target_len = query.size(-2)
    source_len = key.size(-2)

    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    scores = query @ key.transpose(-2, -1)
    scores = scores * scale_factor

    if is_causal:
        causal_mask = torch.ones(
            target_len,
            source_len,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, -math.inf)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, -math.inf)
        else:
            scores = scores + attn_mask.to(device=query.device, dtype=query.dtype)

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    output = attn_weights @ value

    if need_weights:
        return output, attn_weights

    return output, None


def multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    out_proj_weight: Tensor,
    q_proj_bias: Tensor | None = None,
    k_proj_bias: Tensor | None = None,
    v_proj_bias: Tensor | None = None,
    out_proj_bias: Tensor | None = None,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    dropout: float = 0.0,
    training: bool = True,
    need_weights: bool = False,
) -> AttentionOutput:
    batch_size, target_len, embed_dim = query.size()
    source_len = key.size(1)

    if embed_dim % num_heads != 0:
        raise AssertionError('`embed_dim` must be divisible by `num_heads`.')
    if key.size(0) != batch_size or value.size(0) != batch_size:
        raise AssertionError('`query`, `key`, and `value` batch sizes must match.')
    if key.size(1) != value.size(1):
        raise AssertionError('`key` and `value` must have the same sequence length.')

    head_dim = embed_dim // num_heads

    q = query @ q_proj_weight
    k = key @ k_proj_weight
    v = value @ v_proj_weight

    if q_proj_bias is not None:
        q = q + q_proj_bias
    if k_proj_bias is not None:
        k = k + k_proj_bias
    if v_proj_bias is not None:
        v = v + v_proj_bias

    q = q.view(batch_size, target_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, source_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, source_len, num_heads, head_dim).transpose(1, 2)

    head_output, attn_weights = scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout=dropout,
        training=training,
        need_weights=need_weights,
    )  # fmt: skip

    output = head_output.transpose(1, 2)
    output = output.reshape(batch_size, target_len, embed_dim)
    output = output @ out_proj_weight

    if out_proj_bias is not None:
        output = output + out_proj_bias

    if need_weights:
        return output, attn_weights

    return output, None

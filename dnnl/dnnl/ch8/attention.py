import math
from typing import Literal, overload

import torch
from torch import Tensor

__all__ = ['attention', 'scaled_dot_product_attention', 'multi_head_attention']


@overload
def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor: ...
@overload
def attention(
    Q: Tensor, K: Tensor, V: Tensor, return_attn_weights: Literal[False]
) -> Tensor: ...
@overload
def attention(
    Q: Tensor, K: Tensor, V: Tensor, return_attn_weights: Literal[True]
) -> tuple[Tensor, Tensor]: ...


def attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    scores = Q @ K.transpose(-2, -1)
    attn_weights = scores.softmax(dim=-1)
    head_output = attn_weights @ V

    if return_attn_weights:
        return head_output, attn_weights

    return head_output


@overload
def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor: ...
@overload
def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, return_attn_weights: Literal[False]
) -> Tensor: ...
@overload
def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, return_attn_weights: Literal[True]
) -> tuple[Tensor, Tensor]: ...


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    attn_weights = scores.softmax(dim=-1)
    head_output = attn_weights @ V

    if return_attn_weights:
        return head_output, attn_weights

    return head_output


@overload
def multi_head_attention(
    X: Tensor,
    num_heads: int,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    W_O: Tensor,
) -> Tensor: ...
@overload
def multi_head_attention(
    X: Tensor,
    num_heads: int,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    W_O: Tensor,
    return_attn_weights: Literal[False],
) -> Tensor: ...
@overload
def multi_head_attention(
    X: Tensor,
    num_heads: int,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    W_O: Tensor,
    return_attn_weights: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def multi_head_attention(
    X: Tensor,
    num_heads: int,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    W_O: Tensor,
    return_attn_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    batch_size, seq_len, d_model = X.size()
    d_head = d_model // num_heads

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    Q = Q.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)

    if return_attn_weights:
        head_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, return_attn_weights=True
        )
    else:
        head_output = scaled_dot_product_attention(Q, K, V)
        attn_weights = torch.tensor(0)  # dummy tensor

    head_output = head_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output = head_output @ W_O

    if return_attn_weights:
        return output, attn_weights

    return output

import math

import torch
from dnnl.ch8 import attention, multi_head_attention, scaled_dot_product_attention


def test_attention_returns_expected_weights_and_output():
    Q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    V = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output, weights = attention(Q, K, V, return_attn_weights=True)
    expected_weights = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_weights @ V)


def test_scaled_dot_product_attention_matches_manual_computation():
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 5)

    output, weights = scaled_dot_product_attention(Q, K, V, return_attn_weights=True)
    expected_weights = torch.softmax(
        (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1)),
        dim=-1,
    )

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_weights @ V)


def test_multi_head_attention_matches_explicit_head_computation():
    batch_size, seq_len, d_model, num_heads = 2, 3, 4, 2
    d_head = d_model // num_heads
    X = torch.randn(batch_size, seq_len, d_model)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(d_model, d_model)
    W_V = torch.randn(d_model, d_model)
    W_O = torch.randn(d_model, d_model)

    output, weights = multi_head_attention(
        X, num_heads, W_Q, W_K, W_V, W_O, return_attn_weights=True
    )

    Q = (X @ W_Q).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    K = (X @ W_K).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    V = (X @ W_V).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    expected_head_output, expected_weights = scaled_dot_product_attention(
        Q, K, V, return_attn_weights=True
    )
    expected_output = (
        expected_head_output.transpose(1, 2).reshape(batch_size, seq_len, d_model) @ W_O
    )

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_output)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dnnl.nn import MultiheadAttention, SinusoidalPositionalEncoding
from dnnl.nn import functional as dF

batch_size = 4
src_len = 8
tgt_len = 4
d_model = 6
num_heads = 2
key_dim = 6
value_dim = 8
head_dim = d_model // num_heads


def test_naive_attention():
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)

    output, weights = dF.attention(query, key, value)
    expected_weights = F.softmax(query @ key.transpose(-2, -1), dim=-1)

    assert weights is not None
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_weights @ value)


def test_scaled_dot_product_attention():
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)
    scale = 1 / math.sqrt(query.size(-1))

    output, weights = dF.scaled_dot_product_attention(query, key, value)
    expected = F.scaled_dot_product_attention(query, key, value)
    expected_weights = F.softmax((query @ key.transpose(-2, -1)) * scale, dim=-1)

    assert weights is not None
    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(weights, expected_weights)


def test_scaled_dot_product_attention_boolean_mask():
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)
    attn_mask = torch.tensor(
        [
            [
                [True, True, False, True, False, False, True, False],
                [False, True, True, True, False, False, False, True],
                [True, False, True, False, True, False, True, False],
                [False, False, True, False, True, True, False, True],
            ]
        ]
    )

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=~attn_mask)

    assert weights is not None
    assert torch.allclose(output, expected, atol=1e-6)

    masked_weights = weights.masked_select(attn_mask.expand_as(weights))
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))


def test_scaled_dot_product_attention_supports_additive_mask():
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)
    attn_mask = torch.zeros(tgt_len, src_len)
    attn_mask[:, -1] = -torch.inf

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert weights is not None
    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(weights[..., -1], torch.zeros_like(weights[..., -1]))


def test_scaled_dot_product_attention_supports_causal_mode():
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)

    output, weights = dF.scaled_dot_product_attention(query, key, value, is_causal=True)
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    forbidden = torch.ones(tgt_len, src_len, dtype=torch.bool).triu(diagonal=1)

    assert weights is not None
    assert torch.allclose(output, expected, atol=1e-6)

    masked_weights = weights[..., forbidden]
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))


def test_generate_causal_mask_masks_future_positions():
    mask = dF.generate_causal_mask(4)
    expected = torch.tensor(
        [
            [0.0, -torch.inf, -torch.inf, -torch.inf],
            [0.0, 0.0, -torch.inf, -torch.inf],
            [0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert mask.dtype == torch.float32
    assert torch.equal(mask, expected)


def test_scaled_dot_product_attention_causal_uses_boolean_mask():
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)
    mask = torch.ones(tgt_len, src_len, dtype=torch.bool).triu(diagonal=1)

    causal_output, causal_weights = dF.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    masked_output, masked_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )

    assert causal_weights is not None
    assert masked_weights is not None
    assert torch.allclose(causal_output, masked_output)
    assert torch.allclose(causal_weights, masked_weights)


def test_scaled_dot_product_attention_respects_scale_and_training_dropout_flag():
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)

    output, _ = dF.scaled_dot_product_attention(
        query, key, value,
        dropout=0.8,
        training=False,
        scale=0.25,
    )  # fmt: skip
    expected = F.scaled_dot_product_attention(
        query, key, value,
        dropout_p=0.0,
        scale=0.25,
    )  # fmt: skip

    assert torch.allclose(output, expected, atol=1e-6)


def test_multi_head_attention_matches_explicit_cross_attention_with_bias():
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(key_dim, d_model)
    W_V = torch.randn(value_dim, d_model)
    W_O = torch.randn(d_model, d_model)
    b_Q = torch.randn(d_model)
    b_K = torch.randn(d_model)
    b_V = torch.randn(d_model)
    b_O = torch.randn(d_model)

    output, weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads,
        q_proj_weight=W_Q,
        k_proj_weight=W_K,
        v_proj_weight=W_V,
        out_proj_weight=W_O,
        q_proj_bias=b_Q,
        k_proj_bias=b_K,
        v_proj_bias=b_V,
        out_proj_bias=b_O,
        need_weights=True,
    )

    Q = ((query @ W_Q) + b_Q).view(batch_size, tgt_len, num_heads, head_dim)
    K = ((key @ W_K) + b_K).view(batch_size, src_len, num_heads, head_dim)
    V = ((value @ W_V) + b_V).view(batch_size, src_len, num_heads, head_dim)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    expected_head_output, expected_weights = dF.scaled_dot_product_attention(Q, K, V)
    expected_output = expected_head_output.transpose(1, 2).reshape(
        batch_size, tgt_len, d_model
    )
    expected_output = (expected_output @ W_O) + b_O

    assert weights is not None
    assert expected_weights is not None
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_output)


def test_multi_head_attention_matches_torch_cross_attention_with_bias():
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)

    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(key_dim, d_model)
    v_weight = torch.randn(value_dim, d_model)
    out_weight = torch.randn(d_model, d_model)
    q_bias = torch.randn(d_model)
    k_bias = torch.randn(d_model)
    v_bias = torch.randn(d_model)
    out_bias = torch.randn(d_model)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        q_proj_bias=q_bias,
        k_proj_bias=k_bias,
        v_proj_bias=v_bias,
        out_proj_bias=out_bias,
        need_weights=True,
    )
    expected, expected_weights = F.multi_head_attention_forward(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=torch.concat([q_bias, k_bias, v_bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_weight.T,
        out_proj_bias=out_bias,
        training=True,
        need_weights=True,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight.T,
        k_proj_weight=k_weight.T,
        v_proj_weight=v_weight.T,
        average_attn_weights=False,
    )
    expected = expected.transpose(0, 1)

    assert actual_weights is not None
    assert expected_weights is not None
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-6)


def test_multihead_attention_module_matches_torch_module():
    query = torch.randn(batch_size, tgt_len, d_model)
    key_padding_mask = torch.tensor(
        [
            [False, False, True, False],
            [False, True, True, False],
            [False, False, False, True],
            [False, True, False, True],
        ]
    )

    actual = MultiheadAttention(d_model, num_heads)
    expected = nn.MultiheadAttention(d_model, num_heads)

    with torch.inference_mode():
        expected.in_proj_weight.copy_(
            torch.concat(
                [
                    actual.q_proj.weight,
                    actual.k_proj.weight,
                    actual.v_proj.weight,
                ],
                dim=0,
            )
        )
        expected.in_proj_bias.copy_(
            torch.concat(
                [
                    actual.q_proj.bias,
                    actual.k_proj.bias,
                    actual.v_proj.bias,
                ],
                dim=0,
            )
        )
        expected.out_proj.weight.copy_(actual.out_proj.weight)
        expected.out_proj.bias.copy_(actual.out_proj.bias)

    actual_output, actual_weights = actual(
        query,
        query,
        query,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    expected_output, expected_weights = expected(
        query.transpose(0, 1),
        query.transpose(0, 1),
        query.transpose(0, 1),
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    expected_output = expected_output.transpose(0, 1)

    assert torch.allclose(actual_output, expected_output, atol=1e-6)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-6)


def test_multihead_attention_module_matches_torch_without_bias():
    query = torch.randn(batch_size, tgt_len, d_model)

    actual = MultiheadAttention(d_model, num_heads, bias=False)
    expected = nn.MultiheadAttention(d_model, num_heads, bias=False)

    assert actual.bias is False
    assert actual.q_proj.bias is None
    assert actual.k_proj.bias is None
    assert actual.v_proj.bias is None
    assert actual.out_proj.bias is None

    with torch.inference_mode():
        expected.in_proj_weight.copy_(
            torch.concat(
                [
                    actual.q_proj.weight,
                    actual.k_proj.weight,
                    actual.v_proj.weight,
                ],
                dim=0,
            )
        )
        expected.out_proj.weight.copy_(actual.out_proj.weight)

    actual_output, actual_weights = actual(
        query,
        query,
        query,
        need_weights=True,
    )
    expected_output, expected_weights = expected(
        query.transpose(0, 1),
        query.transpose(0, 1),
        query.transpose(0, 1),
        need_weights=True,
    )
    expected_output = expected_output.transpose(0, 1)

    assert torch.allclose(actual_output, expected_output, atol=1e-6)
    assert torch.allclose(actual_weights, expected_weights, atol=1e-6)


def test_sinusoidal_positional_encoding():
    x = torch.zeros(batch_size, src_len, d_model)
    module = SinusoidalPositionalEncoding(d_model, max_len=src_len)

    output = module(x)
    expected = module.pe.expand_as(output)  # type: ignore

    assert output.shape == x.shape
    assert torch.allclose(output, expected)

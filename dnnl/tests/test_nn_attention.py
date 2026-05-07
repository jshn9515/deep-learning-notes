import math

import torch
import torch.nn.functional as F

from dnnl.nn import MultiheadAttention, SinusoidalPositionalEncoding
from dnnl.nn import functional as dF


def test_attention_returns_expected_weights_and_output():
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output, weights = dF.attention(query, key, value, need_weights=True)
    expected_weights = F.softmax(query @ key.transpose(-2, -1), dim=-1)

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_weights @ value)


def test_scaled_dot_product_attention_matches_torch():
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 3, 4)
    value = torch.randn(2, 3, 5)

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, need_weights=True
    )
    expected = F.scaled_dot_product_attention(query, key, value)
    expected_weights = torch.softmax(
        (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1)), dim=-1
    )

    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(weights, expected_weights)


def test_scaled_dot_product_attention_supports_bool_mask_like_torch():
    query = torch.randn(2, 1, 3, 4)
    key = torch.randn(2, 1, 5, 4)
    value = torch.randn(2, 1, 5, 6)
    attn_mask = torch.tensor(
        [
            [
                [True, True, False, True, False],
                [False, True, True, True, False],
                [True, False, True, False, True],
            ]
        ]
    )

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, need_weights=True
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert torch.allclose(output, expected, atol=1e-6)
    masked_weights = weights.masked_select(~attn_mask.expand_as(weights))
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))


def test_scaled_dot_product_attention_supports_additive_mask_like_torch():
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 6)
    attn_mask = torch.zeros(3, 5)
    attn_mask[:, -1] = -torch.inf

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, need_weights=True
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(weights[..., -1], torch.zeros_like(weights[..., -1]))


def test_scaled_dot_product_attention_supports_causal_mode_like_torch():
    query = torch.randn(2, 1, 4, 3)
    key = torch.randn(2, 1, 4, 3)
    value = torch.randn(2, 1, 4, 5)

    output, weights = dF.scaled_dot_product_attention(
        query, key, value, need_weights=True, is_causal=True
    )
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    forbidden = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)

    assert torch.allclose(output, expected, atol=1e-6)
    masked_weights = weights[..., forbidden]
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))


def test_scaled_dot_product_attention_respects_scale_and_training_dropout_flag():
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 3, 4)
    value = torch.randn(2, 3, 5)

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
    batch_size = 2
    target_len = 3
    source_len = 5
    d_model = 4
    key_dim = 6
    value_dim = 8
    num_heads = 2
    d_head = d_model // num_heads
    query = torch.randn(batch_size, target_len, d_model)
    key = torch.randn(batch_size, source_len, key_dim)
    value = torch.randn(batch_size, source_len, value_dim)
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
        W_Q,
        W_K,
        W_V,
        W_O,
        q_proj_bias=b_Q,
        k_proj_bias=b_K,
        v_proj_bias=b_V,
        out_proj_bias=b_O,
        need_weights=True,
    )

    Q = ((query @ W_Q) + b_Q).view(batch_size, target_len, num_heads, d_head)
    K = ((key @ W_K) + b_K).view(batch_size, source_len, num_heads, d_head)
    V = ((value @ W_V) + b_V).view(batch_size, source_len, num_heads, d_head)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    expected_head_output, expected_weights = dF.scaled_dot_product_attention(
        Q, K, V, need_weights=True
    )
    expected_output = (
        expected_head_output.transpose(1, 2).reshape(batch_size, target_len, d_model)
        @ W_O
    ) + b_O

    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_output)


def test_multi_head_attention_matches_torch_cross_attention_with_bias():
    torch.manual_seed(1)
    batch_size = 2
    target_len = 3
    source_len = 5
    embed_dim = 4
    key_dim = 6
    value_dim = 8
    num_heads = 2
    query = torch.randn(batch_size, target_len, embed_dim)
    key = torch.randn(batch_size, source_len, key_dim)
    value = torch.randn(batch_size, source_len, value_dim)

    q_weight = torch.randn(embed_dim, embed_dim)
    k_weight = torch.randn(key_dim, embed_dim)
    v_weight = torch.randn(value_dim, embed_dim)
    out_weight = torch.randn(embed_dim, embed_dim)
    q_bias = torch.randn(embed_dim)
    k_bias = torch.randn(embed_dim)
    v_bias = torch.randn(embed_dim)
    out_bias = torch.randn(embed_dim)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads,
        q_weight,
        k_weight,
        v_weight,
        out_weight,
        q_proj_bias=q_bias,
        k_proj_bias=k_bias,
        v_proj_bias=v_bias,
        out_proj_bias=out_bias,
        need_weights=True,
    )

    torch_mha = torch.nn.MultiheadAttention(
        embed_dim,
        num_heads,
        kdim=key_dim,
        vdim=value_dim,
    )
    with torch.no_grad():
        torch_mha.q_proj_weight.copy_(q_weight.T)
        torch_mha.k_proj_weight.copy_(k_weight.T)
        torch_mha.v_proj_weight.copy_(v_weight.T)
        torch_mha.in_proj_bias.copy_(torch.cat([q_bias, k_bias, v_bias]))
        torch_mha.out_proj.weight.copy_(out_weight.T)
        torch_mha.out_proj.bias.copy_(out_bias)

    expected, expected_weights = torch_mha(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        need_weights=True,
    )
    expected = expected.transpose(0, 1)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(actual_weights.mean(dim=1), expected_weights, atol=1e-6)


def test_multihead_attention_module_matches_torch_module():
    torch.manual_seed(0)
    embed_dim, num_heads = 4, 2
    query = torch.randn(2, 3, embed_dim)
    key_padding_mask = torch.tensor([[False, False, True], [False, True, True]])

    actual = MultiheadAttention(embed_dim, num_heads)
    expected = torch.nn.MultiheadAttention(embed_dim, num_heads)

    with torch.no_grad():
        expected.in_proj_weight.copy_(
            torch.cat(
                [
                    actual.q_proj.weight,
                    actual.k_proj.weight,
                    actual.v_proj.weight,
                ],
                dim=0,
            )
        )
        expected.in_proj_bias.copy_(
            torch.cat(
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


def test_sinusoidal_positional_encoding_adds_sequence_positions():
    module = SinusoidalPositionalEncoding(embed_dim=4, max_len=3)
    x = torch.zeros(2, 3, 4)

    output = module(x)
    expected = module.pe.expand_as(output)

    assert output.shape == x.shape
    assert torch.allclose(output, expected)

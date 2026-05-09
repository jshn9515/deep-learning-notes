import pytest
import torch
import torch.nn as nn

import dnnl.nn.functional as dF
from dnnl.nn import (
    MultiheadAttention,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

batch_size = 2
src_len = 4
tgt_len = 8
d_model = 8
num_heads = 2


def _copy_mha_to_torch(actual: MultiheadAttention, expected: nn.MultiheadAttention):
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
        if expected.in_proj_bias is not None:
            expected.in_proj_bias.copy_(
                torch.concat(
                    [
                        actual.q_proj.bias,
                        actual.k_proj.bias,
                        actual.v_proj.bias,
                    ]
                )
            )
        expected.out_proj.weight.copy_(actual.out_proj.weight)
        if expected.out_proj.bias is not None:
            expected.out_proj.bias.copy_(actual.out_proj.bias)


def _copy_encoder_layer_to_torch(
    actual: TransformerEncoderLayer,
    expected: nn.TransformerEncoderLayer,
):
    _copy_mha_to_torch(actual.self_attn, expected.self_attn)
    expected.linear1.load_state_dict(actual.linear1.state_dict())
    expected.linear2.load_state_dict(actual.linear2.state_dict())
    expected.norm1.load_state_dict(actual.norm1.state_dict())
    expected.norm2.load_state_dict(actual.norm2.state_dict())


def _copy_decoder_layer_to_torch(
    actual: TransformerDecoderLayer,
    expected: nn.TransformerDecoderLayer,
):
    _copy_mha_to_torch(actual.self_attn, expected.self_attn)
    _copy_mha_to_torch(actual.mha_attn, expected.multihead_attn)
    expected.linear1.load_state_dict(actual.linear1.state_dict())
    expected.linear2.load_state_dict(actual.linear2.state_dict())
    expected.norm1.load_state_dict(actual.norm1.state_dict())
    expected.norm2.load_state_dict(actual.norm2.state_dict())
    expected.norm3.load_state_dict(actual.norm3.state_dict())


def _copy_encoder_to_torch(actual: TransformerEncoder, expected: nn.TransformerEncoder):
    z = zip(actual.layers, expected.layers, strict=True)
    for actual_layer, expected_layer in z:
        _copy_encoder_layer_to_torch(actual_layer, expected_layer)  # type: ignore
    if actual.norm is not None and expected.norm is not None:
        expected.norm.load_state_dict(actual.norm.state_dict())


def _copy_decoder_to_torch(actual: TransformerDecoder, expected: nn.TransformerDecoder):
    z = zip(actual.layers, expected.layers, strict=True)
    for actual_layer, expected_layer in z:
        _copy_decoder_layer_to_torch(actual_layer, expected_layer)  # type: ignore
    if actual.norm is not None and expected.norm is not None:
        expected.norm.load_state_dict(actual.norm.state_dict())


@pytest.mark.parametrize('norm_first', [False, True])
def test_transformer_encoder_layer_matches_torch(norm_first: bool):
    src = torch.randn(batch_size, src_len, d_model)
    src_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    src_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    actual = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        norm_first=norm_first,
    )
    expected = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=norm_first,
    )
    _copy_encoder_layer_to_torch(actual, expected)

    actual_output = actual(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
    )
    expected_output = expected(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
    )
    assert torch.allclose(actual_output, expected_output, atol=1e-6)


def test_transformer_encoder_matches_torch_stack_with_norm():
    src = torch.randn(batch_size, src_len, d_model)
    layer1 = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
    )
    norm1 = nn.LayerNorm(d_model)
    actual = TransformerEncoder(layer1, num_layers=2, norm=norm1)

    layer2 = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    norm2 = nn.LayerNorm(d_model)
    expected = nn.TransformerEncoder(layer2, num_layers=2, norm=norm2)
    _copy_encoder_to_torch(actual, expected)

    assert torch.allclose(actual(src), expected(src), atol=1e-6)


@pytest.mark.parametrize('norm_first', [False, True])
def test_transformer_decoder_layer_matches_torch(norm_first: bool):
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    tgt_mask = dF.generate_causal_mask(tgt_len)
    memory_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )

    actual = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        activation='gelu',
        norm_first=norm_first,
    )
    expected = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        activation='gelu',
        batch_first=True,
        norm_first=norm_first,
    )
    _copy_decoder_layer_to_torch(actual, expected)

    actual_output = actual(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    expected_output = expected(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    assert torch.allclose(actual_output, expected_output, atol=1e-6)


def test_transformer_decoder_matches_torch_stack_with_norm():
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    layer1 = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
    )
    norm1 = nn.LayerNorm(8)
    actual = TransformerDecoder(layer1, num_layers=2, norm=norm1)

    layer2 = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    norm2 = nn.LayerNorm(8)
    expected = nn.TransformerDecoder(layer2, num_layers=2, norm=norm2)
    _copy_decoder_to_torch(actual, expected)

    actual_output = actual(tgt, memory)
    expected_output = expected(tgt, memory)
    assert torch.allclose(actual_output, expected_output, atol=1e-6)


def test_transformer_matches_torch_batch_first_transformer():
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)
    src_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    tgt_mask = dF.generate_causal_mask(tgt_len)

    actual = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        norm_first=False,
    )
    expected = nn.Transformer(
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=False,
    )
    _copy_encoder_to_torch(actual.encoder, expected.encoder)
    _copy_decoder_to_torch(actual.decoder, expected.decoder)

    actual_output = actual(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    expected_output = expected(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    assert torch.allclose(actual_output, expected_output, atol=1e-6)


def test_transformer_omits_batch_first_parameter():
    with pytest.raises(TypeError):
        TransformerEncoderLayer(d_model, num_heads, batch_first=False)  # type: ignore

    module = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)
    output = module(src, tgt)
    assert output.shape == tgt.shape

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

from dnnlpy.nn.functional import flash_attention_v1_backward, flash_attention_v1_forward

batch_size = 2
tgt_len = 8
src_len = 4
embed_dim = 6


def _torch_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
) -> Tensor:
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


@pytest.mark.parametrize('is_causal', [False, True])
def test_flash_attention_v1_forward_accepts_batch_input(is_causal: bool):
    query = torch.randn(batch_size, tgt_len, embed_dim)
    key = torch.randn(batch_size, src_len, embed_dim)
    value = torch.randn(batch_size, src_len, embed_dim)

    actual = flash_attention_v1_forward(
        query, key, value, Br=2, Bc=3, is_causal=is_causal
    )
    expected = _torch_attention(query, key, value, is_causal=is_causal)

    assert actual.shape == expected.shape
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_flash_attention_v1_forward_keeps_2d_input_compatible():
    query = torch.randn(tgt_len, embed_dim)
    key = torch.randn(src_len, embed_dim)
    value = torch.randn(src_len, embed_dim)

    actual = flash_attention_v1_forward(query, key, value, Br=3, Bc=2)
    expected = _torch_attention(query, key, value)

    assert actual.shape == expected.shape
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('is_causal', [False, True])
def test_flash_attention_v1_backward_matches_autograd_for_batch_input(is_causal: bool):
    query = torch.randn(batch_size, tgt_len, embed_dim, requires_grad=True)
    key = torch.randn(batch_size, src_len, embed_dim, requires_grad=True)
    value = torch.randn(batch_size, src_len, embed_dim, requires_grad=True)
    dO = torch.randn(batch_size, tgt_len, embed_dim)

    expected_output = _torch_attention(query, key, value, is_causal=is_causal)
    expected_output.backward(dO)

    dQ, dK, dV = flash_attention_v1_backward(
        query.detach(),
        key.detach(),
        value.detach(),
        dO,
        Br=2,
        Bc=3,
        is_causal=is_causal,
    )

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert_close(dQ, query.grad, rtol=1e-5, atol=1e-6)
    assert_close(dK, key.grad, rtol=1e-5, atol=1e-6)
    assert_close(dV, value.grad, rtol=1e-5, atol=1e-6)


def test_flash_attention_v1_backward_rejects_dropout():
    query = torch.randn(batch_size, tgt_len, embed_dim)
    key = torch.randn(batch_size, src_len, embed_dim)
    value = torch.randn(batch_size, src_len, embed_dim)
    dO = torch.randn(2, 3, 4)

    with pytest.raises(NotImplementedError):
        flash_attention_v1_backward(query, key, value, dO, Br=2, Bc=2, dropout=0.1)


@pytest.mark.parametrize(
    ('key', 'value', 'match'),
    [
        (torch.randn(2, 3), torch.randn(2, 3), 'same number of dimensions'),
        (
            torch.randn(2, src_len, embed_dim - 1),
            torch.randn(2, src_len, embed_dim),
            'same embedding dim',
        ),
        (
            torch.randn(2, src_len, embed_dim),
            torch.randn(2, src_len + 1, embed_dim),
            'same sequence length',
        ),
    ],
)
def test_flash_attention_v1_forward_rejects_invalid_tensors(
    key: Tensor, value: Tensor, match: str
):
    query = torch.randn(batch_size, tgt_len, embed_dim)

    with pytest.raises(AssertionError, match=match):
        flash_attention_v1_forward(query, key, value, Br=2, Bc=2)


def test_flash_attention_v1_backward_rejects_invalid_gradient():
    query = torch.randn(batch_size, tgt_len, embed_dim)
    key = torch.randn(batch_size, src_len, embed_dim)
    value = torch.randn(batch_size, src_len, embed_dim)
    dO = torch.randn(batch_size, src_len, embed_dim)

    with pytest.raises(AssertionError, match='output shape'):
        flash_attention_v1_backward(query, key, value, dO, Br=2, Bc=2)

import math

import pytest
import torch
from dnnl.ch10 import flash_attention_v1_backward, flash_attention_v1_forward


def _reference_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))
    scores = (Q @ K.transpose(-2, -1)) * scale
    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, -math.inf)
    return torch.softmax(scores, dim=-1) @ V


@pytest.mark.parametrize('causal', [False, True])
def test_flash_attention_v1_forward_accepts_batch_input(causal):
    Q = torch.randn(2, 5, 4)
    K = torch.randn(2, 5, 4)
    V = torch.randn(2, 5, 4)

    actual = flash_attention_v1_forward(Q, K, V, Br=2, Bc=3, causal=causal)
    expected = _reference_attention(Q, K, V, causal=causal)

    assert actual.shape == V.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_flash_attention_v1_forward_keeps_2d_input_compatible():
    Q = torch.randn(5, 4)
    K = torch.randn(5, 4)
    V = torch.randn(5, 4)

    actual = flash_attention_v1_forward(Q, K, V, Br=3, Bc=2)
    expected = _reference_attention(Q, K, V)

    assert actual.shape == V.shape
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('causal', [False, True])
def test_flash_attention_v1_backward_matches_autograd_for_batch_input(causal):
    Q = torch.randn(2, 5, 4, requires_grad=True)
    K = torch.randn(2, 5, 4, requires_grad=True)
    V = torch.randn(2, 5, 4, requires_grad=True)
    dO = torch.randn(2, 5, 4)

    expected_output = _reference_attention(Q, K, V, causal=causal)
    expected_output.backward(dO)

    dQ, dK, dV = flash_attention_v1_backward(
        Q.detach(),
        K.detach(),
        V.detach(),
        dO,
        Br=2,
        Bc=3,
        causal=causal,
    )

    assert torch.allclose(dQ, Q.grad, atol=1e-6)
    assert torch.allclose(dK, K.grad, atol=1e-6)
    assert torch.allclose(dV, V.grad, atol=1e-6)


def test_flash_attention_v1_backward_rejects_dropout():
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 4)
    dO = torch.randn(2, 3, 4)

    with pytest.raises(NotImplementedError):
        flash_attention_v1_backward(Q, K, V, dO, Br=2, Bc=2, dropout=0.1)

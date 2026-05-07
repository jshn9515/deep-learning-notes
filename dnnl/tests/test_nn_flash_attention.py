import pytest
import torch
import torch.nn.functional as F

from dnnl.nn.functional import flash_attention_v1_backward, flash_attention_v1_forward


def _torch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


@pytest.mark.parametrize('causal', [False, True])
def test_flash_attention_v1_forward_accepts_batch_input(causal):
    Q = torch.randn(2, 5, 4)
    K = torch.randn(2, 5, 4)
    V = torch.randn(2, 5, 4)

    actual = flash_attention_v1_forward(Q, K, V, Br=2, Bc=3, is_causal=causal)
    expected = _torch_attention(Q, K, V, causal=causal)

    assert actual.shape == V.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_flash_attention_v1_forward_keeps_2d_input_compatible():
    Q = torch.randn(5, 4)
    K = torch.randn(5, 4)
    V = torch.randn(5, 4)

    actual = flash_attention_v1_forward(Q, K, V, Br=3, Bc=2)
    expected = _torch_attention(Q, K, V)

    assert actual.shape == V.shape
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('causal', [False, True])
def test_flash_attention_v1_backward_matches_autograd_for_batch_input(causal):
    Q = torch.randn(2, 5, 4, requires_grad=True)
    K = torch.randn(2, 5, 4, requires_grad=True)
    V = torch.randn(2, 5, 4, requires_grad=True)
    dO = torch.randn(2, 5, 4)

    expected_output = _torch_attention(Q, K, V, causal=causal)
    expected_output.backward(dO)

    dQ, dK, dV = flash_attention_v1_backward(
        Q.detach(),
        K.detach(),
        V.detach(),
        dO,
        Br=2,
        Bc=3,
        is_causal=causal,
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


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'Br': 0, 'Bc': 2}, '`Br` must be greater than 0'),
        ({'Br': 2, 'Bc': -1}, '`Bc` must be greater than 0'),
        ({'Br': 2.0, 'Bc': 2}, '`Br` must be an integer'),
        ({'Br': 2, 'Bc': 2, 'dropout': 1.0}, '`dropout` must satisfy'),
        ({'Br': 2, 'Bc': 2, 'scale': 0.0}, '`scale` must be'),
    ],
)
def test_flash_attention_v1_forward_rejects_invalid_parameters(kwargs, match):
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 4)

    with pytest.raises(AssertionError, match=match):
        flash_attention_v1_forward(Q, K, V, **kwargs)


@pytest.mark.parametrize(
    ('key', 'value', 'match'),
    [
        (torch.randn(2, 3), torch.randn(2, 3), 'same number of dimensions'),
        (torch.randn(2, 4, 4), torch.randn(2, 3, 4), 'same shape'),
        (torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(2, 3, 4), 'same dtype'),
        (torch.ones(2, 3, 4, dtype=torch.long), torch.ones(2, 3, 4), 'floating-point'),
    ],
)
def test_flash_attention_v1_forward_rejects_invalid_tensors(key, value, match):
    Q = torch.randn(2, 3, 4)

    with pytest.raises(AssertionError, match=match):
        flash_attention_v1_forward(Q, key, value, Br=2, Bc=2)


def test_flash_attention_v1_backward_rejects_invalid_gradient():
    Q = torch.randn(2, 3, 4)
    K = torch.randn(2, 3, 4)
    V = torch.randn(2, 3, 4)
    dO = torch.randn(2, 3, 5)

    with pytest.raises(AssertionError, match='same shape as `query`'):
        flash_attention_v1_backward(Q, K, V, dO, Br=2, Bc=2)

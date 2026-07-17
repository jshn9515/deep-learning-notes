from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


def test_rotary_positional_embedding_matches_precalculated_rotation_matrices():
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.5, 2.0, -3.0],
                [0.25, -0.75, 1.5, 2.5],
            ],
            [
                [-2.0, 1.0, 0.5, 3.0],
                [4.0, -1.0, -2.0, 0.25],
                [1.25, 2.25, -0.5, -1.5],
            ],
        ],
        dtype=torch.float64,
    )

    # For dim=4 and base=10000, the two pair frequencies are 1 and 0.01.
    # Each sequence position p uses R(p) = diag(R_2(p), R_2(p / 100)).
    rotation_matrices = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.5403023058681398, -0.8414709848078965, 0.0, 0.0],
                [0.8414709848078965, 0.5403023058681398, 0.0, 0.0],
                [0.0, 0.0, 0.9999500004166653, -0.0099998333341667],
                [0.0, 0.0, 0.0099998333341667, 0.9999500004166653],
            ],
            [
                [-0.4161468365471424, -0.9092974268256817, 0.0, 0.0],
                [0.9092974268256817, -0.4161468365471424, 0.0, 0.0],
                [0.0, 0.0, 0.9998000066665778, -0.0199986666933331],
                [0.0, 0.0, 0.0199986666933331, 0.9998000066665778],
            ],
        ],
        dtype=torch.float64,
    )
    expected = torch.einsum('sij,bsj->bsi', rotation_matrices, x)

    actual = dF.rotary_positional_embedding(x)
    assert_close(actual, expected)

    actual = dnn.RotaryPositionalEmbedding(embed_dim=4)(x)
    assert_close(actual, expected)


def _assert_embedding_matches_torch(x: Tensor, weight: Tensor, **kwargs: Any) -> None:
    actual_weight = _copy(weight)
    expected_weight = _copy(weight)

    actual = dF.embedding(x, actual_weight, **kwargs)
    expected = F.embedding(x, expected_weight, **kwargs)

    assert_close(actual, expected)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)

    assert actual_weight.grad is not None
    assert expected_weight.grad is not None
    assert_close(actual_weight.grad, expected_weight.grad)


def test_embedding_function_matches_torch_forward_and_backward():
    x = torch.tensor([[0, 2, 4], [3, 1, 2]])
    weight = torch.randn(5, 3, dtype=torch.float64)

    _assert_embedding_matches_torch(x, weight)


@pytest.mark.parametrize('padding_idx', [1, -1])
def test_embedding_function_matches_torch_with_padding_idx(padding_idx: int):
    x = torch.tensor([[0, 1, 3], [3, 1, 2]])
    weight = torch.randn(4, 2, dtype=torch.float64)

    _assert_embedding_matches_torch(
        x,
        weight,
        padding_idx=padding_idx,
    )


def test_embedding_function_matches_torch_with_scale_grad_by_freq():
    x = torch.tensor([0, 1, 1, 2, 2, 2])
    weight = torch.randn(3, 4, dtype=torch.float64)

    _assert_embedding_matches_torch(
        x,
        weight,
        scale_grad_by_freq=True,
    )


def test_embedding_function_matches_torch_with_max_norm():
    x = torch.tensor([[0, 2], [2, 3]])

    actual_weight = torch.tensor(
        [[3.0, 4.0], [6.0, 8.0], [0.5, 0.0], [0.0, 12.0]],
        requires_grad=True,
    )
    expected_weight = _copy(actual_weight)

    actual = dF.embedding(
        x,
        actual_weight,
        max_norm=1.0,
        norm_type=2.0,
    )
    expected = F.embedding(
        x,
        expected_weight,
        max_norm=1.0,
        norm_type=2.0,
    )

    assert_close(actual, expected)
    assert_close(actual_weight, expected_weight)
    assert_close(actual_weight[1], torch.tensor([6.0, 8.0]))


@pytest.mark.parametrize('padding_idx', [4, -5])
def test_embedding_function_rejects_out_of_range_padding_idx(padding_idx: int):
    with pytest.raises(
        AssertionError,
        match='padding_idx.*num_embeddings',
    ):
        dF.embedding(
            torch.tensor([0, 1]),
            torch.randn(4, 2),
            padding_idx=padding_idx,
        )


def test_embedding_function_rejects_non_matrix_weight():
    with pytest.raises(AssertionError, match='2-dimensional'):
        dF.embedding(torch.tensor([0]), torch.randn(3))


def test_embedding_module_initialization_matches_torch():
    torch.manual_seed(0)
    custom = dnn.Embedding(5, 3, padding_idx=0)
    torch.manual_seed(0)
    reference = nn.Embedding(5, 3, padding_idx=0)

    assert custom.fast is False
    assert_close(custom.weight, reference.weight)
    assert_close(custom.weight[0], torch.zeros(3))


def test_embedding_module_matches_torch_forward_and_backward():
    x = torch.tensor([[0, 1, 3, 3], [2, 1, 3, 0]])

    custom = dnn.Embedding(
        num_embeddings=4,
        embedding_dim=3,
        padding_idx=-1,
        max_norm=1.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
    )
    reference = nn.Embedding(
        num_embeddings=4,
        embedding_dim=3,
        padding_idx=-1,
        max_norm=1.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
    )
    reference.load_state_dict(custom.state_dict())

    actual = custom(x)
    expected = reference(x)

    assert_close(actual, expected)
    assert_close(custom.weight, reference.weight)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)

    assert custom.weight.grad is not None
    assert reference.weight.grad is not None
    assert_close(custom.weight.grad, reference.weight.grad)


def test_fast_embedding_module_matches_torch_forward_and_backward():
    x = torch.tensor([[0, 1, 3, 3], [2, 1, 3, 0]])

    custom = dnn.Embedding(
        num_embeddings=4,
        embedding_dim=3,
        padding_idx=-1,
        max_norm=1.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
        fast=True,
    )
    reference = nn.Embedding(
        num_embeddings=4,
        embedding_dim=3,
        padding_idx=-1,
        max_norm=1.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
    )
    reference.load_state_dict(custom.state_dict())

    actual = custom(x)
    expected = reference(x)

    assert custom.fast is True
    assert_close(actual, expected)
    assert_close(custom.weight, reference.weight)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)

    assert custom.weight.grad is not None
    assert reference.weight.grad is not None
    assert_close(custom.weight.grad, reference.weight.grad)


def test_embedding_module_supports_pretrained_frozen_weight():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    custom = dnn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)

    assert custom.num_embeddings == 3
    assert custom.embedding_dim == 2
    assert custom.padding_idx == 0
    assert not custom.weight.requires_grad
    assert_close(custom(torch.tensor([0, 2])), weight[[0, 2]])


def test_embedding_module_supports_custom_frozen_weight():
    weight = torch.randn(4, 3, dtype=torch.float64)
    custom = dnn.Embedding(4, 3, _weight=weight, _freeze=True)

    assert custom.weight.dtype == torch.float64
    assert custom.weight.device.type == 'cpu'
    assert not custom.weight.requires_grad
    assert_close(custom.weight, weight)


def test_embedding_module_rejects_mismatched_custom_weight():
    with pytest.raises(AssertionError, match='Shape of weight'):
        dnn.Embedding(4, 3, _weight=torch.randn(3, 4))


def test_embedding_module_extra_repr_matches_torch():
    custom = dnn.Embedding(
        5,
        3,
        padding_idx=1,
        max_norm=2.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
        fast=True,
    )
    reference = nn.Embedding(
        5,
        3,
        padding_idx=1,
        max_norm=2.0,
        norm_type=1.0,
        scale_grad_by_freq=True,
    )

    assert custom.extra_repr() == reference.extra_repr()

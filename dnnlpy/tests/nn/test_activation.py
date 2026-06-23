from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

type ActFn = Callable[[Tensor], Tensor]


@pytest.mark.parametrize(
    ('actual_fn', 'expected_fn'),
    [
        (dF.sigmoid, F.sigmoid),
        (dF.tanh, F.tanh),
        (dF.relu, F.relu),
        (dF.gelu, F.gelu),
    ],
)
def test_elementwise_activation_functions_match_torch(
    actual_fn: ActFn, expected_fn: ActFn
):
    x = torch.linspace(-3, 3, steps=13)
    actual = actual_fn(x)
    expected = expected_fn(x)

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmax_function_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dF.softmax(x, dim=dim)
    expected = F.softmax(x, dim=dim)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_log_softmax_function_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dF.log_softmax(x, dim=dim)
    expected = F.log_softmax(x, dim=dim)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ('actual_module', 'expected_module'),
    [
        (dnn.Sigmoid(), nn.Sigmoid()),
        (dnn.Tanh(), nn.Tanh()),
        (dnn.ReLU(), nn.ReLU()),
        (dnn.GELU(), nn.GELU()),
    ],
)
def test_elementwise_activation_modules_match_torch(
    actual_module: ActFn, expected_module: ActFn
):
    x = torch.linspace(-3, 3, steps=13)
    actual = actual_module(x)
    expected = expected_module(x)

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize(
    ('actual_module', 'expected_fn'),
    [
        (dnn.Sigmoid(fast=True), F.sigmoid),
        (dnn.Tanh(fast=True), F.tanh),
        (dnn.ReLU(fast=True), F.relu),
        (dnn.GELU(fast=True), F.gelu),
    ],
)
def test_fast_elementwise_activation_modules_match_torch(
    actual_module: ActFn, expected_fn: ActFn
):
    x = torch.linspace(-3, 3, steps=13)
    actual = actual_module(x)
    expected = expected_fn(x)

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmax(dim=dim)(x)
    expected = nn.Softmax(dim=dim)(x)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_fast_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmax(dim=dim, fast=True)(x)
    expected = F.softmax(x, dim=dim)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_log_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.LogSoftmax(dim=dim)(x)
    expected = nn.LogSoftmax(dim=dim)(x)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_fast_log_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.LogSoftmax(dim=dim, fast=True)(x)
    expected = F.log_softmax(x, dim=dim)

    assert torch.allclose(actual, expected)


def test_softmax_modules_include_dim_in_repr():
    softmax = dnn.Softmax(dim=-1)
    log_softmax = dnn.LogSoftmax(dim=1)
    assert softmax.extra_repr() == 'dim=-1'
    assert log_softmax.extra_repr() == 'dim=1'

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import dnnl.nn as dnn
import dnnl.nn.functional as dF


@pytest.mark.parametrize(
    ('actual_fn', 'expected_fn'),
    [
        (dF.sigmoid, torch.sigmoid),
        (dF.tanh, torch.tanh),
        (dF.relu, F.relu),
        (dF.gelu, F.gelu),
    ],
)
def test_elementwise_activation_functions_match_torch(actual_fn, expected_fn):
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
def test_elementwise_activation_modules_match_torch(actual_module, expected_module):
    x = torch.linspace(-3, 3, steps=13)
    actual = actual_module(x)
    expected = expected_module(x)

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmax(dim=dim)(x)
    expected = nn.Softmax(dim=dim)(x)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_log_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.LogSoftmax(dim=dim)(x)
    expected = nn.LogSoftmax(dim=dim)(x)

    assert torch.allclose(actual, expected)


def test_softmax_modules_include_dim_in_repr():
    softmax = dnn.Softmax(dim=-1)
    log_softmax = dnn.LogSoftmax(dim=1)
    assert softmax.extra_repr() == 'dim=-1'
    assert log_softmax.extra_repr() == 'dim=1'

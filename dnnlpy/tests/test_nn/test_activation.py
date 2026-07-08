from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

type ActFn = Callable[[Tensor], Tensor]


@pytest.mark.parametrize(
    ('actual_fn', 'expected_fn'),
    [
        (dF.celu, F.celu),
        (dF.elu, F.elu),
        (dF.gelu, F.gelu),
        (dF.hardsigmoid, F.hardsigmoid),
        (dF.hardshrink, F.hardshrink),
        (dF.hardswish, F.hardswish),
        (dF.hardtanh, F.hardtanh),
        (dF.leaky_relu, F.leaky_relu),
        (dF.log_sigmoid, F.logsigmoid),
        (dF.mish, F.mish),
        (dF.relu, F.relu),
        (dF.relu6, F.relu6),
        (dF.rrelu, F.rrelu),
        (dF.selu, F.selu),
        (dF.sigmoid, F.sigmoid),
        (dF.silu, F.silu),
        (dF.softplus, F.softplus),
        (dF.softshrink, F.softshrink),
        (dF.softsign, F.softsign),
        (dF.tanh, F.tanh),
        (dF.tanhshrink, F.tanhshrink),
    ],
)
def test_elementwise_activation_functions_match_torch(
    actual_fn: ActFn, expected_fn: ActFn
):
    x = torch.linspace(-3, 3, steps=13)
    actual = actual_fn(x)
    expected = expected_fn(x)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_prelu_function_matches_torch():
    x = torch.linspace(-3, 3, steps=24).reshape(2, 3, 4)
    weight = torch.tensor([0.1, 0.2, 0.3])

    actual = dF.prelu(x, weight)
    expected = F.prelu(x, weight)

    assert_close(actual, expected)


def test_threshold_function_matches_torch():
    x = torch.linspace(-3, 3, steps=13)
    actual = dF.threshold(x, threshold=0.5, value=-2.0)
    expected = F.threshold(x, threshold=0.5, value=-2.0)

    assert_close(actual, expected)


def test_softplus_function_is_stable_for_extreme_inputs():
    x = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], requires_grad=True)
    actual = dF.softplus(x)
    expected = F.softplus(x)

    assert torch.isfinite(actual).all()
    assert_close(actual, expected)

    actual.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gelu_function_matches_torch_tanh_approximation():
    x = torch.linspace(-3, 3, steps=13)
    actual = dF.gelu(x, approximate='tanh')
    expected = F.gelu(x, approximate='tanh')

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_relu_function_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()

    actual = dF.relu(x, inplace=True)
    F.relu(expected, inplace=True)

    assert actual is x
    assert_close(actual, expected)


def test_silu_function_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()

    actual = dF.silu(x, inplace=True)
    F.silu(expected, inplace=True)

    assert actual is x
    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_fn', 'expected_fn'),
    [
        (
            lambda x: dF.celu(x, alpha=0.7, inplace=True),
            lambda x: F.celu(x, alpha=0.7, inplace=True),
        ),
        (
            lambda x: dF.elu(x, alpha=0.7, inplace=True),
            lambda x: F.elu(x, alpha=0.7, inplace=True),
        ),
        (
            lambda x: dF.hardsigmoid(x, inplace=True),
            lambda x: F.hardsigmoid(x, inplace=True),
        ),
        (
            lambda x: dF.hardswish(x, inplace=True),
            lambda x: F.hardswish(x, inplace=True),
        ),
        (
            lambda x: dF.hardtanh(x, min_val=-0.5, max_val=1.5, inplace=True),
            lambda x: F.hardtanh(x, min_val=-0.5, max_val=1.5, inplace=True),
        ),
        (
            lambda x: dF.leaky_relu(x, negative_slope=0.2, inplace=True),
            lambda x: F.leaky_relu(x, negative_slope=0.2, inplace=True),
        ),
        (
            lambda x: dF.mish(x, inplace=True),
            lambda x: F.mish(x, inplace=True),
        ),
        (
            lambda x: dF.relu6(x, inplace=True),
            lambda x: F.relu6(x, inplace=True),
        ),
        (
            lambda x: dF.selu(x, inplace=True),
            lambda x: F.selu(x, inplace=True),
        ),
        (
            lambda x: dF.threshold(x, threshold=0.5, value=-2.0, inplace=True),
            lambda x: F.threshold(x, threshold=0.5, value=-2.0, inplace=True),
        ),
    ],
)
def test_inplace_activation_functions_match_torch(actual_fn: ActFn, expected_fn: ActFn):
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()

    actual = actual_fn(x)
    expected = expected_fn(expected)

    assert actual is x
    assert_close(actual, expected)


def test_rrelu_function_training_samples_negative_slopes_in_range():
    x = torch.tensor([-4.0, -2.0, 0.0, 2.0])
    actual = dF.rrelu(x, lower=0.1, upper=0.3, training=True)

    slopes = actual[:2] / x[:2]
    assert torch.all(slopes >= 0.1)
    assert torch.all(slopes <= 0.3)
    assert_close(actual[2:], x[2:])


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmax_function_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dF.softmax(x, dim=dim)
    expected = F.softmax(x, dim=dim)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmin_function_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dF.softmin(x, dim=dim)
    expected = F.softmin(x, dim=dim)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_log_softmax_function_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dF.log_softmax(x, dim=dim)
    expected = F.log_softmax(x, dim=dim)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_module', 'expected_module'),
    [
        (dnn.CELU(0.7), nn.CELU(0.7)),
        (dnn.ELU(0.7), nn.ELU(0.7)),
        (dnn.GELU(), nn.GELU()),
        (dnn.HardShrink(0.4), nn.Hardshrink(0.4)),
        (dnn.HardSigmoid(), nn.Hardsigmoid()),
        (dnn.HardSwish(), nn.Hardswish()),
        (dnn.HardTanh(-0.5, 1.5), nn.Hardtanh(-0.5, 1.5)),
        (dnn.LeakyReLU(0.2), nn.LeakyReLU(0.2)),
        (dnn.LogSigmoid(), nn.LogSigmoid()),
        (dnn.Mish(), nn.Mish()),
        (dnn.ReLU(), nn.ReLU()),
        (dnn.ReLU6(), nn.ReLU6()),
        (dnn.RReLU(0.1, 0.3), nn.RReLU(0.1, 0.3)),
        (dnn.SELU(), nn.SELU()),
        (dnn.Sigmoid(), nn.Sigmoid()),
        (dnn.SiLU(), nn.SiLU()),
        (dnn.Softplus(beta=2.0, threshold=10.0), nn.Softplus(2.0, 10.0)),
        (dnn.SoftShrink(0.4), nn.Softshrink(0.4)),
        (dnn.SoftSign(), nn.Softsign()),
        (dnn.Tanh(), nn.Tanh()),
        (dnn.TanhShrink(), nn.Tanhshrink()),
        (dnn.Threshold(0.5, -2.0), nn.Threshold(0.5, -2.0)),
    ],
)
def test_elementwise_activation_modules_match_torch(
    actual_module: nn.Module, expected_module: nn.Module
):
    x = torch.linspace(-3, 3, steps=13)
    actual_module.eval()
    expected_module.eval()
    actual = actual_module(x)
    expected = expected_module(x)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_prelu_module_matches_torch():
    actual_module = dnn.PReLU(num_parameters=3, init=0.2)
    expected_module = nn.PReLU(num_parameters=3, init=0.2)
    expected_module.load_state_dict(actual_module.state_dict())
    x = torch.linspace(-3, 3, steps=24).reshape(2, 3, 4)

    actual = actual_module(x)
    expected = expected_module(x)

    assert_close(actual, expected)


def test_prelu_reset_parameters_restores_initial_weight():
    module = dnn.PReLU(num_parameters=3, init=0.2)
    module.weight.data.fill_(1.0)

    module.reset_parameters()

    assert_close(module.weight, torch.full((3,), 0.2))


def test_relu_module_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()
    actual_module = dnn.ReLU(inplace=True)
    expected_module = nn.ReLU(inplace=True)

    actual = actual_module(x)
    expected = expected_module(expected)

    assert actual is x
    assert actual_module.inplace is True
    assert_close(actual, expected)


def test_silu_module_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()
    actual_module = dnn.SiLU(inplace=True)
    expected_module = nn.SiLU(inplace=True)

    actual = actual_module(x)
    expected = expected_module(expected)

    assert actual is x
    assert actual_module.inplace is True
    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_module', 'expected_fn'),
    [
        (dnn.CELU(fast=True), F.celu),
        (dnn.ELU(fast=True), F.elu),
        (dnn.GELU(fast=True), F.gelu),
        (dnn.HardShrink(fast=True), F.hardshrink),
        (dnn.HardSigmoid(fast=True), F.hardsigmoid),
        (dnn.HardSwish(fast=True), F.hardswish),
        (dnn.HardTanh(fast=True), F.hardtanh),
        (dnn.LeakyReLU(fast=True), F.leaky_relu),
        (dnn.LogSigmoid(fast=True), F.logsigmoid),
        (dnn.Mish(fast=True), F.mish),
        (dnn.ReLU(fast=True), F.relu),
        (dnn.ReLU6(fast=True), F.relu6),
        (dnn.RReLU(fast=True), F.rrelu),
        (dnn.SELU(fast=True), F.selu),
        (dnn.Sigmoid(fast=True), F.sigmoid),
        (dnn.SiLU(fast=True), F.silu),
        (dnn.Softplus(fast=True), F.softplus),
        (dnn.SoftShrink(fast=True), F.softshrink),
        (dnn.SoftSign(fast=True), F.softsign),
        (dnn.Tanh(fast=True), F.tanh),
        (dnn.TanhShrink(fast=True), F.tanhshrink),
    ],
)
def test_fast_elementwise_activation_modules_match_torch(
    actual_module: nn.Module, expected_fn: nn.Module
):
    x = torch.linspace(-3, 3, steps=13)
    actual_module.eval()
    actual = actual_module(x)
    expected = expected_fn(x)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fast_gelu_module_matches_torch_tanh_approximation():
    x = torch.linspace(-3, 3, steps=13)
    actual = dnn.GELU(approximate='tanh', fast=True)(x)
    expected = F.gelu(x, approximate='tanh')

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fast_relu_module_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()
    actual_module = dnn.ReLU(inplace=True, fast=True)

    actual = actual_module(x)
    F.relu(expected, inplace=True)

    assert actual is x
    assert actual_module.inplace is True
    assert actual_module.fast is True
    assert_close(actual, expected)


def test_fast_silu_module_supports_inplace():
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()
    actual_module = dnn.SiLU(inplace=True, fast=True)

    actual = actual_module(x)
    F.silu(expected, inplace=True)

    assert actual is x
    assert actual_module.inplace is True
    assert actual_module.fast is True
    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_module', 'expected_module'),
    [
        (dnn.CELU(0.7, inplace=True), nn.CELU(0.7, inplace=True)),
        (dnn.ELU(0.7, inplace=True), nn.ELU(0.7, inplace=True)),
        (dnn.HardSigmoid(inplace=True), nn.Hardsigmoid(inplace=True)),
        (dnn.HardSwish(inplace=True), nn.Hardswish(inplace=True)),
        (dnn.HardTanh(-0.5, 1.5, inplace=True), nn.Hardtanh(-0.5, 1.5, inplace=True)),
        (dnn.LeakyReLU(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True)),
        (dnn.Mish(inplace=True), nn.Mish(inplace=True)),
        (dnn.ReLU6(inplace=True), nn.ReLU6(inplace=True)),
        (dnn.RReLU(0.1, 0.3, inplace=True), nn.RReLU(0.1, 0.3, inplace=True)),
        (dnn.SELU(inplace=True), nn.SELU(inplace=True)),
        (dnn.Threshold(0.5, -2.0, inplace=True), nn.Threshold(0.5, -2.0, inplace=True)),
    ],
)
def test_inplace_activation_modules_match_torch(
    actual_module: nn.Module, expected_module: nn.Module
):
    x = torch.linspace(-3, 3, steps=13)
    expected = x.clone()
    actual_module.eval()
    expected_module.eval()

    actual = actual_module(x)
    expected = expected_module(expected)

    assert actual is x
    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmax(dim=dim)(x)
    expected = nn.Softmax(dim=dim)(x)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_fast_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmax(dim=dim, fast=True)(x)
    expected = F.softmax(x, dim=dim)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_softmin_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmin(dim=dim)(x)
    expected = nn.Softmin(dim=dim)(x)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_fast_softmin_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.Softmin(dim=dim, fast=True)(x)
    expected = F.softmin(x, dim=dim)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_log_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.LogSoftmax(dim=dim)(x)
    expected = nn.LogSoftmax(dim=dim)(x)

    assert_close(actual, expected)


@pytest.mark.parametrize('dim', [0, 1, -1])
def test_fast_log_softmax_module_matches_torch(dim: int):
    x = torch.randn(3, 4, 5)
    actual = dnn.LogSoftmax(dim=dim, fast=True)(x)
    expected = F.log_softmax(x, dim=dim)

    assert_close(actual, expected)


def test_softmax_modules_include_dim_in_repr():
    softmin = dnn.Softmin(dim=0)
    softmax = dnn.Softmax(dim=-1)
    log_softmax = dnn.LogSoftmax(dim=1)
    assert softmin.extra_repr() == 'dim=0'
    assert softmax.extra_repr() == 'dim=-1'
    assert log_softmax.extra_repr() == 'dim=1'

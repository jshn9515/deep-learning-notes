import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


def test_bilinear_function_matches_torch_with_bias():
    input1 = torch.randn(4, 3)
    input2 = torch.randn(4, 2)
    weight = torch.randn(5, 3, 2)
    bias = torch.randn(5)

    actual = dF.bilinear(input1, input2, weight, bias)
    expected = F.bilinear(input1, input2, weight, bias)

    assert_close(actual, expected)


def test_bilinear_function_matches_torch_without_bias_for_batched_inputs():
    input1 = torch.randn(2, 4, 3)
    input2 = torch.randn(2, 4, 2)
    weight = torch.randn(5, 3, 2)

    actual = dF.bilinear(input1, input2, weight)
    expected = F.bilinear(input1, input2, weight)

    assert_close(actual, expected)


def test_bilinear_module_matches_torch_module():
    input1 = torch.randn(2, 4, 3)
    input2 = torch.randn(2, 4, 2)
    actual = dnn.Bilinear(3, 2, 5)
    expected = nn.Bilinear(3, 2, 5)
    expected.load_state_dict(actual.state_dict())

    assert actual.in1_features == expected.in1_features
    assert actual.in2_features == expected.in2_features
    assert actual.out_features == expected.out_features
    assert_close(actual(input1, input2), expected(input1, input2))


def test_fast_bilinear_module_matches_torch_module():
    input1 = torch.randn(2, 4, 3)
    input2 = torch.randn(2, 4, 2)
    actual = dnn.Bilinear(3, 2, 5, fast=True)
    expected = nn.Bilinear(3, 2, 5)
    expected.load_state_dict(actual.state_dict())

    assert actual.fast is True
    assert_close(actual(input1, input2), expected(input1, input2))


def test_bilinear_module_supports_no_bias():
    input1 = torch.randn(4, 3)
    input2 = torch.randn(4, 2)
    actual = dnn.Bilinear(3, 2, 5, bias=False)
    expected = nn.Bilinear(3, 2, 5, bias=False)
    expected.load_state_dict(actual.state_dict())

    assert actual.bias is None
    assert_close(actual(input1, input2), expected(input1, input2))


def test_linear_function_matches_torch_with_bias():
    x = torch.randn(4, 3)
    weight = torch.randn(5, 3)
    bias = torch.randn(5)

    actual = dF.linear(x, weight, bias)
    expected = F.linear(x, weight, bias)

    assert_close(actual, expected)


def test_linear_function_matches_torch_without_bias_for_batched_input():
    x = torch.randn(2, 4, 3)
    weight = torch.randn(5, 3)

    actual = dF.linear(x, weight)
    expected = F.linear(x, weight)

    assert_close(actual, expected)


def test_linear_module_matches_torch_module():
    x = torch.randn(2, 4, 3)
    actual = dnn.Linear(3, 5)
    expected = nn.Linear(3, 5)
    expected.load_state_dict(actual.state_dict())

    assert actual.in_features == expected.in_features
    assert actual.out_features == expected.out_features
    assert_close(actual(x), expected(x))


def test_fast_linear_module_matches_torch_module():
    x = torch.randn(2, 4, 3)
    actual = dnn.Linear(3, 5, fast=True)
    expected = nn.Linear(3, 5)
    expected.load_state_dict(actual.state_dict())

    assert actual.fast is True
    assert_close(actual(x), expected(x))


def test_linear_module_supports_no_bias():
    x = torch.randn(2, 3)
    actual = dnn.Linear(3, 5, bias=False)
    expected = nn.Linear(3, 5, bias=False)
    expected.load_state_dict(actual.state_dict())

    assert actual.bias is None
    assert_close(actual(x), expected(x))


def test_linear_module_initializes_bias_with_symmetric_bound():
    module = dnn.Linear(16, 128)
    bound = 1 / 16**0.5

    assert module.bias is not None
    assert module.bias.min() >= -bound
    assert module.bias.max() <= bound


def test_flatten_module_matches_torch_module():
    x = torch.randn(2, 3, 4, 5)
    actual = dnn.Flatten(start_dim=1, end_dim=2)
    expected = nn.Flatten(start_dim=1, end_dim=2)

    assert actual.start_dim == expected.start_dim
    assert actual.end_dim == expected.end_dim
    assert_close(actual(x), expected(x))


def test_flatten_module_preserves_fast_argument():
    module = dnn.Flatten(fast=True)
    x = torch.randn(2, 3, 4)

    assert module.fast is True
    assert_close(module(x), torch.flatten(x, start_dim=1, end_dim=-1))


def test_unflatten_module_matches_torch_module():
    x = torch.randn(2, 12, 5)
    actual = dnn.Unflatten(dim=1, unflattened_size=(3, 4))
    expected = nn.Unflatten(dim=1, unflattened_size=(3, 4))

    assert actual.dim == expected.dim
    assert actual.unflattened_size == expected.unflattened_size
    assert_close(actual(x), expected(x))


def test_unflatten_module_preserves_fast_argument():
    module = dnn.Unflatten(dim=1, unflattened_size=(3, 4), fast=True)
    x = torch.randn(2, 12, 5)

    assert module.fast is True
    assert_close(module(x), torch.unflatten(x, dim=1, sizes=(3, 4)))


def test_identity_module_returns_same_tensor():
    module = dnn.Identity()
    x = torch.randn(2, 3)

    assert module(x) is x

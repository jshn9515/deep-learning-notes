import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


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


def test_identity_module_returns_same_tensor():
    module = dnn.Identity()
    x = torch.randn(2, 3)

    assert module(x) is x

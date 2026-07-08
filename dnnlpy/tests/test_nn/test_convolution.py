from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd as TConv
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
from dnnlpy.nn.convolution import _ConvNd as DConv

type TupleND = tuple[int, ...]
type Size1D = int | tuple[int]
type Size2D = int | tuple[int, int]
type Size3D = int | tuple[int, int, int]
type SizeND = int | TupleND


def _padding_tuple(padding: SizeND, ndim: int) -> TupleND:
    if isinstance(padding, int):
        padding = (padding,) * ndim
    return tuple(pad for value in reversed(padding) for pad in (value, value))


def _as_tuple(value: SizeND, ndim: int) -> TupleND:
    if isinstance(value, int):
        return (value,) * ndim
    return value


@torch.inference_mode()
def _copy_conv_parameters(module: DConv, torch_module: TConv):
    module.weight.copy_(torch_module.weight)
    if module.bias is not None and torch_module.bias is not None:
        module.bias.copy_(torch_module.bias)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((2,), (1,), (1,), 1),
        ((1,), (2,), (2,), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv1d_matches_torch(
    stride: Size1D, padding: Size1D, dilation: Size1D, groups: int
):
    x = torch.randn(2, 4, 11)
    weight = torch.randn(6, 4 // groups, 3)
    bias = torch.randn(6)
    pad = _padding_tuple(padding, ndim=1)

    actual = dF.conv1d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=1),
        padding=pad,
        dilation=_as_tuple(dilation, ndim=1),
        groups=groups,
    )
    expected = F.conv1d(
        F.pad(x, pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((2, 1), (1, 2), 1, 1),
        ((1, 2), (2, 1), (2, 1), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv2d_matches_torch(
    stride: Size2D, padding: Size2D, dilation: Size2D, groups: int
):
    x = torch.randn(2, 4, 8, 9)
    weight = torch.randn(6, 4 // groups, 3, 3)
    bias = torch.randn(6)
    pad = _padding_tuple(padding, ndim=2)

    actual = dF.conv2d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=2),
        padding=pad,
        dilation=_as_tuple(dilation, ndim=2),
        groups=groups,
    )
    expected = F.conv2d(
        F.pad(x, pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((1, 2, 1), (1, 0, 2), 1, 1),
        ((1, 2, 1), (1, 2, 1), (1, 2, 1), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv3d_matches_torch(
    stride: Size3D, padding: Size3D, dilation: Size3D, groups: int
):
    x = torch.randn(2, 4, 5, 7, 8)
    weight = torch.randn(6, 4 // groups, 3, 3, 3)
    bias = torch.randn(6)
    pad = _padding_tuple(padding, ndim=3)

    actual = dF.conv3d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=3),
        padding=pad,
        dilation=_as_tuple(dilation, ndim=3),
        groups=groups,
    )
    expected = F.conv3d(
        F.pad(x, pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_conv2d_supports_no_bias():
    x = torch.randn(2, 3, 6, 7)
    weight = torch.randn(4, 3, 3, 2)
    pad = _padding_tuple((1, 0), ndim=2)

    actual = dF.conv2d(x, weight, padding=pad)
    expected = F.conv2d(F.pad(x, pad), weight)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('conv_name', 'shape', 'weight_shape', 'padding'),
    [
        ('conv1d', (2, 3, 8), (4, 3, 3), (1,)),
        ('conv2d', (2, 3, 6, 7), (4, 3, 3, 2), (1, 2)),
        ('conv3d', (2, 3, 5, 6, 7), (4, 3, 3, 2, 3), (1, 1, 2)),
    ],
)
@pytest.mark.parametrize('padding_mode', ['reflect', 'replicate', 'circular'])
def test_convolution_supports_padding_mode(
    conv_name: str,
    shape: TupleND,
    weight_shape: TupleND,
    padding: TupleND,
    padding_mode: str,
):
    x = torch.randn(*shape)
    weight = torch.randn(*weight_shape)
    bias = torch.randn(weight_shape[0])
    ndim = len(shape) - 2
    pad = _padding_tuple(padding, ndim=ndim)
    conv = getattr(dF, conv_name)
    torch_conv = getattr(F, conv_name)

    actual = conv(
        x,
        weight,
        bias=bias,
        padding=pad,
        padding_mode=padding_mode,
    )
    expected = torch_conv(F.pad(x, pad, mode=padding_mode), weight, bias=bias)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    (
        'dnn_module',
        'torch_module',
        'shape',
        'kwargs',
    ),
    [
        (
            dnn.Conv1d,
            nn.Conv1d,
            (2, 4, 11),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3,),
                'stride': (2,),
                'padding': (1,),
                'dilation': (1,),
            },
        ),
        (
            dnn.Conv2d,
            nn.Conv2d,
            (2, 4, 8, 9),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 2),
                'dilation': (1, 1),
            },
        ),
        (
            dnn.Conv3d,
            nn.Conv3d,
            (2, 4, 5, 7, 8),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3, 3),
                'stride': (1, 2, 1),
                'padding': (1, 0, 2),
                'dilation': (1, 1, 1),
            },
        ),
        (
            dnn.Conv2d,
            nn.Conv2d,
            (2, 4, 8, 9),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (2, 1),
                'dilation': (2, 1),
            },
        ),
        (
            dnn.Conv3d,
            nn.Conv3d,
            (2, 4, 5, 7, 8),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3, 3),
                'stride': (1, 1, 1),
                'padding': (2, 1, 1),
                'dilation': (2, 1, 1),
            },
        ),
    ],
)
def test_conv_module_matches_torch(
    dnn_module: type[DConv],
    torch_module: type[TConv],
    shape: TupleND,
    kwargs: dict[str, Any],
):
    x = torch.randn(*shape)
    actual_module = dnn_module(**kwargs)
    expected_module = torch_module(**kwargs)
    _copy_conv_parameters(actual_module, expected_module)

    actual = actual_module(x)
    expected = expected_module(x)

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ('dnn_module', 'torch_module', 'shape', 'padding'),
    [
        (dnn.Conv1d, nn.Conv1d, (2, 3, 8), (1,)),
        (dnn.Conv2d, nn.Conv2d, (2, 3, 6, 7), (1, 2)),
        (dnn.Conv3d, nn.Conv3d, (2, 3, 5, 6, 7), (1, 1, 2)),
    ],
)
@pytest.mark.parametrize('padding_mode', ['reflect', 'replicate', 'circular'])
def test_conv_module_supports_padding_mode(
    dnn_module: type[DConv],
    torch_module: type[TConv],
    shape: TupleND,
    padding: TupleND,
    padding_mode: str,
):
    x = torch.randn(*shape)
    kwargs = {
        'in_channels': 3,
        'out_channels': 4,
        'kernel_size': (3,) * (len(shape) - 2),
        'padding': padding,
        'padding_mode': padding_mode,
    }
    actual_module = dnn_module(**kwargs)
    expected_module = torch_module(**kwargs)
    _copy_conv_parameters(actual_module, expected_module)

    actual = actual_module(x)
    expected = expected_module(x)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('dnn_module', 'functional_name', 'shape'),
    [
        (dnn.Conv1d, 'conv1d', (2, 3, 8)),
        (dnn.Conv2d, 'conv2d', (2, 3, 6, 7)),
        (dnn.Conv3d, 'conv3d', (2, 3, 5, 6, 7)),
    ],
)
def test_conv_module_delegates_to_dnnlpy_functional(
    monkeypatch,
    dnn_module: type[nn.Module],
    functional_name: str,
    shape: TupleND,
):
    calls = []
    original = getattr(dF, functional_name)

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(dF, functional_name, wrapped)
    module = dnn_module(
        in_channels=3,
        out_channels=4,
        kernel_size=(3,) * (len(shape) - 2),
        padding=1,
    )

    module(torch.randn(*shape))

    assert len(calls) == 1

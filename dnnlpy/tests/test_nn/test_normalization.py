# pyright: reportOptionalMemberAccess=false

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


@pytest.mark.parametrize(
    'shape',
    [
        (8, 4),
        (8, 4, 5),
        (8, 4, 5, 6),
        (8, 4, 3, 5, 6),
    ],
)
def test_batch_norm_function_matches_torch_training(shape: tuple[int, ...]):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)
    actual_running_mean = torch.zeros(4)
    actual_running_var = torch.ones(4)
    expected_running_mean = actual_running_mean.clone()
    expected_running_var = actual_running_var.clone()

    actual = dF.batch_norm(
        x,
        actual_running_mean,
        actual_running_var,
        weight=weight,
        bias=bias,
        use_batch_stats=True,
        momentum=0.1,
    )
    expected = F.batch_norm(
        x,
        expected_running_mean,
        expected_running_var,
        weight=weight,
        bias=bias,
        training=True,
        momentum=0.1,
    )

    assert_close(actual, expected)
    assert_close(actual_running_mean, expected_running_mean)
    assert_close(actual_running_var, expected_running_var)


def test_batch_norm_function_matches_torch_eval_with_running_stats():
    x = torch.randn(8, 4, 5)
    weight = torch.randn(4)
    bias = torch.randn(4)
    running_mean = torch.randn(4)
    running_var = torch.rand(4) + 0.5

    actual = dF.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_batch_stats=False,
    )
    expected = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
    )

    assert_close(actual, expected)


def test_batch_norm_function_uses_batch_stats_when_running_stats_are_none():
    x = torch.randn(8, 4)
    weight = torch.randn(4)
    bias = torch.randn(4)

    actual = dF.batch_norm(
        x,
        None,
        None,
        weight=weight,
        bias=bias,
        use_batch_stats=True,
    )
    expected = F.batch_norm(
        x,
        None,
        None,
        weight=weight,
        bias=bias,
        training=True,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_cls', 'expected_cls', 'shape', 'training', 'track_running_stats'),
    [
        (actual_cls, expected_cls, shape, training, track_running_stats)
        for actual_cls, expected_cls, shape in [
            (dnn.BatchNorm1d, nn.BatchNorm1d, (8, 4)),
            (dnn.BatchNorm1d, nn.BatchNorm1d, (8, 4, 5)),
            (dnn.BatchNorm2d, nn.BatchNorm2d, (8, 4, 5, 6)),
            (dnn.BatchNorm3d, nn.BatchNorm3d, (8, 4, 3, 5, 6)),
        ]
        for training in [True, False]
        for track_running_stats in [True, False]
    ],
)
def test_batch_norm_modules_match_torch_for_mode_and_tracking(
    actual_cls: type[dnn.BatchNorm1d],
    expected_cls: type[nn.BatchNorm1d],
    shape: tuple[int, ...],
    training: bool,
    track_running_stats: bool,
):
    x = torch.randn(shape)
    actual = actual_cls(4, track_running_stats=track_running_stats)
    expected = expected_cls(4, track_running_stats=track_running_stats)
    expected.load_state_dict(actual.state_dict())

    if not training:
        actual.eval()
        expected.eval()
        if track_running_stats:
            actual.running_mean.copy_(torch.randn(4))
            actual.running_var.copy_(torch.rand(4) + 0.5)
            expected.load_state_dict(actual.state_dict())

    actual_out = actual(x)
    expected_out = expected(x)

    assert_close(actual_out, expected_out)
    if track_running_stats:
        assert_close(actual.running_mean, expected.running_mean)
        assert_close(actual.running_var, expected.running_var)
        assert_close(actual.num_batches_tracked, expected.num_batches_tracked)
    else:
        assert actual.running_mean is None
        assert actual.running_var is None
        assert actual.num_batches_tracked is None


def test_batch_norm_module_supports_no_affine():
    actual = dnn.BatchNorm2d(4, affine=False)
    expected = nn.BatchNorm2d(4, affine=False)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(8, 4, 5, 6)

    assert actual.weight is None
    assert actual.bias is None
    assert_close(actual(x), expected(x))


def test_batch_norm_module_supports_tracking_disabled_in_eval_mode():
    actual = dnn.BatchNorm1d(4, track_running_stats=False)
    expected = nn.BatchNorm1d(4, track_running_stats=False)
    x = torch.randn(8, 4)
    shifted_x = torch.randn(8, 4) + 10
    actual.eval()
    expected.eval()

    assert actual.running_mean is None
    assert actual.running_var is None
    assert actual.num_batches_tracked is None
    assert_close(actual(x), expected(x))
    assert_close(actual(shifted_x), expected(shifted_x))


def test_batch_norm_module_supports_cumulative_moving_average():
    actual = dnn.BatchNorm1d(4, momentum=None)
    expected = nn.BatchNorm1d(4, momentum=None)
    expected.load_state_dict(actual.state_dict())

    for _ in range(3):
        x = torch.randn(8, 4)
        assert_close(actual(x), expected(x))

    assert_close(actual.running_mean, expected.running_mean)
    assert_close(actual.running_var, expected.running_var)


def test_batch_norm_reset_parameters_restores_defaults():
    module = dnn.BatchNorm1d(4)
    module.weight.data.normal_()
    module.bias.data.normal_()
    module.running_mean.normal_()
    module.running_var.normal_()
    module.num_batches_tracked.add_(5)

    module.reset_parameters()

    assert_close(module.weight, torch.ones(4))
    assert_close(module.bias, torch.zeros(4))
    assert_close(module.running_mean, torch.zeros(4))
    assert_close(module.running_var, torch.ones(4))
    assert_close(module.num_batches_tracked, torch.tensor(0))


def test_batch_norm_module_supports_no_bias_extension():
    module = dnn.BatchNorm1d(4, bias=False)
    x = torch.randn(8, 4)

    assert module.bias is None
    assert module(x).shape == x.shape


@pytest.mark.parametrize(
    ('module', 'bad_shape'),
    [
        (dnn.BatchNorm1d(4), (8, 4, 5, 6)),
        (dnn.BatchNorm2d(4), (8, 4, 5)),
        (dnn.BatchNorm3d(4), (8, 4, 5, 6)),
    ],
)
def test_batch_norm_modules_reject_invalid_rank(
    module: dnn.BatchNorm1d,
    bad_shape: tuple[int, ...],
):
    with pytest.raises(AssertionError):
        module(torch.randn(bad_shape))


def test_batch_norm_module_rejects_wrong_channel_count():
    module = dnn.BatchNorm2d(4)

    with pytest.raises(AssertionError):
        module(torch.randn(8, 3, 5, 6))


def test_batch_norm_function_rejects_single_value_per_channel_when_training():
    with pytest.raises(ValueError):
        dF.batch_norm(
            torch.randn(1, 4),
            torch.zeros(4),
            torch.ones(4),
            use_batch_stats=True,
        )


@pytest.mark.parametrize(
    ('shape', 'expected_fn'),
    [
        ((8, 4, 5), F.instance_norm),
        ((8, 4, 5, 6), F.instance_norm),
        ((8, 4, 3, 5, 6), F.instance_norm),
    ],
)
def test_instance_norm_function_matches_torch_training(
    shape: tuple[int, ...],
    expected_fn,
):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)
    actual_running_mean = torch.zeros(4)
    actual_running_var = torch.ones(4)
    expected_running_mean = actual_running_mean.clone()
    expected_running_var = actual_running_var.clone()

    actual = dF.instance_norm(
        x,
        actual_running_mean,
        actual_running_var,
        weight=weight,
        bias=bias,
        use_instance_stats=True,
        momentum=0.1,
    )
    expected = expected_fn(
        x,
        expected_running_mean,
        expected_running_var,
        weight=weight,
        bias=bias,
        use_input_stats=True,
        momentum=0.1,
    )

    assert_close(actual, expected)
    assert_close(actual_running_mean, expected_running_mean)
    assert_close(actual_running_var, expected_running_var)


def test_instance_norm_function_matches_torch_eval_with_running_stats():
    x = torch.randn(8, 4, 5, 6)
    weight = torch.randn(4)
    bias = torch.randn(4)
    running_mean = torch.randn(4)
    running_var = torch.rand(4) + 0.5

    actual = dF.instance_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_instance_stats=False,
    )
    expected = F.instance_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_input_stats=False,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('actual_cls', 'expected_cls', 'shape', 'training', 'track_running_stats'),
    [
        (actual_cls, expected_cls, shape, training, track_running_stats)
        for actual_cls, expected_cls, shape in [
            (dnn.InstanceNorm1d, nn.InstanceNorm1d, (8, 4, 5)),
            (dnn.InstanceNorm2d, nn.InstanceNorm2d, (8, 4, 5, 6)),
            (dnn.InstanceNorm3d, nn.InstanceNorm3d, (8, 4, 3, 5, 6)),
        ]
        for training in [True, False]
        for track_running_stats in [True, False]
    ],
)
def test_instance_norm_modules_match_torch_for_mode_and_tracking(
    actual_cls: type[dnn.InstanceNorm1d],
    expected_cls: type[nn.InstanceNorm1d],
    shape: tuple[int, ...],
    training: bool,
    track_running_stats: bool,
):
    x = torch.randn(shape)
    actual = actual_cls(
        4,
        affine=True,
        track_running_stats=track_running_stats,
    )
    expected = expected_cls(
        4,
        affine=True,
        track_running_stats=track_running_stats,
    )
    expected.load_state_dict(actual.state_dict())

    if not training:
        actual.eval()
        expected.eval()
        if track_running_stats:
            actual.running_mean.copy_(torch.randn(4))
            actual.running_var.copy_(torch.rand(4) + 0.5)
            expected.load_state_dict(actual.state_dict())

    actual_out = actual(x)
    expected_out = expected(x)

    assert_close(actual_out, expected_out)
    if track_running_stats:
        assert_close(actual.running_mean, expected.running_mean)
        assert_close(actual.running_var, expected.running_var)
    else:
        assert actual.running_mean is None
        assert actual.running_var is None


def test_instance_norm_module_supports_no_affine():
    actual = dnn.InstanceNorm2d(4, affine=False)
    expected = nn.InstanceNorm2d(4, affine=False)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(8, 4, 5, 6)

    assert actual.weight is None
    assert actual.bias is None
    assert_close(actual(x), expected(x))


def test_instance_norm_reset_parameters_restores_defaults():
    module = dnn.InstanceNorm1d(4, affine=True, track_running_stats=True)
    module.weight.data.normal_()
    module.bias.data.normal_()
    module.running_mean.normal_()
    module.running_var.normal_()

    module.reset_parameters()

    assert_close(module.weight, torch.ones(4))
    assert_close(module.bias, torch.zeros(4))
    assert_close(module.running_mean, torch.zeros(4))
    assert_close(module.running_var, torch.ones(4))


def test_instance_norm_module_supports_no_bias_extension():
    module = dnn.InstanceNorm1d(4, affine=True, bias=False)
    x = torch.randn(8, 4, 5)

    assert module.bias is None
    assert module(x).shape == x.shape


@pytest.mark.parametrize(
    ('module', 'bad_shape'),
    [
        (dnn.InstanceNorm1d(4), (8, 4)),
        (dnn.InstanceNorm2d(4), (8, 4, 5)),
        (dnn.InstanceNorm3d(4), (8, 4, 5, 6)),
    ],
)
def test_instance_norm_modules_reject_invalid_rank(
    module: dnn.InstanceNorm1d,
    bad_shape: tuple[int, ...],
):
    with pytest.raises(AssertionError):
        module(torch.randn(bad_shape))


def test_instance_norm_module_rejects_wrong_channel_count():
    module = dnn.InstanceNorm2d(4)

    with pytest.raises(AssertionError):
        module(torch.randn(8, 3, 5, 6))


def test_instance_norm_function_rejects_single_spatial_value_when_training():
    with pytest.raises(ValueError):
        dF.instance_norm(
            torch.randn(8, 4, 1),
            torch.zeros(4),
            torch.ones(4),
            use_instance_stats=True,
        )


def test_layer_norm_function_matches_torch():
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3, 4)

    actual = dF.layer_norm(x, (3, 4), weight=weight, bias=bias)
    expected = F.layer_norm(x, (3, 4), weight=weight, bias=bias)

    assert_close(actual, expected)


@pytest.mark.parametrize('normalized_shape', [4, (3, 4)])
def test_layer_norm_module_matches_torch(normalized_shape: int | tuple[int, ...]):
    shape = (2, 3, 4) if isinstance(normalized_shape, tuple) else (2, 3, 4)
    actual = dnn.LayerNorm(normalized_shape)
    expected = nn.LayerNorm(normalized_shape)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(shape)

    assert_close(actual(x), expected(x))


def test_layer_norm_module_supports_no_bias_extension():
    module = dnn.LayerNorm(4, bias=False)
    x = torch.randn(2, 3, 4)

    assert module.bias is None
    assert module(x).shape == x.shape


def test_layer_norm_module_supports_no_affine():
    actual = dnn.LayerNorm((3, 4), elementwise_affine=False)
    expected = nn.LayerNorm((3, 4), elementwise_affine=False)
    x = torch.randn(2, 3, 4)

    assert actual.weight is None
    assert actual.bias is None
    assert_close(actual(x), expected(x))


def test_layer_norm_function_rejects_wrong_normalized_shape():
    with pytest.raises(AssertionError):
        dF.layer_norm(torch.randn(2, 3, 4), (2, 4))


def test_rms_norm_function_matches_torch():
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)

    actual = dF.rms_norm(x, (3, 4), weight=weight)
    expected = F.rms_norm(x, (3, 4), weight=weight)

    assert_close(actual, expected)


@pytest.mark.parametrize('normalized_shape', [4, (3, 4)])
def test_rms_norm_module_matches_torch(normalized_shape: int | tuple[int, ...]):
    shape = (2, 3, 4) if isinstance(normalized_shape, tuple) else (2, 3, 4)
    actual = dnn.RMSNorm(normalized_shape)
    expected = nn.RMSNorm(normalized_shape)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(shape)

    assert_close(actual(x), expected(x))


def test_rms_norm_module_supports_no_affine():
    actual = dnn.RMSNorm((3, 4), elementwise_affine=False)
    expected = nn.RMSNorm((3, 4), elementwise_affine=False)
    x = torch.randn(2, 3, 4)

    assert actual.weight is None
    assert_close(actual(x), expected(x))


def test_rms_norm_reset_parameters_restores_defaults():
    module = dnn.RMSNorm((3, 4))
    module.weight.data.normal_()

    module.reset_parameters()

    assert_close(module.weight, torch.ones(3, 4))


def test_rms_norm_function_rejects_wrong_normalized_shape():
    with pytest.raises(AssertionError):
        dF.rms_norm(torch.randn(2, 3, 4), (2, 4))


@pytest.mark.parametrize('shape', [(2, 4), (2, 4, 5), (2, 4, 5, 6)])
def test_group_norm_function_matches_torch(shape: tuple[int, ...]):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)

    actual = dF.group_norm(x, 2, weight=weight, bias=bias)
    expected = F.group_norm(x, 2, weight=weight, bias=bias)

    assert_close(actual, expected)


def test_group_norm_module_matches_torch():
    actual = dnn.GroupNorm(2, 4)
    expected = nn.GroupNorm(2, 4)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(2, 4, 5, 6)

    assert_close(actual(x), expected(x))


def test_group_norm_module_supports_no_affine():
    actual = dnn.GroupNorm(2, 4, affine=False)
    expected = nn.GroupNorm(2, 4, affine=False)
    x = torch.randn(2, 4, 5, 6)

    assert actual.weight is None
    assert actual.bias is None
    assert_close(actual(x), expected(x))


def test_group_norm_rejects_invalid_group_count():
    with pytest.raises(AssertionError):
        dF.group_norm(torch.randn(2, 4, 5), 0)


def test_group_norm_module_rejects_wrong_channel_count():
    module = dnn.GroupNorm(2, 4)

    with pytest.raises(AssertionError):
        module(torch.randn(2, 3, 5))


@pytest.mark.parametrize('shape', [(2, 4, 5), (2, 4, 5, 6)])
def test_local_response_norm_function_matches_torch(shape: tuple[int, ...]):
    x = torch.randn(shape)

    actual = dF.local_response_norm(x, 3, alpha=1e-3, beta=0.5, k=2.0)
    expected = F.local_response_norm(x, 3, alpha=1e-3, beta=0.5, k=2.0)

    assert_close(actual, expected)


def test_local_response_norm_module_matches_torch():
    actual = dnn.LocalResponseNorm(3, alpha=1e-3, beta=0.5, k=2.0)
    expected = nn.LocalResponseNorm(3, alpha=1e-3, beta=0.5, k=2.0)
    x = torch.randn(2, 4, 5, 6)

    assert_close(actual(x), expected(x))

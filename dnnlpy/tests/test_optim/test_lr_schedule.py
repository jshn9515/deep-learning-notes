import inspect

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.testing import assert_close

from dnnlpy.optim.lr_schedule import CosineAnnealingLR


def _make_optimizer() -> optim.SGD:
    parameters = [nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    return optim.SGD(
        [
            {'params': [parameters[0]], 'lr': 0.5},
            {'params': [parameters[1]], 'lr': 0.1},
        ]
    )


def test_cosine_annealing_lr_api_and_exports():
    assert issubclass(CosineAnnealingLR, lr.LRScheduler)

    actual = inspect.signature(CosineAnnealingLR)
    expected = inspect.signature(lr.CosineAnnealingLR)
    assert list(actual.parameters) == list(expected.parameters)

    for name in actual.parameters:
        assert actual.parameters[name].default == expected.parameters[name].default


def test_cosine_annealing_lr_matches_pytorch_across_cycles():
    actual_optimizer = _make_optimizer()
    expected_optimizer = _make_optimizer()
    actual_scheduler = CosineAnnealingLR(actual_optimizer, T_max=4, eta_min=0.01)
    expected_scheduler = lr.CosineAnnealingLR(expected_optimizer, T_max=4, eta_min=0.01)

    actual = torch.tensor(actual_scheduler.get_last_lr())
    expected = torch.tensor(expected_scheduler.get_last_lr())

    assert_close(actual, expected)

    for _ in range(12):
        actual_optimizer.step()
        expected_optimizer.step()
        actual_scheduler.step()
        expected_scheduler.step()

        actual = torch.tensor(actual_scheduler.get_last_lr())
        expected = torch.tensor(expected_scheduler.get_last_lr())
        assert_close(actual, expected)

        assert actual_scheduler.state_dict() == expected_scheduler.state_dict()


def test_cosine_annealing_lr_matches_pytorch_when_resuming():
    actual_optimizer = _make_optimizer()
    expected_optimizer = _make_optimizer()

    for actual_group, expected_group in zip(
        actual_optimizer.param_groups,
        expected_optimizer.param_groups,
        strict=True,
    ):
        actual_group['initial_lr'] = actual_group['lr']
        expected_group['initial_lr'] = expected_group['lr']

    options = {'T_max': 8, 'eta_min': 0.02, 'last_epoch': 3}
    actual_scheduler = CosineAnnealingLR(actual_optimizer, **options)
    expected_scheduler = lr.CosineAnnealingLR(expected_optimizer, **options)

    actual = torch.tensor(actual_scheduler.get_last_lr())
    expected = torch.tensor(expected_scheduler.get_last_lr())
    assert_close(actual, expected)

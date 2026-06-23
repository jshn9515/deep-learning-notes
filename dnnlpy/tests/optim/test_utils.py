import torch

import dnnlpy.optim as dopt
import dnnlpy.optim.utils as utils


def quadratic_loss(theta: torch.Tensor) -> torch.Tensor:
    return theta.square().sum() / 2


def test_run_optimizer_public_export():
    assert dopt.run_optimizer is utils.run_optimizer


def test_run_optimizer_records_parameter_history():
    params = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SimpleSGD([params], lr=0.1)

    history = dopt.run_optimizer(optimizer, quadratic_loss, steps=2)

    assert history.shape == (3, 2)
    assert torch.allclose(history[0], torch.tensor([1.0, -2.0]))
    assert torch.allclose(history[1], torch.tensor([0.9, -1.8]))
    assert torch.allclose(history[2], torch.tensor([0.81, -1.62]))
    assert torch.allclose(params, torch.tensor([0.81, -1.62]))


def test_optimizer_repr_shows_hyperparameters_without_state():
    params = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SGD([params], lr=0.1, momentum=0.9, weight_decay=0.01)

    assert optimizer.defaults == {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.01,
        'nesterov': False,
    }
    assert repr(optimizer) == (
        'SGD(lr=0.1, momentum=0.9, weight_decay=0.01, nesterov=False)'
    )


def test_optimizer_repr_uses_defaults_for_tuple_hyperparameters():
    params = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.Adam([params], lr=0.01, betas=(0.8, 0.95), eps=1e-7)

    assert optimizer.defaults == {
        'lr': 0.01,
        'betas': (0.8, 0.95),
        'eps': 1e-7,
        'weight_decay': 0.0,
    }
    assert repr(optimizer) == (
        'Adam(lr=0.01, betas=(0.8, 0.95), eps=1e-07, weight_decay=0.0)'
    )

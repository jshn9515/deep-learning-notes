from typing import Any

import pytest
import torch

import dnnl.optim as dopt


@pytest.mark.parametrize(
    ('optimizer_cls', 'kwargs'),
    [
        (dopt.Adagrad, {'lr': 0.1, 'eps': 0.0}),
        (dopt.RMSprop, {'lr': 0.1, 'rho': 0.9, 'eps': 1e-8}),
        (dopt.Adadelta, {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6}),
        (dopt.Adam, {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8}),
    ],
)
def test_adaptive_optimizers_apply_weight_decay_as_gradient_term(
    optimizer_cls: type[dopt.Optimizer],
    kwargs: Any,
):
    actual_param = torch.tensor([1.0, -2.0], requires_grad=True)
    expected_param = actual_param.detach().clone().requires_grad_()
    weight_decay = 0.2

    actual_optimizer = optimizer_cls(
        [actual_param],
        weight_decay=weight_decay,
        **kwargs,
    )
    expected_optimizer = optimizer_cls([expected_param], **kwargs)

    for grad in [torch.tensor([0.5, -0.25]), torch.tensor([0.25, 0.5])]:
        actual_param.grad = grad.clone()
        expected_param.grad = grad + weight_decay * expected_param.detach()

        actual_optimizer.step()
        expected_optimizer.step()

    assert torch.allclose(actual_param, expected_param)


@pytest.mark.parametrize(
    'optimizer_cls',
    [dopt.Adagrad, dopt.RMSprop, dopt.Adadelta, dopt.Adam],
)
def test_adaptive_optimizers_leave_parameters_without_gradients_unchanged(
    optimizer_cls: type[dopt.Optimizer],
):
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = optimizer_cls([trained, skipped], weight_decay=0.2)

    optimizer.step()

    assert torch.allclose(skipped, torch.tensor([2.0]))

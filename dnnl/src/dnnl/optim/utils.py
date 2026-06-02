from collections.abc import Callable

from torch import Tensor

from .adagrad import Adagrad
from .sgd import SimpleSGD, SimpleSGDWithMomentum, SimpleSGDWithNesterovMomentum

type Loss = Callable[[Tensor], Tensor]

__all__ = [
    'run_sgd',
    'run_sgd_with_momentum',
    'run_sgd_with_nesterov_momentum',
    'run_adagrad',
]


def run_sgd(loss_fn: Loss, params: Tensor, lr: float, steps: int) -> list[Tensor]:
    """Run simple SGD on a cloned parameter tensor and record its trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        steps (int): Number of optimization steps to run.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGD([theta], lr=lr)
    history = [theta.clone().detach()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        history.append(theta.detach().clone())

    return history


def run_sgd_with_momentum(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    momentum: float,
    steps: int,
) -> list[Tensor]:
    """Run SGD with momentum and record the parameter trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        momentum (float): Momentum coefficient.
        steps (int): Number of optimization steps to run.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithMomentum([theta], lr=lr, momentum=momentum)
    history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        history.append(theta.detach().clone())

    return history


def run_sgd_with_nesterov_momentum(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    momentum: float,
    steps: int,
) -> list[Tensor]:
    """Run SGD with Nesterov momentum and record the parameter trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        momentum (float): Momentum coefficient.
        steps (int): Number of optimization steps to run.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithNesterovMomentum([theta], lr=lr, momentum=momentum)
    history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        history.append(theta.detach().clone())

    return history


def run_adagrad(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    eps: float = 1e-10,
) -> tuple[list[Tensor], list[list[Tensor]]]:
    """Run Adagrad and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        eps (float, default: 1e-10): Small value added to Adagrad denominators.

    Returns:
        A tuple containing parameter snapshots and per-step effective learning
        rates for each optimized parameter.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adagrad([theta], lr=lr, eps=eps)

    theta_history = [theta.detach().clone()]
    lr_history = []

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())
        lr_history.append(optimizer.get_effective_lr())

    return theta_history, lr_history

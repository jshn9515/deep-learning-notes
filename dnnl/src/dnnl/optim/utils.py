from collections.abc import Callable

from torch import Tensor

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .rmsprop import RMSprop
from .sgd import (
    SimpleSGD,
    SimpleSGDWithMomentum,
    SimpleSGDWithNesterovMomentum,
    SimpleSGDWithWeightDecay,
)

type Loss = Callable[[Tensor], Tensor]

__all__ = [
    'run_sgd',
    'run_sgd_with_weight_decay',
    'run_sgd_with_momentum',
    'run_sgd_with_nesterov_momentum',
    'run_adagrad',
    'run_rmsprop',
    'run_adadelta',
    'run_adam',
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


def run_sgd_with_weight_decay(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    weight_decay: float,
    steps: int,
) -> list[Tensor]:
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithWeightDecay([theta], lr=lr, weight_decay=weight_decay)
    history = [theta.clone().detach()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        history.append(theta.clone().detach())

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
        lr_history.append(optimizer.get_effective_lr()[0])

    return theta_history, lr_history


def run_rmsprop(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    eps: float = 1e-10,
) -> tuple[list[Tensor], list[list[Tensor]]]:
    """Run RMSProp and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        eps (float, default: 1e-10): Small value added to RMSProp denominators.

    Returns:
        A tuple containing parameter snapshots and per-step effective learning
        rates for each optimized parameter.
    """
    theta = params.clone().requires_grad_()
    optimizer = RMSprop([theta], lr=lr, eps=eps)

    theta_history = [theta.detach().clone()]
    lr_history = []

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())
        lr_history.append(optimizer.get_effective_lr()[0])

    return theta_history, lr_history


def run_adadelta(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    eps: float = 1e-10,
) -> list[Tensor]:
    """Run Adadelta and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        eps (float, default: 1e-10): Small value added to Adadelta denominators.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adadelta([theta], lr=lr, eps=eps)
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return theta_history


def run_adam(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-10,
) -> list[Tensor]:
    """Run Adam and record parameter history.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        betas (tuple[float, float], default: (0.9, 0.999)): Coefficients for
            computing running averages of gradient and its square.
        eps (float, default: 1e-10): Small value added to Adam denominators.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adam([theta], lr=lr, betas=betas, eps=eps)
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return theta_history

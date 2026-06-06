from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamw import AdamW
from .muon import Muon
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
    'run_adamw',
    'run_muon',
    'collect_lr_schedule',
    'plot_lr_schedule',
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
    steps: int,
    **kwargs: Any,
) -> list[Tensor]:
    """Run SGD with weight decay and record the parameter trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the optimizer constructor,
            such as ``weight_decay``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithWeightDecay([theta], lr=lr, **kwargs)
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
    steps: int,
    **kwargs: Any,
) -> list[Tensor]:
    """Run SGD with momentum and record the parameter trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the optimizer constructor,
            such as ``momentum``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithMomentum([theta], lr=lr, **kwargs)
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
    steps: int,
    **kwargs: Any,
) -> list[Tensor]:
    """Run SGD with Nesterov momentum and record the parameter trajectory.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the optimizer constructor,
            such as ``momentum``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = SimpleSGDWithNesterovMomentum([theta], lr=lr, **kwargs)
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
    **kwargs: Any,
) -> tuple[list[Tensor], list[list[Tensor]]]:
    """Run Adagrad and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the Adagrad optimizer
            constructor, such as ``eps``.

    Returns:
        A tuple containing parameter snapshots and per-step effective learning
        rates for each optimized parameter.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adagrad([theta], lr=lr, **kwargs)

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
    **kwargs: Any,
) -> tuple[list[Tensor], list[list[Tensor]]]:
    """Run RMSProp and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the RMSprop optimizer
            constructor, such as ``momentum`` and ``eps``.

    Returns:
        A tuple containing parameter snapshots and per-step effective learning
        rates for each optimized parameter.
    """
    theta = params.clone().requires_grad_()
    optimizer = RMSprop([theta], lr=lr, **kwargs)

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
    **kwargs: Any,
) -> list[Tensor]:
    """Run Adadelta and record parameter and effective-learning-rate histories.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the Adadelta optimizer
            constructor, such as ``rho`` and ``eps``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adadelta([theta], lr=lr, **kwargs)
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
    **kwargs: Any,
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
        **kwargs: Additional keyword arguments passed to the Adam optimizer
            constructor, such as ``eps`` and ``weight_decay``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = Adam([theta], lr=lr, betas=betas, **kwargs)
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return theta_history


def run_adamw(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    betas: tuple[float, float] = (0.9, 0.999),
    **kwargs: Any,
) -> list[Tensor]:
    """Run AdamW and record parameter history.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial parameter value. The tensor is cloned before
            optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        betas (tuple[float, float], default: (0.9, 0.999)): Coefficients for
            computing running averages of gradient and its square.
        **kwargs: Additional keyword arguments passed to the AdamW optimizer
            constructor, such as ``eps`` and ``weight_decay``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = AdamW([theta], lr=lr, betas=betas, **kwargs)
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return theta_history


def run_muon(
    loss_fn: Loss,
    params: Tensor,
    lr: float,
    steps: int,
    **kwargs: Any,
) -> list[Tensor]:
    """Run Muon and record parameter history.

    Args:
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        params (Tensor): Initial two-dimensional parameter value. The tensor is
            cloned before optimization.
        lr (float): Base learning rate.
        steps (int): Number of optimization steps to run.
        **kwargs: Additional keyword arguments passed to the Muon optimizer
            constructor, such as ``momentum``, ``ns_coefficients``, and
            ``ns_steps``.

    Returns:
        Parameter snapshots before the first step and after each update.
    """
    theta = params.clone().requires_grad_()
    optimizer = Muon([theta], lr=lr, **kwargs)
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return theta_history


def collect_lr_schedule(
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    num_steps: int = 100,
    metric_values: list[float] | None = None,
) -> list[float]:
    """Collect the learning-rate values produced by a scheduler.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer controlled by
            ``scheduler``.
        scheduler (lr_scheduler.LRScheduler): Learning-rate scheduler to step.
        num_steps (int, default: 100): Number of scheduler steps to collect.
        metric_values (list[float] | None, default: None): Metric values passed
            to ``ReduceLROnPlateau`` schedulers.

    Returns:
        Learning-rate values observed before each scheduler step.
    """
    lr_history = [scheduler.get_last_lr()[0]]

    for step in range(num_steps):
        lr_history.append(scheduler.get_last_lr()[0])

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            if metric_values is None:
                raise AssertionError(
                    '`metric_values` must be provided for ReduceLROnPlateau scheduler.'
                )
            metric = metric_values[step]
            scheduler.step(metric)
        else:
            optimizer.step()
            scheduler.step()

    lr_history = torch.tensor(lr_history)
    return lr_history.tolist()


def plot_lr_schedule(
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    num_steps: int = 100,
    metric_values: list[float] | None = None,
    xlabel: str = 'Epoch',
) -> None:
    """Plot the learning-rate values produced by a scheduler.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer controlled by
            ``scheduler``.
        scheduler (lr_scheduler.LRScheduler): Learning-rate scheduler to plot.
        num_steps (int, default: 100): Number of scheduler steps to collect.
        metric_values (list[float] | None, default: None): Metric values passed
            to ``ReduceLROnPlateau`` schedulers.
        xlabel (str, default: 'Epoch'): Label for the horizontal axis.
    """
    name = type(scheduler).__name__
    history = collect_lr_schedule(optimizer, scheduler, num_steps, metric_values)

    fig = plt.figure(name, figsize=(5, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history)
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel.capitalize())
    ax.set_ylabel('Learning Rate')
    ax.legend([name])
    ax.set_title(f'{name} Learning Rate Schedule')
    plt.show()

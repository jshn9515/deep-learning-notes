from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor

from .base import Optimizer

type Loss = Callable[[Tensor], Tensor]

__all__ = [
    'run_optimizer',
    'collect_lr_schedule',
    'plot_lr_schedule',
]


def run_optimizer(
    optimizer: Optimizer,
    loss_fn: Loss,
    steps: int,
) -> Tensor:
    """Run an optimizer on a cloned parameter tensor and record its trajectory.

    Args:
        optimizer (Optimizer): Optimizer to run. Its ``params`` attribute must
            contain a single tensor to optimize.
        loss_fn (Loss): Function that maps the optimized tensor to a scalar loss.
        steps (int): Number of optimization steps to run.

    Returns:
        Parameter snapshots before the first step and after each update. The
        returned tensor has shape ``(steps + 1, *param.shape)``.
    """
    theta = optimizer.params[0]
    theta_history = [theta.detach().clone()]

    for _ in range(steps):
        loss = loss_fn(theta)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        theta_history.append(theta.detach().clone())

    return torch.stack(theta_history)


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
    name = scheduler.__class__.__name__
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

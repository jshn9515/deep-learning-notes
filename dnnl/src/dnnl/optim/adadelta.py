from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['Adadelta']


class Adadelta(Optimizer):
    """Adadelta optimizer with running averages of gradients and updates."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        """Create an Adadelta optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1.0): Learning rate used to scale each update.
            rho (float, default: 0.9): Decay factor for the moving averages.
            eps (float, default: 1e-6): Small value added to root-mean-square
                terms for numerical stability.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
        """
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avgs = [torch.zeros_like(p) for p in self.params]
        self.accumulate_updates = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using the current Adadelta state."""
        for p, square_avg, accumulate_update in zip(
            self.params,
            self.square_avgs,
            self.accumulate_updates,
            strict=True,
        ):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            square_avg.mul_(self.rho).add_(
                p.grad.square(),
                alpha=1 - self.rho,
            )

            rms_update = (accumulate_update + self.eps).sqrt()
            rms_grad = (square_avg + self.eps).sqrt()
            update = -self.lr * rms_update / rms_grad * p.grad

            p.add_(update)

            accumulate_update.mul_(self.rho).add_(
                update.square(),
                alpha=1 - self.rho,
            )

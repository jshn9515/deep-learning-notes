from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['RMSprop']


class RMSprop(Optimizer):
    """RMSprop optimizer with an exponential average of squared gradients."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        rho: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Create an RMSprop optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-2): Base learning rate.
            rho (float, default: 0.99): Decay factor for the squared-gradient
                moving average.
            eps (float, default: 1e-8): Small value added to the denominator
                for numerical stability.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
        """
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avgs = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using the current RMSprop state."""
        for p, square_avg in zip(self.params, self.square_avgs, strict=True):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            square_avg.mul_(self.rho).add_(
                p.grad.square(),
                alpha=1 - self.rho,
            )

            effective_lr = self.lr / (square_avg.sqrt() + self.eps)
            p.add_(-effective_lr * p.grad)

    @torch.no_grad()
    def get_effective_lr(self) -> list[Tensor]:
        """Return the current per-parameter effective learning rates."""
        return [
            self.lr / (square_avg.sqrt() + self.eps).clone()
            for square_avg in self.square_avgs
        ]

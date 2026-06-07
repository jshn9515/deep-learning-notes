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
        momentum: float = 0.0,
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
            momentum (float, default: 0.0): Momentum coefficient.
        """
        super().__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.ema_of_sq_grads = [torch.zeros_like(p) for p in self.params]
        self.momentum_buffers = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using the current RMSprop state."""
        for p, v, buffer in zip(
            self.params,
            self.ema_of_sq_grads,
            self.momentum_buffers,
            strict=True,
        ):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            v.mul_(self.rho).add_(
                p.grad.square(),
                alpha=1 - self.rho,
            )

            if self.momentum > 0:
                buffer.mul_(self.momentum).addcdiv_(p.grad, v.sqrt() + self.eps)
                p.sub_(buffer, alpha=self.lr)
            else:
                p.addcdiv_(p.grad, v.sqrt() + self.eps, value=-self.lr)

    @torch.no_grad()
    def get_effective_lr(self) -> list[Tensor]:
        """Return per-parameter effective learning rates."""
        effective_lr = []

        for v in self.ema_of_sq_grads:
            lr = self.lr / (v.sqrt() + self.eps).clone()
            effective_lr.append(lr)

        return effective_lr

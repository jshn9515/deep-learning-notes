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
        super().__init__(
            params,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay
        )
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

        self.ema_of_sq_grads = [torch.zeros_like(p) for p in self.params]
        self.ema_of_sq_updates = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using the current Adadelta state."""
        for p, v, u in zip(
            self.params,
            self.ema_of_sq_grads,
            self.ema_of_sq_updates,
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
            delta_x = (u + self.eps).sqrt() / (v + self.eps).sqrt() * p.grad

            u.mul_(self.rho).add_(
                delta_x.square(),
                alpha=1 - self.rho,
            )
            p.sub_(delta_x, alpha=self.lr)

    @torch.no_grad()
    def get_effective_lr(self) -> list[Tensor]:
        """Return per-parameter effective learning rates."""
        effective_lr = []

        for v, u in zip(
            self.ema_of_sq_grads,
            self.ema_of_sq_updates,
            strict=True,
        ):
            lr = self.lr * (u + self.eps).sqrt() / (v + self.eps).sqrt()
            effective_lr.append(lr)

        return effective_lr

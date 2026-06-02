from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['Adagrad']


class Adagrad(Optimizer):
    """Adaptive gradient optimizer with per-parameter learning rates."""

    def __init__(self, params: Sequence[Tensor], lr: float = 1e-2, eps: float = 1e-10):
        """Create an Adagrad optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-2): Base learning rate.
            eps (float, default: 1e-10): Small value added to the denominator
                for numerical stability.
        """
        self.lr = lr
        self.eps = eps
        super().__init__(params)
        self.sum_sq_grads = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Accumulate squared gradients and apply an Adagrad update."""
        for p, sum_sq_grad in zip(self.params, self.sum_sq_grads, strict=True):
            if p.grad is None:
                continue

            sum_sq_grad.add_(p.grad.square())
            effective_lr = self.lr / (sum_sq_grad.sqrt() + self.eps)
            p.add_(-effective_lr * p.grad)

    @torch.no_grad()
    def get_effective_lr(self) -> list[Tensor]:
        """Return the current per-parameter effective learning rates."""
        return [
            self.lr / (sum_sq_grad.sqrt() + self.eps)
            for sum_sq_grad in self.sum_sq_grads
        ]

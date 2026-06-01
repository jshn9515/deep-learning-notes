from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = [
    'SimpleSGD',
    'SGDWithMomentum',
]


class SimpleSGD(Optimizer):
    """Stochastic gradient descent without momentum."""

    def __init__(self, params: list[Tensor], lr: float = 1e-3):
        """Initialize the optimizer.

        Args:
            params (list[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each gradient update.
        """
        self.params = params
        self.lr = lr

    @override
    @torch.no_grad()
    def step(self):
        """Subtract ``lr * grad`` from each parameter with a gradient."""
        for p in self.params:
            if p.grad is None:
                continue

            p.sub_(self.lr * p.grad)

    @override
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients for all optimized parameters.

        Args:
            set_to_none (bool, default: False): If ``True``, replace existing
                gradients with ``None``. Otherwise, zero gradients in place.
        """
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()


class SGDWithMomentum(Optimizer):
    """Stochastic gradient descent with exponential momentum."""

    def __init__(self, params: list[Tensor], lr: float = 1e-3, momentum: float = 0.0):
        """Initialize the optimizer and per-parameter velocity buffers.

        Args:
            params (list[methodTensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each velocity update.
            momentum (float, default: 0.0): Coefficient applied to the previous
                velocity before adding the current gradient.
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update velocities from gradients and apply them to parameters."""
        for p, v in zip(self.params, self.velocity, strict=True):
            if p.grad is None:
                continue

            v.mul_(self.momentum).add_(p.grad)
            p.sub_(self.lr * v)

    @override
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients for all optimized parameters.

        Args:
            set_to_none (bool, default: False): If ``True``, replace existing
                gradients with ``None``. Otherwise, zero gradients in place.
        """
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

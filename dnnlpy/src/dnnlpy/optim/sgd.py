from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = [
    'SimpleSGD',
    'SimpleSGDWithWeightDecay',
    'SimpleSGDWithMomentum',
    'SimpleSGDWithNesterovMomentum',
    'SGD',
]


class SimpleSGD(Optimizer):
    """Stochastic gradient descent without momentum."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3):
        """Create a stochastic gradient descent optimizer without momentum.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each gradient update.
        """
        super().__init__(params, lr=lr)
        self.lr = lr

    @override
    @torch.no_grad()
    def step(self):
        """Subtract ``lr * grad`` from each parameter with a gradient."""
        for p in self.params:
            if p.grad is None:
                continue

            p.sub_(p.grad, alpha=self.lr)


class SimpleSGDWithWeightDecay(Optimizer):
    """Stochastic gradient descent with weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        """Create a stochastic gradient descent optimizer with weight decay.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each gradient update.
            weight_decay (float, default: 0.0): Coefficient applied to the parameters
                before adding them to the gradient. Must be non-negative.
        """
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay

    @override
    @torch.no_grad()
    def step(self):
        """Subtract ``lr * grad`` from each parameter with a gradient."""
        for p in self.params:
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            p.sub_(p.grad, alpha=self.lr)


class SimpleSGDWithMomentum(Optimizer):
    """Stochastic gradient descent with momentum."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ):
        """Create a stochastic gradient descent optimizer with momentum.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each velocity update.
            momentum (float, default: 0.0): Coefficient applied to the previous
                velocity before adding the current gradient.
        """
        super().__init__(params, lr=lr, momentum=momentum)
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
            p.sub_(v, alpha=self.lr)


class SimpleSGDWithNesterovMomentum(Optimizer):
    """Stochastic gradient descent with Nesterov momentum."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ):
        """Create a stochastic gradient descent optimizer with Nesterov momentum.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each velocity update.
            momentum (float, default: 0.0): Coefficient applied to the previous
                velocity before adding the current gradient.
        """
        super().__init__(params, lr=lr, momentum=momentum)
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
            p.grad.add_(v, alpha=self.momentum)
            p.sub_(p.grad, alpha=self.lr)


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum variants."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """Create a stochastic gradient descent optimizer with optional momentum and
        Nesterov momentum.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each velocity update.
            momentum (float, default: 0.0): Coefficient applied to the previous velocity
                before adding the current gradient.
            weight_decay (float, default: 0.0): Coefficient applied to the parameters
                before adding them to the gradient. Must be non-negative.
            nesterov (bool, default: False): Whether to use Nesterov momentum.
        """
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = [torch.zeros_like(p) for p in self.params]
        self.has_velocity = [False for _ in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update velocities from gradients and apply them to parameters."""
        for idx, (p, v) in enumerate(zip(self.params, self.velocity, strict=True)):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            if self.momentum > 0:
                if self.has_velocity[idx]:
                    v.mul_(self.momentum).add_(p.grad)
                else:
                    v.copy_(p.grad)
                    self.has_velocity[idx] = True

                if self.nesterov:
                    p.grad.add_(v, alpha=self.momentum)
                else:
                    p.grad.copy_(v)

            p.sub_(p.grad, alpha=self.lr)

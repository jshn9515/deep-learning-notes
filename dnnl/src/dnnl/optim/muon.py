from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['Muon']

NS_COEFFS_DEFAULT = (3.4445, -4.7750, 2.0315)


def newton_schulz_5(
    X: Tensor,
    ns_steps: int = 5,
    ns_coefficients: tuple[float, float, float] = NS_COEFFS_DEFAULT,
    eps: float = 1e-7,
) -> Tensor:
    """Approximate the orthogonalized Muon update with Newton-Schulz steps.

    Args:
        X (Tensor): Two-dimensional update matrix to orthogonalize.
        ns_steps (int, default: 5): Number of Newton-Schulz iterations.
        ns_coefficients (tuple[float, float, float], default: (3.4445, -4.7750, 2.0315)):
            Coefficients for the quintic Newton-Schulz iteration.
        eps (float, default: 1e-7): Small value used when normalizing ``X``.

    Returns:
        Orthogonalized update matrix with the same shape as ``X``.
    """
    if X.ndim != 2:
        raise ValueError('Muon only supports 2D parameters.')

    a, b, c = ns_coefficients
    X = X / (X.norm() + eps)
    should_transpose = X.size(0) > X.size(1)

    if should_transpose:
        X = X.T

    for _ in range(ns_steps):
        A = torch.mm(X, X.T)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)

    if should_transpose:
        X = X.T

    return X


class Muon(Optimizer):
    """Muon optimizer for two-dimensional parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = NS_COEFFS_DEFAULT,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        """Create a Muon optimizer.

        Args:
            params (Iterable[Tensor]): Two-dimensional parameters to update.
            lr (float, default: 1e-3): Base learning rate.
            weight_decay (float, default: 0.1): Coefficient applied to the
                parameters before adding them to the gradient.
            momentum (float, default: 0.95): Momentum coefficient applied to
                the update buffer.
            nesterov (bool, default: True): Whether to use Nesterov momentum.
            ns_coefficients (tuple[float, float, float], default: (3.4445,
                -4.7750, 2.0315)): Coefficients for the Newton-Schulz
                orthogonalization iteration.
            ns_steps (int, default: 5): Number of Newton-Schulz iterations.
            eps (float, default: 1e-7): Small value used when normalizing the
                update matrix.
        """
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_coefficients = ns_coefficients
        self.ns_steps = ns_steps
        self.eps = eps

        self.momentum_buffers = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using momentum and a Muon orthogonalized step."""
        for p, buffer in zip(self.params, self.momentum_buffers, strict=True):
            if p.grad is None:
                continue

            # Decoupled weight decay: directly shrink parameters.
            if self.weight_decay > 0:
                p.mul_(1 - self.lr * self.weight_decay)

            buffer.mul_(self.momentum).add_(p.grad)
            if self.nesterov:
                direction = p.grad + self.momentum * buffer
            else:
                direction = buffer

            update = newton_schulz_5(
                direction,
                ns_steps=self.ns_steps,
                ns_coefficients=self.ns_coefficients,
                eps=self.eps,
            )

            p.sub_(update, alpha=self.lr)

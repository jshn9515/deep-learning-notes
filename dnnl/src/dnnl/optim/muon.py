from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['Muon']


def newton_schulz_5(
    X: Tensor,
    ns_steps: int = 5,
    eps: float = 1e-7,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> Tensor:
    """Approximate the orthogonalized Muon update with Newton-Schulz steps.

    Args:
        X (Tensor): Two-dimensional update matrix to orthogonalize.
        ns_steps (int, default: 5): Number of Newton-Schulz iterations.
        eps (float, default: 1e-7): Small value used when normalizing ``X``.
        ns_coefficients (tuple[float, float, float], default: (3.4445, -4.7750,
            2.0315)): Coefficients for the quintic Newton-Schulz iteration.

    Returns:
        Orthogonalized update matrix with the same shape as ``X``.
    """
    if X.ndim != 2:
        raise NotImplementedError('Muon only supports 2D parameters.')

    a, b, c = ns_coefficients
    X = X / (X.norm() + eps)
    should_transpose = X.size(0) > X.size(1)

    if should_transpose:
        X = X.T

    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if should_transpose:
        X = X.T

    return X


class Muon(Optimizer):
    """Muon optimizer for two-dimensional parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.95,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        """Create a Muon optimizer.

        Args:
            params (Iterable[Tensor]): Two-dimensional parameters to update.
            lr (float, default: 1e-3): Base learning rate.
            momentum (float, default: 0.95): Momentum coefficient applied to
                the update buffer.
            ns_coefficients (tuple[float, float, float], default: (3.4445,
                -4.7750, 2.0315)): Coefficients for the Newton-Schulz
                orthogonalization iteration.
            ns_steps (int, default: 5): Number of Newton-Schulz iterations.
            eps (float, default: 1e-7): Small value used when normalizing the
                update matrix.
        """
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.ns_coefficients = ns_coefficients
        self.ns_steps = ns_steps
        self.eps = eps

        for p in self.params:
            if p.ndim != 2:
                raise NotImplementedError('Muon only supports 2D parameters.')

        self.momentum_buffers = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using momentum and a Muon orthogonalized step."""
        for p, buffer in zip(self.params, self.momentum_buffers, strict=True):
            if p.grad is None:
                continue

            buffer.mul_(self.momentum).add_(p.grad)
            update = newton_schulz_5(
                buffer,
                ns_steps=self.ns_steps,
                eps=self.eps,
                ns_coefficients=self.ns_coefficients,
            )

            p.add_(update, alpha=-self.lr)

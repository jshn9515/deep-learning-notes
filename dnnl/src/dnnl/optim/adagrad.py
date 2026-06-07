from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['Adagrad']


class Adagrad(Optimizer):
    """Adaptive gradient optimizer with per-parameter learning rates."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ):
        """Create an Adagrad optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-2): Base learning rate.
            lr_decay (float, default: 0.0): Learning-rate decay multiplier.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
            initial_accumulator_value (float, default: 0.0): Initial value for
                each squared-gradient accumulator.
            eps (float, default: 1e-10): Small value added to the denominator
                for numerical stability.
        """
        super().__init__(params)
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

        self.step_count = 0
        self.sum_of_sq_grads = [
            torch.full_like(p, initial_accumulator_value) for p in self.params
        ]

    @override
    @torch.no_grad()
    def step(self):
        """Accumulate squared gradients and apply an Adagrad update."""
        self.step_count += 1
        clr = self.lr / (1 + (self.step_count - 1) * self.lr_decay)

        for p, s in zip(self.params, self.sum_of_sq_grads, strict=True):
            if p.grad is None:
                continue

            if self.weight_decay > 0:
                p.grad.add_(self.weight_decay * p)

            s.add_(p.grad.square())
            p.sub_(clr / (s.sqrt() + self.eps) * p.grad)

    @torch.no_grad()
    def get_effective_lr(self) -> list[Tensor]:
        """Return per-parameter effective learning rates."""
        clr = self.lr / (1 + max(self.step_count - 1, 0) * self.lr_decay)
        effective_lr = []

        for s in self.sum_of_sq_grads:
            lr = clr / (s.sqrt() + self.eps).clone()
            effective_lr.append(lr)

        return effective_lr

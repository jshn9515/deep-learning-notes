from collections.abc import Iterable
from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = ['AdamW']


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """Create an AdamW optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Base learning rate.
            betas (tuple[float, float], default: (0.9, 0.999)): Coefficients
                used to compute running averages of the gradient and its square.
            eps (float, default: 1e-8): Small value added to the denominator
                for numerical stability.
            weight_decay (float, default: 1e-2): Decoupled weight decay
                coefficient applied directly to parameters before the Adam
                update.
        """
        super().__init__(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

        self.step_count = 0
        self.first_moments = [torch.zeros_like(p) for p in self.params]
        self.second_moments = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        """Update parameters using Adam moments and decoupled weight decay."""
        self.step_count += 1

        for p, m, v in zip(
            self.params,
            self.first_moments,
            self.second_moments,
            strict=True,
        ):
            if p.grad is None:
                continue

            # Decoupled weight decay: directly shrink parameters.
            if self.weight_decay > 0:
                p.mul_(1 - self.lr * self.weight_decay)

            m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)

            bias_correction1 = 1 - pow(self.beta1, self.step_count)
            bias_correction2 = 1 - pow(self.beta2, self.step_count)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Adam update: use the original gradient statistics.
            p.addcdiv_(
                m_hat,
                v_hat.sqrt() + self.eps,
                value=-self.lr,
            )

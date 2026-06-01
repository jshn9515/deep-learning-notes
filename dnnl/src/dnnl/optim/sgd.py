from typing import override

import torch
from torch import Tensor

from .base import Optimizer

__all__ = [
    'SimpleSGD',
    'SGDWithMomentum',
]


class SimpleSGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    @override
    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            p.sub_(self.lr * p.grad)

    @override
    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()


class SGDWithMomentum(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, beta: float):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @override
    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocity, strict=True):
            if p.grad is None:
                continue

            v.mul_(self.beta).add_(p.grad)
            p.sub_(self.lr * v)

    @override
    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

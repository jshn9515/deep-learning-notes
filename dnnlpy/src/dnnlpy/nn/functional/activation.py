import math

import torch
from torch import Tensor

__all__ = [
    'sigmoid',
    'tanh',
    'relu',
    'gelu',
    'softmax',
    'log_softmax',
]


def sigmoid(x: Tensor) -> Tensor:
    """Apply the sigmoid function element-wise."""
    return 1 / (1 + (-x).exp())


def tanh(x: Tensor) -> Tensor:
    """Apply the hyperbolic tangent function element-wise."""
    return x.tanh()


def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the rectified linear unit function element-wise."""
    if inplace:
        return x.clamp_(min=0)
    return x.clamp(min=0)


def gelu(x: Tensor, approximate: str = 'none') -> Tensor:
    """Apply the Gaussian Error Linear Unit function element-wise."""
    if approximate == 'tanh':
        scale = math.sqrt(2 / math.pi)
        return 0.5 * x * (1.0 + tanh(scale * (x + 0.044715 * x.pow(3))))
    else:
        return 0.5 * x * (1.0 + (x / math.sqrt(2)).erf())


def softmax(x: Tensor, dim: int) -> Tensor:
    """Apply the softmax function along the specified dimension."""
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = (x - max_x).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def log_softmax(x: Tensor, dim: int) -> Tensor:
    """Apply the log-softmax function along the specified dimension."""
    max_x = x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.logsumexp(x - max_x, dim=dim, keepdim=True)
    return x - max_x - log_sum_exp

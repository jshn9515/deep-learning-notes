import torch.nn as nn
from torch import Tensor

from . import functional as dF

__all__ = [
    'Sigmoid',
    'Tanh',
    'ReLU',
    'GELU',
    'Softmax',
    'LogSoftmax',
]


class Sigmoid(nn.Module):
    """Apply the sigmoid function element-wise."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return dF.sigmoid(x)


class Tanh(nn.Module):
    """Apply the hyperbolic tangent function element-wise."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return dF.tanh(x)


class ReLU(nn.Module):
    """Apply the rectified linear unit function element-wise."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return dF.relu(x)


class GELU(nn.Module):
    """Apply the Gaussian Error Linear Unit function element-wise."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return dF.gelu(x)


class Softmax(nn.Module):
    """Apply the softmax function along a specified dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return dF.softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(nn.Module):
    """Apply the log-softmax function along a specified dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return dF.log_softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

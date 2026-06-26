import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal

from . import functional as dF

type Approx = Literal['none', 'tanh']

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

    def __init__(self, *, fast: bool = False):
        """Initialize the sigmoid activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.sigmoid(x)
        return dF.sigmoid(x)


class Tanh(nn.Module):
    """Apply the hyperbolic tangent function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the hyperbolic tangent activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.tanh(x)
        return dF.tanh(x)


class ReLU(nn.Module):
    """Apply the rectified linear unit function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the ReLU activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.relu(x, self.inplace)
        return dF.relu(x, self.inplace)


class GELU(nn.Module):
    """Apply the Gaussian Error Linear Unit function element-wise."""

    def __init__(self, approximate: Approx = 'none', *, fast: bool = False):
        """Initialize the GELU activation function.

        Args:
            approximate (Literal['none', 'tanh'], default: 'none'): The approximation
                method to use. Options are 'none' for the exact GELU function or 'tanh'
                for the tanh approximation.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.approximate = approximate
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.gelu(x, approximate=self.approximate)
        return dF.gelu(x, self.approximate)


class Softmax(nn.Module):
    """Apply the softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        """Initialize the softmax activation function.

        Args:
            dim (int): Dimension along which softmax will be computed.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softmax(x, dim=self.dim)
        return dF.softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(nn.Module):
    """Apply the log-softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        """Initialize the log-softmax activation function.

        Args:
            dim (int): Dimension along which log-softmax will be computed.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.log_softmax(x, dim=self.dim)
        return dF.log_softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

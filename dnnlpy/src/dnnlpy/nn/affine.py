import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = [
    'Identity',
    'Linear',
]


class Identity(nn.Module):
    """A module that returns the input as is."""

    def __init__(self, *, fast: bool = False):
        """Initialize the Identity module."""
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        """Return the input as is."""
        return x


class Linear(nn.Module):
    """Apply an affine transformation to the incoming data."""

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize the weight and optional bias parameters.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, default: True): Whether to learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast = fast

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters using the same default initialization as PyTorch."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias)  # A simplified version

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear transformation."""
        if self.fast:
            return F.linear(x, self.weight, self.bias)
        return dF.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}'
        )

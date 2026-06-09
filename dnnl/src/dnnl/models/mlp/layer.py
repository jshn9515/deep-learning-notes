import math
from typing import override

import numpy as np

from .base import Module, Parameter

__all__ = [
    'Flatten',
    'Linear',
]

rng = np.random.default_rng()


class Flatten(Module):
    """Flatten the input while keeping the batch dimension unchanged."""

    def __init__(self):
        """Initialize the flatten layer."""
        super().__init__()

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return a two-dimensional view with flattened non-batch axes."""
        self.save_to_context(x.shape)
        return x.reshape(x.shape[0], -1)

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Restore gradients to the original input shape."""
        assert self.ctx is not None, 'Must call forward before backward.'
        original_shape = self.load_from_context()
        return grad.reshape(original_shape)


class Linear(Module):
    """Fully connected affine layer for batched NumPy inputs."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize weights and biases for a linear transformation.

        Args:
            in_features (int): Number of input features per sample.
            out_features (int): Number of output features per sample.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        W = rng.standard_normal((in_features, out_features))
        W = W * math.sqrt(2.0 / in_features)
        b = np.zeros(out_features)

        self.W = Parameter(W)
        self.b = Parameter(b)

    @override
    def extra_repr(self) -> str:
        """Return feature sizes for ``repr(layer)``."""
        return f'in_features={self.in_features},\nout_features={self.out_features}'

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``x @ W + b`` and cache ``x`` for backpropagation."""
        self.save_to_context(x)
        return x @ self.W + self.b

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Store parameter gradients and return input gradients."""
        assert self.ctx is not None, 'Must call forward before backward.'
        x = self.load_from_context()

        self.W.grad = x.T @ grad  # dW
        self.b.grad = np.sum(grad, axis=0)  # db
        return grad @ self.W.T  # dx

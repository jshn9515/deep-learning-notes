from typing import override

import numpy as np

from .base import Module

rng = np.random.default_rng(42)

__all__ = ['Linear']


class Linear(Module):
    """Fully connected affine layer for batched NumPy inputs."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize weights and biases for a linear transformation.

        Args:
            in_features (int): Number of input features per sample.
            out_features (int): Number of output features per sample.
        """
        self.in_features = in_features
        self.out_features = out_features

        self.W = rng.standard_normal((in_features, out_features))
        self.W = self.W * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        self.x = None
        self.dW = None
        self.db = None

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``x @ W + b`` and cache ``x`` for the backward pass."""
        self.x = x
        return x @ self.W + self.b

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Accumulate parameter gradients and return input gradients."""
        assert self.x is not None, 'Must call forward before backward.'
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0)
        dx = grad @ self.W.T
        return dx

    @override
    def parameters(self) -> list[np.ndarray]:
        """Return the weight and bias arrays."""
        return [self.W, self.b]

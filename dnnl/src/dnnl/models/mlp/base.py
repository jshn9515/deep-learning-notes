from abc import ABC, abstractmethod

import numpy as np

__all__ = ['Module']


class Module(ABC):
    """Base class for NumPy MLP layers with manual backpropagation."""

    def __init__(self):
        """Initialize the layer context storage."""
        self.ctx = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute layer outputs from input values."""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Propagate output gradients back to this layer's inputs."""
        pass

    def parameters(self) -> list[np.ndarray]:
        """Return trainable parameter arrays owned by this layer."""
        return []

# pyright: reportIncompatibleMethodOverride=false

from typing import override

import numpy as np

from .activation import softmax
from .base import Module

__all__ = ['cross_entropy', 'CrossEntropyLoss']


def cross_entropy(
    probs: np.ndarray, targets: np.ndarray, eps: float = 1e-12
) -> np.floating:
    """Compute mean negative log likelihood for class probabilities.

    Args:
        probs (np.ndarray): Batch of class probabilities with shape
            ``(batch_size, num_classes)``.
        targets (np.ndarray): Integer class indices with shape ``(batch_size,)``.
        eps (float, default: 1e-12): Small value added before the logarithm.
    """
    batch_size = probs.shape[0]
    correct_probs = probs[np.arange(batch_size), targets]
    return -np.mean(np.log(correct_probs + eps))


class CrossEntropyLoss(Module):
    """Softmax cross-entropy loss for batched classification logits."""

    def __init__(self, eps: float = 1e-12):
        """Initialize the loss.

        Args:
            eps (float, default: 1e-12): Small value added before the logarithm
                for numerical stability.
        """
        self.eps = eps
        self.probs = None
        self.targets = None

    @override
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.floating:
        """Return mean cross-entropy loss for logits and integer targets."""
        self.targets = targets
        self.probs = softmax(logits)
        loss = cross_entropy(self.probs, targets, self.eps)
        return loss

    @override
    def backward(self) -> np.ndarray:
        """Return gradients of the loss with respect to the logits."""
        if self.probs is None or self.targets is None:
            raise AssertionError('Must call forward before backward.')

        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.targets] -= 1
        return grad / batch_size

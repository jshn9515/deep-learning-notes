from abc import ABC, abstractmethod

__all__ = ['Optimizer']


class Optimizer(ABC):
    """Abstract interface for gradient-based parameter optimizers."""

    @abstractmethod
    def step(self):
        """Update parameters in place using their current gradients."""
        pass

    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        """Clear stored parameter gradients.

        Args:
            set_to_none (bool, default: False): If ``True``, replace existing
                gradients with ``None``. Otherwise, zero gradients in place.
        """
        pass

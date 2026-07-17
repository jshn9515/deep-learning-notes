import math

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ['CosineAnnealingLR']


class CosineAnnealingLR(LRScheduler):
    """Anneal each parameter group's learning rate with a cosine schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """Set the learning rate of each parameter group using a cosine annealing schedule.

        Args:
            optimizer (Optimizer): Optimizer whose learning rate is scheduled.
            T_max (int): Maximum number of iterations in the cosine cycle.
            eta_min (float, default: 0.0): Minimum learning rate.
            last_epoch (int, default: -1): Index of the last completed epoch.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute learning rates from the recursive cosine schedule."""
        if getattr(self, '_is_initial', self.last_epoch == 0):
            return [group['lr'] for group in self.optimizer.param_groups]

        if self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]

        # The usual recurrence divides by zero at each cosine minimum.
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr']
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(
                    self.base_lrs, self.optimizer.param_groups, strict=True
                )
            ]

        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group['lr'] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

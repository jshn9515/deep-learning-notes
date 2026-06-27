from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

type Reduction = Literal['mean', 'sum', 'none']

__all__ = ['CrossEntropyLoss']


class CrossEntropyLoss(nn.Module):
    """Combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class."""

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: Reduction = 'mean',
        *,
        fast: bool = False,
    ):
        """Initializes the CrossEntropyLoss module.

        Args:
            weight (Tensor | None): A manual rescaling weight given to each class.If given,
                it has to be a 1D Tensor assigning weight to each of the classes. Default: None.
            reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
                - 'none': no reduction will be applied;
                - 'mean': the sum of the output will be divided by the number of elements in the output;
                - 'sum': the output will be summed.
                Default: 'mean'.
            fast (bool, default: False): If set to True, will use the fast implementation
                from torch.nn.functional. Default: False.
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.cross_entropy(
                x,
                target,
                weight=self.weight,
                reduction=self.reduction,
            )
        return dF.cross_entropy(
            x,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )

from torch import Tensor

from .activation import log_softmax

__all__ = ['cross_entropy']


def cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: str = 'mean',
) -> Tensor:
    """Compute cross entropy loss between logits and class indices.

    Args:
        x (Tensor): Logits tensor of shape (N, C) or (C,) where N is the batch size and C is
            the number of classes.
        target (Tensor): Class indices tensor of shape (N,) or scalar for single sample.
        weight (Tensor, optional): A manual rescaling weight given to each class. If given,
            it has to be a 1D tensor of size C.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' |
            'mean' | 'sum'. Default: 'mean'.

    Returns:
        Tensor: The computed cross entropy loss. If reduction is 'none', the shape will be
            (N,) for batch input or scalar for single sample. If reduction is 'mean' or 'sum',
            the shape will be scalar.
    """
    if reduction not in {'mean', 'sum', 'none'}:
        raise NotImplementedError('`reduction` must be `mean`, `sum`, or `none`.')

    if x.ndim == 1:
        log_probs = log_softmax(x, dim=0)
        loss = -log_probs[target]
        if weight is not None:
            loss = loss * weight[target]
            total_weight = weight[target]
        else:
            total_weight = None
    else:
        log_probs = log_softmax(x, dim=1)
        gather_index = target.unsqueeze(1)
        loss = -log_probs.gather(1, gather_index).squeeze(1)
        if weight is not None:
            sample_weight = weight[target]
            loss = loss * sample_weight
            total_weight = sample_weight.sum()
        else:
            total_weight = None

    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    if total_weight is not None:
        return loss.sum() / total_weight
    return loss.mean()

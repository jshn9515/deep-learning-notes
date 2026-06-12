import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import dnnl.nn as dnn
import dnnl.nn.functional as dF


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_cross_entropy_function_matches_torch(reduction: str):
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.cross_entropy(x, target, weight=weight, reduction=reduction)
    expected = F.cross_entropy(x, target, weight=weight, reduction=reduction)

    assert torch.allclose(actual, expected)


def test_cross_entropy_function_matches_torch_without_weight():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])

    actual = dF.cross_entropy(x, target)
    expected = F.cross_entropy(x, target)

    assert torch.allclose(actual, expected)


def test_cross_entropy_module_matches_torch_module():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dnn.CrossEntropyLoss(weight=weight, reduction='sum')
    expected = nn.CrossEntropyLoss(weight=weight, reduction='sum')

    assert torch.allclose(actual(x, target), expected(x, target))


def test_cross_entropy_rejects_invalid_reduction():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])

    with pytest.raises(NotImplementedError, match='reduction'):
        dF.cross_entropy(x, target, reduction='batchmean')

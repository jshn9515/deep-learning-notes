import torch

import dnnl.optim as dopt
import dnnl.optim.utils as utils


def quadratic_loss(theta: torch.Tensor) -> torch.Tensor:
    return theta.square().sum() / 2


def test_run_optimizer_public_export():
    assert dopt.run_optimizer is utils.run_optimizer


def test_run_optimizer_records_parameter_history():
    params = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SimpleSGD([params], lr=0.1)

    history = dopt.run_optimizer(optimizer, quadratic_loss, steps=2)

    assert history.shape == (3, 2)
    assert torch.allclose(history[0], torch.tensor([1.0, -2.0]))
    assert torch.allclose(history[1], torch.tensor([0.9, -1.8]))
    assert torch.allclose(history[2], torch.tensor([0.81, -1.62]))
    assert torch.allclose(params, torch.tensor([0.81, -1.62]))

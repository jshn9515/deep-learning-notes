import inspect

import torch

import dnnl.optim as dopt
import dnnl.optim.adam as adam


def test_adam_module_has_docstrings():
    for name in adam.__all__:
        member = getattr(adam, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'


def test_adam_public_export():
    assert dopt.Adam is adam.Adam


def test_adam_accumulates_moments_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.Adam([param], lr=0.1, betas=(0.9, 0.999), eps=0.0)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert optimizer.step_count == 1
    assert torch.allclose(optimizer.exp_avg[0], torch.tensor([0.05, -0.025]))
    assert torch.allclose(optimizer.exp_avg_sq[0], torch.tensor([0.00025, 0.0000625]))
    assert torch.allclose(param, torch.tensor([0.9, -1.9]))


def test_adam_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.Adam([trained, skipped], lr=0.1, betas=(0.9, 0.999), eps=0.0)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.9]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.exp_avg[1], torch.zeros_like(skipped))
    assert torch.equal(optimizer.exp_avg_sq[1], torch.zeros_like(skipped))

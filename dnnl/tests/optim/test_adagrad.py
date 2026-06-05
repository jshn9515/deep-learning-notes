import inspect

import torch

import dnnl.optim as dopt
import dnnl.optim.adagrad as adagrad


def test_adagrad_module_has_docstrings():
    for name in adagrad.__all__:
        member = getattr(adagrad, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'


def test_adagrad_public_export():
    assert dopt.Adagrad is adagrad.Adagrad


def test_adagrad_accumulates_squared_gradients_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.Adagrad([param], lr=0.1, eps=0.0)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.allclose(optimizer.sum_sq_grads[0], torch.tensor([0.25, 0.0625]))
    assert torch.allclose(param, torch.tensor([0.9, -1.9]))

    param.grad = torch.tensor([0.5, 0.25])
    optimizer.step()

    expected_sum_sq_grads = torch.tensor([0.5, 0.125])
    expected_effective_lr = 0.1 / expected_sum_sq_grads.sqrt()
    assert torch.allclose(optimizer.sum_sq_grads[0], expected_sum_sq_grads)
    assert torch.allclose(param, torch.tensor([0.8292893, -1.9707106]))
    assert torch.allclose(optimizer.get_effective_lr()[0], expected_effective_lr)


def test_adagrad_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.Adagrad([trained, skipped], lr=0.1, eps=0.0)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.9]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.sum_sq_grads[1], torch.zeros_like(skipped))

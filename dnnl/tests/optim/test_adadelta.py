import inspect

import torch

import dnnl.optim as dopt
import dnnl.optim.adadelta as adadelta


def test_adadelta_module_has_docstrings():
    for name in adadelta.__all__:
        member = getattr(adadelta, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'


def test_adadelta_public_export():
    assert dopt.Adadelta is adadelta.Adadelta


def test_adadelta_accumulates_state_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.Adadelta([param], lr=1.0, rho=0.9, eps=1e-6)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    expected_square_avg = torch.tensor([0.025, 0.00625])
    expected_update = torch.tensor([-0.0031622, 0.0031620])
    expected_accumulate_update = 0.1 * expected_update.square()
    assert torch.allclose(optimizer.square_avgs[0], expected_square_avg)
    assert torch.allclose(optimizer.accumulate_updates[0], expected_accumulate_update)
    assert torch.allclose(param, torch.tensor([0.9968378, -1.9968380]))


def test_adadelta_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.Adadelta([trained, skipped], lr=1.0, rho=0.9, eps=1e-6)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.9968378]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.square_avgs[1], torch.zeros_like(skipped))
    assert torch.equal(optimizer.accumulate_updates[1], torch.zeros_like(skipped))

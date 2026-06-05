import inspect

import pytest
import torch

import dnnl.optim as dopt
import dnnl.optim.base as base
import dnnl.optim.sgd as sgd


def test_sgd_modules_have_docstrings():
    modules = [base, sgd]

    for module in modules:
        for name in module.__all__:
            member = getattr(module, name)
            assert inspect.getdoc(member), name

            for method_name, method in inspect.getmembers(member, inspect.isfunction):
                if method.__qualname__.startswith(f'{member.__name__}.'):
                    assert inspect.getdoc(method), f'{name}.{method_name}'


def test_sgd_public_exports():
    assert dopt.Optimizer is base.Optimizer
    assert dopt.SimpleSGD is sgd.SimpleSGD
    assert dopt.SimpleSGDWithMomentum is sgd.SimpleSGDWithMomentum
    assert dopt.SimpleSGDWithNesterovMomentum is sgd.SimpleSGDWithNesterovMomentum
    assert dopt.SGD is sgd.SGD


def test_optimizer_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        dopt.Optimizer([])  # type: ignore[abstract]


def test_optimizer_zero_grad_can_zero_or_remove_gradients():
    param = torch.tensor([1.0, 2.0], requires_grad=True)
    untouched = torch.tensor([3.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])

    optimizer = dopt.SimpleSGD([param, untouched])
    optimizer.zero_grad()

    assert param.grad is not None
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert untouched.grad is None

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.zero_grad(set_to_none=True)

    assert param.grad is None


def test_simple_sgd_step_updates_parameters_and_skips_missing_gradients():
    trainable = torch.tensor([1.0, -2.0], requires_grad=True)
    frozen = torch.tensor([3.0], requires_grad=True)
    trainable.grad = torch.tensor([0.25, -0.5])

    optimizer = dopt.SimpleSGD([trainable, frozen], lr=0.1)
    optimizer.step()

    assert torch.allclose(trainable, torch.tensor([0.975, -1.95]))
    assert torch.allclose(frozen, torch.tensor([3.0]))


def test_simple_sgd_with_momentum_accumulates_velocity():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SimpleSGDWithMomentum([param], lr=0.1, momentum=0.9)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.allclose(optimizer.velocity[0], torch.tensor([0.5, -0.25]))
    assert torch.allclose(param, torch.tensor([0.95, -1.975]))

    param.grad = torch.tensor([0.1, 0.2])
    optimizer.step()

    assert torch.allclose(optimizer.velocity[0], torch.tensor([0.55, -0.025]))
    assert torch.allclose(param, torch.tensor([0.895, -1.9725]))


def test_simple_sgd_with_nesterov_momentum_uses_lookahead_update():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SimpleSGDWithNesterovMomentum([param], lr=0.1, momentum=0.9)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.allclose(optimizer.velocity[0], torch.tensor([0.5, -0.25]))
    assert torch.allclose(param, torch.tensor([0.905, -1.9525]))


@pytest.mark.parametrize('nesterov', [False, True])
def test_sgd_matches_simple_momentum_variants(nesterov: bool):
    actual_param = torch.tensor([1.0, -2.0], requires_grad=True)
    expected_param = actual_param.detach().clone().requires_grad_()
    optimizer = dopt.SGD([actual_param], lr=0.1, momentum=0.9, nesterov=nesterov)

    if nesterov:
        expected_optimizer_cls = dopt.SimpleSGDWithNesterovMomentum
    else:
        expected_optimizer_cls = dopt.SimpleSGDWithMomentum

    expected_optimizer = expected_optimizer_cls(
        [expected_param],
        lr=0.1,
        momentum=0.9,
    )

    for grad in [torch.tensor([0.5, -0.25]), torch.tensor([0.1, 0.2])]:
        actual_param.grad = grad.clone()
        expected_param.grad = grad.clone()
        optimizer.step()
        expected_optimizer.step()

    assert torch.allclose(actual_param, expected_param)
    assert torch.allclose(optimizer.velocity[0], expected_optimizer.velocity[0])


def test_sgd_with_zero_momentum_matches_simple_sgd():
    actual_param = torch.tensor([1.0, -2.0], requires_grad=True)
    expected_param = actual_param.detach().clone().requires_grad_()
    actual_param.grad = torch.tensor([0.5, -0.25])
    expected_param.grad = actual_param.grad.clone()

    dopt.SGD([actual_param], lr=0.1).step()
    dopt.SimpleSGD([expected_param], lr=0.1).step()

    assert torch.allclose(actual_param, expected_param)

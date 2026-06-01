import inspect

import pytest
import torch

import dnnl.optim as optim


def test_public_optimizer_api_has_docstrings():
    public_members = [
        optim.Optimizer,
        optim.Optimizer.step,
        optim.Optimizer.zero_grad,
        optim.SimpleSGD,
        optim.SimpleSGD.__init__,
        optim.SimpleSGD.step,
        optim.SimpleSGD.zero_grad,
        optim.SGDWithMomentum,
        optim.SGDWithMomentum.__init__,
        optim.SGDWithMomentum.step,
        optim.SGDWithMomentum.zero_grad,
    ]

    for member in public_members:
        assert inspect.getdoc(member)


def test_sgd_with_momentum_uses_pytorch_parameter_name():
    signature = inspect.signature(optim.SGDWithMomentum)

    assert 'momentum' in signature.parameters
    assert 'beta' not in signature.parameters
    assert signature.parameters['momentum'].default == 0.0


def test_optimizer_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        optim.Optimizer()  # type: ignore[call-arg]


def test_simple_sgd_step_updates_parameters_and_skips_missing_gradients():
    trainable = torch.tensor([1.0, -2.0], requires_grad=True)
    frozen = torch.tensor([3.0], requires_grad=True)
    trainable.grad = torch.tensor([0.25, -0.5])

    optimizer = optim.SimpleSGD([trainable, frozen], lr=0.1)
    optimizer.step()

    assert torch.allclose(trainable, torch.tensor([0.975, -1.95]))
    assert torch.allclose(frozen, torch.tensor([3.0]))


def test_simple_sgd_zero_grad_can_zero_or_remove_gradients():
    param = torch.tensor([1.0, 2.0], requires_grad=True)
    untouched = torch.tensor([3.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])

    optimizer = optim.SimpleSGD([param, untouched], lr=0.1)
    optimizer.zero_grad()

    assert param.grad is not None
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert untouched.grad is None

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.zero_grad(set_to_none=True)

    assert param.grad is None


def test_sgd_with_momentum_initializes_velocity_buffers():
    params = [torch.ones(2), torch.arange(3.0)]
    optimizer = optim.SGDWithMomentum(params, lr=0.1, momentum=0.9)

    assert optimizer.params == params
    assert optimizer.lr == 0.1
    assert optimizer.momentum == 0.9
    assert len(optimizer.velocity) == len(params)
    for velocity, param in zip(optimizer.velocity, params, strict=True):
        assert torch.equal(velocity, torch.zeros_like(param))


def test_sgd_with_momentum_accumulates_velocity_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = optim.SGDWithMomentum([param], lr=0.1, momentum=0.9)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.allclose(optimizer.velocity[0], torch.tensor([0.5, -0.25]))
    assert torch.allclose(param, torch.tensor([0.95, -1.975]))

    param.grad = torch.tensor([0.1, 0.2])
    optimizer.step()

    assert torch.allclose(optimizer.velocity[0], torch.tensor([0.55, -0.025]))
    assert torch.allclose(param, torch.tensor([0.895, -1.9725]))


def test_sgd_with_momentum_step_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = optim.SGDWithMomentum([trained, skipped], lr=0.1, momentum=0.9)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.95]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.velocity[1], torch.zeros_like(skipped))


def test_sgd_with_momentum_zero_grad_can_zero_or_remove_gradients():
    param = torch.tensor([1.0], requires_grad=True)
    param.grad = torch.tensor([0.5])
    optimizer = optim.SGDWithMomentum([param], lr=0.1, momentum=0.9)

    optimizer.zero_grad()

    assert param.grad is not None
    assert torch.equal(param.grad, torch.zeros_like(param))

    param.grad = torch.tensor([0.5])
    optimizer.zero_grad(set_to_none=True)

    assert param.grad is None

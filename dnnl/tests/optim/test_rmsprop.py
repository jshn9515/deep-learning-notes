import inspect

import torch

import dnnl.optim as dopt
import dnnl.optim.rmsprop as rmsprop


def test_rmsprop_module_has_docstrings():
    for name in rmsprop.__all__:
        member = getattr(rmsprop, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'


def test_rmsprop_public_export():
    assert dopt.RMSprop is rmsprop.RMSprop


def test_rmsprop_accumulates_squared_gradients_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.RMSprop([param], lr=0.1, rho=0.9, eps=0.0)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    expected_square_avg = torch.tensor([0.025, 0.00625])
    expected_effective_lr = torch.tensor([0.6324555, 1.264911])
    assert torch.allclose(optimizer.ema_of_sq_grads[0], expected_square_avg)
    assert torch.allclose(param, torch.tensor([0.6837722, -1.6837722]))
    assert torch.allclose(optimizer.get_effective_lr()[0], expected_effective_lr)


def test_rmsprop_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.RMSprop([trained, skipped], lr=0.1, rho=0.9, eps=0.0)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.6837722]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.ema_of_sq_grads[1], torch.zeros_like(skipped))
    assert torch.equal(optimizer.momentum_buffers[1], torch.zeros_like(skipped))


def test_rmsprop_accumulates_momentum_buffer():
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = dopt.RMSprop([param], lr=0.1, rho=0.9, eps=0.0, momentum=0.5)

    param.grad = torch.tensor([0.5])
    optimizer.step()

    assert torch.allclose(optimizer.ema_of_sq_grads[0], torch.tensor([0.025]))
    assert torch.allclose(optimizer.momentum_buffers[0], torch.tensor([3.1622777]))
    assert torch.allclose(param, torch.tensor([0.6837722]))

    param.grad = torch.tensor([0.25])
    optimizer.step()

    expected_ema = torch.tensor([0.02875])
    expected_buffer = 0.5 * torch.tensor([3.1622777]) + 0.25 / expected_ema.sqrt()
    assert torch.allclose(optimizer.ema_of_sq_grads[0], expected_ema)
    assert torch.allclose(optimizer.momentum_buffers[0], expected_buffer)
    assert torch.allclose(param, torch.tensor([0.6837722]) - 0.1 * expected_buffer)

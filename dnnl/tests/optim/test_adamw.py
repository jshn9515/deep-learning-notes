import inspect

import torch

import dnnl.optim as dopt
import dnnl.optim.adamw as adamw


def test_adamw_module_has_docstrings():
    for name in adamw.__all__:
        member = getattr(adamw, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'


def test_adamw_public_export():
    assert dopt.AdamW is adamw.AdamW


def test_adamw_applies_decoupled_weight_decay_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.AdamW(
        [param],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=0.0,
        weight_decay=0.2,
    )

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert optimizer.step_count == 1
    assert torch.allclose(optimizer.first_moments[0], torch.tensor([0.05, -0.025]))
    assert torch.allclose(
        optimizer.second_moments[0],
        torch.tensor([0.00025, 0.0000625]),
    )
    assert torch.allclose(param, torch.tensor([0.88, -1.86]))


def test_adamw_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.AdamW([trained, skipped], lr=0.1, eps=0.0, weight_decay=0.2)

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([0.88]))
    assert torch.allclose(skipped, torch.tensor([2.0]))
    assert torch.equal(optimizer.first_moments[1], torch.zeros_like(skipped))
    assert torch.equal(optimizer.second_moments[1], torch.zeros_like(skipped))

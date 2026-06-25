import inspect

import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt
import dnnlpy.optim.adamw as adamw


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

    state = optimizer.state[param]
    assert state['step'] == 1
    assert_close(state['exp_avg'], torch.tensor([0.05, -0.025]))
    assert_close(state['exp_avg_sq'], torch.tensor([0.00025, 0.0000625]))
    assert_close(param, torch.tensor([0.88, -1.86]))


def test_adamw_skips_parameters_without_gradients():
    trained = torch.tensor([1.0], requires_grad=True)
    skipped = torch.tensor([2.0], requires_grad=True)
    trained.grad = torch.tensor([0.5])
    optimizer = dopt.AdamW([trained, skipped], lr=0.1, eps=0.0, weight_decay=0.2)

    optimizer.step()

    assert_close(trained, torch.tensor([0.88]))
    assert_close(skipped, torch.tensor([2.0]))
    assert optimizer.state[skipped] == {}

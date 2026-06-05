import inspect
from collections.abc import Callable

import torch.optim as optim

import dnnl.optim as dopt


def _default(function: Callable, name: str):
    return inspect.signature(function).parameters[name].default


def test_optimizer_defaults_match_torch_optim_defaults():
    assert _default(dopt.SGD.__init__, 'lr') == _default(optim.SGD.__init__, 'lr')
    assert _default(dopt.SGD.__init__, 'momentum') == _default(
        optim.SGD.__init__,
        'momentum',
    )
    assert _default(dopt.SGD.__init__, 'weight_decay') == _default(
        optim.SGD.__init__,
        'weight_decay',
    )
    assert _default(dopt.SGD.__init__, 'nesterov') == _default(
        optim.SGD.__init__,
        'nesterov',
    )

    assert _default(dopt.Adagrad.__init__, 'lr') == _default(
        optim.Adagrad.__init__,
        'lr',
    )
    assert _default(dopt.Adagrad.__init__, 'eps') == _default(
        optim.Adagrad.__init__,
        'eps',
    )
    assert _default(dopt.Adagrad.__init__, 'weight_decay') == _default(
        optim.Adagrad.__init__,
        'weight_decay',
    )

    assert _default(dopt.RMSprop.__init__, 'lr') == _default(
        optim.RMSprop.__init__,
        'lr',
    )
    assert _default(dopt.RMSprop.__init__, 'rho') == _default(
        optim.RMSprop.__init__,
        'alpha',
    )
    assert _default(dopt.RMSprop.__init__, 'eps') == _default(
        optim.RMSprop.__init__,
        'eps',
    )
    assert _default(dopt.RMSprop.__init__, 'weight_decay') == _default(
        optim.RMSprop.__init__,
        'weight_decay',
    )

    assert _default(dopt.Adadelta.__init__, 'lr') == _default(
        optim.Adadelta.__init__,
        'lr',
    )
    assert _default(dopt.Adadelta.__init__, 'rho') == _default(
        optim.Adadelta.__init__,
        'rho',
    )
    assert _default(dopt.Adadelta.__init__, 'eps') == _default(
        optim.Adadelta.__init__,
        'eps',
    )
    assert _default(dopt.Adadelta.__init__, 'weight_decay') == _default(
        optim.Adadelta.__init__,
        'weight_decay',
    )

    assert _default(dopt.Adam.__init__, 'lr') == _default(optim.Adam.__init__, 'lr')
    assert _default(dopt.Adam.__init__, 'betas') == _default(
        optim.Adam.__init__,
        'betas',
    )
    assert _default(dopt.Adam.__init__, 'eps') == _default(
        optim.Adam.__init__,
        'eps',
    )
    assert _default(dopt.Adam.__init__, 'weight_decay') == _default(
        optim.Adam.__init__,
        'weight_decay',
    )

    assert _default(dopt.AdamW.__init__, 'lr') == _default(
        optim.AdamW.__init__,
        'lr',
    )
    assert _default(dopt.AdamW.__init__, 'betas') == _default(
        optim.AdamW.__init__,
        'betas',
    )
    assert _default(dopt.AdamW.__init__, 'eps') == _default(
        optim.AdamW.__init__,
        'eps',
    )
    assert _default(dopt.AdamW.__init__, 'weight_decay') == _default(
        optim.AdamW.__init__,
        'weight_decay',
    )


def test_utils_defaults_match_wrapped_optimizer_defaults():
    assert _default(dopt.run_adagrad, 'eps') == _default(dopt.Adagrad.__init__, 'eps')
    assert _default(dopt.run_rmsprop, 'eps') == _default(dopt.RMSprop.__init__, 'eps')
    assert _default(dopt.run_adadelta, 'eps') == _default(dopt.Adadelta.__init__, 'eps')
    assert _default(dopt.run_adam, 'betas') == _default(dopt.Adam.__init__, 'betas')
    assert _default(dopt.run_adam, 'eps') == _default(dopt.Adam.__init__, 'eps')
    assert _default(dopt.run_adamw, 'betas') == _default(dopt.AdamW.__init__, 'betas')
    assert _default(dopt.run_adamw, 'eps') == _default(dopt.AdamW.__init__, 'eps')
    assert _default(dopt.run_adamw, 'weight_decay') == _default(
        dopt.AdamW.__init__,
        'weight_decay',
    )

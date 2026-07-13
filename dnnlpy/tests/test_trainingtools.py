from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.testing import assert_close
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError

from dnnlpy.trainingtools import Trainer


def make_regression_loader() -> DataLoader:
    x = torch.linspace(-1.0, 1.0, steps=8).unsqueeze(1)
    y = 2 * x + 1
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_trainer_fits_plain_module_and_returns_metric_logs(capsys):
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(max_epochs=2, device='cpu', verbose=True)

    history = trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        train_metrics={'mse': MeanSquaredError()},
    )

    captured = capsys.readouterr()
    assert len(history) == 2
    assert 'Training on cpu' in captured.out
    assert 'Epoch [1/2]' in captured.out
    assert set(history[-1]) == {'train_loss', 'train_mse'}


def test_trainer_clips_gradients(monkeypatch):
    calls = []

    def fake_clip_grad_norm_(parameters, max_norm):
        calls.append((list(parameters), max_norm))

    monkeypatch.setattr(nn.utils, 'clip_grad_norm_', fake_clip_grad_norm_)

    model = nn.Linear(1, 1)
    trainer = Trainer(
        max_epochs=1,
        device='cpu',
        gradient_clip_val=0.5,
        verbose=False,
    )
    trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
    )

    assert calls
    assert calls[0][1] == 0.5


def test_trainer_configures_deterministic_seed(monkeypatch):
    calls = []

    def fake_set_seed(seed=42, *, deterministic=False, benchmark=False, warn_only=True):
        calls.append((seed, deterministic, benchmark, warn_only))
        return torch.default_generator

    monkeypatch.setattr('dnnlpy.trainingtools.set_seed', fake_set_seed)

    Trainer(
        max_epochs=1,
        device='cpu',
        seed=123,
        deterministic=True,
        benchmark=True,
        verbose=False,
    )

    assert calls == [(123, True, True, True)]


def test_trainer_steps_lr_scheduler_each_epoch():
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = lr.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(max_epochs=2, device='cpu', verbose=False)

    trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    assert optimizer.param_groups[0]['lr'] == 0.025


def test_trainer_uses_short_precision_names():
    trainer = Trainer(max_epochs=1, device='cpu', precision='bf16', verbose=False)

    assert trainer.precision == 'bf16'
    assert trainer._amp_dtype is torch.bfloat16


def test_trainer_repr_shows_configuration():
    trainer = Trainer(
        device='cpu',
        amp=True,
        precision='bf16',
        seed=123,
        deterministic=True,
        benchmark=True,
        gradient_clip_val=0.5,
        gradient_clip_algorithm='value',
        max_epochs=3,
        checkpoint_path='checkpoints',
        checkpoint_every_n_epochs=2,
        verbose=False,
    )

    text = repr(trainer)

    assert text.startswith('Trainer(')
    assert "device=device(type='cpu')" in text
    assert 'amp=True' in text
    assert "precision='bf16'" in text
    assert 'seed=123' in text
    assert 'deterministic=True' in text
    assert 'benchmark=True' in text
    assert 'gradient_clip_val=0.5' in text
    assert "gradient_clip_algorithm='value'" in text
    assert 'max_epochs=3' in text
    assert f'checkpoint_path={Path("checkpoints")!r}' in text
    assert 'checkpoint_every_n_epochs=2' in text
    assert 'verbose=False' in text
    assert 'history' not in text


def test_trainer_saves_and_loads_checkpoint(tmp_path):
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = lr.StepLR(optimizer, step_size=1)
    trainer = Trainer(
        max_epochs=1,
        device='cpu',
        checkpoint_path=tmp_path,
        verbose=False,
    )

    history = trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    checkpoint_path = tmp_path / 'epoch=1.pth'
    assert checkpoint_path.exists()

    restored_model = nn.Linear(1, 1)
    restored_optimizer = optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = lr.StepLR(restored_optimizer, step_size=1)
    restored_trainer = Trainer(max_epochs=1, device='cpu', verbose=False)

    checkpoint = restored_trainer.load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    assert checkpoint['epoch'] == 1
    assert restored_trainer.history == history
    assert restored_scheduler.state_dict() == lr_scheduler.state_dict()
    for restored, trained in zip(restored_model.parameters(), model.parameters()):
        assert_close(restored, trained)


def test_trainer_resume_from_checkpoint_continues_history(tmp_path):
    model = nn.Linear(1, 1)
    trainer = Trainer(
        max_epochs=1, device='cpu', checkpoint_path=tmp_path, verbose=False
    )
    trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
    )

    resumed_trainer = Trainer(max_epochs=2, device='cpu', verbose=False)
    resumed_trainer.fit(
        model,
        make_regression_loader(),
        loss_fn=nn.MSELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        resume_from_checkpoint=tmp_path / 'epoch=1.pth',
    )

    assert len(resumed_trainer.history) == 2

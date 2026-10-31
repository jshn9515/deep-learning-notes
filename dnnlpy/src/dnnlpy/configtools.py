__lazy_modules__ = ['torch', 'torch.accelerator']

import os
import sys
from functools import cache
from pathlib import Path

import platformdirs as pdi
import torch
import torch.accelerator as accl

__all__ = [
    'get_data_dir',
    'get_data_path',
    'get_default_device',
    'get_experiment_dir',
    'get_experiment_path',
    'get_num_workers',
    'has_gil',
    'set_seed',
]


@cache
def has_gil() -> bool:
    """Check if the current Python interpreter has a Global Interpreter Lock (GIL)."""
    if sys.version_info >= (3, 13):
        return sys._is_gil_enabled()
    return True


def set_seed(
    seed: int | None = None,
    *,
    deterministic: bool = False,
    benchmark: bool = False,
    warn_only: bool = True,
) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed (int, default: None): Seed value to apply to all supported random
            number generators.
        deterministic (bool, default: False): Whether to request deterministic
            PyTorch algorithms.
        benchmark (bool, default: False): Whether to enable cuDNN benchmark mode.
        warn_only (bool, default: True): Whether nondeterministic PyTorch
            operations should warn instead of raising an error.

    Returns:
        Generator: The PyTorch generator returned by :func:`torch.manual_seed`.
    """
    if seed is not None:
        torch_rng = torch.manual_seed(seed)
    else:
        torch_rng = torch.default_generator

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    return torch_rng


@cache
def get_default_device() -> torch.device:
    """Return the current accelerator device, or CPU when none is available."""
    device = accl.current_accelerator(check_available=True)
    if device is not None:
        index = accl.current_device_index()
        return torch.device(device.type, index)
    else:
        return torch.device('cpu')


@cache
def get_data_path() -> Path:
    """Return the dataset root directory, creating it when necessary.

    This function is similar to :func:`get_data_dir`, but it returns a Path object.
    The `DNNL_DATA_ROOT` environment variable overrides the default `~/datasets` location.
    """
    default_root = pdi.user_data_path('dnnlpy', appauthor=False)
    root = Path(os.getenv('DNNL_DATA_ROOT', default_root))
    root.mkdir(parents=True, exist_ok=True)
    return root


@cache
def get_data_dir() -> str:
    """Return the dataset root directory, creating it when necessary.

    This function is similar to :func:`get_data_path`, but it returns a string instead
    of a Path object. The `DNNL_DATA_ROOT` environment variable overrides the default
    `~/datasets` location.
    """
    root = get_data_path()
    return root.as_posix()


@cache
def get_experiment_path() -> Path:
    """Return the experiment root directory, creating it when necessary.

    This function is similar to :func:`get_experiment_dir`, but it returns a Path object.
    The ``DNNL_EXPERIMENT_ROOT`` environment variable overrides the default
    ``~/dnnlpy/experiments`` location.
    """
    default_root = pdi.user_data_path('dnnlpy', appauthor=False) / 'experiments'
    root = Path(os.getenv('DNNL_EXPERIMENT_ROOT', default_root))
    root.mkdir(parents=True, exist_ok=True)
    return root


@cache
def get_experiment_dir() -> str:
    """Return the experiment root directory, creating it when necessary.

    This function is similar to :func:`get_experiment_root`, but it returns a string instead
    of a Path object. The ``DNNL_EXPERIMENT_ROOT`` environment variable overrides the default
    ``~/dnnlpy/experiments`` location.
    """
    root = get_experiment_path()
    return root.as_posix()


def get_num_workers(num_workers: int | None = None) -> int:
    """Get the number of worker threads to use for parallel processing.

    Args:
        num_workers (int, optional): The number of workers to use. This function can
            automatically determine the number of workers based on whether the Python
            interpreter has a Global Interpreter Lock (GIL) and the number of CPU cores
            available. If None, the default will be all available CPU cores if GIL is not
            present, or 1/2 of the available CPU cores if GIL is present.

    Returns:
        int: The number of worker threads to use for parallel processing.
    """
    if num_workers is None:
        if sys.version_info >= (3, 13):
            num_workers = os.process_cpu_count() or 1
        else:
            num_workers = os.cpu_count() or 1

        if has_gil() and num_workers > 1:
            num_workers = max(1, num_workers // 2)

    return num_workers

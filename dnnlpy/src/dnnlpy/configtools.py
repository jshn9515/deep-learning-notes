import os
import random

import numpy as np
import torch
import torch.accelerator as accl

__all__ = [
    'set_seed',
    'get_default_device',
    'get_data_root',
]


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
        Generator: The PyTorch generator returned by `torch.manual_seed`.
    """
    random.seed(seed)
    np.random.seed(seed)

    if seed is not None:
        torch_rng = torch.manual_seed(seed)
    else:
        torch_rng = torch.default_generator

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    return torch_rng


def get_default_device() -> torch.device:
    """Return the current accelerator device, or CPU when none is available."""
    device = accl.current_accelerator(check_available=True)
    if device is not None:
        return device
    return torch.device('cpu')


def get_data_root() -> str:
    """Return the dataset root directory, creating it when necessary.

    The `DNNL_DATA_ROOT` environment variable overrides the default
    `~/datasets` location.
    """
    root = os.getenv('DNNL_DATA_ROOT', os.path.expanduser('~/datasets'))
    if not os.path.exists(root):
        os.mkdir(root)
    return root

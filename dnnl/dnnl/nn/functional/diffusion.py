import torch
from torch import Tensor

__all__ = [
    'add_noise_v1',
    'add_noise_v2',
    'denoise_v1',
    'denoise_v2',
]


def add_noise_v1(x0: Tensor, betas: Tensor) -> Tensor:
    """Iteratively add Gaussian noise according to a beta schedule."""
    xt = x0.clone()
    for beta in betas:
        noise = torch.randn_like(x0)
        xt = (1 - beta).sqrt() * xt + beta.sqrt() * noise
    return xt


def add_noise_v2(x0: Tensor, betas: Tensor, timestep: int) -> Tensor:
    """Sample ``x_t`` directly from ``x_0`` at a given diffusion timestep."""
    noise = torch.randn_like(x0)
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    return alpha_bars[timestep].sqrt() * x0 + (1 - alpha_bars[timestep]).sqrt() * noise


def denoise_v1(x0: Tensor, xt: Tensor, timestep: int, betas: Tensor) -> Tensor:
    """Compute one DDPM reverse mean step with deterministic variance offset."""
    alphas = 1.0 - betas
    alpha_t = alphas[timestep]
    alpha_bars = alphas.cumprod(dim=0)
    alpha_bar_t = alpha_bars[timestep]
    alpha_bar_prev_t = (
        alpha_bars[timestep - 1]
        if timestep > 0
        else torch.tensor(1.0, device=x0.device)
    )
    beta_t = betas[timestep]

    param1 = alpha_bar_prev_t.sqrt() * beta_t / (1 - alpha_bar_t)
    param2 = alpha_t.sqrt() * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
    mean = param1 * x0 + param2 * xt
    variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t

    if timestep > 0:
        return mean + variance.sqrt()
    return mean


def denoise_v2(x0: Tensor, xt: Tensor, timestep: int, betas: Tensor) -> Tensor:
    """Compute one DDPM reverse step and sample noise when ``timestep > 0``."""
    alphas = 1.0 - betas
    alpha_t = alphas[timestep]
    alpha_bars = alphas.cumprod(dim=0)
    alpha_bar_t = alpha_bars[timestep]
    alpha_bar_prev_t = (
        alpha_bars[timestep - 1]
        if timestep > 0
        else torch.tensor(1.0, device=x0.device)
    )
    beta_t = betas[timestep]

    param1 = alpha_bar_prev_t.sqrt() * beta_t / (1 - alpha_bar_t)
    param2 = alpha_t.sqrt() * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
    mean = param1 * x0 + param2 * xt
    variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t

    if timestep > 0:
        noise = torch.randn_like(x0)
        return mean + variance.sqrt() * noise
    return mean

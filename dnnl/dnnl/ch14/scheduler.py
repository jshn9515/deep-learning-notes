import torch
from torch import Tensor
from torch.types import Device

__all__ = [
    'add_noise_v1',
    'add_noise_v2',
    'denoise_v1',
    'denoise_v2',
    'DDPMScheduler',
]


def add_noise_v1(x0: Tensor, betas: Tensor) -> Tensor:
    xt = x0.clone()
    for beta in betas:
        noise = torch.randn_like(x0)
        xt = (1 - beta).sqrt() * xt + beta.sqrt() * noise
    return xt


def add_noise_v2(x0: Tensor, betas: Tensor, timestep: int) -> Tensor:
    noise = torch.randn_like(x0)
    t = timestep
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    xt = alpha_bars[t].sqrt() * x0 + (1 - alpha_bars[t]).sqrt() * noise
    return xt


def denoise_v1(x0: Tensor, xt: Tensor, timestep: int, betas: Tensor) -> Tensor:
    t = timestep
    alphas = 1.0 - betas
    alpha_t = alphas[t]
    alpha_bars = alphas.cumprod(dim=0)
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev_t = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
    beta_t = betas[t]

    param1 = alpha_bar_prev_t.sqrt() * beta_t / (1 - alpha_bar_t)
    param2 = alpha_t.sqrt() * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
    mean = param1 * x0 + param2 * xt
    variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t

    if t > 0:
        return mean + variance.sqrt()
    else:
        return mean


def denoise_v2(x0: Tensor, xt: Tensor, timestep: int, betas: Tensor) -> Tensor:
    t = timestep
    alphas = 1.0 - betas
    alpha_t = alphas[t]
    alpha_bars = alphas.cumprod(dim=0)
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev_t = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
    beta_t = betas[t]

    param1 = alpha_bar_prev_t.sqrt() * beta_t / (1 - alpha_bar_t)
    param2 = alpha_t.sqrt() * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
    mean = param1 * x0 + param2 * xt
    variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t

    if t > 0:
        # Add noise for all steps except the last one
        z = torch.randn_like(x0)
        return mean + variance.sqrt() * z
    else:
        return mean


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """Scheduler for the Denoising Diffusion Probabilistic Models (DDPM) that defines
        the noise schedule and provides a method to add noise to the original samples based
        on the time steps.

        Args:
            num_train_timesteps (int): The total number of time steps used during training,
                which determines the length of the noise schedule.
            beta_start (float): The starting value of the noise variance (beta) at time step 0.
            beta_end (float): The ending value of the noise variance (beta) at the final time step.
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)

        self.num_inference_steps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if original_samples.shape != noise.shape:
            raise AssertionError('original_samples and noise must have the same shape.')

        if timesteps.ndim != 1:
            raise AssertionError('timesteps must be a 1D tensor of shape (batch_size,)')

        self.alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        sqrt_alpha_bar = self.alphas_cumprod[timesteps].sqrt()
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_bar = (1.0 - self.alphas_cumprod)[timesteps].sqrt()
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)

        noisy_samples = (
            sqrt_alpha_bar * original_samples + sqrt_one_minus_alpha_bar * noise
        )
        return noisy_samples

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Device = 'cpu',
    ):
        if num_inference_steps > self.num_train_timesteps:
            raise AssertionError(
                f'num_inference_steps must be in the range (0, {self.num_train_timesteps}].'
            )

        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device,
        )

    def previous_timestep(self, timestep: int) -> int:
        if self.num_inference_steps != self.num_train_timesteps:
            index = (self.timesteps == timestep).float().argmax()
            if index == len(self.timesteps) - 1:
                prev = -1
            else:
                prev = int(self.timesteps[index + 1])
        else:
            prev = timestep - 1
        return prev

    def step(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """Perform a single reverse diffusion step to compute the previous sample given the
        model's output, the current time step, and the current sample.

        Args:
            model_output (Tensor): The output from the diffusion model, which is typically
                the predicted noise component at the current time step.
            timestep (int): The current time step in the reverse diffusion process.
            sample (Tensor): The current noisy sample at the given time step.
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        if prev_t >= 0:
            alpha_bar_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=sample.device)

        pred_original_sample = (
            sample - (1 - alpha_bar_t).sqrt() * model_output
        ) / alpha_bar_t.sqrt()

        param1 = alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t)
        param2 = alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = param1 * pred_original_sample + param2 * sample

        if prev_t >= 0:
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
        else:
            variance = torch.tensor(0.0, device=sample.device)

        if timestep > 0:
            noise = torch.randn_like(sample)
            prev_sample = mean + variance.sqrt() * noise
        else:
            prev_sample = mean

        return prev_sample

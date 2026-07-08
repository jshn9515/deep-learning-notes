import torch
from torch.testing import assert_close

from dnnlpy.models.ddpm import (
    DDPMScheduler,
    SinusoidalTimestepEmbedding,
    UNet2DModel,
    add_noise,
    denoise,
)


def test_add_noise_keeps_expected_shape():
    x0 = torch.ones(2, 1, 4, 4)
    betas = torch.tensor([0.1, 0.2, 0.3])

    torch.manual_seed(0)
    xt = add_noise(x0, betas, timestep=0)

    assert xt.shape == x0.shape


def test_denoise_is_deterministic_at_timestep_zero():
    x0 = torch.ones(2, 1, 4, 4)
    xt = torch.full_like(x0, 0.5)
    betas = torch.tensor([0.1, 0.2, 0.3])

    output = denoise(x0, xt, timestep=0, betas=betas)

    assert output.shape == x0.shape
    assert torch.isfinite(output).all()


def test_ddpm_scheduler_add_noise_and_timestep_schedule():
    scheduler = DDPMScheduler(num_train_timesteps=10, beta_start=0.1, beta_end=0.2)
    original_samples = torch.ones(2, 1, 2, 2)
    noise = torch.zeros_like(original_samples)
    timesteps = torch.tensor([0, 9])

    noisy = scheduler.add_noise(original_samples, noise, timesteps)
    expected = scheduler.alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)

    assert_close(noisy, expected * original_samples)

    scheduler.set_timesteps(5)
    assert scheduler.timesteps.tolist() == [9, 6, 4, 2, 0]
    assert scheduler.previous_timestep(9) == 6
    assert scheduler.previous_timestep(0) == -1


def test_ddpm_scheduler_step_at_timestep_zero_has_no_noise():
    scheduler = DDPMScheduler(num_train_timesteps=3, beta_start=0.1, beta_end=0.2)
    sample = torch.ones(2, 1, 2, 2)
    model_output = torch.zeros_like(sample)

    previous = scheduler.step(model_output, timestep=0, sample=sample)

    assert previous.shape == sample.shape
    assert torch.isfinite(previous).all()


def test_sinusoidal_embeddings_and_unet_forward_shapes():
    timesteps = torch.tensor([0, 1])
    embeddings = SinusoidalTimestepEmbedding(8)(timesteps)
    model = UNet2DModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(8, 16),
        time_emb_dim=16,
    )
    x = torch.randn(2, 1, 8, 8)

    output = model(x, timesteps)

    assert embeddings.shape == (2, 8)
    assert output.shape == x.shape

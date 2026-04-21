import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['VAE', 'vae_loss']


class VAE(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.latent_dim = latent_dim
        input_dim = math.prod(input_shape)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, input_shape),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = torch.addcmul(mu, std, eps)
        return latent

    def decode(self, z: Tensor) -> Tensor:
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        x_hat = self.decode(latent)
        return x_hat, mu, logvar


def vae_loss(
    x_hat: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    loss_fn: Literal['mse', 'bce'] = 'bce',
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if loss_fn == 'mse':
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    elif loss_fn == 'bce':
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    else:
        raise NotImplementedError(f'Unsupported loss function: {loss_fn}')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    batch_size = x.size(0)
    return loss / batch_size, recon_loss / batch_size, kl_loss / batch_size

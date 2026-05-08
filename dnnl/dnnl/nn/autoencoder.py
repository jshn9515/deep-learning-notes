import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['AutoEncoder', 'VAE']


class AutoEncoder(nn.Module):
    """A fully connected autoencoder for small image tensors."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        """Initialize encoder and decoder networks.

        Args:
            input_shape (tuple[int, int, int]): Per-sample input shape, excluding batch size.
            hidden_dim (int, default: 256): Width of the hidden fully connected layer.
            latent_dim (int, default: 32): Size of the latent representation.
        """
        super().__init__()
        self.input_shape = input_shape
        input_dim = math.prod(input_shape)
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),  # 28x28 -> 784
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
            nn.Unflatten(1, input_shape),  # 784 -> 28x28
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs into latent vectors."""
        z = self.encoder(x)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors back to input-shaped tensors."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct inputs through the encoder and decoder."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class VAE(nn.Module):
    """A fully connected variational autoencoder for small image tensors."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        """Initialize encoder, latent heads, and decoder.

        Args:
            input_shape (tuple[int, int, int]): Per-sample input shape, excluding batch size.
            hidden_dim (int, default: 256): Width of the hidden fully connected layer.
            latent_dim (int, default: 32): Size of the Gaussian latent distribution.
        """
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
        """Encode inputs into latent mean and log-variance tensors."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent vectors with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = torch.addcmul(mu, std, eps)
        return latent

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors back to input-shaped tensors."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct inputs and return reconstruction, mean, and log-variance."""
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        x_hat = self.decode(latent)
        return x_hat, mu, logvar

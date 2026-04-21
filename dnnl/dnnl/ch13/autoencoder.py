import math

import torch.nn as nn
from torch import Tensor

__all__ = ['AutoEncoder']


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
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
        z = self.encoder(x)
        return z

    def decode(self, z: Tensor) -> Tensor:
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

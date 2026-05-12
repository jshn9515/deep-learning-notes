import torch.nn as nn
from torch import Tensor

__all__ = [
    'ViTLinearPatchEmbedding',
    'ViTConvPatchEmbedding',
]


class ViTLinearPatchEmbedding(nn.Module):
    """Embed image patches by unfolding them and applying a linear projection."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        """Initialize an unfold-based patch embedding layer.

        Args:
            image_size (int): Height and width of the square input image.
            patch_size (int): Height and width of each square image patch.
            in_channels (int): Number of input image channels.
            embed_dim (int): Output embedding dimension for each patch.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise AssertionError('`image_size` must be divisible by `patch_size`.')

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Convert images of shape ``(batch, channels, height, width)`` to patch tokens."""
        patches = self.unfold(x)  # (B, C*P*P, N)
        patches = patches.transpose(1, 2)  # (B, N, C*P*P)
        output = self.proj(patches)
        return output


class ViTConvPatchEmbedding(nn.Module):
    """Embed image patches with a strided convolution."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        """Initialize a convolution-based patch embedding layer.

        Args:
            image_size (int): Height and width of the square input image.
            patch_size (int): Height and width of each square image patch.
            in_channels (int): Number of input image channels.
            embed_dim (int): Output embedding dimension for each patch.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise AssertionError('`image_size` must be divisible by `patch_size`.')

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert images of shape ``(batch, channels, height, width)`` to patch tokens."""
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

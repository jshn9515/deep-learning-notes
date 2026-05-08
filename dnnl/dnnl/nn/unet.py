import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiheadAttention
from .positional_encoding import SinusoidalTimestepEmbedding

__all__ = ['UNet2DModel']


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding='same',
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)  # (B, out_ch)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, num_channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, num_channels)
        self.attn = MultiheadAttention(num_channels, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.attn(h, h, h)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return x + attn_out


class Downsample(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            num_channels,
            num_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet2DModel(nn.Module):
    """A compact 2D U-Net with timestep conditioning and attention blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (64, 128, 256, 512),
        time_emb_dim: int = 256,
    ):
        """Initialize the U-Net down path, bottleneck, and up path.

        Args:
            in_channels (int, default: 3): Number of input image channels.
            out_channels (int, default: 3): Number of output image channels.
            block_out_channels (tuple[int, ...], default: (64, 128, 256, 512)):
                Channel sizes for each U-Net resolution level.
            time_emb_dim (int, default: 256): Dimension of timestep embeddings.
        """
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        first_ch = block_out_channels[0]
        self.init_conv = nn.Conv2d(
            in_channels,
            first_ch,
            kernel_size=3,
            padding=1,
        )

        # Down path
        self.downs = nn.ModuleList()
        in_ch = block_out_channels[0]
        skip_channels = []

        for out_ch in block_out_channels:
            is_last_ch = out_ch == block_out_channels[-1]
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(in_ch, out_ch, time_emb_dim),
                        ResBlock(out_ch, out_ch, time_emb_dim),
                        AttentionBlock(out_ch),
                        Downsample(out_ch) if not is_last_ch else nn.Identity(),
                    ]
                )
            )
            in_ch = out_ch
            skip_channels.append(out_ch)

        # Middle
        last_ch = block_out_channels[-1]
        self.mid_block1 = ResBlock(last_ch, last_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(last_ch)
        self.mid_block2 = ResBlock(last_ch, last_ch, time_emb_dim)

        # Up path
        self.ups = nn.ModuleList()
        in_ch = block_out_channels[-1]

        for out_ch in reversed(skip_channels):
            is_first_ch = out_ch == skip_channels[0]
            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(in_ch + out_ch, out_ch, time_emb_dim),
                        ResBlock(out_ch, out_ch, time_emb_dim),
                        AttentionBlock(out_ch),
                        Upsample(out_ch) if not is_first_ch else nn.Identity(),
                    ]
                )
            )
            in_ch = out_ch

        self.final_block = Block(in_ch, in_ch)
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        """Run the U-Net on a batch of images and timesteps."""
        if x.size(0) != timesteps.size(0):
            raise AssertionError(
                f'Batch size of x and timesteps must match, '
                f'but got {x.size(0)} and {timesteps.size(0)}.'
            )

        t_emb = self.time_embedding(timesteps)
        x = self.init_conv(x)

        skips = []
        for block1, block2, attn, down in self.downs:  # type: ignore
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, up in self.ups:  # type: ignore
            skip = skips.pop()
            x = torch.concat([x, skip], dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            x = up(x)

        x = self.final_block(x)
        x = self.final_conv(x)
        return x

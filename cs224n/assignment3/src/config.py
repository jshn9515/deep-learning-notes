from dataclasses import asdict, dataclass, field
from typing import Any

import dnnlpy.cs224n.assignment3 as a3
from torch.types import Device

import dnnlpy

__all__ = ['GPT2TrainingConfig']


DEVICE = dnnlpy.get_default_device()


@dataclass(kw_only=True)
class GPT2TrainingConfig:
    # Dataset parameters
    path: str = 'roneneldan/TinyStories'
    num_stories: int | str | None = None

    # Model parameters
    model_size: str = 'small'
    model_cfg: a3.GPT2Config = field(init=False)

    # Training parameters
    batch_size: int = 32
    chunk_size: int | None = None
    max_steps: int | None = None

    # Optimizer parameters
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_norm: float = 1.0  # gradient clipping

    # Other parameters
    seed: int | None = None
    device: Device = DEVICE

    def __post_init__(self) -> None:
        match self.model_size:
            case 'tiny':
                self.model_cfg = a3.GPT2_TINY
            case 'small':
                self.model_cfg = a3.GPT2_SMALL
            case 'medium':
                self.model_cfg = a3.GPT2_MEDIUM
            case 'large':
                self.model_cfg = a3.GPT2_LARGE
            case 'xl':
                self.model_cfg = a3.GPT2_XL
            case _:
                raise RuntimeError(f'Invalid model size: {self.model_size}.')

        if self.chunk_size is None:
            self.chunk_size = self.model_cfg.context_length
        else:
            if self.chunk_size < 2:
                raise AssertionError(
                    '`chunk_size` must be at least 2 for next-token prediction.'
                )
            if self.chunk_size > self.model_cfg.context_length + 1:
                raise AssertionError(
                    '`chunk_size` cannot exceed the model context length plus one target token.'
                )

        if self.seed is not None:
            dnnlpy.set_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

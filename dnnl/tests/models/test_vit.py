import pytest
import torch

from dnnl.models.vit import ViTConvPatchEmbedding, ViTLinearPatchEmbedding


def test_vit_linear_patch_embedding_returns_patch_tokens():
    module = ViTLinearPatchEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
    )
    x = torch.randn(2, 3, 8, 8)

    output = module(x)

    assert module.num_patches == 4
    assert output.shape == (2, 4, 6)


def test_vit_conv_patch_embedding_returns_patch_tokens():
    module = ViTConvPatchEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
    )
    x = torch.randn(2, 3, 8, 8)

    output = module(x)

    assert module.num_patches == 4
    assert output.shape == (2, 4, 6)


def test_vit_patch_embedding_requires_divisible_image_size():
    with pytest.raises(AssertionError, match='divisible'):
        ViTLinearPatchEmbedding(
            image_size=7,
            patch_size=4,
            in_channels=3,
            embed_dim=6,
        )

import pytest
import torch

from dnnl.models.vit import (
    ViTConvPatchEmbedding,
    ViTEmbedding,
    ViTLinearPatchEmbedding,
    ViTPositionalEmbedding,
    patchify,
)


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


def test_patchify_splits_images_into_flattened_patches():
    x = torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4)

    patches = patchify(x, patch_size=2)

    assert patches.shape == (2, 4, 4)
    assert torch.equal(patches[0, 0], torch.tensor([0, 1, 4, 5]))
    assert torch.equal(patches[0, 3], torch.tensor([10, 11, 14, 15]))


def test_patchify_requires_divisible_image_size():
    x = torch.randn(2, 3, 5, 4)

    with pytest.raises(AssertionError, match='divisible'):
        patchify(x, patch_size=2)


def test_vit_positional_embedding_adds_positions_and_validates_length():
    module = ViTPositionalEmbedding(num_patches=4, embed_dim=6, use_cls_token=True)
    x = torch.zeros(2, 5, 6)

    output = module(x)

    assert output.shape == x.shape
    assert torch.allclose(output, module.pos_embed.expand_as(output))

    with pytest.raises(AssertionError, match='Expected sequence length'):
        module(torch.zeros(2, 4, 6))


def test_vit_positional_embedding_interpolates_with_and_without_cls_token():
    with_cls = ViTPositionalEmbedding(num_patches=4, embed_dim=6, use_cls_token=True)
    without_cls = ViTPositionalEmbedding(
        num_patches=4,
        embed_dim=6,
        use_cls_token=False,
    )

    cls_output = with_cls.interpolate((2, 2), (3, 3))
    patch_output = without_cls.interpolate((2, 2), (3, 3))

    assert cls_output.shape == (1, 10, 6)
    assert patch_output.shape == (1, 9, 6)
    assert torch.allclose(cls_output[:, :1], with_cls.pos_embed[:, :1])


def test_vit_positional_embedding_rejects_mismatched_old_grid():
    module = ViTPositionalEmbedding(num_patches=4, embed_dim=6)

    with pytest.raises(AssertionError, match='Expected old grid'):
        module.interpolate((1, 3), (2, 2))


def test_vit_embedding_returns_patch_and_class_tokens():
    module = ViTEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
        dropout=0.0,
    )
    x = torch.randn(2, 3, 8, 8)

    output = module(x)
    resized_pos_embed = module.interpolate_pos_embedding((2, 2), (3, 3))

    assert output.shape == (2, 5, 6)
    assert resized_pos_embed.shape == (1, 10, 6)

import pytest
import torch

from dnnl.models.vit import (
    VisionTransformer,
    ViTClassificationHead,
    ViTConvPatchEmbedding,
    ViTEmbedding,
    ViTEncoder,
    ViTEncoderLayer,
    ViTLinearPatchEmbedding,
    ViTMLP,
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


def test_vit_mlp_uses_default_hidden_dimension():
    module = ViTMLP(embed_dim=6, dropout=0.0)
    x = torch.randn(2, 5, 6)

    output = module(x)

    assert module.net[0].out_features == 24
    assert output.shape == x.shape


def test_vit_encoder_layer_returns_token_sequence():
    module = ViTEncoderLayer(
        embed_dim=6,
        num_heads=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    x = torch.randn(2, 5, 6)

    output = module(x)

    assert output.shape == x.shape


def test_vit_encoder_stacks_layers_and_normalizes_output():
    module = ViTEncoder(
        embed_dim=6,
        num_heads=2,
        num_layers=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    x = torch.randn(2, 5, 6)

    output = module(x)

    assert len(module.layers) == 2
    assert output.shape == x.shape


def test_vit_classification_head_uses_class_token():
    module = ViTClassificationHead(embed_dim=3, num_classes=2)
    with torch.no_grad():
        module.head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        module.head.bias.zero_()
    x = torch.tensor(
        [
            [[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]],
            [[5.0, 7.0, 11.0], [13.0, 17.0, 19.0]],
        ]
    )

    output = module(x)

    assert torch.equal(output, torch.tensor([[2.0, 3.0], [5.0, 7.0]]))


def test_vision_transformer_returns_class_logits():
    module = VisionTransformer(
        image_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=7,
        embed_dim=6,
        num_heads=2,
        num_layers=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    x = torch.randn(2, 3, 8, 8)

    output = module(x)

    assert output.shape == (2, 7)

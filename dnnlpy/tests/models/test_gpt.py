import pytest
import torch
import torch.nn.functional as F

from dnnlpy.models.gpt import MiniGPT


def test_minigpt_forward_returns_vocab_logits():
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 17, (2, 5))

    logits = model(input_ids)

    assert logits.shape == (2, 5, 17)


def test_minigpt_ties_token_embedding_and_lm_head_by_default():
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )

    assert model.weight_tying is True
    assert model.lm_head.weight is model.token_embed.weight


def test_minigpt_can_disable_weight_tying():
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
        weight_tying=False,
    )

    assert model.weight_tying is False
    assert model.lm_head.weight is not model.token_embed.weight


def test_minigpt_forward_rejects_sequences_past_block_size():
    model = MiniGPT(
        vocab_size=17,
        block_size=4,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 17, (2, 5))

    with pytest.raises(AssertionError, match='exceeds block_size'):
        model(input_ids)


def test_minigpt_loss_returns_scalar():
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 17, (2, 5))

    loss = model.loss(input_ids)

    assert loss.ndim == 0


def test_minigpt_loss_shifts_inputs_when_targets_are_omitted():
    torch.manual_seed(0)
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 17, (2, 5))

    loss = model.loss(input_ids)
    logits = model(input_ids)[:, :-1, :]
    expected = F.cross_entropy(
        logits.reshape(-1, 17),
        input_ids[:, 1:].reshape(-1),
    )

    assert torch.allclose(loss, expected)


def test_minigpt_loss_uses_provided_targets_without_shifting():
    torch.manual_seed(0)
    model = MiniGPT(
        vocab_size=17,
        block_size=6,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 17, (2, 5))
    targets = torch.randint(0, 17, (2, 5))

    loss = model.loss(input_ids, targets)
    expected = F.cross_entropy(
        model(input_ids).reshape(-1, 17),
        targets.reshape(-1),
    )

    assert torch.allclose(loss, expected)


def test_minigpt_generate_returns_greedy_next_token_from_last_position():
    model = MiniGPT(
        vocab_size=5,
        block_size=3,
        embed_dim=8,
        num_layers=1,
        num_heads=2,
        hidden_dim=16,
        dropout=0.0,
    )
    logits = torch.tensor(
        [
            [[0.0, 9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 7.0, 0.0, 0.0]],
            [[8.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 6.0, 0.0]],
        ]
    )

    next_token = model.generate(logits, greedy=True)

    assert next_token.tolist() == [[2], [3]]

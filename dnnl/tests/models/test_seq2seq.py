import torch

from dnnl.models.seq2seq import Seq2SeqTransformer


def test_seq2seq_transformer_forward_returns_vocab_logits():
    model = Seq2SeqTransformer(
        src_vocab_size=11,
        tgt_vocab_size=13,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_len=6,
    )
    src = torch.randint(0, 11, (2, 4))
    tgt = torch.randint(0, 13, (2, 5))

    logits = model(src, tgt)

    assert logits.shape == (2, 5, 13)

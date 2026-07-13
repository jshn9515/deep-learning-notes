from typing import cast

from tokenizers import Tokenizer
from tokenizers.models import BPE

import dnnlpy.tokenizers as dltk
import dnnlpy.tokenizers.base as dltk_base


def test_bpe_encode_matches_hf_tokenizers():
    vocab = {
        '<unk>': 0,
        'l': 1,
        'o': 2,
        'w': 3,
        'lo': 4,
        'low': 5,
        'e': 6,
        'r': 7,
        'er': 8,
        'lower': 9,
    }
    merges = [('l', 'o'), ('lo', 'w'), ('e', 'r'), ('low', 'er')]

    tokenizer = dltk.Tokenizer(
        dltk.BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    text = 'lower'

    actual = tokenizer.encode(text, add_special_tokens=False)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)
    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_encode_unknown_piece_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]

    tokenizer = dltk.Tokenizer(
        dltk.BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    text = 'lowx'

    actual = tokenizer.encode(text, add_special_tokens=False)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)
    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_decode_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]

    tokenizer = dltk.Tokenizer(
        dltk.BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    ids = [5, 0]

    actual = tokenizer.decode(ids, skip_special_tokens=False)
    expected = hf_tokenizer.decode(ids, skip_special_tokens=False)
    assert actual == expected


def test_bpe_train_from_iterator_accepts_batches():
    tokenizer = dltk.Tokenizer(
        dltk.BPE(), normalizer=dltk.StripNormalizer(), num_workers=1
    )
    texts = [[' low ', ' lower '], [' lowest ']]

    tokenizer.train_from_iterator(texts, vocab_size=20)

    assert 'low' in tokenizer.vocab
    assert 'lower' in tokenizer.vocab


def test_bpe_train_from_iterator_includes_initial_alphabet():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['z'])

    assert 'z' in tokenizer.vocab


def test_bpe_train_from_iterator_uses_first_initial_alphabet_character():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['xyz'])

    assert 'x' in tokenizer.vocab
    assert 'xyz' not in tokenizer.vocab


def test_bpe_train_records_merge_when_merged_token_already_exists():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(
        ['abab'],
        vocab_size=6,
        special_tokens=['<unk>', 'ab'],
    )

    model = cast(dltk.BPE, tokenizer.model)

    assert hasattr(tokenizer.model, 'merges')
    assert model.merges == [('a', 'b'), ('ab', 'ab')]
    assert tokenizer.vocab == {'<unk>': 0, 'ab': 1, 'a': 2, 'b': 3, 'abab': 4}


def test_bpe_train_does_not_duplicate_existing_vocab_token():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(
        ['ab'],
        vocab_size=6,
        special_tokens=['<unk>', 'ab'],
    )

    model = cast(dltk.BPE, tokenizer.model)

    assert hasattr(tokenizer.model, 'merges')
    assert model.merges == [('a', 'b')]
    assert list(tokenizer.vocab) == ['<unk>', 'ab', 'a', 'b']


def test_bpe_training_ignores_empty_texts():
    tokenizer = dltk.Tokenizer(
        dltk.BPE(),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(),
        num_workers=1,
    )

    tokenizer.train_from_iterator(['', 'low lower', ''], vocab_size=20)

    model = cast(dltk.BPE, tokenizer.model)

    assert model.merges


def test_pre_token_counts_are_processed_in_bounded_batches(monkeypatch):
    batch_sizes = []

    def serial_parallel_map(func, values, num_workers):
        assert num_workers == 2
        for value in values:
            batch_sizes.append(len(value))
            yield func(value)

    monkeypatch.setattr(dltk_base, 'parallel_map', serial_parallel_map)

    pre_tokenizer = dltk.ByteLevelPreTokenizer()
    tokenizer = dltk.Tokenizer(dltk.BPE(), pre_tokenizer=pre_tokenizer, num_workers=2)
    texts = ['hello'] * 2050

    counts = tokenizer._count_pre_tokens(texts)

    assert counts == pre_tokenizer.count_tokens(texts)
    assert batch_sizes == [1024, 1024, 2]

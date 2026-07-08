from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE as HFBPE

from dnnlpy.tokenizers import BPE, StripNormalizer, Tokenizer


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
    tokenizer = Tokenizer(
        BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = HFTokenizer(HFBPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    text = 'lower'

    actual = tokenizer.encode(text, add_special_tokens=False)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)
    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_encode_unknown_piece_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]
    tokenizer = Tokenizer(
        BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = HFTokenizer(HFBPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    text = 'lowx'

    actual = tokenizer.encode(text, add_special_tokens=False)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)
    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_decode_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]
    tokenizer = Tokenizer(
        BPE(vocab=vocab, merges=merges, unk_token='<unk>'),
        num_workers=1,
    )
    hf_tokenizer = HFTokenizer(HFBPE(vocab=vocab, merges=merges, unk_token='<unk>'))

    ids = [5, 0]

    actual = tokenizer.decode(ids, skip_special_tokens=False)
    expected = hf_tokenizer.decode(ids, skip_special_tokens=False)
    assert actual == expected


def test_bpe_train_from_iterator_accepts_batches():
    tokenizer = Tokenizer(BPE(), normalizer=StripNormalizer(), num_workers=1)
    texts = [[' low ', ' lower '], [' lowest ']]

    tokenizer.train_from_iterator(texts, vocab_size=20)

    assert 'low' in tokenizer.vocab
    assert 'lower' in tokenizer.vocab


def test_bpe_train_from_iterator_includes_initial_alphabet():
    tokenizer = Tokenizer(BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['z'])

    assert 'z' in tokenizer.vocab


def test_bpe_train_from_iterator_uses_first_initial_alphabet_character():
    tokenizer = Tokenizer(BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['xyz'])

    assert 'x' in tokenizer.vocab
    assert 'xyz' not in tokenizer.vocab


def test_bpe_train_records_merge_when_merged_token_already_exists():
    tokenizer = Tokenizer(BPE(), num_workers=1)

    tokenizer.train_from_iterator(
        ['abab'],
        vocab_size=6,
        special_tokens=['<unk>', 'ab'],
    )

    assert tokenizer.model.merges == [('a', 'b'), ('ab', 'ab')]
    assert tokenizer.vocab == {'<unk>': 0, 'ab': 1, 'a': 2, 'b': 3, 'abab': 4}


def test_bpe_train_does_not_duplicate_existing_vocab_token():
    tokenizer = Tokenizer(BPE(), num_workers=1)

    tokenizer.train_from_iterator(
        ['ab'],
        vocab_size=6,
        special_tokens=['<unk>', 'ab'],
    )

    assert tokenizer.model.merges == [('a', 'b')]
    assert list(tokenizer.vocab) == ['<unk>', 'ab', 'a', 'b']

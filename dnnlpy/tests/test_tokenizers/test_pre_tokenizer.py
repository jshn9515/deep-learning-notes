from collections import Counter

from tokenizers.pre_tokenizers import ByteLevel, Whitespace

import dnnlpy.tokenizers as dltk


def test_byte_level_pre_tokenizer_matches_hf_tokenizers():
    pre_tokenizer = dltk.ByteLevelPreTokenizer(add_prefix_space=False)
    hf_pre_tokenizer = ByteLevel(add_prefix_space=False)

    text = 'Once upon a time!'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_pre_tokenizer_adds_virtual_prefix_space():
    pre_tokenizer = dltk.ByteLevelPreTokenizer(add_prefix_space=True)
    hf_pre_tokenizer = ByteLevel(add_prefix_space=True)

    text = 'Once upon'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_pre_tokenizer_can_skip_regex_splitting():
    pre_tokenizer = dltk.ByteLevelPreTokenizer(use_regex=False)
    hf_pre_tokenizer = ByteLevel(use_regex=False)

    text = 'Once upon!'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_token_only_pre_tokenization_matches_tokens_with_offsets():
    cases = [
        (dltk.ByteLevelPreTokenizer(add_prefix_space=False), 'Once upon a time!'),
        (dltk.ByteLevelPreTokenizer(add_prefix_space=True), 'Once upon'),
        (dltk.ByteLevelPreTokenizer(use_regex=False), 'Once upon!'),
    ]

    for pre_tokenizer, text in cases:
        actual = pre_tokenizer.pre_tokenize_tokens(text)
        expected = [token for token, _ in pre_tokenizer.pre_tokenize(text)]

        assert actual == expected


def test_byte_level_pre_tokenizer_accepts_empty_text():
    pre_tokenizer = dltk.ByteLevelPreTokenizer(add_prefix_space=True)

    assert pre_tokenizer.pre_tokenize('') == []
    assert pre_tokenizer.pre_tokenize_tokens('') == []


def test_byte_level_token_counts_match_individual_pre_tokenization():
    texts = ['Once upon a time!', 'Once upon another time!']

    for pre_tokenizer in (
        dltk.ByteLevelPreTokenizer(add_prefix_space=False),
        dltk.ByteLevelPreTokenizer(add_prefix_space=True),
        dltk.ByteLevelPreTokenizer(use_regex=False),
    ):
        expected = Counter()
        for text in texts:
            for token, _ in pre_tokenizer.pre_tokenize(text):
                expected[token] += 1

        actual = pre_tokenizer.count_tokens(texts)
        assert actual == expected


def test_byte_level_alphabet_matches_hf_tokenizers():
    actual = dltk.ByteLevelPreTokenizer.alphabet()
    expected = ByteLevel.alphabet()

    assert len(actual) == 256
    assert len(set(actual)) == 256
    assert set(actual) == set(expected)


def test_whitespace_pre_tokenizer_matches_hf_tokenizers():
    pre_tokenizer = dltk.WhitespacePreTokenizer()
    hf_pre_tokenizer = Whitespace()

    text = "Hello, y'all! How are you?"

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_whitespace_pre_tokenizer_ignores_surrounding_whitespace():
    pre_tokenizer = dltk.WhitespacePreTokenizer()
    hf_pre_tokenizer = Whitespace()

    text = '  deep   learning\nnotes  '

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_default_token_only_pre_tokenization_discards_offsets():
    pre_tokenizer = dltk.WhitespacePreTokenizer()

    actual = pre_tokenizer.pre_tokenize_tokens('deep learning')
    expected = ['deep', 'learning']
    assert actual == expected


def test_default_pre_tokenizer_token_counts():
    pre_tokenizer = dltk.WhitespacePreTokenizer()

    actual = pre_tokenizer.count_tokens(['deep learning', 'deep'])
    expected = Counter({'deep': 2, 'learning': 1})
    assert actual == expected

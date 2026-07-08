from tokenizers.pre_tokenizers import (
    ByteLevel as HFByteLevel,
    Whitespace as HFWhitespace,
)

from dnnlpy.tokenizers.pre_tokenizer import (
    ByteLevelPreTokenizer,
    WhitespacePreTokenizer,
)


def test_byte_level_pre_tokenizer_matches_hf_tokenizers():
    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    hf_pre_tokenizer = HFByteLevel(add_prefix_space=False)

    text = 'Once upon a time!'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_pre_tokenizer_adds_virtual_prefix_space():
    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True)
    hf_pre_tokenizer = HFByteLevel(add_prefix_space=True)

    text = 'Once upon'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_pre_tokenizer_can_skip_regex_splitting():
    pre_tokenizer = ByteLevelPreTokenizer(use_regex=False)
    hf_pre_tokenizer = HFByteLevel(use_regex=False)

    text = 'Once upon!'

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_byte_level_alphabet_matches_hf_tokenizers():
    actual = ByteLevelPreTokenizer.alphabet()
    expected = HFByteLevel.alphabet()

    assert len(actual) == 256
    assert len(set(actual)) == 256
    assert set(actual) == set(expected)


def test_whitespace_pre_tokenizer_matches_hf_tokenizers():
    pre_tokenizer = WhitespacePreTokenizer()
    hf_pre_tokenizer = HFWhitespace()

    text = "Hello, y'all! How are you?"

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected


def test_whitespace_pre_tokenizer_ignores_surrounding_whitespace():
    pre_tokenizer = WhitespacePreTokenizer()
    hf_pre_tokenizer = HFWhitespace()

    text = '  deep   learning\nnotes  '

    actual = pre_tokenizer.pre_tokenize(text)
    expected = hf_pre_tokenizer.pre_tokenize_str(text)
    assert actual == expected

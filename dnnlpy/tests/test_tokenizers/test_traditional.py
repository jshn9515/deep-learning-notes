import pytest

import dnnlpy.tokenizers as dltk


@pytest.mark.parametrize(
    ('tokenizer_class', 'text', 'expected_vocab'),
    [
        (
            dltk.CharacterTokenizer,
            ['deep', 'learning'],
            {
                '<unk>': 0,
                'a': 1,
                'd': 2,
                'e': 3,
                'g': 4,
                'i': 5,
                'l': 6,
                'n': 7,
                'p': 8,
                'r': 9,
            },
        ),
        (
            dltk.WordTokenizer,
            ['deep learning', 'deep networks'],
            {'<unk>': 0, 'deep': 1, 'learning': 2, 'networks': 3},
        ),
    ],
)
def test_from_text_builds_sorted_vocab_from_multiple_texts(
    tokenizer_class: dltk.TraditionalTokenizer,
    text: str | list[str],
    expected_vocab: dict[str, int],
):
    tokenizer = tokenizer_class.from_text(text)

    assert tokenizer.vocab == expected_vocab


def test_character_tokenizer_encodes_and_decodes_characters():
    tokenizer = dltk.CharacterTokenizer.from_text('deep')

    ids = tokenizer.encode('peed')

    assert ids == [tokenizer.vocab[token] for token in 'peed']
    assert tokenizer.decode(ids) == 'peed'


def test_word_tokenizer_encodes_and_decodes_whitespace_separated_words():
    tokenizer = dltk.WordTokenizer.from_text('deep learning')

    ids = tokenizer.encode('  deep\tlearning  ')

    assert ids == [tokenizer.vocab['deep'], tokenizer.vocab['learning']]
    assert tokenizer.decode(ids) == 'deep learning'


@pytest.mark.parametrize(
    ('tokenizer', 'text', 'expected'),
    [
        (dltk.CharacterTokenizer.from_text('deep'), 'x', ''),
        (dltk.WordTokenizer.from_text('deep learning'), 'unknown', ''),
    ],
)
def test_unknown_tokens_use_unk_token(
    tokenizer: dltk.TraditionalTokenizer, text: str, expected: str
):
    ids = tokenizer.encode(text)

    assert ids == [tokenizer.unk_id]
    assert tokenizer.decode(ids) == expected
    assert tokenizer.decode(ids, skip_special_tokens=False) == '<unk>'


@pytest.mark.parametrize(
    ('tokenizer', 'text', 'expected_without_special', 'expected_with_special'),
    [
        (dltk.CharacterTokenizer.from_text('ab'), 'a^b', 'ab', 'a^b'),
        (
            dltk.WordTokenizer.from_text('deep learning'),
            'deep <eos>',
            'deep',
            'deep <eos>',
        ),
    ],
)
def test_decode_can_include_or_skip_added_special_tokens(
    tokenizer: dltk.TraditionalTokenizer,
    text: str,
    expected_without_special: str,
    expected_with_special: str,
):
    special_token = '^' if isinstance(tokenizer, dltk.CharacterTokenizer) else '<eos>'
    tokenizer.add_special_tokens([special_token])
    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == expected_without_special
    assert tokenizer.decode(ids, skip_special_tokens=False) == expected_with_special


@pytest.mark.parametrize(
    'tokenizer_class', [dltk.CharacterTokenizer, dltk.WordTokenizer]
)
def test_constructor_requires_unk_token_in_vocab(
    tokenizer_class: type[dltk.TraditionalTokenizer],
):
    with pytest.raises(KeyError, match='Unknown token'):
        tokenizer_class({'known': 0})

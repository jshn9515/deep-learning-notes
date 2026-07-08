from tokenizers.decoders import ByteLevel as HFByteLevel

from dnnlpy.tokenizers.decoder import ByteLevelDecoder


def test_byte_level_decoder_matches_hf_tokenizers():
    decoder = ByteLevelDecoder()
    hf_decoder = HFByteLevel()

    tokens = ['Once', 'Ġupon', 'Ġa', 'Ġtime', '!']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_restores_unicode_text():
    decoder = ByteLevelDecoder()
    hf_decoder = HFByteLevel()

    tokens = ['CafÃ©', 'ĠðŁĺģ']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_restores_leading_space():
    decoder = ByteLevelDecoder()
    hf_decoder = HFByteLevel()

    tokens = ['ĠHello']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_concatenates_tokens_without_separator():
    decoder = ByteLevelDecoder()
    hf_decoder = HFByteLevel()

    tokens = ['hello', 'world']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected

from tokenizers.normalizers import (
    ByteLevel as HFByteLevel,
    Lowercase as HFLowercase,
    Strip as HFStrip,
)

from dnnlpy.tokenizers.normalizer import (
    ByteLevelNormalizer,
    LowercaseNormalizer,
    StripNormalizer,
)


def test_byte_level_normalizer_matches_hf_tokenizers():
    normalizer = ByteLevelNormalizer()
    hf_normalizer = HFByteLevel()

    text = 'Café 😁'

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_lowercase_normalizer_matches_hf_tokenizers():
    normalizer = LowercaseNormalizer()
    hf_normalizer = HFLowercase()

    text = 'Hello WORLD'

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_matches_hf_tokenizers():
    normalizer = StripNormalizer()
    hf_normalizer = HFStrip()

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_strip_left_only():
    normalizer = StripNormalizer(left=True, right=False)
    hf_normalizer = HFStrip(left=True, right=False)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_strip_right_only():
    normalizer = StripNormalizer(left=False, right=True)
    hf_normalizer = HFStrip(left=False, right=True)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_disable_stripping():
    normalizer = StripNormalizer(left=False, right=False)
    hf_normalizer = HFStrip(left=False, right=False)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected

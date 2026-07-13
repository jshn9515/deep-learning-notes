from collections import Counter
from collections.abc import Iterable
from typing import override

import regex as re

from .base import PreTokenizer
from .utils import BYTE_TO_UNICODE, bytes_to_unicode

type Offset = tuple[int, int]

__all__ = [
    'ByteLevelPreTokenizer',
    'WhitespacePreTokenizer',
]

WHITESPACE = re.compile(
    r'\w+'
    r'|[^\w\s]+'
)
GPT2PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r'| ?\p{L}+'
    r'| ?\p{N}+'
    r'| ?[^\s\p{L}\p{N}]+'
    r'|\s+(?!\S)'
    r'|\s+'
)


class ByteLevelPreTokenizer(PreTokenizer):
    """Byte-level pre-tokenizer."""

    def __init__(
        self,
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        use_regex: bool = True,
    ):
        """Create a byte-level pre-tokenizer.

        Args:
            add_prefix_space (bool, default: False): Add one leading space when the input
                does not already start with whitespace.
            trim_offsets (bool, default: True): Trim leading and trailing whitespace offsets
                from produced tokens.
            use_regex (bool, default: True): Use regex to split text into tokens.
        """
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets
        self.use_regex = use_regex

    @staticmethod
    def alphabet() -> list[str]:
        """Return the byte-level Unicode alphabet."""
        return list(BYTE_TO_UNICODE.values())

    @override
    def pre_tokenize(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Pre-tokenize text before converting each piece to UTF-8 bytes."""
        if not text:
            return []  # Return an empty list for empty input

        original_text = text
        original_length = len(original_text)
        add_prefix_space = self.add_prefix_space and not text[0].isspace()

        if add_prefix_space:
            text = f' {text}'

        prefix_shift = 1 if add_prefix_space else 0

        if self.use_regex:
            pieces = [
                (match.group(), match.start(), match.end())
                for match in GPT2PATTERN.finditer(text)
            ]
        else:
            pieces = [(text, 0, len(text))]

        output = []
        for token, start, end in pieces:
            offset_start = max(0, start - prefix_shift)
            offset_end = max(0, end - prefix_shift)

            offset_start = min(offset_start, original_length)
            offset_end = min(offset_end, original_length)

            token = bytes_to_unicode(token)
            output.append((token, (offset_start, offset_end)))

        return output

    @override
    def pre_tokenize_tokens(self, text: str) -> list[str]:
        """Pre-tokenize text without computing offsets."""
        if not text:
            return []  # Return an empty list for empty input

        if self.add_prefix_space and not text[0].isspace():
            text = f' {text}'

        if not self.use_regex:
            return [bytes_to_unicode(text)]

        return list(map(bytes_to_unicode, GPT2PATTERN.findall(text)))

    @override
    def count_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count byte-level tokens while converting each unique piece only once."""
        raw_counts = Counter()

        for text in texts:
            if text and self.add_prefix_space and not text[0].isspace():
                text = f' {text}'

            if self.use_regex:
                raw_counts.update(GPT2PATTERN.findall(text))
            else:
                raw_counts[text] += 1

        return Counter(
            {
                bytes_to_unicode(token): frequency
                for token, frequency in raw_counts.items()
            }
        )


class WhitespacePreTokenizer(PreTokenizer):
    """Split text into non-whitespace spans."""

    @override
    def pre_tokenize(self, text: str) -> list[tuple[str, Offset]]:
        """Pre-tokenize text on whitespace."""
        if not text:
            return []  # Return an empty list for empty input

        return [(match.group(), match.span()) for match in WHITESPACE.finditer(text)]

    @override
    def pre_tokenize_tokens(self, text: str) -> list[str]:
        """Pre-tokenize text on whitespace without computing offsets."""
        if not text:
            return []  # Return an empty list for empty input

        return WHITESPACE.findall(text)

    @override
    def count_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count whitespace tokens while converting each unique piece only once."""
        raw_counts = Counter()

        for text in texts:
            raw_counts.update(WHITESPACE.findall(text))

        return raw_counts

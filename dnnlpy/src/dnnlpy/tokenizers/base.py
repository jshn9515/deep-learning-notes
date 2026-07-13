from __future__ import annotations

import itertools as it
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from typing import Self, Sequence

from .utils import get_num_workers, parallel_map

type Offset = tuple[int, int]

__all__ = [
    'Decoder',
    'Encoding',
    'Model',
    'Normalizer',
    'PostProcessor',
    'PreTokenizer',
    'Tokenizer',
    'TraditionalTokenizer',
    'Trainer',
]


class Normalizer(ABC):
    """Abstract base class for normalizers."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize a string."""


class PreTokenizer(ABC):
    """Abstract base class for pre-tokenizers."""

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[tuple[str, Offset]]:
        """Pre-tokenize a string into a list of (token, (start, end)) tuples."""

    def pre_tokenize_tokens(self, text: str) -> list[str]:
        """Pre-tokenize a string without retaining offsets."""
        if not text:
            return []  # Return an empty list for empty input
        return [token for token, _ in self.pre_tokenize(text)]

    def count_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count pre-tokenized strings across an iterable of texts."""
        counts = Counter()
        for text in texts:
            counts.update(self.pre_tokenize_tokens(text))
        return counts


class Model(ABC):
    """Abstract base class for tokenizer models."""

    unk_token: str
    vocab: dict[str, int] | None = None

    @abstractmethod
    def encode(
        self,
        tokenizer: Tokenizer,
        text: str | Sequence[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding: ...

    @abstractmethod
    def decode(
        self,
        tokenizer: Tokenizer,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str: ...

    @abstractmethod
    def train_from_iterator(
        self,
        tokenizer: Tokenizer,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        special_tokens: list[str] | None = None,
        initial_alphabet: list[str] | None = None,
    ) -> None: ...


class Trainer(ABC):
    """Abstract base class for tokenizer trainers."""

    @abstractmethod
    def train(self, inputs: Iterable[str | Iterable[str]]) -> None:
        """Train the tokenizer model from an iterator of input texts."""


class Decoder(ABC):
    """Decoder protocol for decoding tokens back into text."""

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """Decode tokens back into text."""


class PostProcessor(ABC):
    """Abstract base class for post-processors."""

    @abstractmethod
    def process(self, encoding: Encoding) -> Encoding:
        """Process an encoding and return a new encoding."""


class Encoding:
    def __init__(
        self,
        ids: list[int],
        tokens: list[str],
        offsets: list[tuple[int, int]] | None = None,
        type_ids: list[int] | None = None,
        attention_mask: list[int] | None = None,
        special_tokens_mask: list[int] | None = None,
    ):
        """Initialize an Encoding object.

        Args:
            ids (list[int]): List of token IDs.
            tokens (list[str]): List of token strings.
            offsets (list[tuple[int, int]], optional): List of (start, end) offsets for
                each token. Defaults to None, which will create offsets of (0, 0).
            type_ids (list[int], optional): List of type IDs for each token. Defaults to
                None, which will create type IDs of 0.
            attention_mask (list[int], optional): List of attention mask values for each
                token. Defaults to None, which will create attention masks of 1.
            special_tokens_mask (list[int], optional): List of special token mask values
                for each token. Defaults to None, which will create special token masks of 0.
        """
        self.ids = ids
        self.tokens = tokens
        self.offsets = offsets or [(0, 0)] * len(ids)
        self.type_ids = type_ids or [0] * len(ids)
        self.attention_mask = attention_mask or [1] * len(ids)
        self.special_tokens_mask = special_tokens_mask or [0] * len(ids)

    def __len__(self) -> int:
        return len(self.ids)


class TraditionalTokenizer(ABC):
    """Base class for traditional tokenizers that split text into words or characters.

    A tokenizer owns a vocabulary mapping string tokens to integer IDs and
    defines the common encode/decode interface. Subclasses decide how text is
    split into tokens, while this base class provides vocabulary lookup,
    special-token bookkeeping, and batch helpers.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = '<unk>',
    ):
        """Create a tokenizer from an existing vocabulary.

        Args:
            vocab (dict[str, int]): Mapping from token strings to integer IDs.
            unk_token (str, default: '<unk>'): Token used when encoding unknown
                input tokens.

        Raises:
            ValueError: If ``unk_token`` is not present in ``vocab``.
        """
        self._token_to_id = dict(vocab)
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

        if unk_token not in self._token_to_id:
            raise KeyError(f'Unknown token {unk_token!r} is not in vocab.')

        self.unk_token = unk_token
        self.unk_id = self._token_to_id[unk_token]

        self.special_tokens = [unk_token]
        self.special_token_ids = {self.unk_id}

    @property
    def vocab(self) -> dict[str, int]:
        """Vocabulary mapping tokens to integer IDs."""
        return self._token_to_id

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._token_to_id)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def extra_repr(self) -> str:
        """Return tokenizer metadata displayed inside ``repr``."""
        return (
            f'vocab_size={self.vocab_size}, '
            f'unk_token={self.unk_token!r}, '
            f'special_tokens={self.special_tokens!r}'
        )

    def __repr__(self) -> str:
        """Return a compact tokenizer representation."""
        extra = self.extra_repr()
        if extra:
            return f'{self.__class__.__name__}({extra})'
        return f'{self.__class__.__name__}()'

    def token_to_id(self, token: str) -> int:
        """Return the ID for ``token``, or the unknown-token ID if missing.

        Args:
            token (str): Token to look up.
        """
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, index: int) -> str:
        """Return the token for ``index``.

        Args:
            index (int): Token ID to look up.

        Raises:
            KeyError: If ``index`` is not in the vocabulary.
        """
        if index not in self._id_to_token:
            raise KeyError(f'Unknown token ID: {index}.')
        return self._id_to_token[index]

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        """Map a list of tokens to token IDs.

        Args:
            tokens (list[str]): Tokens to look up.
        """
        return [self.token_to_id(token) for token in tokens]

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        """Map a list of token IDs to tokens.

        Args:
            indices (list[int]): Token IDs to look up.
        """
        return [self.id_to_token(index) for index in indices]

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add tokens to the vocabulary and mark them as special.

        Existing vocabulary entries are not duplicated, but they are still
        marked as special. Special tokens are skipped by default during decode.

        Args:
            tokens (list[str]): Tokens to add or mark as special.

        Returns:
            The number of new vocabulary entries added.
        """
        added_count = 0

        for token in tokens:
            if token not in self._token_to_id:
                new_id = self._next_token_id()

                self._token_to_id[token] = new_id
                self._id_to_token[new_id] = token

                added_count += 1

            if token not in self.special_tokens:
                self.special_tokens.append(token)

            self.special_token_ids.add(self._token_to_id[token])

        return added_count

    def _next_token_id(self) -> int:
        if not self._id_to_token:
            return 0
        return max(self._id_to_token) + 1

    @classmethod
    @abstractmethod
    def from_text(cls, text: str | list[str], *args, **kwargs) -> Self:
        """Build a tokenizer from one text string or a list of text strings.

        Args:
            text (str | list[str]): Training corpus.
            *args: Additional tokenizer-specific positional arguments.
            **kwargs: Additional tokenizer-specific keyword arguments.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text (str): Text to encode.
        """
        pass

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of text strings.

        Args:
            texts (list[str]): Text strings to encode.
        """
        return [self.encode(text) for text in texts]

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into text.

        Args:
            ids (list[int]): Token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.
        """
        pass

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token ID sequences.

        Args:
            batch_ids (list[list[int]]): Batch of token ID sequences to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]


class Tokenizer:
    """Base class for tokenizers that provides methods for encoding and decoding text."""

    def __init__(
        self,
        model: Model,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        post_processor: PostProcessor | None = None,
        decoder: Decoder | None = None,
        num_workers: int | None = None,
    ):
        """Initialize a Tokenizer instance.

        This class is similar to the `tokenizers.Tokenizer` class from the Hugging Face
        `tokenizers` library, but it is implemented in pure Python and does not rely on
        Rust bindings.

        Args:
            model (Model): The tokenizer model to use for encoding and decoding.
            normalizer (Normalizer | None, optional): The normalizer to use for text
                normalization. Defaults to None.
            pre_tokenizer (PreTokenizer | None, optional): The pre-tokenizer to use for
                pre-tokenization. Defaults to None.
            post_processor (PostProcessor | None, optional): The post-processor to use
                for post-processing. Defaults to None.
            decoder (Decoder | None, optional): The decoder to use for decoding tokens
                back into text. Defaults to None.
            num_workers (int | None, optional): The number of worker threads to use for
                parallel processing. If None, defaults to the number of CPU cores. This
                function automatically choose to use multithreading or multiprocessing
                based on the presence of the GIL. Defaults to None.
        """
        self.model = model
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.post_processor = post_processor
        self.decoder = decoder
        self.num_workers = get_num_workers(num_workers)

        self.unk_token = model.unk_token
        self.special_tokens = [self.unk_token]

        self._vocab = model.vocab or {self.unk_token: 0}
        self._refresh_id_lookup()

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    @vocab.setter
    def vocab(self, vocab: dict[str, int]) -> None:
        self.set_vocab(vocab)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    @property
    def unk_id(self) -> int:
        unk_id = self.token_to_id(self.unk_token)
        if unk_id is None:
            raise KeyError('Missing [UNK] token from the vocabulary.')
        return unk_id

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token in self.special_tokens
            if (token_id := self.token_to_id(token))
        }

    def __len__(self) -> int:
        return self.vocab_size

    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.vocab_size}, '
            f'unk_token={self.unk_token!r}, '
            f'special_tokens={self.special_tokens!r}'
        )

    def __repr__(self) -> str:
        extra = self.extra_repr()
        if extra:
            return f'{self.__class__.__name__}({extra})'
        return f'{self.__class__.__name__}()'

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def set_vocab(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self._refresh_id_lookup()

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def token_to_id(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def id_to_token(self, index: int) -> str | None:
        return self._id_to_token.get(index)

    def lookup_indices(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert a sequence of token IDs to their corresponding tokens.

        Args:
            ids (Sequence[int]): A sequence of token IDs to convert.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during conversion.

        Returns:
            list[str]: A list of tokens corresponding to the input IDs.
        """
        if skip_special_tokens:
            special_ids = self.special_token_ids
        else:
            special_ids = set()

        tokens = []
        for index in ids:
            if skip_special_tokens and index in special_ids:
                continue

            token = self.id_to_token(index)
            if token is not None:
                tokens.append(token)

        return tokens

    def lookup_tokens(self, tokens: Sequence[str]) -> list[int | None]:
        """Convert a sequence of tokens to their corresponding token IDs.

        Args:
            tokens (Sequence[str]): A sequence of tokens to convert.

        Returns:
            list[int | None]: A list of token IDs corresponding to the input tokens.
                If a token is not found in the vocabulary, None is returned for that token.
        """
        return [self.token_to_id(token) for token in tokens]

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add special tokens to the tokenizer's vocabulary.

        Args:
            tokens (list[str]): A list of special tokens to add.

        Returns:
            count (int): The number of new special tokens added to the vocabulary.
        """
        count = self.add_tokens(tokens)
        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)
        return count

    def encode(
        self,
        text: str | Sequence[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode a string or sequence of strings into an ``Encoding`` object.

        Args:
            text (str | Sequence[str]): The input text or pretokenized sequence.
            is_pretokenized (bool, default: False): Whether the input is pretokenized.
            add_special_tokens (bool, default: True): Whether to add special tokens.

        Returns:
            Encoding: An Encoding object containing token IDs, tokens, and offsets.

        Example:
            >>> tokenizer = Tokenizer(model=BPE())
            >>> encoding = tokenizer.encode('Hello, world!')
            >>> print(encoding.ids)
            >>> print(encoding.tokens)
        """
        return self.model.encode(
            self,
            text,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        """Encode a batch of text strings.

        Args:
            texts (list[str]): A list of input text strings to encode.
            is_pretokenized (bool, default: False): Whether the input is pretokenized.
            add_special_tokens (bool, default: True): Whether to add special tokens.

        Returns:
            list[Encoding]: A list of Encoding objects for each input string.

        Example:
            >>> tokenizer = Tokenizer(model=BPE())
            >>> encodings = tokenizer.encode_batch(['Hello, world!', 'Goodbye!'])
            >>> for encoding in encodings:
                    print(encoding.ids)
                    print(encoding.tokens)
        """
        return list(
            parallel_map(
                lambda text: self.encode(
                    text,
                    is_pretokenized=is_pretokenized,
                    add_special_tokens=add_special_tokens,
                ),
                texts,
                num_workers=self.num_workers,
            )
        )

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into text.

        Args:
            ids (Sequence[int]): A sequence of token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.

        Returns:
            str: The decoded string.

        Example:
            >>> tokenizer = Tokenizer(model=BPE())
            >>> text = tokenizer.decode([1, 2, 3])
            >>> print(text)
        """
        return self.model.decode(self, ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode a batch of token ID sequences.

        Args:
            sequences (list[list[int]]): A list of token ID sequences to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.

        Returns:
            list[str]: A list of decoded strings corresponding to each input sequence.
        """
        return [
            self.decode(sequence, skip_special_tokens=skip_special_tokens)
            for sequence in sequences
        ]

    def train_from_iterator(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        special_tokens: list[str] | None = None,
        initial_alphabet: list[str] | None = None,
    ) -> None:
        """Train the tokenizer model from an iterator.

        Args:
            texts (Iterable[str | Iterable[str]]): An iterable of input texts or
                a batch of input texts.
            vocab_size (int, default: 100): The desired vocabulary size.
            min_frequency (int, default: 0): The minimum frequency for a token to be
                included in the vocabulary.
            special_tokens (list[str] | None, default: None): A list of special tokens
                to include in the vocabulary.
            initial_alphabet (list[str] | None, default: None): A list of characters to
                include in the initial alphabet, even if they are not present in the
                training texts. If an entry has multiple characters, only the first
                character is used.
        """
        self.model.train_from_iterator(
            self,
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
        )

    def add_tokens(self, tokens: list[str]) -> int:
        """Add new tokens to the tokenizer's vocabulary.

        Args:
            tokens (list[str]): A list of tokens to add.

        Returns:
            count (int): The number of new tokens added to the vocabulary.
        """
        count = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                count += 1
        if count:
            self._refresh_id_lookup()
        return count

    def _refresh_id_lookup(self) -> None:
        """Refresh the reverse mapping from IDs to tokens."""
        self._token_to_id = self._vocab
        self._id_to_token = {index: token for token, index in self._vocab.items()}

    def _normalize(self, text: str) -> str:
        """Normalize a string with the registered normalizer."""
        if self.normalizer is None:
            return text
        return self.normalizer.normalize(text)

    def _pre_tokenize(self, text: str) -> list[tuple[str, Offset]]:
        """Pre-tokenize a string with the registered pre-tokenizer."""
        if self.pre_tokenizer is None:
            return [(text, (0, len(text)))]
        return self.pre_tokenizer.pre_tokenize(text)

    def _pre_tokenize_tokens(self, text: str) -> list[str]:
        """Pre-tokenize a string without retaining offsets."""
        if self.pre_tokenizer is None:
            return [text]
        return self.pre_tokenizer.pre_tokenize_tokens(text)

    def _post_process(self, encoding: Encoding) -> Encoding:
        """Post-process an encoding with the registered post-processor."""
        if self.post_processor is None:
            return encoding
        return self.post_processor.process(encoding)

    def _decode(self, tokens: list[str]) -> str:
        """Decode a list of tokens into a string."""
        if self.decoder is None:
            return ''.join(tokens)
        return self.decoder.decode(tokens)

    def _count_pre_tokens(
        self,
        texts: Iterable[str],
        batch_size: int = 1024,
    ) -> Counter[str]:
        """Count pre-tokens in an iterable of texts using parallel processing."""
        batches = it.batched(texts, batch_size)

        counts = Counter()
        for batch_counts in parallel_map(
            self._count_pre_tokens_batch,
            batches,
            num_workers=self.num_workers,
        ):
            counts.update(batch_counts)

        return counts

    def _count_pre_tokens_batch(self, texts: tuple[str, ...]) -> Counter[str]:
        """Count pre-tokens in one pickle-safe batch."""
        if self.pre_tokenizer is None:
            return Counter(texts)
        return self.pre_tokenizer.count_tokens(texts)

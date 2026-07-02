from abc import ABC, abstractmethod
from typing import Self

__all__ = ['Tokenizer']


class Tokenizer(ABC):
    """Base class for text tokenizers.

    A tokenizer owns a vocabulary mapping string tokens to integer IDs and
    defines the common encode/decode interface. Subclasses decide how text is
    split into tokens, while this base class provides vocabulary lookup,
    special-token bookkeeping, and batch helpers.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = '<unk>',
    ) -> None:
        """Create a tokenizer from an existing vocabulary.

        Args:
            vocab (dict[str, int]): Mapping from token strings to integer IDs.
            unk_token (str, default: '<unk>'): Token used when encoding unknown
                input tokens.

        Raises:
            ValueError: If `unk_token` is not present in `vocab`.
        """
        self.token_to_id = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        if unk_token not in self.token_to_id:
            raise ValueError(f'Unknown token {unk_token!r} is not in vocab.')

        self.unk_token = unk_token
        self.unk_id = self.token_to_id[unk_token]

        self.special_tokens = [unk_token]
        self.special_token_ids = {self.unk_id}

    @property
    def vocab(self) -> dict[str, int]:
        """Vocabulary mapping tokens to integer IDs."""
        return self.token_to_id

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def extra_repr(self) -> str:
        """Return tokenizer metadata displayed inside `repr`."""
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

    def token2id(self, token: str) -> int:
        """Return the ID for `token`, or the unknown-token ID if missing.

        Args:
            token (str): Token to look up.
        """
        return self.token_to_id.get(token, self.unk_id)

    def id2token(self, index: int) -> str:
        """Return the token for `index`.

        Args:
            index (int): Token ID to look up.

        Raises:
            KeyError: If `index` is not in the vocabulary.
        """
        if index not in self.id_to_token:
            raise KeyError(f'Unknown token ID: {index}.')
        return self.id_to_token[index]

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        """Map a list of tokens to token IDs.

        Args:
            tokens (list[str]): Tokens to look up.
        """
        return [self.token2id(token) for token in tokens]

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        """Map a list of token IDs to tokens.

        Args:
            indices (list[int]): Token IDs to look up.
        """
        return [self.id2token(index) for index in indices]

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
            if token not in self.token_to_id:
                new_id = self._next_token_id()

                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

                added_count += 1

            if token not in self.special_tokens:
                self.special_tokens.append(token)

            self.special_token_ids.add(self.token_to_id[token])

        return added_count

    def _next_token_id(self) -> int:
        if not self.id_to_token:
            return 0
        return max(self.id_to_token) + 1

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

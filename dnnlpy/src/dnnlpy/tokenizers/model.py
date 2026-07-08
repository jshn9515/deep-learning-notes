import itertools as it
from collections.abc import Iterable, Sequence
from typing import Literal, overload, override

from .base import Encoding, Model, Tokenizer
from .trainer import BPETrainer

type Pair = tuple[str, str]
type Offset = tuple[int, int]
type SymbolSpan = tuple[str, int, int]

__all__ = ['BPE']


class BPE(Model):
    """Creates a Byte-Pair Encoding (BPE) model for tokenization."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[Pair] | None = None,
        unk_token: str = '<unk>',
    ):
        """Initialize the BPE model with a vocabulary, merges, and an unknown token.

        Args:
            vocab (dict[str, int], optional): A dictionary mapping tokens to their IDs.
            merges (list[Pair], optional): A list of merge pairs for BPE.
            unk_token (str, default: '<unk>'): The token to use for unknown tokens.
        """
        self.vocab = vocab or {unk_token: 0}
        self.merges = merges or []
        self.unk_token = unk_token
        self._refresh_merge_ranks()

    def _refresh_merge_ranks(self) -> None:
        """Refresh the mapping of merge pairs to their ranks based on the current merges."""
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @overload
    def encode(
        self,
        tokenizer: Tokenizer,
        text: str,
        is_pretokenized: Literal[False] = False,
        add_special_tokens: bool = True,
    ) -> Encoding: ...

    @overload
    def encode(
        self,
        tokenizer: Tokenizer,
        text: Sequence[str],
        is_pretokenized: Literal[True],
        add_special_tokens: bool = True,
    ) -> Encoding: ...

    @override
    def encode(
        self,
        tokenizer: Tokenizer,
        text: str | Sequence[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode the input text into a sequence of token IDs.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            text (str | Sequence[str]): The input text or pretokenized sequence.
            is_pretokenized (bool, default: False): Whether the input is pretokenized.
            add_special_tokens (bool, default: True): Whether to add special tokens.

        Returns:
            Encoding: An Encoding object containing token IDs, tokens, and offsets.
        """
        tokens, offsets = self._prepare_input(
            tokenizer, text, is_pretokenized=is_pretokenized
        )

        ids = []
        output_tokens = []
        output_offsets = []

        for token, token_offset in zip(tokens, offsets, strict=True):
            for model_token, start, end in self._encode_token(token):
                output_token = model_token
                token_id = tokenizer.token_to_id(output_token)

                if token_id is None:
                    token_id = tokenizer.unk_id
                    output_token = self.unk_token

                ids.append(token_id)
                output_tokens.append(output_token)
                output_offsets.append((token_offset[0] + start, token_offset[0] + end))

        encoding = Encoding(
            ids=ids,
            tokens=output_tokens,
            offsets=output_offsets,
        )

        if add_special_tokens and tokenizer.post_processor is not None:
            return tokenizer._post_process(encoding)
        return encoding

    def _prepare_input(
        self,
        tokenizer: Tokenizer,
        text: str | Sequence[str],
        is_pretokenized: bool,
    ) -> tuple[list[str], list[Offset]]:
        """Prepare input tokens and offsets for encoding.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            text (str | Sequence[str]): The input text or pretokenized sequence.
            is_pretokenized (bool): Whether the input is pretokenized.

        Returns:
            tuple[list[str], list[Offset]]: A tuple containing the list of tokens and
                their corresponding offsets.

        Raises:
            TypeError: If the input types do not match the expected types based on
                `is_pretokenized` flag.
        """
        if is_pretokenized:
            if not isinstance(text, Sequence):
                raise TypeError(
                    '`text` must be a sequence of strings when `is_pretokenized=True`.'
                )
            tokens = list(text)
            return tokens, [(0, 0)] * len(tokens)

        else:
            if not isinstance(text, str):
                raise TypeError('`text` must be a string when `is_pretokenized=False`.')

            normalized = tokenizer._normalize(text)
            pieces = tokenizer._pre_tokenize(normalized)
            tokens = [piece for piece, _ in pieces]
            offsets = [offset for _, offset in pieces]
            return tokens, offsets

    def _encode_token(self, token: str) -> list[SymbolSpan]:
        """Encode a single token into subword units using BPE merges.

        Args:
            token (str): The input token to encode.

        Returns:
            list[SymbolSpan]: A list of tuples containing the subword units and their
                corresponding start and end offsets within the original token.

        Example:
            >>> vocab = {'l': 0, 'o': 1, 'w': 2, 'lo': 3, 'w</w>': 4}
            >>> merges = [('l', 'o'), ('lo', 'w</w>')]
            >>> bpe = BPE(vocab=vocab, merges=merges)
            >>> bpe._encode_token('low')
            [('lo', 0, 2), ('w</w>', 2, 5)]
        """
        symbols = [(char, index, index + 1) for index, char in enumerate(token)]

        while len(symbols) > 1:
            available_pairs = [
                (left[0], right[0])
                for left, right in it.pairwise(symbols)
                if (left[0], right[0]) in self._merge_ranks
            ]
            if not available_pairs:
                break

            best_pair = min(available_pairs, key=self._merge_ranks.__getitem__)
            symbols = self._merge_symbol_spans(symbols, best_pair)

        return symbols

    def _merge_symbol_spans(
        self,
        symbols: list[SymbolSpan],
        pair: Pair,
    ) -> list[SymbolSpan]:
        """Merge adjacent symbol spans based on the specified pair.

        Args:
            symbols (list[SymbolSpan]): A list of symbol spans to merge.
            pair (Pair): A tuple representing the pair of symbols to merge.

        Returns:
            list[SymbolSpan]: A new list of symbol spans after merging the specified pair.

        Example:
            >>> symbols = [('l', 0, 1), ('o', 1, 2), ('w</w>', 2, 5)]
            >>> pair = ('l', 'o')
            >>> bpe._merge_symbol_spans(symbols, pair)
            [('lo', 0, 2), ('w</w>', 2, 5)]
        """
        merged = []
        index = 0

        while index < len(symbols):
            current, start, _ = symbols[index]

            if index + 1 < len(symbols) and (current, symbols[index + 1][0]) == pair:
                *_, end = symbols[index + 1]
                merged.append((''.join(pair), start, end))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return merged

    @override
    def decode(
        self,
        tokenizer: Tokenizer,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a sequence of token IDs back into a string.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            ids (Sequence[int]): A sequence of token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.

        Returns:
            decoded (str): The decoded string.

        Example:
            >>> vocab = {'l': 0, 'o': 1, 'w': 2, 'lo': 3, 'w</w>': 4}
            >>> merges = [('l', 'o'), ('lo', 'w</w>')]
            >>> bpe = BPE(vocab=vocab, merges=merges)
            >>> tokenizer = Tokenizer(model=bpe)
            >>> tokenizer.add_special_tokens(['<unk>'])
            >>> tokenizer.train_from_iterator(['low', 'lower', 'newest', 'widest'])
            >>> ids = tokenizer.encode('lowest').ids
            >>> bpe.decode(tokenizer, ids)
            'lowest'
        """
        tokens = tokenizer.lookup_indices(ids, skip_special_tokens)
        if tokenizer.decoder is not None:
            return tokenizer._decode(tokens)
        return ' '.join(tokens)

    @override
    def train_from_iterator(
        self,
        tokenizer: Tokenizer,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        special_tokens: list[str] | None = None,
        initial_alphabet: list[str] | None = None,
    ) -> None:
        """Train the BPE model from an iterator of texts.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            texts (Iterable[str | Iterable[str]]): An iterable of texts or pretokenized
                sequences.
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
        trainer = BPETrainer(
            model=self,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
        )
        trainer.train(texts)

from __future__ import annotations

import itertools as it
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, override

from .base import Tokenizer, Trainer

if TYPE_CHECKING:
    from .model import BPE

type Pair = tuple[str, str]
type WordSymbols = tuple[str, ...]
type WordCountMapping = dict[WordSymbols, int]

__all__ = ['BPETrainer']


class BPETrainer(Trainer):
    """Internal trainer containing the complete BPE training pipeline."""

    def __init__(
        self,
        model: BPE,
        tokenizer: Tokenizer,
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
        initial_alphabet: list[str] | None = None,
    ):
        """Initialize the BPETrainer with the given parameters.

        Args:
            model (BPE): The BPE model to train.
            tokenizer (Tokenizer): The tokenizer instance to use for training.
            vocab_size (int): The maximum size of the vocabulary to build.
            min_frequency (int, default: 2): The minimum frequency for a token to be
                included in the vocabulary.
            special_tokens (list[str] | None, optinal): A list of special tokens to
                include in the vocabulary. If None, only the unknown token will be included.
            initial_alphabet (list[str] | None, optional): Characters to include in the
                initial alphabet, even when they are not present in the training corpus.
                If an entry contains multiple characters, only the first one is used.

        Raises:
            ValueError: If `vocab_size` is less than 1 or `min_frequency` is negative.
        """
        if vocab_size < 1:
            raise ValueError('`vocab_size` must be at least 1.')
        if min_frequency < 0:
            raise ValueError('`min_frequency` must be non-negative.')

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.num_workers = tokenizer.num_workers
        self.special_tokens = self._prepare_special_tokens(special_tokens)
        self.initial_alphabet = self._prepare_initial_alphabet(initial_alphabet)

    def _prepare_special_tokens(self, special_tokens: list[str] | None) -> list[str]:
        """Prepare special tokens for training.

        Args:
            special_tokens (list[str] | None): List of special tokens to include.

        Returns:
            list[str]: A list of **unique** special tokens, ensuring the unknown token
                is included.
        """
        if special_tokens is None:
            tokens = [self.model.unk_token]
        else:
            tokens = list(special_tokens)

        if self.model.unk_token not in tokens:
            tokens.insert(0, self.model.unk_token)

        return list(dict.fromkeys(tokens))

    def _prepare_initial_alphabet(
        self,
        initial_alphabet: list[str] | None = None,
    ) -> list[str]:
        """Prepare unique one-character alphabet tokens for initial vocabulary.

        Args:
            initial_alphabet (list[str] | None): A list of characters to include in the
                initial alphabet, even if they are not present in the training texts.

        Returns:
            alphabet (list[str]): A list of unique one-character tokens to include in the
                initial vocabulary. If an entry has multiple characters, only the first
                character is used.
        """
        if initial_alphabet is None:
            return []

        alphabet = []
        seen_alphabet = set()
        for token in initial_alphabet:
            if not token:
                continue

            char = token[0]
            if char not in seen_alphabet:
                alphabet.append(char)
                seen_alphabet.add(char)

        return alphabet

    @override
    def train(self, inputs: Iterable[str | Iterable[str]]) -> None:
        """Train the BPE model from an iterator of input texts.

        The whole training process is as follows:

        1. Flatten the input texts into a single iterator of strings.
        2. Collect word counts from the input texts, normalizing and pre-tokenizing them.
        3. Initialize the vocabulary with special tokens and the byte-level alphabet.
        4. Iteratively count adjacent symbol pairs, select the most frequent pair, and merge
        it into a new token until the desired vocabulary size is reached or no more eligible
        pairs are found.
        5. Save the trained vocabulary and merges to the model and tokenizer.

        Args:
            inputs (Iterable[str | Iterable[str]]): An iterable of input texts or
                pre-tokenized sequences.

        Raises:
            ValueError: If `vocab_size` is less than 1 or `min_frequency` is negative.
        """
        texts = self._iter_texts(inputs)
        word_counts = self._collect_word_counts(texts)
        word_symbols = {tuple(word): freq for word, freq in sorted(word_counts.items())}
        vocab_tokens, known_tokens = self._initialize_vocab(word_counts)

        merges = []

        while len(vocab_tokens) < self.vocab_size:
            pair_counts = self._count_pairs(word_symbols)
            best_pair = self._select_best_pair(pair_counts)

            if best_pair is None:
                break

            new_token = ''.join(best_pair)

            if new_token not in known_tokens:
                vocab_tokens.append(new_token)
                known_tokens.add(new_token)

            merges.append(best_pair)

            word_symbols = {
                self._merge_pair(symbols, best_pair): freq
                for symbols, freq in word_symbols.items()
            }

        self._save_result(vocab_tokens, merges)

    def _iter_texts(self, inputs: Iterable[str | Iterable[str]]) -> Iterator[str]:
        """Flatten an iterable of iterables of strings into a single iterator of strings."""
        for item in inputs:
            if isinstance(item, str):
                yield item
            else:
                yield from item

    def _split_words(self, text: str) -> list[str]:
        """Normalize and pre-tokenize a single text into a list of words."""
        text = self.tokenizer._normalize(text)
        words = []
        for piece, _ in self.tokenizer._pre_tokenize(text):
            words.append(piece)
        return words

    def _collect_word_counts(self, texts: Iterable[str]) -> Counter[str]:
        """Collect word counts from the input texts. Normalize and pre-tokenize the input
        texts, then count how often eachcpre-tokenized word appears in the corpus.

        Each text is processed in parallel when multiple workers are available. The offset
        information returned by the pre-tokenizer is ignored because only the token strings
        are needed for frequency counting.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            texts (Iterable[str]): An iterable of input texts to process.

        Returns:
            Counter[str]: A counter mapping each pre-tokenized word to its frequency in
                the input texts.

        Examples:
            >>> texts = ['Hello world!', 'Hello again.']
            >>> _collect_word_counts(texts)
            Counter({'Hello': 2, 'world!': 1, 'again.': 1})
        """
        counts = Counter()

        word_batches = map(self._split_words, texts)
        for words in word_batches:
            counts.update(words)

        return counts

    def _initialize_vocab(
        self,
        word_counts: Counter[str],
    ) -> tuple[list[str], set[str]]:
        """Initialize the vocabulary with special tokens and the byte-level alphabet."""
        alphabet = {char for word in word_counts for char in word}
        alphabet = alphabet.union(self.initial_alphabet)
        alphabet = sorted(alphabet)

        vocab_tokens = list(self.special_tokens)
        known_tokens = set(vocab_tokens)

        for token in alphabet:
            if token not in known_tokens:
                vocab_tokens.append(token)
                known_tokens.add(token)

        return vocab_tokens, known_tokens

    def _count_pairs(self, word_symbols: WordCountMapping) -> Counter[Pair]:
        """Count frequencies of adjacent token pairs.

        Args:
            word_symbols (WordCountMapping): A mapping from tuples of symbols to word
                frequencies.
            num_workers (int, default: 1): The number of parallel workers to use for counting
                pairs. If 1, counting is done in a single thread.

        Returns:
            Counter[Pair]: A counter mapping each adjacent token pair to its frequency.

        Examples:
            >>> tokenizer = Tokenizer(BPE())
            >>> word_symbols = {
            ...     ('l', 'o', 'w'): 5,
            ...     ('l', 'o', 'w', 'e', 'r'): 2,
            ... }
            >>> _count_pairs(word_symbols)
            Counter({
                ('l', 'o'): 7,
                ('o', 'w'): 7,
                ('w', 'e'): 2,
                ('e', 'r'): 2,
            })
        """
        counts = Counter()

        for symbols, frequency in word_symbols.items():
            for pair in it.pairwise(symbols):
                counts[pair] += frequency

        return counts

    def _select_best_pair(self, pair_counts: Counter[Pair]) -> Pair | None:
        """Select the most frequent adjacent symbol pair that meets the minimum frequency
        requirement.

        Args:
            pair_counts (Counter[Pair]): A counter mapping adjacent symbol pairs to their
                frequencies.

        Returns:
            Pair | None: The most frequent adjacent symbol pair that meets the minimum frequency requirement, or None if no such pair exists.
        """
        eligible_pairs = [
            (pair, count)
            for pair, count in pair_counts.items()
            if count >= self.min_frequency
        ]
        if not eligible_pairs:
            return None

        pair, _ = max(eligible_pairs, key=lambda item: (item[1], item[0]))
        return pair

    def _merge_pair(self, symbols: WordSymbols, pair: Pair) -> WordSymbols:
        """Merge the first occurrence of a symbol pair in a tuple of symbols.

        Args:
            symbols (WordSymbols): A tuple of symbols to merge.
            pair (Pair): A pair of symbols to merge.

        Returns:
            WordSymbols: A new tuple of symbols with the first occurrence of the pair merged.

        Examples:
            >>> symbols = ('l', 'o', 'w', 'e', 'r')
            >>> pair = ('l', 'o')
            >>> _merge_pair(symbols, pair)
            ('lo', 'w', 'e', 'r')
            >>> symbols = ('l', 'o', 'w', 'e', 'r')
            >>> pair = ('o', 'w')
            >>> _merge_pair(symbols, pair)
            ('l', 'ow', 'e', 'r')
        """
        index = 0
        merged = []

        while index < len(symbols):
            if index < len(symbols) - 1 and symbols[index : index + 2] == pair:
                merged.append(''.join(pair))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return tuple(merged)

    def _save_result(self, vocab_tokens: list[str], merges: list[Pair]) -> None:
        """Save the trained vocabulary and merges to the model and tokenizer."""
        self.model.merges = list(merges)
        self.model._refresh_merge_ranks()

        vocab = {token: index for index, token in enumerate(vocab_tokens)}
        self.tokenizer.vocab = vocab
        self.tokenizer.special_tokens = list(self.special_tokens)

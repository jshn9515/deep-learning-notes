from __future__ import annotations

import heapq
import itertools as it
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, override

from .base import Tokenizer, Trainer

if TYPE_CHECKING:
    from .model import BPE

type Pair = tuple[str, str]
type WordSymbols = tuple[str, ...]
type WordCountMapping = dict[WordSymbols, int]
type PairWordIndices = dict[Pair, set[int]]
type PairFrequencyBuckets = dict[int, set[Pair]]

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
        4. Count adjacent symbol pairs, then incrementally update the counts while merging
        the most frequent pair until the desired vocabulary size is reached or no more
        eligible pairs are found.
        5. Save the trained vocabulary and merges to the model and tokenizer.

        Args:
            inputs (Iterable[str | Iterable[str]]): An iterable of input texts or
                pre-tokenized sequences.

        Raises:
            ValueError: If `vocab_size` is less than 1 or `min_frequency` is negative.
        """
        texts = self._iter_texts(inputs)
        word_counts = self._collect_word_counts(texts)
        sorted_word_counts = sorted(word_counts.items())
        word_symbols = [tuple(word) for word, _ in sorted_word_counts]
        word_frequencies = [frequency for _, frequency in sorted_word_counts]

        vocab_tokens, known_tokens = self._initialize_vocab(word_counts)
        pair_counts, pair_word_indices = self._initialize_pair_counts(
            word_symbols,
            word_frequencies,
        )
        pair_frequency_buckets, frequency_heap = self._initialize_pair_frequency_index(
            pair_counts
        )

        merges = []

        while len(vocab_tokens) < self.vocab_size:
            best_pair = self._select_best_pair(
                pair_counts,
                pair_frequency_buckets,
                frequency_heap,
            )

            if best_pair is None:
                break

            new_token = ''.join(best_pair)

            if new_token not in known_tokens:
                vocab_tokens.append(new_token)
                known_tokens.add(new_token)

            merges.append(best_pair)

            affected_word_indices = tuple(pair_word_indices.pop(best_pair))
            for word_index in affected_word_indices:
                self._merge_word_and_update_pair_counts(
                    word_index,
                    best_pair,
                    word_symbols,
                    word_frequencies,
                    pair_counts,
                    pair_word_indices,
                    pair_frequency_buckets,
                    frequency_heap,
                )

        self._save_result(vocab_tokens, merges)

    def _iter_texts(self, inputs: Iterable[str | Iterable[str]]) -> Iterator[str]:
        """Flatten an iterable of iterables of strings into a single iterator of strings."""
        for item in inputs:
            if isinstance(item, str):
                yield item
            else:
                yield from item

    def _collect_word_counts(self, texts: Iterable[str]) -> Counter[str]:
        """Collect word counts from the input texts. Normalize and pre-tokenize the input
        texts, then count how often eachcpre-tokenized word appears in the corpus.

        The registered pre-tokenizer may count tokens directly so training does not need to
        materialize offsets or a complete encoded-token list for every text.

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
        normalized_texts = map(self.tokenizer._normalize, texts)
        return self.tokenizer._count_pre_tokens(normalized_texts)

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

    def _initialize_pair_counts(
        self,
        word_symbols: list[WordSymbols],
        word_frequencies: list[int],
    ) -> tuple[Counter[Pair], PairWordIndices]:
        """Count pairs once and index the words containing each pair."""
        pair_counts = Counter()
        pair_word_indices = defaultdict(set)

        for word_index, (symbols, frequency) in enumerate(
            zip(word_symbols, word_frequencies, strict=True)
        ):
            word_pairs = set()
            for pair in it.pairwise(symbols):
                pair_counts[pair] += frequency
                word_pairs.add(pair)

            for pair in word_pairs:
                pair_word_indices[pair].add(word_index)

        return pair_counts, dict(pair_word_indices)

    def _initialize_pair_frequency_index(
        self,
        pair_counts: Counter[Pair],
    ) -> tuple[PairFrequencyBuckets, list[int]]:
        """Group pairs by frequency and build a max-frequency heap."""
        pair_frequency_buckets = defaultdict(set)

        for pair, count in pair_counts.items():
            pair_frequency_buckets[count].add(pair)

        if sys.version_info >= (3, 14):
            frequency_heap = [count for count in pair_frequency_buckets]
            heapq.heapify_max(frequency_heap)
        else:
            frequency_heap = [-count for count in pair_frequency_buckets]
            heapq.heapify(frequency_heap)

        return dict(pair_frequency_buckets), frequency_heap

    def _merge_word_and_update_pair_counts(
        self,
        word_index: int,
        pair: Pair,
        word_symbols: list[WordSymbols],
        word_frequencies: list[int],
        pair_counts: Counter[Pair],
        pair_word_indices: PairWordIndices,
        pair_frequency_buckets: PairFrequencyBuckets,
        frequency_heap: list[int],
    ) -> None:
        """Merge a pair in one word and update only the pair counts that changed."""
        old_symbols = word_symbols[word_index]
        new_symbols = self._merge_pair(old_symbols, pair)
        frequency = word_frequencies[word_index]
        old_pair_occurrences = Counter(it.pairwise(old_symbols))
        new_pair_occurrences = Counter(it.pairwise(new_symbols))

        for changed_pair in old_pair_occurrences.keys() | new_pair_occurrences.keys():
            old_occurrences = old_pair_occurrences[changed_pair]
            new_occurrences = new_pair_occurrences[changed_pair]
            occurrence_delta = new_occurrences - old_occurrences

            if occurrence_delta:
                self._update_pair_count(
                    changed_pair,
                    occurrence_delta * frequency,
                    pair_counts,
                    pair_frequency_buckets,
                    frequency_heap,
                )

            if new_occurrences:
                pair_word_indices.setdefault(changed_pair, set()).add(word_index)
            else:
                indices = pair_word_indices.get(changed_pair)
                if indices is not None:
                    indices.discard(word_index)
                    if not indices:
                        del pair_word_indices[changed_pair]

        word_symbols[word_index] = new_symbols

    def _update_pair_count(
        self,
        pair: Pair,
        count_delta: int,
        pair_counts: Counter[Pair],
        pair_frequency_buckets: PairFrequencyBuckets,
        frequency_heap: list[int],
    ) -> None:
        """Update a pair count and its frequency-bucket membership."""
        old_count = pair_counts[pair]
        if old_count:
            old_bucket = pair_frequency_buckets[old_count]
            old_bucket.remove(pair)
            if not old_bucket:
                del pair_frequency_buckets[old_count]

        new_count = old_count + count_delta
        if not new_count:
            pair_counts.pop(pair, None)
            return

        pair_counts[pair] = new_count
        new_bucket = pair_frequency_buckets.get(new_count)

        if new_bucket is None:
            new_bucket = set()
            pair_frequency_buckets[new_count] = new_bucket

            if sys.version_info >= (3, 14):
                heapq.heappush_max(frequency_heap, new_count)
            else:
                heapq.heappush(frequency_heap, -new_count)

        new_bucket.add(pair)

    def _select_best_pair(
        self,
        pair_counts: Counter[Pair],
        pair_frequency_buckets: PairFrequencyBuckets | None = None,
        frequency_heap: list[int] | None = None,
    ) -> Pair | None:
        """Select the most frequent adjacent symbol pair that meets the minimum frequency
        requirement.

        Args:
            pair_counts (Counter[Pair]): A counter mapping adjacent symbol pairs to their
                frequencies.

        Returns:
            Pair | None: The most frequent adjacent symbol pair that meets the minimum frequency requirement, or None if no such pair exists.
        """
        if pair_frequency_buckets is None or frequency_heap is None:
            eligible_pairs = (
                (count, pair)
                for pair, count in pair_counts.items()
                if count >= self.min_frequency
            )
            return max(eligible_pairs, default=(0, None))[1]

        while frequency_heap:
            if sys.version_info >= (3, 14):
                best_count = frequency_heap[0]
                pairs = pair_frequency_buckets.get(best_count)
                if pairs is None:
                    heapq.heappop_max(frequency_heap)
                    continue
            else:
                best_count = -frequency_heap[0]
                pairs = pair_frequency_buckets.get(best_count)
                if pairs is None:
                    heapq.heappop(frequency_heap)
                    continue

            if best_count < self.min_frequency:
                return None

            return max(pairs)

        return None

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

import gzip
import os
from collections.abc import Iterable
from typing import Self, overload

import numpy as np
import numpy.typing as npt

type Key = int | str
type VectorOrKey = Key | np.ndarray

__all__ = [
    'KeyedVectors',
    'load_word2vec_format',
]

MISSING = object()


def _as_items(value: VectorOrKey | Iterable[VectorOrKey] | None) -> list[VectorOrKey]:
    if value is None:
        return []
    if isinstance(value, int | str):
        return [value]
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return [value]
    return list(value)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


class KeyedVectors:
    """Store dense vectors and query their cosine similarities.

    String and integer keys map to vector rows. Lookup also accepts an occupied
    row index; an explicitly stored integer key takes precedence over that index.
    """

    def __init__(
        self,
        vector_size: int,
        count: int = 0,
        dtype: npt.DTypeLike = np.float32,
    ):
        """Initialize vector storage and empty key mappings.

        Args:
            vector_size (int): Positive number of components in each vector.
            count (int, default: 0): Number of empty rows to preallocate. Insertion
                fills these rows before growing the array.
            dtype (npt.DTypeLike, default: np.float32): Vector storage dtype. Use a
                floating-point dtype for normalization and similarity queries.

        Attributes:
            vector_size (int): Number of components in each vector.
            vectors (np.ndarray): Vector matrix of shape `(len(self), vector_size)`.
            idx2key (list): Keys in row order, with `MISSING` for unfilled rows.
            key2idx (dict): Mapping from stored keys to row indices.
            norms (np.ndarray | None): Cached row L2 norms, computed on demand.
            next_index (int): Next preallocated row to fill.

        Raises:
            AssertionError: If `vector_size` is not positive or `count` is negative.

        Examples:
            >>> vectors = KeyedVectors(2)
            >>> vectors.add_vector('east', np.array([1.0, 0.0]))
            0
            >>> vectors.add_vector('north', np.array([0.0, 1.0]))
            1
            >>> vectors.similarity('east', 'north')
            0.0
        """
        if vector_size < 1:
            raise AssertionError('`vector_size` must be positive.')
        if count < 0:
            raise AssertionError('`count` cannot be negative.')

        self.vector_size = vector_size
        self.vectors = np.zeros((count, vector_size), dtype=dtype)

        self.idx2key = [MISSING] * count
        self.key2idx = {}

        self.norms = None
        self.next_index = 0

    def __len__(self) -> int:
        """Return the number of rows, including unfilled preallocated rows."""
        return len(self.idx2key)

    def __contains__(self, key: int | str) -> bool:
        """Check for a stored string key or an occupied integer row index.

        Integer membership checks row positions rather than stored integer keys.
        Use `has_index_for` to check whether a key can be resolved by lookup.
        """
        if isinstance(key, int):
            return 0 <= key < len(self.idx2key) and self.idx2key[key] is not MISSING

        if isinstance(key, str):
            return key in self.key2idx

        return False

    def __getitem__(self, key_or_keys: Key | Iterable[Key]) -> np.ndarray:
        """Return one vector or stack vectors in the requested order.

        Args:
            key_or_keys (Key | Iterable[Key]): A key or row index, or a nonempty
                iterable of keys or row indices.

        Returns:
            vectors (np.ndarray): A read-only vector of shape `(vector_size,)`
                for one key, or a new matrix of shape `(n_keys, vector_size)`.

        Raises:
            KeyError: If a requested key or row index cannot be resolved.
            ValueError: If the iterable is empty.
        """
        if isinstance(key_or_keys, int | str):
            return self.get_vector(key_or_keys)
        else:
            return np.vstack([self.get_vector(key) for key in key_or_keys])

    def get_index(self, key: Key, default: int | None = None) -> int:
        """Resolve a stored key or occupied row index.

        Args:
            key (Key): Stored key or integer row index. Stored keys take
                precedence over positional lookup.
            default (int | None, default: None): Value returned for a missing
                key. If `None`, a missing key raises an exception.

        Returns:
            index (int): Resolved row index, or `default` for a missing key.

        Raises:
            KeyError: If the key is missing and no default is supplied.
        """
        idx = self.key2idx.get(key)
        if idx is not None:
            return idx

        if isinstance(key, int) and key in self:
            return key

        if default is not None:
            return default

        raise KeyError(f'Requested key {key!r} not found in vocabulary.')

    def has_index_for(self, key: Key) -> bool:
        """Return whether a stored key or occupied row index can be resolved."""
        return self.get_index(key, -1) >= 0

    def get_vector(self, key: Key, norm: bool = False) -> np.ndarray:
        """Return a read-only vector, optionally normalized to unit length.

        Args:
            key (Key): Stored key or occupied row index.
            norm (bool, default: False): Whether to divide by the L2 norm.
                Zero vectors remain zero.

        Returns:
            vector (np.ndarray): Read-only array of shape `(vector_size,)`.
                Without normalization, it shares the underlying vector storage.

        Raises:
            KeyError: If the key or row index cannot be resolved.
        """
        idx = self.get_index(key)
        vector = self.vectors[idx]

        if norm:
            self.fill_norms()
            assert self.norms is not None

            divisor = self.norms[idx]
            vector = vector if divisor == 0.0 else vector / divisor

        result = vector.view()
        result.setflags(write=False)
        return result

    def add_vector(self, key: Key, vector: np.ndarray) -> int:
        """Insert one vector, keeping the existing value if the key is present.

        Args:
            key (Key): String or integer key to insert.
            vector (np.ndarray): Array of shape `(vector_size,)`,
                converted to the storage dtype.

        Returns:
            index (int): Row index of the new or existing key.

        Raises:
            TypeError: If `key` is not a string or integer, or `vector` is not
                a NumPy array.
            AssertionError: If the vector does not have shape `(vector_size,)`.
        """
        if not isinstance(key, int | str):
            raise TypeError('`key` must be a string or integer.')

        if not isinstance(vector, np.ndarray):
            raise TypeError('`vector` must be a NumPy array.')

        if vector.ndim != 1 or vector.size != self.vector_size:
            raise AssertionError(
                f'Expected `vector` to have shape ({self.vector_size},), '
                f'but got {vector.shape}.'
            )

        self.add_vectors([key], vector.reshape(1, -1))
        return self.key2idx[key]

    def add_vectors(
        self,
        keys: Iterable[Key],
        weights: np.ndarray | Iterable[np.ndarray],
        replace: bool = False,
    ) -> None:
        """Insert vectors and optionally replace values for existing keys.

        New keys fill preallocated rows before growing the matrix. Existing
        keys retain their row indices. This method clears the cached norms.

        Args:
            keys (Iterable[Key]): Collection of keys corresponding to the vector
                rows. For one vector, pass a collection containing one key.
            weights (np.ndarray | Iterable[np.ndarray]): Array of vector values
                of shape `(n_keys, vector_size)`, converted to the storage dtype.
                A collection containing one vector must still be two-dimensional.
            replace (bool, default: False): Whether to overwrite existing keys.
                Otherwise, their supplied vectors are ignored.

        Raises:
            TypeError: If `keys` is a single string or integer, or `weights` is
                not a NumPy array.
            AssertionError: If the matrix shape does not match the key count
                and vector size.
        """
        if isinstance(keys, int | str):
            raise TypeError('`keys` must be a collection of keys.')
        if isinstance(weights, np.ndarray) and weights.ndim == 1:
            raise TypeError('`weights` must be a collection of vectors.')
        elif any(not isinstance(weight, np.ndarray) for weight in weights):
            raise TypeError('`weights` must be a collection of NumPy arrays.')

        key_list = list(keys)

        matrix = np.asarray(weights, dtype=self.vectors.dtype)
        if matrix.shape != (len(key_list), self.vector_size):
            raise AssertionError(
                f'Expected `weights` to have shape '
                f'({len(key_list)}, {self.vector_size}), but got {matrix.shape}.'
            )

        for key, vector in zip(key_list, matrix, strict=True):
            idx = self.key2idx.get(key)
            if idx is not None:
                if replace:
                    self.vectors[idx] = vector
                continue

            if (
                self.next_index < len(self.idx2key)
                and self.idx2key[self.next_index] is MISSING
            ):
                idx = self.next_index
                self.idx2key[idx] = key
                self.vectors[idx] = vector
            else:
                idx = len(self.idx2key)
                self.idx2key.append(key)
                self.vectors = np.vstack([self.vectors, vector])

            self.key2idx[key] = idx
            self.next_index = idx + 1

        self.norms = None

    def fill_norms(self, force: bool = False) -> None:
        """Cache the L2 norm of each vector row.

        Args:
            force (bool, default: False): Recompute even if norms are cached.
                Use this after modifying `vectors` directly.
        """
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    def get_normed_vectors(self) -> np.ndarray:
        """Return a normalized copy of the vector matrix.

        Returns:
            vectors (np.ndarray): Array of shape `(len(self), vector_size)`
                with each nonzero row normalized to unit L2 norm. Zero rows
                remain zero. The stored vectors are unchanged.
        """
        self.fill_norms()
        assert self.norms is not None

        divisors = self.norms[:, np.newaxis]
        return np.divide(
            self.vectors,
            divisors,
            out=np.zeros_like(self.vectors),
            where=divisors != 0.0,
        )

    def similarity(self, word1: Key, word2: Key) -> float:
        """Compute the cosine similarity between two vectors.

        Args:
            word1 (Key): First stored key or occupied row index.
            word2 (Key): Second stored key or occupied row index.

        Returns:
            similarity (float): Dot product of the unit vectors, normally in
                `[-1, 1]`. Returns zero if either vector is zero.

        Raises:
            KeyError: If either key or row index cannot be resolved.
        """
        word_vec1 = _unit_vector(self[word1])
        word_vec2 = _unit_vector(self[word2])
        return float(np.dot(word_vec1, word_vec2))

    def distance(self, word1: Key, word2: Key) -> float:
        """Compute cosine distance as one minus cosine similarity.

        Args:
            word1 (Key): First stored key or occupied row index.
            word2 (Key): Second stored key or occupied row index.

        Returns:
            distance (float): Cosine distance, normally in `[0, 2]`. Returns
                one if either vector is zero.

        Raises:
            KeyError: If either key or row index cannot be resolved.
        """
        return 1.0 - self.similarity(word1, word2)

    @overload
    def most_similar(
        self,
        positive: VectorOrKey | Iterable[VectorOrKey] | None = None,
        negative: VectorOrKey | Iterable[VectorOrKey] | None = None,
        top_k: int = 10,
        restrict_vocab: int | None = None,
    ) -> list[tuple[Key, float]]: ...

    @overload
    def most_similar(
        self,
        positive: VectorOrKey | Iterable[VectorOrKey] | None = None,
        negative: VectorOrKey | Iterable[VectorOrKey] | None = None,
        top_k: None = None,
        restrict_vocab: int | None = None,
    ) -> np.ndarray: ...

    def most_similar(
        self,
        positive: VectorOrKey | Iterable[VectorOrKey] | None = None,
        negative: VectorOrKey | Iterable[VectorOrKey] | None = None,
        top_k: int | None = 10,
        restrict_vocab: int | None = None,
    ) -> list[tuple[Key, float]] | np.ndarray:
        """Find vectors closest to a combination of positive and negative inputs.

        Each input is normalized before combining it with weight +1 for
        `positive` or -1 for `negative`. The combined vector is normalized again
        before computing cosine scores. Custom weights are not supported.
        A zero combined vector produces all-zero scores. Fill all preallocated
        rows before requesting ranked keys.

        Args:
            positive (VectorOrKey | Iterable[VectorOrKey] | None): Keys, row
                indices, or NumPy vectors to add. Accepts one input or an
                iterable; `None` contributes no inputs. Each raw vector must
                have shape `(vector_size,)`.
            negative (VectorOrKey | Iterable[VectorOrKey] | None): Inputs to
                subtract, with the same accepted forms as `positive`.
            top_k (int | None, default: 10): Maximum number of ranked results.
                Nonpositive integers return an empty list. `None` returns all
                scores in row order, including scores for input keys.
            restrict_vocab (int | None, default: None): Search only the first
                this many rows. Must be nonnegative; values larger than the
                matrix size search all rows. `None` searches all rows.

        Returns:
            result (list[tuple[Key, float]] | np.ndarray): Key-score pairs in
                descending score order, excluding resolved input keys, or a
                one-dimensional score array when `top_k` is `None`. Raw vector
                inputs do not exclude any keys. Zero candidate vectors score
                zero.

        Raises:
            AssertionError: If both input collections are empty, unless
                `top_k` is a nonpositive integer.
            KeyError: If an input key or row index cannot be resolved.
        """
        if isinstance(top_k, int) and top_k < 1:
            return []

        positive_items = _as_items(positive)
        negative_items = _as_items(negative)
        if not positive_items and not negative_items:
            raise AssertionError('Cannot compute similarity with no input.')

        mean = np.zeros(self.vector_size, dtype=self.vectors.dtype)
        input_indices = set()

        for items, weight in ((positive_items, 1.0), (negative_items, -1.0)):
            for item in items:
                if isinstance(item, np.ndarray):
                    vector = item
                else:
                    idx = self.get_index(item)
                    input_indices.add(idx)
                    vector = self.get_vector(item)

                mean += weight * _unit_vector(vector)

        mean = _unit_vector(mean)

        end = (
            len(self.vectors)
            if restrict_vocab is None
            else min(restrict_vocab, len(self.vectors))
        )
        self.fill_norms()
        assert self.norms is not None

        scores = np.divide(
            self.vectors[:end] @ mean,
            self.norms[:end],
            out=np.zeros(end, dtype=self.vectors.dtype),
            where=self.norms[:end] != 0.0,
        )
        if top_k is None:
            return scores

        count = min(top_k + len(input_indices), end)
        if count == 0:
            return []

        if count == end:
            best = np.argsort(scores, stable=True, descending=True)
        else:
            # TODO: Use `descending` argument in `np.argpartition` in NumPy v2.6
            partition = np.argpartition(-scores, count - 1)[:count]
            best = partition[
                np.argsort(scores[partition], stable=True, descending=True)
            ]

        result = []
        for idx in best:
            if idx not in input_indices:
                result.append((self.idx2key[idx], scores[idx].item()))
                if len(result) == top_k:
                    break

        return result

    @classmethod
    def load_word2vec_format(
        cls,
        fname: str | os.PathLike[str],
        encoding: str = 'utf-8',
        unicode_errors: str = 'strict',
        limit: int | None = None,
        dtype: npt.DTypeLike = np.float32,
        no_header: bool = False,
    ) -> Self:
        """Load vectors from a gzip-compressed text word2vec or GloVe file.

        Each record contains a word, a space, and whitespace-separated numeric
        components. By default, the first line gives the vocabulary size and
        vector size. Duplicate words keep their first vector. If fewer unique
        words are read than expected, storage is trimmed to the loaded rows.

        Args:
            fname (str | os.PathLike[str]): Path to a gzip-compressed text file.
            encoding (str, default: 'utf-8'): Encoding used to decode words and
                the header.
            unicode_errors (str, default: 'strict'): Error handling policy for
                decoding, such as 'strict', 'ignore', or 'replace'.
            limit (int | None, default: None): Nonnegative maximum number of
                unique vectors to load from a file with a header. Without a
                header, only the first `limit` records are considered. `None`
                uses the header count or all headerless records.
            dtype (npt.DTypeLike, default: np.float32): Vector storage dtype.
            no_header (bool, default: False): Whether the file has no header.
                Reads all records into memory and infers the vector size from
                the first selected record.

        Returns:
            model (Self): Loaded vectors and key mappings in file order, ready
                for lookup, similarity queries, or further insertion.

        Raises:
            OSError: If the file cannot be opened or is not a valid gzip file.
            UnicodeDecodeError: If decoding fails under the selected policy.
            AssertionError: If the parsed dimensions are invalid.
            RuntimeError: If the header or numeric values cannot be parsed.
            RuntimeError: If no headerless records are selected, a record has
                no word/value separator, or a vector has the wrong size.
        """
        with gzip.open(fname, 'rb') as stream:
            if no_header:
                records = list(stream)

                if limit is not None:
                    records = records[:limit]
                if not records:
                    raise RuntimeError('Word2Vec file is empty.')

                _, first_values = records[0].rstrip().split(b' ', 1)
                vector_size = len(first_values.split())
                vocab_size = len(records)

            else:
                header = stream.readline().decode(encoding, errors=unicode_errors)
                vocab_size, vector_size = (int(value) for value in header.split())
                if limit is not None:
                    vocab_size = min(vocab_size, limit)
                records = stream

            result = cls(vector_size, count=vocab_size, dtype=dtype)
            loaded = 0

            for raw_line in records:
                if loaded == vocab_size:
                    break
                try:
                    raw_word, raw_values = raw_line.rstrip().split(b' ', 1)
                except ValueError:
                    raise RuntimeError(f'Invalid word2vec record at row {loaded + 1}.')

                word = raw_word.decode(encoding, errors=unicode_errors)
                vector = np.fromstring(raw_values, sep=' ', dtype=dtype)
                if vector.size != vector_size:
                    raise RuntimeError(
                        f'Expected vector for {word!r} to have {vector_size} '
                        f'values, but got {vector.size}.'
                    )

                if word in result.key2idx:
                    continue

                result.idx2key[loaded] = word
                result.key2idx[word] = loaded
                result.vectors[loaded] = vector
                loaded += 1

        if loaded != vocab_size:
            result.idx2key = result.idx2key[:loaded]
            result.vectors = np.ascontiguousarray(result.vectors[:loaded])

        result.next_index = loaded
        return result


load_word2vec_format = KeyedVectors.load_word2vec_format

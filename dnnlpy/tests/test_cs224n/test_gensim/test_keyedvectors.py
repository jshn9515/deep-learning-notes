import math
from collections.abc import Iterable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from dnnlpy.cs224n.gensim import KeyedVectors


@pytest.fixture
def vectors() -> KeyedVectors:
    model = KeyedVectors(2)
    model.add_vectors(
        ['east', 'north', 'northeast', 'west', 'zero'],
        np.array([[3, 0], [0, 4], [1, 1], [-2, 0], [0, 0]]),
    )
    return model


@pytest.mark.parametrize('vector_size, count', [(0, 0), (-1, 0), (2, -1)])
def test_invalid_dimensions(vector_size: int, count: int):
    with pytest.raises(AssertionError):
        KeyedVectors(vector_size, count=count)


def test_preallocation_and_growth():
    model = KeyedVectors(2, count=2, dtype=np.float64)
    assert 0 not in model
    assert not model.has_index_for(0)

    assert model.add_vector('east', np.array([3, 0])) == 0
    assert 0 in model
    assert 1 not in model
    model.add_vectors(['north', 'west'], np.array([[0, 4], [-2, 0]]))

    assert len(model) == 3
    assert model.idx2key == ['east', 'north', 'west']
    assert model.key2idx == {'east': 0, 'north': 1, 'west': 2}
    assert model.vectors.dtype == np.float64
    assert_array_equal(model.vectors, [[3, 0], [0, 4], [-2, 0]])


def test_lookup(vectors: KeyedVectors):
    assert 'east' in vectors
    assert 'missing' not in vectors
    assert -1 not in vectors
    assert len(vectors) not in vectors
    assert vectors.get_index('east') == 0
    assert vectors.get_index(1) == 1
    assert vectors.get_index('missing', default=42) == 42
    assert vectors.has_index_for('north')
    assert not vectors.has_index_for('missing')
    assert_array_equal(vectors['east'], [3, 0])
    assert_array_equal(vectors[1], [0, 4])
    assert_array_equal(vectors[['north', 'east']], [[0, 4], [3, 0]])


@pytest.mark.parametrize('key', ['missing', -1, 5])
def test_missing_lookup(vectors: KeyedVectors, key: int | str):
    with pytest.raises(KeyError, match='not found in vocabulary'):
        vectors.get_vector(key)


def test_integer_key():
    model = KeyedVectors(2)
    model.add_vector(42, np.array([1, 2]))
    assert model.get_index(42) == 0
    assert model.has_index_for(42)
    assert_array_equal(model[42], [1, 2])


@pytest.mark.parametrize('norm', [False, True])
def test_vectors_are_read_only(vectors: KeyedVectors, norm: bool):
    vector = vectors.get_vector('east', norm=norm)
    assert_allclose(vector, [1, 0] if norm else [3, 0])

    with pytest.raises(ValueError, match='read-only'):
        vector[0] = 99

    assert_array_equal(vectors['east'], [3, 0])


def test_duplicate_and_replacement_invalidate_norms(vectors: KeyedVectors):
    vectors.fill_norms()
    vectors.add_vector('east', np.array([0, 2]))
    assert_array_equal(vectors['east'], [3, 0])

    vectors.add_vectors(['east'], np.array([[0, 2]]), replace=True)
    assert len(vectors) == 5
    assert vectors.get_index('east') == 0
    assert_allclose(vectors.get_vector('east', norm=True), [0, 1])
    assert vectors.similarity('east', 'north') == pytest.approx(1)


@pytest.mark.parametrize(
    'weights',
    [np.array([[1, 2, 3]]), np.array([[1, 2], [3, 4]])],
)
def test_invalid_vector_shape(vectors: KeyedVectors, weights: np.ndarray):
    before = vectors.vectors.copy()

    with pytest.raises(AssertionError, match='Expected `weights` to have shape'):
        vectors.add_vectors(['new'], weights)

    assert 'new' not in vectors
    assert_array_equal(vectors.vectors, before)


@pytest.mark.parametrize(
    'vector', [np.array([[1, 2]]), np.array([1, 2, 3]), np.array(1)]
)
def test_add_vector_requires_one_vector(vectors: KeyedVectors, vector: np.ndarray):
    before = vectors.vectors.copy()

    with pytest.raises(AssertionError, match='Expected `vector` to have shape'):
        vectors.add_vector('new', vector)

    assert 'new' not in vectors
    assert_array_equal(vectors.vectors, before)


def test_add_vector_requires_one_key(vectors: KeyedVectors):
    with pytest.raises(TypeError, match='`key` must be a string or integer'):
        vectors.add_vector(['new'], np.array([1, 2]))  # type: ignore[arg-type]

    assert 'new' not in vectors


@pytest.mark.parametrize('keys', ['new', 42])
def test_add_vectors_requires_key_collection(vectors: KeyedVectors, keys: int | str):
    before = vectors.vectors.copy()

    with pytest.raises(TypeError, match='`keys` must be a collection'):
        vectors.add_vectors(keys, np.array([[1, 2]]))  # type: ignore[arg-type]

    assert 'new' not in vectors
    assert 42 not in vectors
    assert_array_equal(vectors.vectors, before)


def test_add_vectors_singleton_collection():
    model = KeyedVectors(2)
    model.add_vectors([42], np.array([[1, 2]]))

    assert model.idx2key == [42]
    assert_array_equal(model[42], [1, 2])


@pytest.mark.parametrize(
    'weights', [np.array([1, 2]), np.array([1, 2, 3]), np.array(1)]
)
def test_add_vectors_rejects_noncollection_array(
    vectors: KeyedVectors, weights: np.ndarray
):
    before = vectors.vectors.copy()

    with pytest.raises(TypeError):
        vectors.add_vectors(['new'], weights)

    assert 'new' not in vectors
    assert_array_equal(vectors.vectors, before)


@pytest.mark.parametrize(
    'weights',
    [
        [np.array([1, 2]), np.array([3, 4])],
        (np.array([1, 2]), np.array([3, 4])),
    ],
)
def test_add_vectors_accepts_numpy_array_collection(weights: Iterable[np.ndarray]):
    model = KeyedVectors(2)
    model.add_vectors(['first', 'second'], weights)

    assert model.idx2key == ['first', 'second']
    assert_array_equal(model.vectors, [[1, 2], [3, 4]])


@pytest.mark.parametrize('vector', [[1, 2], (1, 2)])
def test_add_vector_requires_numpy_array(
    vectors: KeyedVectors, vector: Iterable[float]
):
    before = vectors.vectors.copy()

    with pytest.raises(TypeError, match='`vector` must be a NumPy array'):
        vectors.add_vector('new', vector)  # type: ignore[arg-type]

    assert 'new' not in vectors
    assert_array_equal(vectors.vectors, before)


@pytest.mark.parametrize('weights', [[[1, 2]], ((1, 2),)])
def test_add_vectors_requires_numpy_array_elements(
    vectors: KeyedVectors, weights: Iterable[Iterable[float]]
):
    before = vectors.vectors.copy()

    with pytest.raises(TypeError, match='must be a collection of NumPy arrays'):
        vectors.add_vectors(['new'], weights)  # type: ignore[arg-type]

    assert 'new' not in vectors
    assert_array_equal(vectors.vectors, before)


def test_normalization_and_zero_vectors(vectors: KeyedVectors):
    expected = [[1, 0], [0, 1], [2**-0.5, 2**-0.5], [-1, 0], [0, 0]]

    assert_allclose(vectors.get_normed_vectors(), expected)
    assert_array_equal(vectors.get_vector('zero', norm=True), [0, 0])

    assert vectors.similarity('zero', 'east') == 0
    assert vectors.distance('zero', 'east') == 1


def test_force_refresh_norms(vectors: KeyedVectors):
    vectors.fill_norms()
    vectors.vectors[0] = [0, 2]
    vectors.fill_norms(force=True)
    assert_allclose(vectors.get_vector('east', norm=True), [0, 1])


@pytest.mark.parametrize(
    'other, expected',
    [('east', 1), ('north', 0), ('northeast', 2**-0.5), ('west', -1)],
)
def test_cosine_similarity(vectors: KeyedVectors, other: str, expected: float):
    similarity = vectors.similarity('east', other)
    distance = vectors.distance('east', other)

    assert similarity == pytest.approx(expected)
    assert distance == pytest.approx(1 - expected)


@pytest.mark.parametrize('top_k', [1, 10])
def test_most_similar_excludes_inputs(vectors: KeyedVectors, top_k: int):
    result = vectors.most_similar(positive=['east', 'north'], top_k=top_k)

    assert result[0][0] == 'northeast'
    assert result[0][1] == pytest.approx(1)
    assert len(result) == min(top_k, 3)
    assert not {'east', 'north'} & {key for key, _ in result}
    assert [score for _, score in result] == sorted(
        [score for _, score in result], reverse=True
    )


def test_negative_and_mixed_queries(vectors: KeyedVectors):
    result = vectors.most_similar(negative='west', top_k=1)
    assert result[0][0] == 'east'
    assert result[0][1] == pytest.approx(1)

    result = vectors.most_similar(positive=['east', 'east'], negative='north', top_k=1)
    assert result[0][0] == 'northeast'
    assert result[0][1] == pytest.approx(1 / math.sqrt(10))


def test_cancelling_query(vectors: KeyedVectors):
    scores = vectors.most_similar(positive='east', negative='east', top_k=None)
    assert_array_equal(scores, np.zeros(len(vectors)))


def test_raw_query_scores_and_vocabulary_limit(vectors: KeyedVectors):
    scores1 = vectors.most_similar(np.array([5, 0]), top_k=None)
    scores2 = vectors.most_similar(np.array([5, 0]), top_k=None, restrict_vocab=2)

    assert_allclose(scores1, [1, 0, 1 / math.sqrt(2), -1, 0])
    assert_allclose(scores2, [1, 0])

    assert vectors.most_similar('east', restrict_vocab=2) == [('north', 0)]
    assert vectors.most_similar('east', restrict_vocab=0) == []


@pytest.mark.parametrize('top_k', [0, -1])
def test_nonpositive_top_k(vectors: KeyedVectors, top_k: int):
    assert vectors.most_similar('east', top_k=top_k) == []


def test_invalid_similarity_queries(vectors: KeyedVectors):
    with pytest.raises(AssertionError, match='no input'):
        vectors.most_similar()
    with pytest.raises(KeyError):
        vectors.most_similar('missing')

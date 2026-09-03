import gzip
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dnnlpy.cs224n.gensim import KeyedVectors, load_word2vec_format


@pytest.mark.parametrize('no_header', [False, True])
@pytest.mark.parametrize('limit', [None, 1, 10])
def test_load_vectors(tmp_path: Path, no_header: bool, limit: int | None):
    path = tmp_path / 'vectors.gz'
    records = 'café 1 2\nnorth 0 4\n'

    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        stream.write(records if no_header else '2 2\n' + records)

    model = load_word2vec_format(
        path,
        no_header=no_header,
        limit=limit,
        dtype=np.float64,
    )
    count = 1 if limit == 1 else 2

    assert model.idx2key == ['café', 'north'][:count]
    assert model.vector_size == 2
    assert model.vectors.dtype == np.float64
    assert_array_equal(model.vectors, [[1, 2], [0, 4]][:count])
    assert model.add_vector('new', np.array([3, 5])) == count


def test_duplicate_and_short_file(tmp_path: Path):
    path = tmp_path / 'vectors.gz'
    with gzip.open(path, 'wt') as stream:
        stream.write('5 2\neast 1 0\neast 9 9\nnorth 0 1\n')

    model = KeyedVectors.load_word2vec_format(path)
    assert len(model) == 2
    assert model.idx2key == ['east', 'north']
    assert model.key2idx == {'east': 0, 'north': 1}
    assert_array_equal(model.vectors, [[1, 0], [0, 1]])
    assert model.vectors.flags.c_contiguous
    assert model.add_vector('west', np.array([-1, 0])) == 2


@pytest.mark.parametrize(
    'contents, no_header, message',
    [
        (b'', True, 'Word2Vec file is empty'),
        (b'1 2\ninvalid\n', False, 'Invalid word2vec record at row 1'),
        (b'1 2\neast 1\n', False, 'to have 2 values, but got 1'),
        (b'east 1 0\nnorth 1\n', True, 'to have 2 values, but got 1'),
    ],
)
def test_invalid_records(
    tmp_path: Path, contents: bytes, no_header: bool, message: str
):
    path = tmp_path / 'invalid.gz'
    path.write_bytes(gzip.compress(contents))

    with pytest.raises(RuntimeError, match=message):
        load_word2vec_format(path, no_header=no_header)


def test_encoding_and_unicode_errors(tmp_path: Path):
    path = tmp_path / 'encoded.gz'
    path.write_bytes(gzip.compress(b'1 2\ncaf\xe9 1 2\n'))

    model = load_word2vec_format(path, encoding='latin-1')
    assert_array_equal(model['café'], [1, 2])

    with pytest.raises(UnicodeDecodeError):
        load_word2vec_format(path)

    model = load_word2vec_format(path, unicode_errors='replace')
    assert_array_equal(model['caf\ufffd'], [1, 2])

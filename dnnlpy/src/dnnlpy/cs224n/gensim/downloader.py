import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .keyedvectors import KeyedVectors

__all__ = ['load']

_MODEL_NAME = 'glove-wiki-gigaword-200'
_FILE_NAME = f'{_MODEL_NAME}.gz'
_DOWNLOAD_URL = f'https://github.com/RaRe-Technologies/gensim-data/releases/download/{_MODEL_NAME}/{_FILE_NAME}'
_MD5 = '59652db361b7a87ee73834a6c391dfc1'


def _base_dir() -> Path:
    gensim_data_dir = os.getenv('GENSIM_DATA_DIR')

    if gensim_data_dir is not None:
        return Path(gensim_data_dir).expanduser()
    else:
        return Path.home() / 'gensim-data'


def _checksum(path: Path) -> str:
    digest = hashlib.md5()

    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)

    return digest.hexdigest()


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f'{_MODEL_NAME}-', dir=dest.parent) as temp:
        temp = Path(temp) / _FILE_NAME
        with (
            urllib.request.urlopen(_DOWNLOAD_URL) as response,
            temp.open('wb') as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)

        if _checksum(temp) != _MD5:
            raise RuntimeError('Downloaded model failed its MD5 checksum.')

        os.replace(temp, dest)


def load(name: str) -> KeyedVectors:
    """Download and load `glove-wiki-gigaword-200`.

    Args:
        name (str): Model name. Only `glove-wiki-gigaword-200` is supported.

    Returns:
        model (KeyedVectors): Loaded vectors or the cached file path.
    """
    if name != _MODEL_NAME:
        raise NotImplementedError(
            f'Unsupported model {name!r}. Only {_MODEL_NAME!r} is supported.'
        )

    path = _base_dir() / name / _FILE_NAME
    if not path.is_file():
        _download(path)

    return KeyedVectors.load_word2vec_format(path)

"""A small vendored module for accessing the pretrained Word2Vec model used in the word-vector notebook.

Since gensim is no longer actively developed and has compatibility issues with Python 3.14, this module provides the minimal functionality needed to load and use the pretrained Word2Vec model without depending on gensim.
"""

from .downloader import load as load
from .keyedvectors import (
    KeyedVectors as KeyedVectors,
    load_word2vec_format as load_word2vec_format,
)

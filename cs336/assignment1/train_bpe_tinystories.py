import time
from collections.abc import Iterator

import datasets as ds
import dnnlpy.tokenizers as dltk

__all__ = ['main']

NUM_STORIES = 2119719
VOCAB_SIZE = 10000
INITIAL_ALPHABET = dltk.ByteLevelPreTokenizer.alphabet()


def batch_iterator(dataset: ds.Dataset, batch_size: int = 1024) -> Iterator[list[str]]:
    """Yield batches of story strings to train_from_iterator."""
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield dataset[start:end]['text']


def main():
    dataset = ds.load_dataset('roneneldan/TinyStories', split='train', num_proc=2)
    assert len(dataset) == NUM_STORIES

    tokenizer = dltk.Tokenizer(
        dltk.BPE(unk_token='<unk>'),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(add_prefix_space=False),
        decoder=dltk.ByteLevelDecoder(),
    )
    text_iterator = batch_iterator(dataset)

    start = time.perf_counter()
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=VOCAB_SIZE,
        initial_alphabet=INITIAL_ALPHABET,
    )
    end = time.perf_counter()
    print(f'Training completed in {end - start:.4f} seconds.')

    file_name = 'bpe_tinystories.json'
    tokenizer.save(file_name)
    print(f'Tokenizer saved to {file_name}.')


if __name__ == '__main__':
    main()

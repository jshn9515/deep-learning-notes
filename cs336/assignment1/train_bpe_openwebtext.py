import time
from collections.abc import Iterator

import datasets as ds
import dnnlpy.tokenizers as dltk

__all__ = ['main']

NUM_STORIES = 8013769
VOCAB_SIZE = 32000
INITIAL_ALPHABET = dltk.ByteLevelPreTokenizer.alphabet()


def batch_iterator(dataset: ds.Dataset, batch_size: int = 1024) -> Iterator[list[str]]:
    """Yield batches of story strings to train_from_iterator."""
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield dataset[start:end]['text']


def main():
    dataset = ds.load_dataset('Skylion007/openwebtext', split='train', num_proc=4)
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
    elapsed = end - start

    print(f'Training completed in {elapsed:.4f} seconds.')

    tokenizer.save('bpe_openwebtext.json')


if __name__ == '__main__':
    main()

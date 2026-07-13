import os
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache

__all__ = [
    'bytes_to_unicode',
    'get_num_workers',
    'has_gil',
    'parallel_map',
    'unicode_to_bytes',
]


def has_gil() -> bool:
    """Check if the current Python interpreter has a Global Interpreter Lock (GIL)."""
    if sys.version_info >= (3, 13):
        return sys._is_gil_enabled()
    return True


def get_num_workers(num_workers: int | None = None) -> int:
    """Get the number of worker threads to use for parallel processing.

    Args:
        num_workers (int | None, optional): The number of workers to use. This function
            can automatically determine the number of workers based on whether the Python
            interpreter has a Global Interpreter Lock (GIL) and the number of CPU cores
            available. If None, the default will be all available CPU cores if GIL is not
            present, or 1/2 of the available CPU cores if GIL is present.

    Returns:
        num_workers (int): The number of worker threads to use for parallel processing.
    """
    if num_workers is None:
        if sys.version_info >= (3, 13):
            num_workers = os.process_cpu_count() or 1
        else:
            num_workers = os.cpu_count() or 1

        if has_gil() and num_workers > 1:
            num_workers = max(1, num_workers // 2)

    return num_workers


def _byte_to_unicode() -> dict[int, str]:
    """Create a mapping from byte values (0-255) to Unicode characters."""
    byte_encoder = {
        byte: chr(codepoint)
        for byte, codepoint in [
            *[(byte, byte) for byte in range(ord('!'), ord('~') + 1)],
            *[(byte, byte) for byte in range(ord('¡'), ord('¬') + 1)],
            *[(byte, byte) for byte in range(ord('®'), ord('ÿ') + 1)],
        ]
    }

    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_encoder:
            byte_encoder[byte] = chr(next_codepoint)
            next_codepoint += 1

    return byte_encoder


BYTE_TO_UNICODE = _byte_to_unicode()
UNICODE_TO_BYTE = {char: byte for byte, char in BYTE_TO_UNICODE.items()}


@lru_cache(maxsize=100_000)
def bytes_to_unicode(text: str) -> str:
    return ''.join(BYTE_TO_UNICODE[byte] for byte in text.encode('utf-8'))


@lru_cache(maxsize=100_000)
def unicode_to_bytes(text: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[char] for char in text)


def parallel_map[T, R](
    func: Callable[[T], R],
    values: Iterable[T],
    num_workers: int | None = None,
    buffersize: int | None = None,
) -> Iterator[R]:
    """Map a function over an iterable using threads or processes.

    Args:
        func (Callable): A callable that takes a single argument and returns a value.
        values (Iterable): An iterable of input values to process.
        num_workers (int, default: 1): The number of worker threads to use for
            parallel processing.
        buffersize (int | None, optional): The maximum number of results to buffer
            before yielding. If None, defaults to `num_workers * 4`.
    """
    num_workers = get_num_workers(num_workers)

    if buffersize is None:
        buffersize = num_workers * 4

    if num_workers <= 1:
        for value in values:
            yield func(value)
    else:
        if has_gil():
            executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)

        with executor:
            if sys.version_info >= (3, 14):
                yield from executor.map(func, values, buffersize=buffersize)
            else:
                yield from _buffered_map(executor, func, values, buffersize)


# TODO: Remove this function when Python 3.14 is the minimum supported version.
def _buffered_map[T, R](
    executor: Executor,
    func: Callable[[T], R],
    values: Iterable[T],
    buffersize: int,
) -> Iterator[R]:
    """Map lazily while keeping at most buffer_size pending futures.

    This function is used by Python < 3.14, where Executor.map does not support
    `buffersize` argument.
    """
    if buffersize < 1:
        raise AssertionError('`buffer_size` must be at least 1.')

    iterator = iter(values)
    pending = deque()

    for _ in range(buffersize):
        try:
            value = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(func, value))

    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            value = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(func, value))

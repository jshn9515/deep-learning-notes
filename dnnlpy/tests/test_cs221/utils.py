import time
from functools import wraps
from typing import Any

__all__ = ['timeout']


def timeout(func: Any | None = None, *, seconds: float = 5) -> Any:
    if func is None:

        def decorator(f: Any) -> Any:
            return timeout(f, seconds=seconds)

        return decorator

    else:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if elapsed > seconds:
                raise TimeoutError(
                    f'{func.__name__} took {elapsed:.2f}s, exceeding {seconds}s.'
                )

            return result

    return wrapper

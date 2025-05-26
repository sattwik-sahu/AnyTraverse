from functools import wraps
from time import perf_counter_ns
from typing import Callable, TypeVar


TReturn = TypeVar("TReturn")


def timeit(func: Callable[..., TReturn]) -> Callable[..., TReturn]:
    @wraps(func)
    def _f(verbose: bool = True, *args, **kwargs) -> TReturn:
        if not verbose:
            return func(*args, **kwargs)
        start_time = perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = perf_counter_ns()
        print(f"{func.__name__} took: {(end_time - start_time) * 1e-6:.4f} ms")
        return result

    return _f

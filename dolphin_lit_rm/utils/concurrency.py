from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Callable, List, Any

def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    max_workers: int = 4,
    chunksize: int = 1,
) -> List[Any]:
    """
    Calls `fn` over `items` with up to `max_workers` threads and
    returns results in *input* order.

    Suitable for I/O-bound work (LLM HTTP calls).  If `max_workers`
    is 1 the call behaves exactly like `map`.
    """
    if max_workers == 1:
        return [fn(x) for x in items]

    futures = {}
    results = [None] * len(items)                     # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for idx, item in enumerate(items):
            futures[pool.submit(fn, item)] = idx
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

"""Bounded CPU batch preparation; no model calls, GPU work or image cache."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


class PrefetchLoader:
    def __init__(self, base, batches):
        self.base = base
        self.dataset = base.dataset
        self.batches = batches
        self._active = False

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        if self._active:
            raise RuntimeError("Concurrent iteration over a prefetch loader is unsupported")
        self._active = True
        pool = None
        pending = deque()
        try:
            source = iter(self.base)
            if not self.batches:
                yield from source
                return
            sentinel = object()
            # Only this one producer calls next(source). Ready batches are capped
            # by the future window; running tasks return small patches, not HRs.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sr-prefetch")
            for _ in range(self.batches):
                pending.append(pool.submit(next, source, sentinel))
            while pending:
                batch = pending.popleft().result()
                if batch is sentinel:
                    break
                pending.append(pool.submit(next, source, sentinel))
                yield batch
        finally:
            for future in pending:
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            self.dataset.close()
            self._active = False

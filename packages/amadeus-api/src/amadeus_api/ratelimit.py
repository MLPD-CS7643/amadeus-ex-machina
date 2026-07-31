"""In-memory sliding-window rate limiter (per client key)."""

import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        if self.max_per_minute <= 0:
            return True
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.max_per_minute:
                return False
            hits.append(now)
            return True

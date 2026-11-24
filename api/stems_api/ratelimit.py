"""Rate limits: at most N events per key in a sliding window, kept in memory.

A key is something like ``upload:owner:<id>`` or ``upload:ip:<address>``. The limits are
per API process; several processes behind a load balancer would each allow the limit, and
would need a shared store instead. Timestamps come from a monotonic clock.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    key: str
    max_events: int
    window_seconds: float


class RateLimited(Exception):
    """A limit is used up. ``retry_after`` is whole seconds until one event is allowed."""

    def __init__(self, retry_after: int, limit: Limit) -> None:
        super().__init__(f"rate limit reached for {limit.key}; retry in {retry_after} s")
        self.retry_after = retry_after
        self.limit = limit


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, limits: Sequence[Limit]) -> None:
        """Count one event against every limit, or count nothing and raise ``RateLimited``.

        All-or-nothing: a request refused by its IP limit does not use up its owner limit.
        """
        now = self._clock()
        with self._lock:
            waits: list[tuple[float, Limit]] = []
            for limit in limits:
                events = self._events.get(limit.key)
                if events is None:
                    continue
                while events and events[0] <= now - limit.window_seconds:
                    events.popleft()
                if not events:
                    del self._events[limit.key]
                elif len(events) >= limit.max_events:
                    # Allowed again when the oldest event that counts leaves the window.
                    oldest = events[len(events) - limit.max_events]
                    waits.append((oldest + limit.window_seconds - now, limit))
            closed = [limit for limit in limits if limit.max_events <= 0]
            if closed:  # a limit of zero switches the thing off
                raise RateLimited(max(1, math.ceil(closed[0].window_seconds)), closed[0])
            if waits:
                wait, limit = max(waits, key=lambda pair: pair[0])
                raise RateLimited(max(1, math.ceil(wait)), limit)
            for limit in limits:
                self._events.setdefault(limit.key, deque()).append(now)

    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._events)

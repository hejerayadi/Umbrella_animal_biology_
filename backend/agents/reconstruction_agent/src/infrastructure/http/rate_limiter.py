"""Client-side request pacing.

NCBI and EMBL-EBI both publish request budgets and will block a caller that
ignores them. Staying under the limit is our responsibility, not something to
discover through 429s.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class AsyncRateLimiter:
    """Token bucket allowing `rate` requests per second, with burst tolerance.

    A bucket rather than a fixed sleep between calls: real usage is bursty, and
    a bucket lets a handful of requests go out immediately after an idle period
    while still holding the long-run average under the limit.

    One limiter instance must be shared by every caller of a given service -
    per-client limiters would each permit the full rate and collectively
    exceed it.
    """

    rate: float
    capacity: float | None = None
    _tokens: float = field(init=False, default=0.0)
    _updated_at: float = field(init=False, default_factory=time.monotonic)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("Rate must be positive.")
        if self.capacity is None:
            self.capacity = self.rate
        self._tokens = float(self.capacity)

    async def acquire(self) -> None:
        """Block until one request may proceed."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._updated_at = now

                assert self.capacity is not None
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Time until the bucket holds a whole token again.
                wait = (1.0 - self._tokens) / self.rate

            # Slept outside the lock so other callers can still be served the
            # moment their own tokens become available.
            await asyncio.sleep(wait)

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

"""Retry rules for external calls.

Retrying the wrong thing is worse than not retrying: replaying a request that
failed deterministically just multiplies the load and delays the error the
caller needs to see.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from ...configuration.logging import get_logger
from ...domain.exceptions import ExternalServiceError, RateLimitError

_log = get_logger(__name__)

T = TypeVar("T")

# 429 and the 5xx family are transient by definition. 4xx others are the
# caller's fault and will fail identically on replay.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with jitter.

    Jitter matters here because this agent fires several searches concurrently:
    without it, retries from the same batch stay synchronised and hit the
    service in the same bursts that caused the failure.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before `attempt` (1-based)."""
        raw = self.initial_delay * (self.multiplier ** (attempt - 1))
        capped = min(raw, self.max_delay)
        spread = capped * self.jitter
        return max(0.0, capped + random.uniform(-spread, spread))

    def should_retry(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, ExternalServiceError):
            return error.retryable
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in _RETRYABLE_STATUS
        # A connect/read timeout may succeed on replay; a malformed request
        # will not, and httpx raises different types for the two.
        return isinstance(error, httpx.TransportError)


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    service: str,
) -> T:
    """Run `operation`, retrying per `policy`.

    The final failure is re-raised unchanged so the caller sees the real cause
    rather than a wrapper hiding it.
    """
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except Exception as error:  # noqa: BLE001 - re-raised below if not retryable
            last_error = error

            if attempt == policy.max_attempts or not policy.should_retry(error):
                raise

            # A 429 that names its own retry-after knows better than our curve.
            delay = policy.delay_for(attempt)
            if isinstance(error, RateLimitError) and error.retry_after:
                delay = max(delay, error.retry_after)

            _log.warning(
                "%s call failed (attempt %d/%d), retrying in %.1fs: %s",
                service,
                attempt,
                policy.max_attempts,
                delay,
                error,
            )
            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error if last_error else ExternalServiceError(service, "retry loop exhausted")

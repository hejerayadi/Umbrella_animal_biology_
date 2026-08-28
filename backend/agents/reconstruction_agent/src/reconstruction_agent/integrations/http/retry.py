"""Transport-level retry for outbound HTTP.

This handles one narrow thing: a request that failed for a reason that has
nothing to do with the science, and might well succeed if repeated - a dropped
connection, a 503, a rate limit. That is a *retry*.

It is deliberately not the same mechanism as a replan. A replan happens when
the request succeeded and the evidence it returned was inadequate, which no
amount of repetition fixes. Keeping the two separate is what stops the agent
from burning its budget re-running a search that worked perfectly and simply
found nothing useful. The agent-level counterpart lives in `orchestration/`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from reconstruction_agent.domain.exceptions import ReconstructionError

#: Statuses worth repeating. Everything else - 400, 401, 404 - describes the
#: request itself, and repeating it produces the same answer more slowly.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to retry, and how long to wait between attempts."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    #: Random fraction added to each delay. Without it, several clients that
    #: failed together retry together, and the upstream service is hit by the
    #: same burst it just rejected.
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before `attempt` (1-based)."""
        base = min(self.initial_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        return base * (1.0 + random.random() * self.jitter)  # noqa: S311 - not cryptographic


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    deadline_remaining: Callable[[], float] | None = None,
) -> T:
    """Run `operation`, repeating it while it fails retryably.

    `deadline_remaining` lets the caller cut the loop short: sleeping four
    seconds before a retry is pointless when only two seconds of the run
    remain, and doing it anyway turns a partial result into no result.
    """
    last_error: ReconstructionError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except ReconstructionError as error:
            if not error.retryable or attempt == policy.max_attempts:
                raise
            last_error = error

            # A server that said how long to wait outranks any curve we would
            # otherwise compute - it knows when it will be ready and we do not.
            retry_after = getattr(error, "retry_after", None)
            delay = float(retry_after) if retry_after else policy.delay_for(attempt)

            if deadline_remaining is not None and delay >= deadline_remaining():
                raise

            await asyncio.sleep(delay)

    # Unreachable: the loop either returns or raises. Present so the function
    # has no implicit None path for a type checker to complain about.
    raise last_error if last_error else RuntimeError("retry loop exited without a result")

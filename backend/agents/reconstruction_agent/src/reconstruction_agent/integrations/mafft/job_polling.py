"""Waiting for an EMBL-EBI job to finish.

EMBL-EBI job services share one shape - POST to run, poll a status endpoint,
then fetch a result. Only MAFFT is still used here: homology search moved to
the NCBI BLAST URL API, which is a different shape (an opaque request id and a
much stricter polling etiquette) and does its own waiting.

The timeout is a real deadline, not a formality. This agent answers inside a
single HTTP call with a fixed budget, so a poll that outlives the run is worse
than useless: it holds the run open past the point where a partial answer could
still have been returned. Callers pass the time they can actually afford.
"""

from __future__ import annotations

import asyncio
import time

from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError, ServiceTimeoutError
from reconstruction_agent.integrations.http.client import ServiceClient

#: Terminal states meaning the result can be collected.
_DONE = frozenset({"FINISHED"})
#: Terminal states meaning it never will be.
_FAILED = frozenset({"ERROR", "FAILURE", "NOT_FOUND"})


async def poll_until_complete(
    client: ServiceClient,
    service: str,
    job_id: str,
    *,
    interval: float = 5.0,
    timeout: float = 280.0,
    timeout_code: ErrorCode = ErrorCode.UPSTREAM_UNAVAILABLE,
) -> str:
    """Block until `job_id` finishes, returning its final status.

    Raises rather than returning a failed status: a caller that has to check
    both an exception and a return value has two things to get wrong.
    """
    deadline = time.monotonic() + timeout

    while True:
        status = (await client.get_text(f"/{service}/status/{job_id}")).strip().upper()

        if status in _DONE:
            return status

        if status in _FAILED:
            raise ExternalServiceError(
                service,
                f"job {job_id} ended in state {status}",
                retryable=False,
                details={"job_id": job_id, "status": status},
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceTimeoutError(
                service,
                f"job {job_id} did not finish within {timeout:.0f}s (last status {status})",
                code=timeout_code,
                details={"job_id": job_id, "status": status},
            )

        # Never sleep past the deadline - the remaining time belongs to
        # finalising a partial answer, not to one more hopeful poll.
        await asyncio.sleep(min(interval, remaining))

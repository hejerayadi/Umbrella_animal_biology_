"""The submit-then-poll loop shared by EMBL-EBI's BLAST and MAFFT services.

Both work the same way: POST a job, get an id, poll until the status stops
being RUNNING, then GET the result. Only the endpoint paths differ, so the
loop lives here once.
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum

from ...configuration.logging import get_logger
from ...domain.exceptions import ExternalServiceError, JobTimeoutError
from ..http.client import ServiceClient

_log = get_logger(__name__)


class JobStatus(str, Enum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    FAILURE = "FAILURE"
    NOT_FOUND = "NOT_FOUND"

    @classmethod
    def parse(cls, raw: str) -> JobStatus:
        """Map a status string, treating anything unrecognised as still running.

        Optimistic on purpose: an unknown status is far more likely to be a new
        transient state than a silent success, and the poll timeout is what
        bounds the wait either way.
        """
        try:
            return cls(raw.strip().upper())
        except ValueError:
            _log.debug("Unrecognised EMBL-EBI job status %r; treating as RUNNING.", raw)
            return cls.RUNNING

    @property
    def is_terminal(self) -> bool:
        return self is not JobStatus.RUNNING


async def poll_until_complete(
    client: ServiceClient,
    service: str,
    job_id: str,
    *,
    interval: float = 5.0,
    timeout: float = 600.0,
) -> None:
    """Block until `job_id` leaves RUNNING, or raise.

    Raises `JobTimeoutError` on timeout (retryable - the job may simply be
    queued) and `ExternalServiceError` when the job itself failed (not
    retryable - resubmitting identical input fails identically).
    """
    started = time.monotonic()

    while True:
        raw = await client.get_text(f"/{service}/status/{job_id}")
        status = JobStatus.parse(raw)

        if status is JobStatus.FINISHED:
            _log.info(
                "%s job %s finished in %.1fs.", service, job_id, time.monotonic() - started
            )
            return

        if status.is_terminal:
            raise ExternalServiceError(
                service, f"job {job_id} ended with status {status.value}", retryable=False
            )

        waited = time.monotonic() - started
        if waited > timeout:
            raise JobTimeoutError(service, job_id, waited)

        await asyncio.sleep(interval)

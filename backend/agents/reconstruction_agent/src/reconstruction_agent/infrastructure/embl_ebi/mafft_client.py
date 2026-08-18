"""EMBL-EBI MAFFT REST service: multiple sequence alignment."""
from __future__ import annotations

from ...configuration.logging import get_logger
from ...configuration.settings import EMBLEBISettings
from ...domain.exceptions import ExternalServiceError
from ..http.client import ServiceClient
from ..http.retry import RetryPolicy
from .job_polling import poll_until_complete

_log = get_logger(__name__)

_SERVICE = "mafft"


class MafftClient:
    """Aligns a FASTA block and returns the aligned FASTA."""

    def __init__(self, settings: EMBLEBISettings, *, timeout: float = 30.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service=_SERVICE,
            base_url=settings.base_url,
            timeout=timeout,
            requests_per_second=1.0,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=2.0),
        )

    async def submit(self, fasta: str, *, output_format: str = "fasta") -> str:
        """Queue an alignment of the sequences in `fasta`, returning the job id."""
        if not self._settings.contact_email:
            raise ExternalServiceError(
                _SERVICE,
                "EMBL_EBI_CONTACT_EMAIL is required; EMBL-EBI rejects anonymous submissions.",
                retryable=False,
            )

        response = await self._client.post(
            f"/{_SERVICE}/run",
            data={
                "email": self._settings.contact_email,
                "stype": "dna",
                "sequence": fasta,
                "format": output_format,
            },
        )
        job_id = response.text.strip()
        if not job_id:
            raise ExternalServiceError(_SERVICE, "run returned an empty job id")

        _log.info("Submitted MAFFT job %s.", job_id)
        return job_id

    async def result(self, job_id: str, *, result_type: str = "aln-fasta") -> str:
        """The aligned FASTA for `job_id`, waiting for it if still running."""
        await poll_until_complete(
            self._client,
            _SERVICE,
            job_id,
            interval=self._settings.poll_interval_seconds,
            timeout=self._settings.poll_timeout_seconds,
        )
        return await self._client.get_text(f"/{_SERVICE}/result/{job_id}/{result_type}")

    async def align(self, fasta: str) -> str:
        """Submit and wait in one call - the common case."""
        job_id = await self.submit(fasta)
        return await self.result(job_id)

    async def aclose(self) -> None:
        await self._client.aclose()

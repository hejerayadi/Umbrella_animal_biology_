"""EMBL-EBI MAFFT REST service: multiple sequence alignment.

The one EMBL-EBI service the agent still uses. Homology search runs against the
NCBI BLAST URL API; alignment stays here because NCBI publishes no alignment
service and no MAFFT binary is installed anywhere in this repository.

MAFFT aligns and nothing more. It does not reconstruct anything, and it has no
concept of the gap being filled: it is handed the target flanks and a set of
homologues, and it reports which residues line up with which. All of the
scientific content is in reading that output, which happens in the alignment
service rather than here.
"""

from __future__ import annotations

from reconstruction_agent.config.settings import EmblEbiSettings
from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError
from reconstruction_agent.integrations.http.client import ServiceClient
from reconstruction_agent.integrations.http.retry import RetryPolicy
from reconstruction_agent.integrations.mafft.job_polling import poll_until_complete

SERVICE = "mafft"


class MafftClient:
    """Aligns a FASTA block and returns the aligned FASTA."""

    def __init__(self, settings: EmblEbiSettings, *, timeout: float = 30.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service=SERVICE,
            base_url=settings.base_url,
            timeout=timeout,
            requests_per_second=1.0,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=2.0),
        )

    async def submit(self, fasta: str, *, output_format: str = "fasta") -> str:
        """Queue an alignment of the sequences in `fasta`, returning the job id."""
        if not self._settings.contact_email:
            raise ExternalServiceError(
                SERVICE,
                "EMBL_EBI_CONTACT_EMAIL is required; EMBL-EBI rejects anonymous submissions.",
                code=ErrorCode.ALIGNMENT_FAILED,
                retryable=False,
            )

        response = await self._client.post(
            f"/{SERVICE}/run",
            data={
                "email": self._settings.contact_email,
                "stype": "dna",
                "sequence": fasta,
                "format": output_format,
            },
        )
        job_id = response.text.strip()
        if not job_id:
            raise ExternalServiceError(
                SERVICE, "run returned an empty job id", code=ErrorCode.ALIGNMENT_FAILED
            )
        return job_id

    async def result(
        self,
        job_id: str,
        *,
        result_type: str = "aln-fasta",
        timeout: float | None = None,
    ) -> str:
        """The aligned FASTA for `job_id`, waiting for it if still running."""
        await poll_until_complete(
            self._client,
            SERVICE,
            job_id,
            interval=self._settings.poll_interval_seconds,
            timeout=timeout if timeout is not None else self._settings.poll_timeout_seconds,
            timeout_code=ErrorCode.ALIGNMENT_FAILED,
        )
        return await self._client.get_text(f"/{SERVICE}/result/{job_id}/{result_type}")

    async def align(self, fasta: str, *, timeout: float | None = None) -> str:
        """Submit and wait in one call - the common case."""
        job_id = await self.submit(fasta)
        return await self.result(job_id, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

"""EMBL-EBI NCBI-BLAST REST service: find sequences homologous to a query."""
from __future__ import annotations

from ...configuration.logging import get_logger
from ...configuration.settings import EMBLEBISettings
from ...domain.exceptions import ExternalServiceError
from ..http.client import ServiceClient
from ..http.retry import RetryPolicy
from .job_polling import poll_until_complete

_log = get_logger(__name__)

_SERVICE = "ncbiblast"


class BlastClient:
    """Submits a BLAST search and returns the raw result payload.

    Returns text; turning it into `Reference` objects is `tools/blast/mapper`.
    """

    def __init__(self, settings: EMBLEBISettings, *, timeout: float = 30.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service=_SERVICE,
            base_url=settings.base_url,
            timeout=timeout,
            # EBI publishes no hard public rate; this is self-imposed courtesy
            # pacing so a multi-gap run cannot hammer the service.
            requests_per_second=1.0,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=2.0),
        )

    async def submit(
        self,
        sequence: str,
        *,
        database: str = "em_rel",
        program: str = "blastn",
        max_hits: int = 50,
        expect: float = 1e-5,
    ) -> str:
        """Queue a search, returning its job id."""
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
                "program": program,
                "database": database,
                "stype": "dna",
                "sequence": sequence,
                "alignments": max_hits,
                "scores": max_hits,
                "exp": expect,
            },
        )
        job_id = response.text.strip()
        if not job_id:
            raise ExternalServiceError(_SERVICE, "run returned an empty job id")

        _log.info("Submitted BLAST job %s (%d bp query).", job_id, len(sequence))
        return job_id

    async def result(self, job_id: str, *, result_type: str = "json") -> str:
        """The finished result for `job_id`, waiting for it if still running."""
        await poll_until_complete(
            self._client,
            _SERVICE,
            job_id,
            interval=self._settings.poll_interval_seconds,
            timeout=self._settings.poll_timeout_seconds,
        )
        return await self._client.get_text(f"/{_SERVICE}/result/{job_id}/{result_type}")

    async def search(self, sequence: str, **kwargs: object) -> str:
        """Submit and wait in one call - the common case."""
        job_id = await self.submit(sequence, **kwargs)  # type: ignore[arg-type]
        return await self.result(job_id)

    async def aclose(self) -> None:
        await self._client.aclose()

"""EMBL-EBI NCBI-BLAST REST service: find sequences homologous to a query."""
from __future__ import annotations

import math

from configuration.logging import get_logger
from configuration.settings import EMBLEBISettings
from domain.exceptions import ExternalServiceError
from infrastructure.embl_ebi.job_polling import poll_until_complete
from infrastructure.http.client import ServiceClient
from infrastructure.http.retry import RetryPolicy

_log = get_logger(__name__)

_SERVICE = "ncbiblast"

# EMBL-EBI validates `exp`, `alignments` and `scores` against fixed lists and
# rejects anything else with a bare HTTP 400 naming the parameter. Two traps
# here: Python renders 1e-5 as the string "1e-05", which is NOT in the list;
# and any hit count that is not one of these exact values fails. Both are
# snapped rather than passed through, so a caller can ask for 37 hits or an
# e-value of 2e-6 and get the nearest legal setting instead of a 400.
#
# Live list: GET /ncbiblast/parameterdetails/{exp,alignments,scores}
_EXPECT_VALUES: tuple[tuple[float, str], ...] = (
    (1e-200, "1e-200"),
    (1e-100, "1e-100"),
    (1e-50, "1e-50"),
    (1e-10, "1e-10"),
    (1e-5, "1e-5"),
    (1e-4, "1e-4"),
    (1e-3, "1e-3"),
    (1e-2, "1e-2"),
    (1e-1, "1e-1"),
    (1.0, "1.0"),
    (10.0, "10"),
    (100.0, "100"),
    (1000.0, "1000"),
)

_HIT_COUNTS: tuple[int, ...] = (0, 5, 10, 20, 50, 100, 150, 200, 250, 500, 750, 1000)


def _snap_expect(expect: float) -> str:
    """The allowed `exp` string closest to `expect`, compared in log space.

    Log space because these values span 200 orders of magnitude: on a linear
    scale every value below 1 would collapse onto the same neighbour.
    """
    target = math.log10(expect) if expect > 0 else -300.0
    return min(_EXPECT_VALUES, key=lambda item: abs(math.log10(item[0]) - target))[1]


def _snap_hit_count(count: int) -> int:
    """The smallest allowed hit count that is at least `count`.

    Rounds up so a caller never silently gets fewer hits than they asked for.
    """
    for allowed in _HIT_COUNTS:
        if allowed >= count:
            return allowed
    return _HIT_COUNTS[-1]


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
        database: str = "em_cds_std_vrt",
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

        hits = _snap_hit_count(max_hits)
        response = await self._client.post(
            f"/{_SERVICE}/run",
            data={
                "email": self._settings.contact_email,
                "program": program,
                "database": database,
                "stype": "dna",
                "sequence": sequence,
                "alignments": hits,
                "scores": hits,
                "exp": _snap_expect(expect),
            },
        )
        job_id = response.text.strip()
        if not job_id:
            raise ExternalServiceError(_SERVICE, "run returned an empty job id")

        _log.info("blast_submitted", job_id=job_id, query_length=len(sequence))
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

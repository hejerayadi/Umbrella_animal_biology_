"""NCBI E-utilities access: find accessions, fetch the sequences behind them."""
from __future__ import annotations

from typing import Any

from configuration.logging import get_logger
from configuration.settings import NCBISettings
from domain.exceptions import ExternalServiceError
from infrastructure.http.client import ServiceClient
from infrastructure.http.retry import RetryPolicy

_log = get_logger(__name__)


class NCBIClient:
    """Thin wrapper over esearch/efetch.

    Speaks HTTP and returns raw text; parsing FASTA into domain objects is the
    tool mapper's job, not this layer's.
    """

    def __init__(self, settings: NCBISettings, *, timeout: float = 30.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service="ncbi",
            base_url=settings.base_url,
            timeout=timeout,
            # Honours the 3/s anonymous and 10/s keyed budgets.
            requests_per_second=settings.requests_per_second,
            retry_policy=RetryPolicy(max_attempts=3),
        )

    def _identity_params(self) -> dict[str, str]:
        """NCBI asks every caller to identify itself; a keyed caller gets 10/s."""
        params: dict[str, str] = {"tool": self._settings.tool_name}
        if self._settings.contact_email:
            params["email"] = self._settings.contact_email
        if self._settings.api_key:
            params["api_key"] = self._settings.api_key
        return params

    async def search(
        self, term: str, *, database: str = "nuccore", limit: int = 20
    ) -> list[str]:
        """Accession-level ids matching `term`, best match first."""
        payload = await self._client.get_json(
            "/esearch.fcgi",
            params={
                **self._identity_params(),
                "db": database,
                "term": term,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        try:
            return list(payload["esearchresult"]["idlist"])
        except (KeyError, TypeError) as error:
            raise ExternalServiceError(
                "ncbi", f"unexpected esearch payload: {payload!r}"
            ) from error

    async def fetch_fasta(self, identifiers: list[str], *, database: str = "nuccore") -> str:
        """Raw multi-FASTA for the given ids.

        Batched into one call rather than one per id: efetch accepts a
        comma-separated list, and each round trip costs against the rate budget.
        """
        if not identifiers:
            return ""

        return await self._client.get_text(
            "/efetch.fcgi",
            params={
                **self._identity_params(),
                "db": database,
                "id": ",".join(identifiers),
                "rettype": "fasta",
                "retmode": "text",
            },
        )

    async def fetch_region(
        self,
        identifier: str,
        start: int,
        stop: int,
        *,
        database: str = "nuccore",
        strand: int = 1,
    ) -> str:
        """FASTA for one subject's coordinate range.

        A BLAST hit names a region of a record, not the whole record - and the
        whole record can be a chromosome. Asking for just the range keeps the
        response in kilobytes and, since everything gathered is checkpointed,
        keeps the checkpoint small too.

        `strand=2` asks NCBI for the minus strand, which is cheaper and less
        error-prone than reverse-complementing the result here.
        """
        params: dict[str, Any] = {
            **self._identity_params(),
            "db": database,
            "id": identifier,
            "rettype": "fasta",
            "retmode": "text",
            "seq_start": max(1, start),
            "seq_stop": max(1, stop),
        }
        if strand < 0:
            params["strand"] = 2

        return await self._client.get_text("/efetch.fcgi", params=params)

    async def fetch_taxonomy(self, identifier: str) -> str:
        """The taxonomy record for one taxid, as XML.

        XML rather than JSON because `esummary` on the taxonomy database does
        not return the lineage, and `efetch` only speaks XML for it.
        """
        return await self._client.get_text(
            "/efetch.fcgi",
            params={
                **self._identity_params(),
                "db": "taxonomy",
                "id": identifier,
                "retmode": "xml",
            },
        )

    async def fetch_summary(
        self, identifiers: list[str], *, database: str = "nuccore"
    ) -> dict[str, Any]:
        """Metadata (title, organism, length) without pulling the residues."""
        if not identifiers:
            return {}

        payload = await self._client.get_json(
            "/esummary.fcgi",
            params={
                **self._identity_params(),
                "db": database,
                "id": ",".join(identifiers),
                "retmode": "json",
            },
        )
        return payload.get("result", {}) if isinstance(payload, dict) else {}

    async def aclose(self) -> None:
        await self._client.aclose()

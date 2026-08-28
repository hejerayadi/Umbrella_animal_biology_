"""The official NCBI BLAST URL API - a second homology search provider.

A different service from EMBL-EBI `ncbiblast`, despite the confusing name. EBI
runs NCBI's BLAST *program* over ENA divisions; this is NCBI's own public
server, searching NCBI's own databases (`nt`, `core_nt`, `refseq_genomes`).

Two things make it a genuinely better fit for this agent than EBI, and one
makes it worse.

**Better: taxonomic scoping is exact.** EBI offers fixed taxonomic divisions
and the agent has to work out which one covers the target by resolving a
collection label to a clade. NCBI takes `ENTREZ_QUERY=txid40674[ORGN]` and
restricts the search to that clade directly - the tax id the agent already
holds, with no label matching and no guessing. Any rank can be requested, so
the search can be scoped at genus, family or order rather than whatever
granularity a provider happened to publish.

**Better: the target can be excluded.** `NOT <accession>[ACCN]` removes the
record being repaired from its own search. A record cannot be evidence about
its own unresolved region, and in a ground-truth benchmark - where bases are
withheld artificially but the public record still has them - failing to
exclude it turns the whole measurement into a lookup.

**Worse: the pacing rules are strict.** NCBI asks callers not to contact the
server more than once every 10 seconds, and not to poll a single search more
than once a minute. Both are honoured here. The consequence is that completion
is *detected* slowly even when the search itself was fast, which is a real cost
and shows up directly in any latency comparison against EBI.
"""

from __future__ import annotations

import asyncio
import re
import time

from reconstruction_agent.config.settings import NcbiBlastSettings
from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ExternalServiceError, ServiceTimeoutError
from reconstruction_agent.integrations.http.client import ServiceClient
from reconstruction_agent.integrations.http.retry import RetryPolicy
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)

SERVICE = "ncbi_blast"

#: The submission response is an HTML page with the identifiers in a comment
#: block. There is no JSON form of it, so they are read out by pattern.
_RID = re.compile(r"^\s*RID\s*=\s*(\S+)", re.MULTILINE)
#: Request Time Of Execution - NCBI's own estimate, in seconds. Used to wait
#: roughly the right amount before the first poll instead of polling blind.
_RTOE = re.compile(r"^\s*RTOE\s*=\s*(\d+)", re.MULTILINE)
_STATUS = re.compile(r"\s*Status=(\w+)")
_THERE_ARE_HITS = re.compile(r"\s*ThereAreHits=(\w+)")
#: NCBI reports a rejected submission as an HTML page rather than a 4xx, so
#: the reason has to be lifted out of the markup. Without this the caller
#: sees a megabyte of HTML and no explanation.
_PAGE_ERROR = re.compile(r"(Message ID#\d+[^<]*)")


class NcbiBlastClient:
    """Submits a search to the NCBI BLAST URL API and returns the raw XML."""

    def __init__(self, settings: NcbiBlastSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service=SERVICE,
            base_url=settings.base_url,
            timeout=timeout,
            # NCBI asks for no more than one request every 10 seconds. This is
            # not a courtesy setting: exceeding their limits earns a block, and
            # a blocked address takes the sequence and taxonomy calls down with
            # it because they share the host.
            requests_per_second=settings.requests_per_second,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay=10.0),
        )

    def _identity(self) -> dict[str, str]:
        """The tool/email parameters NCBI asks every caller to send.

        No API key: NCBI issues keys for E-utilities only, and the BLAST URL
        API neither accepts nor is rate-limited by one. Identification plus the
        pacing above is the whole of what this service asks for.
        """
        params = {"tool": self._settings.tool_name}
        if self._settings.contact_email:
            params["email"] = self._settings.contact_email
        return params

    async def submit(
        self,
        sequence: str,
        *,
        database: str | None = None,
        program: str | None = None,
        max_hits: int = 50,
        expect: float = 1e-5,
        entrez_query: str | None = None,
    ) -> tuple[str, int]:
        """Queue a search, returning its request id and NCBI's time estimate."""
        params = {
            **self._identity(),
            "CMD": "Put",
            "PROGRAM": program or self._settings.program,
            "DATABASE": database or self._settings.database,
            "QUERY": sequence,
            "HITLIST_SIZE": str(max_hits),
            "EXPECT": str(expect),
            "FILTER": "L",
        }
        if self._settings.megablast:
            params["MEGABLAST"] = "on"
        if entrez_query:
            params["ENTREZ_QUERY"] = entrez_query

        # POST rather than GET: a 1 kb query plus an Entrez expression exceeds
        # what is safe in a URL, and NCBI silently truncates rather than
        # rejecting, which would search a shortened query without saying so.
        response = await self._client.post(self._settings.endpoint_path, data=params)
        body = response.text

        match = _RID.search(body)
        if not match:
            reported = _PAGE_ERROR.search(body)
            detail = reported.group(1).strip() if reported else body[:200]
            raise ExternalServiceError(
                SERVICE,
                f"submission was rejected: {detail}",
                code=ErrorCode.BLAST_SUBMISSION_FAILED,
                retryable=False,
            )

        rid = match.group(1)
        estimate = int(m.group(1)) if (m := _RTOE.search(body)) else 30
        _log.info(
            "ncbi_blast_submitted",
            rid=rid,
            estimate_seconds=estimate,
            database=params["DATABASE"],
            entrez_query=entrez_query,
            query_length=len(sequence),
        )
        return rid, estimate

    async def wait(self, rid: str, *, estimate: int = 30, timeout: float) -> bool:
        """Block until `rid` finishes, reporting whether it found anything.

        Waits out NCBI's own estimate before the first poll, then polls at the
        interval NCBI asks for. Polling faster would finish sooner and is
        exactly what gets an address blocked.
        """
        deadline = time.monotonic() + timeout
        first_wait = min(max(estimate, 5), max(timeout - 1.0, 1.0))
        await asyncio.sleep(first_wait)

        while True:
            body = await self._client.get_text(
                self._settings.endpoint_path,
                params={
                    **self._identity(),
                    "CMD": "Get",
                    "RID": rid,
                    "FORMAT_OBJECT": "SearchInfo",
                },
            )
            status = m.group(1).upper() if (m := _STATUS.search(body)) else "UNKNOWN"

            if status == "READY":
                hits = _THERE_ARE_HITS.search(body)
                return bool(hits and hits.group(1).lower() == "yes")

            if status in {"FAILED", "UNKNOWN"}:
                raise ExternalServiceError(
                    SERVICE,
                    f"search {rid} ended in state {status}",
                    code=ErrorCode.BLAST_SUBMISSION_FAILED,
                    retryable=False,
                    details={"rid": rid, "status": status},
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceTimeoutError(
                    SERVICE,
                    f"search {rid} did not finish within {timeout:.0f}s (last status {status})",
                    code=ErrorCode.BLAST_TIMEOUT,
                    details={"rid": rid, "status": status},
                )
            # Never sleep past the deadline - the time left belongs to
            # finalising a partial answer, not to one more hopeful poll.
            await asyncio.sleep(min(self._settings.poll_interval_seconds, remaining))

    async def result(self, rid: str) -> str:
        """The finished search as BLAST XML."""
        return await self._client.get_text(
            self._settings.endpoint_path,
            params={
                **self._identity(),
                "CMD": "Get",
                "RID": rid,
                "FORMAT_TYPE": self._settings.format_type,
            },
        )

    async def search(
        self,
        sequence: str,
        *,
        database: str | None = None,
        max_hits: int = 50,
        expect: float = 1e-5,
        entrez_query: str | None = None,
        timeout: float = 600.0,
    ) -> str:
        """Submit and wait in one call - the common case.

        Returns an empty BLAST XML document when the search completed with no
        hits, so a caller never has to distinguish "nothing found" from "no
        result fetched".
        """
        rid, estimate = await self.submit(
            sequence,
            database=database,
            max_hits=max_hits,
            expect=expect,
            entrez_query=entrez_query,
        )
        found = await self.wait(rid, estimate=estimate, timeout=timeout)
        if not found:
            return ""
        return await self.result(rid)

    async def aclose(self) -> None:
        await self._client.aclose()

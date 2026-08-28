"""NCBI E-utilities: sequences, record metadata, and taxonomy lookups.

NCBI asks every caller to identify itself and to stay under a request rate:
three per second anonymously, ten with an API key. Both are handled by the
shared `ServiceClient` plus the `tool`/`email` parameters attached to every
call. Exceeding the rate earns a block rather than a 429, so the pacing here is
not merely polite.
"""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

from reconstruction_agent.config.settings import NcbiSettings
from reconstruction_agent.domain.enums import MoleculeType
from reconstruction_agent.domain.exceptions import ExternalServiceError, SequenceNotFoundError
from reconstruction_agent.domain.models.sequence import SequenceRecord
from reconstruction_agent.integrations.http.client import ServiceClient
from reconstruction_agent.integrations.http.retry import RetryPolicy

SERVICE = "ncbi"

#: NCBI's `genome` summary field, mapped onto the molecule types we model.
#: Anything unlisted stays UNKNOWN rather than being guessed - a wrong molecule
#: type steers the reference search at the wrong collection entirely.
_GENOME_FIELD_TO_MOLECULE = {
    "mitochondrion": MoleculeType.MITOCHONDRION,
    "chloroplast": MoleculeType.CHLOROPLAST,
    "plastid": MoleculeType.CHLOROPLAST,
    "plasmid": MoleculeType.PLASMID,
    "genomic": MoleculeType.GENOMIC_DNA,
    "chromosome": MoleculeType.GENOMIC_DNA,
}


class NcbiClient:
    """Read access to NCBI E-utilities."""

    def __init__(self, settings: NcbiSettings, *, timeout: float = 30.0) -> None:
        self._settings = settings
        self._client = ServiceClient(
            service=SERVICE,
            base_url=settings.base_url,
            timeout=timeout,
            requests_per_second=settings.requests_per_second,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=1.0),
        )

    def _identity(self) -> dict[str, str]:
        """The tool/email parameters NCBI asks every caller to send."""
        params = {"tool": self._settings.tool_name}
        if self._settings.contact_email:
            params["email"] = self._settings.contact_email
        if self._settings.api_key:
            params["api_key"] = self._settings.api_key.get_secret_value()
        return params

    async def esearch(self, db: str, term: str, *, retmax: int = 20) -> list[str]:
        """Ids in `db` matching `term`, most relevant first."""
        payload = await self._client.get_json(
            "/esearch.fcgi",
            params={
                **self._identity(),
                "db": db,
                "term": term,
                "retmode": "json",
                "retmax": retmax,
            },
        )
        result = (payload or {}).get("esearchresult") or {}
        return [str(item) for item in result.get("idlist", [])]

    async def esummary(self, db: str, identifier: str) -> dict[str, Any]:
        """The document summary for one record."""
        payload = await self._client.get_json(
            "/esummary.fcgi",
            params={**self._identity(), "db": db, "id": identifier, "retmode": "json"},
        )
        result = (payload or {}).get("result") or {}
        uids = result.get("uids") or []
        if not uids:
            raise SequenceNotFoundError(
                f"NCBI has no {db} record for {identifier!r}.",
                details={"accession": identifier, "db": db},
            )
        summary = result.get(str(uids[0])) or {}
        return dict(summary)

    async def efetch_text(self, db: str, identifier: str, **extra: str) -> str:
        """A record in a text format (FASTA, GenBank flat file, ...)."""
        return await self._client.get_text(
            "/efetch.fcgi",
            params={**self._identity(), "db": db, "id": identifier, **extra},
        )

    async def fetch_sequence_window(self, accession: str, start: int, end: int) -> str:
        """The residues of `accession` between `start` and `end`, 1-based inclusive.

        A BLAST hit against a mammalian genomic record names an accession that
        can be hundreds of kilobases long, while the region that actually
        aligned is a few hundred bases of it. Fetching the whole record to use
        that fragment costs the bandwidth, the deadline, and - the reason this
        method exists - the alignment itself: MAFFT handed a 1 kb query and a
        181 kb subject produces an alignment in which the query is a rounding
        error, and the gap columns read off it are meaningless.

        NCBI applies the range server-side via `seq_start`/`seq_stop`, so the
        oversized record is never transferred at all.
        """
        low, high = (start, end) if start <= end else (end, start)
        fasta = await self.efetch_text(
            "nuccore",
            accession,
            rettype="fasta",
            retmode="text",
            seq_start=str(max(low, 1)),
            seq_stop=str(max(high, 1)),
        )
        return _residues_from_fasta(fasta)

    async def efetch_xml(self, db: str, identifier: str) -> ElementTree.Element:
        """A record as parsed XML.

        Taxonomy is only served as XML - there is no JSON representation that
        carries the full lineage - so this is the path every lineage lookup
        takes.
        """
        raw = await self.efetch_text(db, identifier, retmode="xml")
        try:
            return ElementTree.fromstring(raw)
        except ElementTree.ParseError as error:
            raise ExternalServiceError(
                SERVICE, f"could not parse {db} XML for {identifier!r}: {error}"
            ) from error

    async def fetch_sequence(self, accession: str) -> SequenceRecord:
        """One nucleotide record, with its identity and its bases.

        Two calls rather than one: the summary carries the organism, tax id and
        molecule type, and the FASTA carries the residues. Fetching the full
        GenBank flat file would supply both, but for a scaffold it is orders of
        magnitude larger than the parts actually needed.
        """
        summary = await self.esummary("nuccore", accession)
        fasta = await self.efetch_text("nuccore", accession, rettype="fasta", retmode="text")
        residues = _residues_from_fasta(fasta)
        if not residues:
            raise SequenceNotFoundError(
                f"NCBI returned no sequence for {accession!r}.",
                details={"accession": accession},
            )

        tax_id = summary.get("taxid")
        return SequenceRecord(
            accession=str(summary.get("accessionversion") or accession),
            residues=residues,
            description=str(summary.get("title") or ""),
            organism=str(summary.get("organism") or "") or None,
            tax_id=int(tax_id) if tax_id else None,
            molecule_type=_molecule_type_from_summary(summary),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _residues_from_fasta(raw: str) -> str:
    """The bases from a single-record FASTA, upper-cased and unwrapped."""
    lines = [line.strip() for line in raw.splitlines()]
    return "".join(line for line in lines if line and not line.startswith(">")).upper()


def _molecule_type_from_summary(summary: dict[str, Any]) -> MoleculeType:
    """The molecule type NCBI reports for a record.

    `genome` names the replicon ("mitochondrion", "chromosome"); `biomol` is
    consulted only as a fallback, because it distinguishes DNA from RNA rather
    than nuclear from organellar, which is the distinction that matters here.
    """
    genome = str(summary.get("genome") or "").strip().lower()
    if genome in _GENOME_FIELD_TO_MOLECULE:
        return _GENOME_FIELD_TO_MOLECULE[genome]

    biomol = str(summary.get("biomol") or "").strip().lower()
    if biomol in {"genomic", "genomic dna"}:
        return MoleculeType.GENOMIC_DNA
    return MoleculeType.UNKNOWN

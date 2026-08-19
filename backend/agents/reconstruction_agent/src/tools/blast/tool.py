"""Homology search over a gap's flanking context, as a selectable tool."""
from __future__ import annotations

import asyncio

from configuration.logging import get_logger
from contracts.output import EvidenceItem
from domain.exceptions import ReconstructionError
from domain.models import Reference
from infrastructure.embl_ebi.blast_client import BlastClient
from infrastructure.ncbi.client import NCBIClient
from tools.blast.mapper import HitSpan, to_references
from tools.blast.schemas import BlastSearchInput, BlastSearchOutput
from tools.contracts import Tool
from tools.ncbi.mapper import to_references as fasta_to_references

_log = get_logger(__name__)

#: How many hits are worth paying a fetch for. MAFFT aligns at most eight
#: references, so retrieving residues for every one of fifty hits would spend
#: dozens of round trips on sequences that will never be aligned. Ranked hits
#: come back best-first, so the ones taken are the ones that would be used.
_MAX_SEQUENCE_FETCHES = 12

#: A single fetched region longer than this is not the local homology we asked
#: for. Everything gathered is checkpointed, so an unbounded region would end
#: up serialised into Postgres.
_MAX_FETCHED_RESIDUES = 100_000


class BlastSearchTool(Tool[BlastSearchInput, BlastSearchOutput]):
    """Finds sequences homologous to the query, ranked by alignment quality.

    The central evidence-gathering step: unlike a name-based NCBI lookup, this
    establishes that a reference actually aligns to the region around the gap,
    which is what licenses reading a reconstruction out of it.

    It also *retrieves* what it finds. BLAST answers with alignment statistics
    and, for a gap, two HSPs bracketing the missing segment - never the segment
    itself. Without a fetch the references arrive with no residues, and the
    alignment step silently discards every one of them.
    """

    name = "blast_search"
    description = (
        "Run a BLAST homology search against nucleotide databases using a gap's flanking "
        "context as the query, and retrieve the matching sequence for each hit. Returns "
        "hits ranked by identity and coverage, ready to align. Use when the informative "
        "relatives are unknown, or to confirm that a named reference really is homologous "
        "over this region. Slow: submits a job and polls, often 30-120 seconds."
    )
    estimated_seconds = 60.0

    def __init__(self, client: BlastClient, ncbi: NCBIClient | None = None) -> None:
        self._client = client
        # Optional so the tool still runs - degraded, metadata only - when no
        # NCBI client is wired, which is how most unit tests construct it.
        self._ncbi = ncbi

    async def run(self, payload: BlastSearchInput) -> BlastSearchOutput:
        try:
            job_id = await self._client.submit(
                payload.sequence,
                database=payload.database,
                program=payload.program,
                max_hits=payload.max_hits,
                expect=payload.expect,
            )
            raw = await self._client.result(job_id)
            references, pending = to_references(
                raw, query_length=len(payload.sequence), gap_length=payload.gap_length
            )
            total = len(references)

            references = await self._attach_residues(references, pending)
            with_sequence = [reference for reference in references if reference.has_sequence]

            return BlastSearchOutput(
                succeeded=True,
                references=references,
                total_hits=total,
                job_id=job_id,
                evidence=[
                    EvidenceItem(
                        source="blast",
                        reference_id=reference.accession,
                        organism=reference.organism,
                        identity=reference.identity,
                        note=(
                            f"BLAST hit over the flanking context of "
                            f"{payload.gap_id or 'the query'}."
                        ),
                    )
                    for reference in references
                ],
                diagnostics={
                    "gap_id": payload.gap_id,
                    "query_length": len(payload.sequence),
                    # The four counts that make the evidence path observable.
                    # `with_sequence` far below `total` means the fetch is
                    # failing, and that is invisible from the hit count alone.
                    "blast_hits_total": total,
                    "blast_hits_with_sequence": len(with_sequence),
                    "blast_fetches_attempted": min(len(pending), _MAX_SEQUENCE_FETCHES),
                },
            )

        except ReconstructionError as error:
            _log.warning("blast_failed", gap_id=payload.gap_id, error=str(error))
            return BlastSearchOutput(succeeded=False, error=str(error))

    async def _attach_residues(
        self, references: list[Reference], pending: list[HitSpan]
    ) -> list[Reference]:
        """Fetch the subject regions the hits bracket, best-ranked hits first.

        Failures are absorbed: a reference without residues is still evidence
        that a homologue exists, and one unreachable accession must not lose
        the rest of the search.
        """
        if not pending or self._ncbi is None:
            return references

        wanted = {span.accession: span for span in pending}
        order = [r.accession for r in references if r.accession in wanted]
        selected = [wanted[accession] for accession in order[:_MAX_SEQUENCE_FETCHES]]

        fetched = await asyncio.gather(
            *(self._fetch_span(span) for span in selected), return_exceptions=True
        )
        residues = {
            span.accession: result
            for span, result in zip(selected, fetched, strict=True)
            if isinstance(result, str) and result
        }
        if not residues:
            _log.warning("blast_sequence_fetch_empty", requested=len(selected))
            return references

        return [
            reference
            if reference.has_sequence or reference.accession not in residues
            else _with_residues(reference, residues[reference.accession])
            for reference in references
        ]

    async def _fetch_span(self, span: HitSpan) -> str | None:
        """One subject region as plain residues, or None if it cannot be had."""
        if self._ncbi is None or span.length <= 0:
            return None

        try:
            raw = await self._ncbi.fetch_region(
                span.accession, span.start, span.stop, strand=span.strand
            )
        except ReconstructionError as error:
            _log.info("blast_sequence_fetch_failed", accession=span.accession, error=str(error))
            return None

        records = fasta_to_references(raw)
        if not records or not records[0].residues:
            return None
        return records[0].residues[:_MAX_FETCHED_RESIDUES]


def _with_residues(reference: Reference, residues: str) -> Reference:
    """`reference` with its retrieved residues attached.

    A new object rather than a mutation: references are frozen, and the same
    one may already be held in a previous iteration's evidence.
    """
    return Reference(
        accession=reference.accession,
        organism=reference.organism,
        description=reference.description,
        residues=residues,
        identity=reference.identity,
        coverage=reference.coverage,
        e_value=reference.e_value,
        bit_score=reference.bit_score,
        relatedness=reference.relatedness,
        source=reference.source,
        metadata=reference.metadata,
        strand=reference.strand,
        iteration=reference.iteration,
        attempt=reference.attempt,
    )

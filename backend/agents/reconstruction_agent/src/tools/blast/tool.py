"""Homology search over a gap's flanking context, as a selectable tool."""
from __future__ import annotations

from configuration.logging import get_logger
from contracts.output import EvidenceItem
from domain.exceptions import ReconstructionError
from infrastructure.embl_ebi.blast_client import BlastClient
from tools.blast.mapper import to_references
from tools.blast.schemas import BlastSearchInput, BlastSearchOutput
from tools.contracts import Tool

_log = get_logger(__name__)


class BlastSearchTool(Tool[BlastSearchInput, BlastSearchOutput]):
    """Finds sequences homologous to the query, ranked by alignment quality.

    The central evidence-gathering step: unlike a name-based NCBI lookup, this
    establishes that a reference actually aligns to the region around the gap,
    which is what licenses reading a reconstruction out of it.
    """

    name = "blast_search"
    description = (
        "Run a BLAST homology search against nucleotide databases using a gap's flanking "
        "context as the query. Returns hits ranked by identity and coverage. Use when the "
        "informative relatives are unknown, or to confirm that a named reference really is "
        "homologous over this region. Slow: submits a job and polls, often 30-120 seconds."
    )
    estimated_seconds = 60.0

    def __init__(self, client: BlastClient) -> None:
        self._client = client

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
            references = to_references(raw, query_length=len(payload.sequence))

            return BlastSearchOutput(
                succeeded=True,
                references=references,
                total_hits=len(references),
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
                diagnostics={"gap_id": payload.gap_id, "query_length": len(payload.sequence)},
            )

        except ReconstructionError as error:
            _log.warning("blast_failed", gap_id=payload.gap_id, error=str(error))
            return BlastSearchOutput(succeeded=False, error=str(error))

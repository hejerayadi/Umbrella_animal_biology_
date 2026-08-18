"""Multiple sequence alignment of a gap's context against its references."""
from __future__ import annotations

from ...configuration.logging import get_logger
from ...domain.exceptions import ReconstructionError
from ...infrastructure.embl_ebi.mafft_client import MafftClient
from ..contracts import Tool
from .mapper import build_fasta, to_alignment
from .schemas import AlignmentInput, AlignmentOutput

_log = get_logger(__name__)


class MafftAlignmentTool(Tool[AlignmentInput, AlignmentOutput]):
    """Aligns the gap's flanks with each reference, locating the gap's columns.

    This is the step that converts "these references look related" into
    "these specific bases sit where the gap is".
    """

    name = "mafft_align"
    description = (
        "Align a gap's flanking context against selected reference sequences with MAFFT, "
        "and locate the alignment columns that span the gap. Run after references have been "
        "found and ranked; its output is what candidate reconstructions are read from. "
        "Slow: submits a job and polls."
    )
    estimated_seconds = 45.0

    def __init__(self, client: MafftClient) -> None:
        self._client = client

    async def run(self, payload: AlignmentInput) -> AlignmentOutput:
        if not payload.references:
            return AlignmentOutput(
                succeeded=False, error="No reference sequences supplied to align against."
            )

        try:
            fasta = build_fasta(payload.target_id, payload.target_sequence, payload.references)
            job_id = await self._client.submit(fasta)
            aligned = await self._client.result(job_id)

            alignment = to_alignment(
                aligned,
                gap_id=payload.gap_id,
                target_id=payload.target_id,
                left_flank_length=payload.left_flank_length,
            )

            if alignment is None:
                return AlignmentOutput(
                    succeeded=False,
                    error="Alignment did not contain a usable target row.",
                    job_id=job_id,
                )

            if not alignment.spans_gap:
                # A real outcome worth reporting distinctly: the references
                # aligned, but none of them carry bases across the gap.
                return AlignmentOutput(
                    succeeded=True,
                    alignment=alignment,
                    job_id=job_id,
                    aligned_count=alignment.reference_count,
                    diagnostics={
                        "spans_gap": False,
                        "note": "No reference contributed bases across the gap.",
                    },
                )

            return AlignmentOutput(
                succeeded=True,
                alignment=alignment,
                job_id=job_id,
                aligned_count=alignment.reference_count,
                diagnostics={
                    "spans_gap": True,
                    "mean_identity": alignment.mean_identity(),
                },
            )

        except ReconstructionError as error:
            _log.warning("MAFFT alignment failed for %s: %s", payload.gap_id, error)
            return AlignmentOutput(succeeded=False, error=str(error))

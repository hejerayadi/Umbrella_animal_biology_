"""Turn aligned FASTA into an `Alignment`, and locate the gap's columns.

Finding the gap columns is the load-bearing part: everything downstream reads
the reconstruction out of exactly those columns, so getting them wrong yields
a confident reconstruction of the wrong region.
"""
from __future__ import annotations

from ...domain.models import AlignedPair, Alignment
from ...domain.models.alignment import GAP_CHARACTER
from ..ncbi.mapper import parse_fasta


def to_alignment(
    aligned_fasta: str,
    *,
    gap_id: str,
    target_id: str,
    left_flank_length: int,
    tool: str = "mafft",
) -> Alignment | None:
    """Build an `Alignment` from MAFFT's aligned FASTA output.

    Returns None when the target row is absent - without it there is no
    coordinate system to locate the gap in.
    """
    records = parse_fasta(aligned_fasta)
    if not records:
        return None

    rows = {accession: residues for accession, _, residues in records}
    target_row = rows.pop(target_id, None)
    if target_row is None:
        return None

    pairs = [
        AlignedPair(
            target_id=target_id,
            reference_id=accession,
            target_aligned=target_row,
            reference_aligned=reference_row,
        )
        for accession, reference_row in rows.items()
        # Rows of unequal length are malformed output, not a usable alignment.
        if len(reference_row) == len(target_row)
    ]

    start, end = _locate_gap_columns(target_row, left_flank_length)

    return Alignment(
        gap_id=gap_id,
        pairs=pairs,
        gap_column_start=start,
        gap_column_end=end,
        tool=tool,
    )


def _locate_gap_columns(target_row: str, left_flank_length: int) -> tuple[int | None, int | None]:
    """Find the columns between the two flanks in the aligned target row.

    The target was submitted as left+right flank joined, with nothing between
    them. Any columns the aligner inserted at that junction are where the
    references carry bases the target lacks - the gap.

    Walking by consumed non-gap characters (rather than by column index) is
    what makes this correct: the aligner inserts gap characters into the target
    row, so column position and sequence position diverge.
    """
    if left_flank_length <= 0:
        return None, None

    consumed = 0
    junction_column: int | None = None

    for column, character in enumerate(target_row):
        if character != GAP_CHARACTER:
            consumed += 1
            if consumed == left_flank_length:
                junction_column = column + 1
                break

    if junction_column is None:
        # The left flank was never fully consumed: the alignment does not cover
        # the region we care about.
        return None, None

    # The gap spans the run of alignment gaps immediately after the junction.
    end = junction_column
    while end < len(target_row) and target_row[end] == GAP_CHARACTER:
        end += 1

    # No inserted columns means the references agree the target has no missing
    # bases here - a real answer, but not one with anything to read out.
    if end == junction_column:
        return None, None

    return junction_column, end


def build_fasta(target_id: str, target_sequence: str, references: dict[str, str]) -> str:
    """Assemble the FASTA block submitted to MAFFT.

    The target goes first so it is easy to find in the output regardless of how
    the aligner orders the rest.
    """
    blocks = [f">{target_id}\n{target_sequence}"]
    blocks.extend(
        f">{accession}\n{residues}" for accession, residues in references.items() if residues
    )
    return "\n".join(blocks) + "\n"

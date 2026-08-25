"""Turn aligned FASTA into an `Alignment`, and locate the gap's columns.

Finding the gap columns is the load-bearing part: everything downstream reads
the reconstruction out of exactly those columns, so getting them wrong yields
a confident reconstruction of the wrong region.
"""
from __future__ import annotations

from domain.models import AlignedPair, Alignment
from domain.models.alignment import GAP_CHARACTER
from tools.ncbi.mapper import parse_fasta


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

    start, end, slide = _locate_gap_columns(target_row, left_flank_length)

    return Alignment(
        gap_id=gap_id,
        pairs=pairs,
        gap_column_start=start,
        gap_column_end=end,
        junction_slide=slide,
        tool=tool,
    )


#: How far from the flank junction an inserted run may start and still be read
#: as the missing segment.
#:
#: Not a fudge factor - it is the same allowance `tools/blast/mapper` already
#: makes, for the same reason. When the last base of the left flank equals the
#: last base of the missing segment, two placements of the indel describe the
#: identical sequence and an aligner is free to pick either. Measured on the
#: polar bear mitogenome: the left flank ends in `A` and the true fill
#: `TTTG...CGTA` also ends in `A`, so MAFFT placed the 45-column run one base
#: early. Demanding exact adjacency found no run at all and the whole
#: reconstruction was reported as "no reference aligned across the gap" while
#: eight references sat in the alignment carrying every base of it.
_JUNCTION_TOLERANCE = 10


def _locate_gap_columns(
    target_row: str, left_flank_length: int
) -> tuple[int | None, int | None, int]:
    """The columns between the two flanks, and how far they slid.

    The target was submitted as left+right flank joined, with nothing between
    them. Any columns the aligner inserted near that junction are where the
    references carry bases the target lacks - the gap.

    Walking by consumed non-gap characters (rather than by column index) is
    what makes this correct: the aligner inserts gap characters into the target
    row, so column position and sequence position diverge.

    The third value is the slide: how many bases earlier than the nominal
    junction the run begins. It is not cosmetic - the fill read out of those
    columns is correct *at the column it was found*, and expressing it between
    the flanks the caller actually holds means rotating it by exactly this
    much. See `Alignment.anchor_fill`.
    """
    if left_flank_length <= 0:
        return None, None, 0

    best: tuple[int, int, int] | None = None
    consumed = 0
    column = 0
    length = len(target_row)

    while column < length:
        if target_row[column] != GAP_CHARACTER:
            consumed += 1
            column += 1
            continue

        run_start = column
        while column < length and target_row[column] == GAP_CHARACTER:
            column += 1

        # Positive when the run begins before the flanks were fully consumed.
        slide = left_flank_length - consumed
        if abs(slide) <= _JUNCTION_TOLERANCE and (best is None or abs(slide) < abs(best[2])):
            best = (run_start, column, slide)

    if best is None:
        # Either the left flank was never fully consumed, or no run was
        # inserted near the junction - the references agree the target is
        # missing nothing here. A real answer, but not one to read a fill out of.
        return None, None, 0

    return best


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

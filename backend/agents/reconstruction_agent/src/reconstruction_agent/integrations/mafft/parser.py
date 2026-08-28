"""Reading aligned FASTA, and writing the block that gets aligned.

Deliberately plain string handling. The records here are short - two flanks and
a few dozen homologues - and hand-rolling the parse keeps the identifier
convention under our control, which matters because the identifier is how the
query row is found again in the output.
"""

from __future__ import annotations

from collections.abc import Iterable

from reconstruction_agent.domain.models.alignment import AlignedRow, Alignment
from reconstruction_agent.domain.models.homology import HomologHit

#: Identifier given to the row carrying the target flanks. Short, and free of
#: characters an aligner might rewrite, so it survives the round trip.
QUERY_ID = "query"

#: Characters an aligner may use for a gap.
GAP_CHARS = "-."


def build_fasta(query: str, hits: Iterable[HomologHit]) -> str:
    """The FASTA block to align: the target flanks, then each homologue.

    Only hits whose sequence has actually been fetched are included. A hit
    without residues contributes nothing to an alignment and would appear as an
    empty row, which reads as a homologue that failed to align rather than one
    that was never supplied.
    """
    blocks = [f">{QUERY_ID}\n{query}"]
    blocks.extend(f">{hit.accession}\n{hit.subject_sequence}" for hit in hits if hit.has_sequence)
    return "\n".join(blocks) + "\n"


def parse_alignment(aligned_fasta: str, *, query_id: str = QUERY_ID) -> Alignment:
    """An aligned FASTA payload as an `Alignment`."""
    rows = tuple(
        AlignedRow(identifier=identifier, aligned_sequence=sequence)
        for identifier, sequence in _records(aligned_fasta)
    )
    return Alignment(query_id=query_id, rows=rows)


def _records(fasta: str) -> list[tuple[str, str]]:
    """`(identifier, sequence)` for each record, sequences upper-cased.

    The identifier is the first whitespace-delimited token of the header, which
    is what aligners preserve; everything after it is free-text description that
    they are free to reformat.
    """
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    chunks: list[str] = []

    for raw in fasta.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks).upper()))
            identifier = line[1:].split(None, 1)[0] if line[1:].strip() else ""
            chunks = []
        elif identifier is not None:
            chunks.append(line)

    if identifier is not None:
        records.append((identifier, "".join(chunks).upper()))

    return [(name, seq) for name, seq in records if name and seq]

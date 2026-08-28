"""Reading an alignment to find what the references put where the target has nothing.

This is where a reconstruction actually comes from, and it is the step most
easily got wrong, because a plausible-looking alignment and a usable one are
indistinguishable until the columns are read carefully.

Two things have to be right.

**Which references span the gap.** A homologue that matches one flank
beautifully and stops there proves the flank is conserved and says nothing
about the bases between. Counting it as support is how confident nonsense gets
produced, so a reference must carry aligned residues on *both* sides of the
missing region to contribute anything.

**Where the missing region actually sits.** The obvious approach - take the
columns between the last left-flank base and the first right-flank base - fails
in a specific and silent way. When the flank and the missing segment share an
end base, an aligner may legitimately place the indel one base either side of
the junction. Demanding an exact junction match then reports references that
carry every base of the answer as "nothing aligned across the gap". So the
inserted run is located with a tolerance and the fill is re-anchored against
the flanks afterwards.
"""

from __future__ import annotations

from collections import Counter

from reconstruction_agent.domain.models.alignment import (
    AlignedRow,
    Alignment,
    AlignmentSupport,
    ReferenceFill,
)
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.integrations.mafft.parser import GAP_CHARS

#: How far from the computed junction an inserted run may start and still be
#: taken as the gap. Small on purpose: this absorbs the one-base ambiguity an
#: aligner has when the flank and the fill share an end base, and nothing more.
#: A large tolerance would start matching unrelated indels in the flanks.
DEFAULT_SLIDE_TOLERANCE = 3


def analyze_gap(
    alignment: Alignment,
    context: GapContext,
    hits: tuple[HomologHit, ...] = (),
    *,
    slide_tolerance: int = DEFAULT_SLIDE_TOLERANCE,
) -> AlignmentSupport:
    """What the aligned references say about the missing region of `context`."""
    query = alignment.query_row
    if query is None or not alignment.reference_rows:
        return AlignmentSupport(gap_id=context.gap_id)

    columns = _ungapped_to_column(query.aligned_sequence)
    junction = len(context.left_flank)
    if not columns or junction >= len(columns):
        return AlignmentSupport(gap_id=context.gap_id)

    region = _locate_region(query, columns, junction, slide_tolerance)
    if region is None:
        # The references aligned, but none of them inserted anything between
        # the flanks. That is a real answer: there is no gap-spanning support.
        return AlignmentSupport(gap_id=context.gap_id)

    start, end = region
    query_in_region = _residues(query, start, end)
    organisms = {hit.accession: hit for hit in hits}

    fills = tuple(
        _fill_for(row, start, end, query_in_region, organisms) for row in alignment.reference_rows
    )
    spanning = tuple(fill for fill in fills if fill.spans_gap)

    return AlignmentSupport(
        gap_id=context.gap_id,
        gap_column_start=start,
        gap_column_end=end,
        fills=fills,
        conservation=_conservation(spanning),
        conflicting_positions=_conflicts(spanning),
    )


def _ungapped_to_column(aligned: str) -> list[int]:
    """Alignment column of each ungapped position of a row."""
    return [index for index, char in enumerate(aligned) if char not in GAP_CHARS]


def _locate_region(
    query: AlignedRow, columns: list[int], junction: int, tolerance: int
) -> tuple[int, int] | None:
    """The half-open column span the references inserted across the gap.

    Anchored on the flanks rather than on the raw junction, then widened to
    whichever run of query gap characters lies near it. The two rarely differ,
    and when they do it is the aligner exercising its freedom over where to put
    an indel - not evidence that the references failed to span anything.
    """
    left_anchor = columns[junction - 1] if junction > 0 else -1
    right_anchor = columns[junction]

    run = _gap_run_near(query.aligned_sequence, right_anchor, tolerance)
    if run is None:
        return None

    start = min(run[0], left_anchor + 1)
    end = max(run[1], right_anchor)
    return (start, end) if end > start else None


def _gap_run_near(aligned: str, anchor: int, tolerance: int) -> tuple[int, int] | None:
    """The run of gap characters in `aligned` closest to column `anchor`.

    Searched outward from the anchor so the nearest run wins, which is what
    keeps a tolerated slide from picking up an unrelated indel further along
    the flank.
    """
    best: tuple[int, int] | None = None
    best_distance = tolerance + 1

    for start, end in _gap_runs(aligned):
        # Distance from the run to the anchor, zero when the run abuts it.
        distance = 0 if start <= anchor <= end else min(abs(anchor - end), abs(start - anchor))
        if distance <= tolerance and distance < best_distance:
            best, best_distance = (start, end), distance

    return best


def _gap_runs(aligned: str) -> list[tuple[int, int]]:
    """Every maximal run of gap characters, as half-open column spans."""
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for index, char in enumerate(aligned):
        if char in GAP_CHARS:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None

    if start is not None:
        runs.append((start, len(aligned)))
    return runs


def _residues(row: AlignedRow, start: int, end: int) -> str:
    """The row residues across a column span, gap characters removed."""
    return "".join(char for char in row.columns(start, end) if char not in GAP_CHARS)


def _fill_for(
    row: AlignedRow,
    start: int,
    end: int,
    query_in_region: str,
    hits: dict[str, HomologHit],
) -> ReferenceFill:
    """What one reference contributes across the gap region."""
    hit = hits.get(row.identifier)
    return ReferenceFill(
        accession=row.identifier,
        organism=hit.organism if hit else None,
        organism_tax_id=hit.organism_tax_id if hit else None,
        bases=_reanchor(_residues(row, start, end), query_in_region),
        spans_gap=_spans(row, start, end),
    )


def _reanchor(reference_region: str, query_region: str) -> str:
    """Strip flank bases the query also holds in the region.

    Normally `query_region` is empty and this returns the reference bases
    unchanged. It is non-empty only when the aligner slid the indel, in which
    case the region has swallowed a base or two of flank; those bases are not
    missing from the target, so returning them would shift the fill.
    """
    if not query_region:
        return reference_region
    if reference_region.startswith(query_region):
        return reference_region[len(query_region) :]
    if reference_region.endswith(query_region):
        return reference_region[: -len(query_region)]
    return reference_region


def _spans(row: AlignedRow, start: int, end: int) -> bool:
    """Whether a reference carries aligned residues on both sides of the region.

    The whole distinction between a homologue and a *usable* homologue.
    """
    before = any(char not in GAP_CHARS for char in row.aligned_sequence[:start])
    after = any(char not in GAP_CHARS for char in row.aligned_sequence[end:])
    return before and after


def _conservation(fills: tuple[ReferenceFill, ...]) -> float:
    """Agreement among the spanning references, 0..1.

    Averaged per position over the length they agree on. Disagreement here is
    the signal that competing candidates exist, so it is measured rather than
    smoothed away.
    """
    usable = [fill.bases for fill in fills if fill.bases]
    if not usable:
        return 0.0
    if len(usable) == 1:
        # A single reference agrees with itself; that is not evidence of
        # conservation, so it scores neutrally rather than perfectly.
        return 0.5

    width = min(len(bases) for bases in usable)
    if width == 0:
        return 0.0

    agreements = [
        Counter(bases[position] for bases in usable).most_common(1)[0][1] / len(usable)
        for position in range(width)
    ]
    return sum(agreements) / width


def _conflicts(fills: tuple[ReferenceFill, ...]) -> tuple[int, ...]:
    """Positions where the spanning references disagree on the base."""
    usable = [fill.bases for fill in fills if fill.bases]
    if len(usable) < 2:
        return ()

    width = min(len(bases) for bases in usable)
    return tuple(
        position for position in range(width) if len({bases[position] for bases in usable}) > 1
    )

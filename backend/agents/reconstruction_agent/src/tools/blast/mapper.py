"""Translate EMBL-EBI BLAST results into domain objects.

The query BLAST is given is a gap's two flanks joined with the N-run removed
(`GapContext.query_sequence`). Against a reference that actually carries the
missing segment, that produces **two HSPs, one per flank** - a large insertion
breaks the local alignment rather than being spanned. The missing residues
therefore lie *between* the HSPs in subject coordinates.

That shape is why this module keeps every HSP instead of the first one. Reading
`hit_hsps[0]` alone threw away the very thing the search was run to find, and
left every reference without residues - which the MAFFT selector then filtered
out, so the default `blast_search -> mafft_align` plan could never align
anything at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from domain.models import Reference
from domain.models.sequence import reverse_complement

#: What BLAST writes into an aligned row where the other sequence has no
#: residue. Both spellings appear in the JSON depending on the service version.
_ALIGNMENT_GAPS = frozenset("-.")


@dataclass(frozen=True, slots=True)
class HitSpan:
    """The subject region a hit's HSPs bracket, on the subject's own strand.

    Returned alongside a reference that has no residues yet, so the tool knows
    what to fetch and how much. `start`/`stop` are 1-based inclusive, matching
    both BLAST's coordinates and NCBI's `seq_start`/`seq_stop`.
    """

    accession: str
    start: int
    stop: int
    strand: int

    @property
    def length(self) -> int:
        return max(0, self.stop - self.start + 1)


def _fraction(value: Any) -> float | None:
    """Normalise a percentage or fraction onto 0..1.

    EBI reports identity as a percentage in JSON output but some result types
    give a fraction; accepting both avoids a silent 100x error in ranking.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(number / 100.0, 1.0) if number > 1.0 else number


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_json_result(raw: str) -> list[dict[str, Any]]:
    """The hit list out of an EBI BLAST JSON payload.

    Returns an empty list rather than raising on malformed JSON: a search that
    produced nothing usable is a normal outcome the tool reports as zero hits.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        return payload.get("hits", []) or []
    return []


def _hsp_strand(hsp: dict[str, Any]) -> int:
    """Which subject strand this HSP is on.

    EBI reports strand as a string like `"plus/minus"`; some result types
    instead give numeric frames. Both are read, and anything unrecognised is
    treated as the plus strand - the common case, and the one where guessing
    wrong is cheapest to detect downstream.
    """
    raw = hsp.get("hsp_strand")
    if isinstance(raw, str):
        # The subject's half is what matters; the query is always plus here.
        return -1 if raw.lower().split("/")[-1].startswith("minus") else 1

    frame = _int_or_none(hsp.get("hsp_hit_frame"))
    if frame is not None and frame < 0:
        return -1

    start, stop = _int_or_none(hsp.get("hsp_hit_from")), _int_or_none(hsp.get("hsp_hit_to"))
    if start is not None and stop is not None and stop < start:
        return -1
    return 1


#: How far from the flank junction an insertion may start and still be read as
#: the missing segment. An aligner is free to slide an indel by a few bases
#: when the surrounding residues repeat, so demanding the exact column would
#: reject genuine carriers; allowing more would start crediting unrelated
#: indels elsewhere in the flank.
_JUNCTION_TOLERANCE = 10


def _gap_bases_carried(hsp: dict[str, Any], left_flank_length: int) -> int:
    """How many subject bases sit at the query's flank junction, if any.

    The query submitted to BLAST is the two flanks joined, so a reference that
    still carries the missing segment aligns with a gap run in the *query* row
    at the junction, and its own bases opposite. Counting them answers the only
    question that matters when choosing what to align: does this reference
    actually have anything to contribute across the gap?

    Returns 0 when the HSP does not reach the junction, which is the common
    case - most hits are homologous to one flank and carry nothing.
    """
    qseq = hsp.get("hsp_qseq")
    hseq = hsp.get("hsp_hseq")
    query_from = _int_or_none(hsp.get("hsp_query_from"))

    if not isinstance(qseq, str) or not isinstance(hseq, str) or query_from is None:
        return 0
    # Rows of unequal length are malformed output; reading them would misplace
    # every column that follows.
    if len(qseq) != len(hseq):
        return 0

    # Query residues consumed so far, as a 0-based count. `hsp_query_from` is
    # 1-based, so the HSP opens having already consumed `query_from - 1`.
    consumed = query_from - 1
    column = 0

    while column < len(qseq):
        if qseq[column] not in _ALIGNMENT_GAPS:
            consumed += 1
            column += 1
            continue

        run = 0
        while column + run < len(qseq) and qseq[column + run] in _ALIGNMENT_GAPS:
            run += 1

        if abs(consumed - left_flank_length) <= _JUNCTION_TOLERANCE:
            return sum(
                1 for character in hseq[column : column + run]
                if character not in _ALIGNMENT_GAPS
            )

        column += run

    return 0


def _residues_from_hsp(hsp: dict[str, Any], strand: int) -> str | None:
    """The subject's residues for one HSP, ungapped and on the target's strand.

    `hsp_hseq` is the aligned subject with gap characters; MAFFT wants plain
    residues, and will reintroduce whatever gaps the alignment needs.
    """
    aligned = hsp.get("hsp_hseq")
    if not isinstance(aligned, str) or not aligned.strip():
        return None

    residues = aligned.replace("-", "").replace(".", "").upper()
    if not residues:
        return None
    return reverse_complement(residues) if strand < 0 else residues


def to_references(
    raw: str,
    *,
    query_length: int | None = None,
    gap_length: int = 0,
    left_flank_length: int | None = None,
) -> tuple[list[Reference], list[HitSpan]]:
    """Domain references for every BLAST hit, plus the spans still to fetch.

    A reference comes back with residues already attached when one HSP carried
    enough of the subject to be worth aligning. When the HSPs *bracket* a
    region instead - the informative case for a gap - the residues are not in
    the response at all, and the corresponding `HitSpan` says which subject
    range the tool should fetch.
    """
    references: list[Reference] = []
    pending: list[HitSpan] = []

    for hit in parse_json_result(raw):
        hsps = [h for h in (hit.get("hit_hsps") or []) if isinstance(h, dict)]
        if not hsps:
            continue

        accession = str(hit.get("hit_acc") or hit.get("hit_id") or "unknown")
        # Best HSP by e-value, falling back to the first: the statistics quoted
        # for the reference should describe its strongest alignment, not
        # whichever one the service happened to list first.
        best = min(hsps, key=lambda h: _float_or_none(h.get("hsp_expect")) or float("inf"))
        strand = _hsp_strand(best)

        coverage = _fraction(hit.get("hit_coverage"))
        if coverage is None and query_length:
            # Summed over every HSP: a hit that matches both flanks separately
            # covers both, and crediting only the first understates it.
            aligned = sum(_int_or_none(h.get("hsp_align_len")) or 0 for h in hsps)
            if aligned:
                coverage = min(aligned / query_length, 1.0)

        residues = _residues_from_hsp(best, strand)
        span = _subject_span(hsps, accession, strand, gap_length)

        # Measured across every HSP, not just the best one: the HSP that
        # carries the missing segment is not always the one with the strongest
        # e-value, and one that does carry it settles the question for the hit.
        gap_bases = (
            max(
                (_gap_bases_carried(hsp, left_flank_length) for hsp in hsps),
                default=0,
            )
            if left_flank_length is not None
            else None
        )

        # Prefer a fetch whenever the HSPs bracket a region: that gap between
        # them is the missing segment, and no single HSP contains it.
        if span is not None and len(hsps) > 1:
            residues = None

        references.append(
            Reference(
                accession=accession,
                organism=hit.get("hit_os"),
                description=hit.get("hit_desc"),
                residues=residues,
                identity=_fraction(best.get("hsp_identity")),
                coverage=coverage,
                e_value=_float_or_none(best.get("hsp_expect")),
                bit_score=_float_or_none(best.get("hsp_bit_score")),
                source="blast",
                strand=strand,
                gap_bases=gap_bases,
                metadata=_metadata(hit, hsps, span),
            )
        )

        if residues is None and span is not None:
            pending.append(span)

    return references, pending


def _subject_span(
    hsps: list[dict[str, Any]], accession: str, strand: int, gap_length: int
) -> HitSpan | None:
    """The subject region the HSPs cover, widened to include the missing part.

    Widened by the gap's own length because the segment we are after sits
    between the flanks and is by definition absent from every HSP. Without the
    margin the fetched region stops exactly where the evidence starts.
    """
    starts = [_int_or_none(h.get("hsp_hit_from")) for h in hsps]
    stops = [_int_or_none(h.get("hsp_hit_to")) for h in hsps]
    bounds = [value for value in (*starts, *stops) if value is not None]
    if not bounds:
        return None

    margin = max(gap_length, 0)
    return HitSpan(
        accession=accession,
        start=max(1, min(bounds) - margin),
        stop=max(bounds) + margin,
        strand=strand,
    )


def _metadata(
    hit: dict[str, Any], hsps: list[dict[str, Any]], span: HitSpan | None
) -> dict[str, str]:
    """Hit facts worth keeping for filtering and for the audit trail."""
    metadata: dict[str, str] = {"hsp_count": str(len(hsps))}

    length = _int_or_none(hit.get("hit_len"))
    if length is not None:
        metadata["subject_length"] = str(length)
    if span is not None:
        metadata["subject_span"] = f"{span.start}-{span.stop}"

    return metadata

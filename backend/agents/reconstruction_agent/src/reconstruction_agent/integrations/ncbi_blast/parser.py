"""Turning NCBI BLAST XML into the same normalised hits EBI produces.

Both providers feed one domain model, so everything downstream - ranking,
alignment, scoring - is written once and neither provider leaks into it.

The conversions differ from the EBI parser in one way that matters and would be
silent if got wrong: **`Hsp_identity` is a count of matching positions, not a
percentage.** EBI reports 95.7 for the same alignment where NCBI reports 957
out of an `Hsp_align-len` of 1000. Dividing by 100 here, as the EBI path does,
would record an identity of 9.57 - which the domain model rejects outright, so
the mistake fails loudly rather than scoring a candidate ten times too high.
Identity is therefore computed as a ratio of the two fields.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree

from reconstruction_agent.domain.exceptions import ExternalServiceError
from reconstruction_agent.domain.models.homology import HomologHit

SERVICE = "ncbi_blast"

#: A binomial: capitalised genus, lowercase specific epithet. Searched for
#: rather than anchored, because real descriptions carry status prefixes -
#: "UNVERIFIED: Ursus maritimus isolate ...", "TPA:", "PREDICTED:" - and
#: anchoring loses the organism on exactly those records.
_BINOMIAL = re.compile(r"([A-Z][a-z]{2,} [a-z][a-z-]{2,})")


def parse_hits(
    payload: str,
    *,
    database_code: str,
    query_length: int,
) -> tuple[HomologHit, ...]:
    """Every hit in a BLAST XML result, best HSP per hit.

    An empty payload means the search completed and found nothing, which is a
    result rather than a failure - the caller gets an empty tuple.
    """
    if not payload.strip():
        return ()

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ExternalServiceError(SERVICE, f"BLAST result was not valid XML: {error}") from error

    # NCBI reports the query length it actually used. Preferred over the
    # caller's count, because a truncated submission would otherwise be scored
    # as full coverage of a query that was never searched.
    reported = _int(root.findtext("./BlastOutput_query-len"))
    effective = reported or query_length

    hits = [
        hit
        for element in root.findall(".//Hit")
        if (hit := _hit_from(element, database_code, effective)) is not None
    ]
    return tuple(hits)


def _hit_from(
    element: ElementTree.Element, database_code: str, query_length: int
) -> HomologHit | None:
    """One hit, built from its best-scoring HSP."""
    hsps = element.findall("./Hit_hsps/Hsp")
    if not hsps:
        return None

    best = max(hsps, key=lambda hsp: _float(hsp.findtext("Hsp_bit-score")))

    accession = (element.findtext("Hit_accession") or "").strip()
    if not accession:
        # Fall back to the composite id, whose last populated field is the
        # accession: "gi|123|ref|NC_003428.1|".
        parts = [p for p in (element.findtext("Hit_id") or "").split("|") if p]
        accession = parts[-1] if parts else ""
    if not accession:
        return None

    description = (element.findtext("Hit_def") or "").strip()

    align_len = _int(best.findtext("Hsp_align-len"))
    matches = _int(best.findtext("Hsp_identity"))
    # A count over a length, not a percentage. See the module docstring.
    identity = (matches / align_len) if align_len > 0 else 0.0

    query_start, query_end = _span(best.findtext("Hsp_query-from"), best.findtext("Hsp_query-to"))
    subject_start, subject_end = _span(best.findtext("Hsp_hit-from"), best.findtext("Hsp_hit-to"))

    coverage = min(align_len / query_length, 1.0) if query_length > 0 else 0.0

    return HomologHit(
        accession=accession,
        description=description,
        organism=organism_from_description(description),
        identity=min(max(identity, 0.0), 1.0),
        query_coverage=coverage,
        e_value=max(_float(best.findtext("Hsp_evalue")), 0.0),
        bit_score=_float(best.findtext("Hsp_bit-score")),
        query_start=query_start,
        query_end=query_end,
        subject_start=subject_start,
        subject_end=subject_end,
        source_database=database_code,
        source_provider="NCBI",
    )


def _span(raw_start: str | None, raw_end: str | None) -> tuple[int, int]:
    """A 1-based inclusive BLAST span as 0-based half-open, orientation-normalised."""
    start = _int(raw_start)
    end = _int(raw_end)
    if start > end:
        # A minus-strand HSP is reported high-to-low. Ordering it here means no
        # arithmetic downstream has to know about strand.
        start, end = end, start
    return max(start - 1, 0), max(end, 0)


def _int(value: str | None) -> int:
    try:
        return int(float(value)) if value else 0
    except (TypeError, ValueError):
        return 0


def _float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def organism_from_description(description: str) -> str | None:
    """The organism a hit description names, or None if it names none.

    Prefers the first binomial in the text, which is how ENA and NCBI
    descriptions are conventionally written. The name is only ever a lookup key
    - what it means taxonomically is decided by NCBI Taxonomy, not here.
    """
    text = description.strip()
    if not text:
        return None
    if match := _BINOMIAL.search(text):
        return match.group(1)

    # No binomial: fall back to the leading clause, which for a
    # non-conventional description is still the best organism guess available.
    head = re.split(r"[,;(]", text, maxsplit=1)[0].strip()
    return head or None

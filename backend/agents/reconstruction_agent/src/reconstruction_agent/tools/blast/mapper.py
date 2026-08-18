"""Translate EMBL-EBI BLAST results into domain objects."""
from __future__ import annotations

import json
from typing import Any

from ...domain.models import Reference


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


def to_references(raw: str, *, query_length: int | None = None) -> list[Reference]:
    """Domain `Reference` objects for every BLAST hit.

    Coverage is derived from the alignment length against the query when EBI
    does not report it directly - the ranker weighs coverage, and a missing
    value would quietly score as zero.
    """
    references: list[Reference] = []

    for hit in parse_json_result(raw):
        alignments = hit.get("hit_hsps") or []
        best = alignments[0] if alignments else {}

        coverage = _fraction(hit.get("hit_coverage"))
        if coverage is None and query_length:
            aligned = best.get("hsp_align_len")
            if aligned:
                coverage = min(float(aligned) / query_length, 1.0)

        references.append(
            Reference(
                accession=str(hit.get("hit_acc") or hit.get("hit_id") or "unknown"),
                organism=hit.get("hit_os"),
                description=hit.get("hit_desc"),
                identity=_fraction(best.get("hsp_identity")),
                coverage=coverage,
                e_value=_float_or_none(best.get("hsp_expect")),
                bit_score=_float_or_none(best.get("hsp_bit_score")),
                source="blast",
            )
        )

    return references


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

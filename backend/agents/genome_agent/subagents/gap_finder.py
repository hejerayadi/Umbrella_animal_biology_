"""
Gap Finder — real NCBI subagent that locates unresolved-region (N-gap)
coordinates for an assembly and enriches each with flanking sequence.

This is the missing link between "this assembly is Scaffold/Contig level"
(the coarse, per-assembly signal `get_genome_metadata_node` already checks)
and "here are the actual gap coordinates" that the Reconstruction Agent
needs. Nothing upstream of this module produced per-gap positions before.

Where the gap coordinates come from
------------------------------------
NCBI's Feature Table format (`efetch?db=nuccore&rettype=ft&retmode=text`)
lists every annotated feature on a sequence, including `gap` (INSDC feature
table key) / `assembly_gap` (GenBank flat-file key) features with their
start/end and an `estimated_length` qualifier. Fetching the feature table
is deliberately used here instead of downloading the full sequence and
scanning for runs of "N": the feature table is small (KB-scale) regardless
of how large the underlying chromosome is (GB-scale), so it respects the
same "don't pull GB-scale sequence" constraint that `sequence_window.py`
was built around.

Resolving the assembly accession (e.g. "GCF_000687225.1") to a fetchable
Nuccore sequence follows the same two-hop dance documented in
`sequence_window.py` (esearch db=assembly -> elink assembly-to-nuccore ->
a Nuccore UID). That UID is also esummary'd here to recover the actual
`NW_...`/`NC_...` sequence accession string — the internal UID
`sequence_window.py` stops at is not something a caller can cite or hand
to another agent.

Flanking sequence around each gap is fetched via
`sequence_window.fetch_sequence_window`, wiring that standalone subagent
into real use for the first time.
"""

from __future__ import annotations

import asyncio
import logging
import re

from ._ncbi_client import ncbi_get
from .sequence_window import (
    NoLinkedSequenceError,
    _resolve_assembly_uid,
    _resolve_nuccore_id,
    fetch_sequence_window,
)

logger = logging.getLogger(__name__)

# Cap the number of gaps enriched with flanking sequence per request. Some
# Scaffold/Contig assemblies carry thousands of gaps; the Reconstruction
# Agent works on a handful of "selected unresolved regions" at a time, and
# each enriched gap costs two extra NCBI round trips (left + right flank).
DEFAULT_MAX_GAPS = 5

# How much sequence to pull on each side of a gap.
DEFAULT_FLANK_BP = 50

_FEATURE_LINE_RE = re.compile(r"^(\d+)\t(\d+)\t(assembly_gap|gap)\s*$")
_ESTIMATED_LENGTH_RE = re.compile(r"^\t+estimated_length\t(\S+)\s*$")


class GapFinderError(Exception):
    """Raised when gaps can't be located or enriched for an assembly."""


def _parse_feature_table_gaps(feature_table_text: str) -> list[dict]:
    """Pull gap coordinates out of an NCBI feature-table (.ft) response.

    Pure/offline so it can be unit-tested without hitting NCBI. Feature
    table lines that start a new feature begin in column 0
    (`"<start>\\t<end>\\t<key>"`); qualifier lines for that feature are
    indented with a leading tab. A gap's `estimated_length` qualifier is
    preferred over the raw coordinate span when it's a real number,
    since NCBI sometimes reports "unknown"-length gaps whose recorded
    span is a placeholder rather than the true gap size.
    """
    gaps: list[dict] = []
    current: dict | None = None

    for line in feature_table_text.splitlines():
        if not line.strip():
            continue

        if not line.startswith("\t"):
            # A new top-level feature line closes out whatever gap we were
            # tracking (its qualifiers are done), whether or not this new
            # line is itself a gap.
            if current is not None:
                gaps.append(current)
                current = None

            match = _FEATURE_LINE_RE.match(line)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                lo, hi = min(start, end), max(start, end)
                current = {"start": lo, "end": hi, "length": hi - lo + 1}
            continue

        if current is not None:
            qualifier_match = _ESTIMATED_LENGTH_RE.match(line)
            if qualifier_match and qualifier_match.group(1).isdigit():
                current["length"] = int(qualifier_match.group(1))

    if current is not None:
        gaps.append(current)

    return gaps


async def _fetch_accession_version(nuccore_id: str) -> str:
    """Resolve a Nuccore UID to its real accession string (e.g. NW_007907101.1)."""
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "nuccore",
            "id": nuccore_id,
            "retmode": "json",
        },
    )
    data = resp.json()
    doc = (data.get("result") or {}).get(nuccore_id) or {}
    return doc.get("accessionversion") or doc.get("caption") or nuccore_id


async def _fetch_feature_table(nuccore_id: str) -> str:
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "efetch.fcgi",
            "db": "nuccore",
            "id": nuccore_id,
            "rettype": "ft",
            "retmode": "text",
        },
    )
    return resp.text


def _extract_fasta_sequence(fasta_text: str) -> str:
    """Strip the `>` header line(s) out of a FASTA response, keeping only bases."""
    return "".join(
        line.strip() for line in fasta_text.splitlines() if line and not line.startswith(">")
    )


async def _resolve_nuccore(assembly_id: str) -> str:
    """Shared assembly -> Nuccore UID resolution, raising GapFinderError on failure."""
    assembly_uid = await _resolve_assembly_uid(assembly_id)
    if assembly_uid is None:
        raise GapFinderError(f"Assembly '{assembly_id}' not found on NCBI.")

    try:
        nuccore_id = await _resolve_nuccore_id(assembly_id, assembly_uid)
    except NoLinkedSequenceError as exc:
        raise GapFinderError(str(exc)) from exc

    if nuccore_id is None:
        raise GapFinderError(
            f"No Nuccore sequence is linked to assembly '{assembly_id}'."
        )
    return nuccore_id


async def _attach_flanks(
    assembly_id: str,
    gap: dict,
    flank_bp: int,
) -> dict:
    left_start = max(1, gap["start"] - flank_bp)
    left_end = gap["start"] - 1
    right_start = gap["end"] + 1
    right_end = gap["end"] + flank_bp

    left_flank = ""
    if left_end >= left_start:
        left_raw = await fetch_sequence_window(assembly_id, left_start, left_end)
        left_flank = _extract_fasta_sequence(left_raw)

    right_raw = await fetch_sequence_window(assembly_id, right_start, right_end)
    right_flank = _extract_fasta_sequence(right_raw)

    return {**gap, "left_flank": left_flank, "right_flank": right_flank}


async def find_target_gaps(
    assembly_id: str,
    max_gaps: int = DEFAULT_MAX_GAPS,
    flank_bp: int = DEFAULT_FLANK_BP,
) -> dict:
    """Find real gap coordinates for `assembly_id` and enrich them with flanks.

    Returns:
        {
            "sequence_accession": "NW_007907101.1",
            "target_gaps": [
                {"start": ..., "end": ..., "length": ..., "left_flank": "...", "right_flank": "..."},
                ...
            ],
        }

    Raises GapFinderError if the assembly can't be resolved to a Nuccore
    sequence at all. An assembly that resolves but has no annotated gaps
    (feature table simply has none) is not an error — it returns an empty
    `target_gaps` list, since some Scaffold-level assemblies genuinely have
    no N-runs annotated as such.
    """
    nuccore_id = await _resolve_nuccore(assembly_id)
    accession = await _fetch_accession_version(nuccore_id)

    feature_table_text = await _fetch_feature_table(nuccore_id)
    gaps = _parse_feature_table_gaps(feature_table_text)[:max_gaps]

    if not gaps:
        logger.info(
            "[find_target_gaps] no annotated gaps found for assembly=%r (accession=%r)",
            assembly_id,
            accession,
        )
        return {"sequence_accession": accession, "target_gaps": []}

    enriched = await asyncio.gather(
        *(_attach_flanks(assembly_id, gap, flank_bp) for gap in gaps)
    )
    return {"sequence_accession": accession, "target_gaps": list(enriched)}


if __name__ == "__main__":

    async def _quick_test():
        print("--- Gap Finder live NCBI test ---")
        result = await find_target_gaps("GCF_000687225.1", max_gaps=2)
        print("sequence_accession:", result["sequence_accession"])
        print("target_gaps:", result["target_gaps"])

    asyncio.run(_quick_test())

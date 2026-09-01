"""
Gap Finder — real NCBI subagent that locates unresolved-region (N-run)
coordinates for an assembly and enriches each with flanking sequence.

This is the missing link between "this assembly is Scaffold/Contig level"
(the coarse, per-assembly signal `get_genome_metadata_node` already checks)
and "here are the actual gap coordinates" that the Reconstruction Agent
needs. Nothing upstream of this module produces per-gap positions.

Where the gap coordinates come from
------------------------------------
The sequence itself. An assembly gap *is* a run of N, and reading the
residues is the only way to find one that is reliable across RefSeq's
record types.

This module previously parsed NCBI's Feature Table (`efetch?rettype=ft`)
looking for `gap`/`assembly_gap` features, on the reasoning that the feature
table is KB-scale no matter how large the record is. It is - but for the
RefSeq WGS/CON scaffolds this agent actually resolves, it carries no gap
features at all. Measured against the polar bear's largest scaffold
(NW_024426341.1, 1,153,480 bp): its feature table is 923 lines of
`gene`/`exon`/`CDS` and contains zero `gap` or `assembly_gap` entries, while
the FASTA for the same record contains 30 runs of N totalling 8,184 bases,
the longest 2,784. Those features live in the GenBank flat file for records
that store their own sequence; a CON record like this one is a `join()` of
WGS contigs and carries neither. Parsing the feature table therefore
returned an empty list for every assembly, every time.

Reading residues is bounded here rather than unbounded: the record is pulled
in `MAX_WINDOW_BP` slices through `sequence_window.py` and capped at
`_SCAFFOLD_MAX_BP`, so a chromosome-scale record costs a fixed handful of
requests instead of a GB-scale download.

Which record gets scanned
-------------------------
The largest genomic record *belonging to this assembly*, within a size band,
via nuccore's `[Assembly]` field sorted by `SLEN`.

The previous implementation took whatever NCBI's `elink` returned first,
which is not the largest record or any particular one - for GCF_017311325.1
that is NW_024425908.1, an unplaced 1,112 bp fragment with no gaps in it,
while the assembly's largest scaffold is 1,153,480 bp and holds the 30 runs
above. Gaps are what joins contigs into scaffolds, so the big scaffolds are
where the unresolved regions are; the small unplaced fragments measured
gap-free.

The upper bound matters because the Reconstruction Agent fetches the whole
record before it can work: a chromosome-scale scaffold made the request hang
until the HTTP timeout. The lower bound skips the unplaced fragments.

What this module reports about its own filtering
------------------------------------------------
Scanning one record of thousands, screening at a length floor and capping the
result are three separate reductions, and `target_gaps` shows none of them.
`find_target_gaps` therefore returns the counts at each stage plus a
`selection_policy` describing the thresholds, so a consumer can tell "this
assembly has ten gaps" from "here are ten of thirty, from one record of
3,899". The filtering is unchanged; only its visibility is.
"""

from __future__ import annotations

import asyncio
import logging
import re

from ._ncbi_client import ncbi_get
from .sequence_window import MAX_WINDOW_BP, _resolve_assembly_uid, fetch_window_by_accession

logger = logging.getLogger(__name__)

# Cap the number of gaps handed on per request. Some Scaffold/Contig
# assemblies carry thousands. The Reconstruction Agent triages the list it is
# given - measured on NW_024426341.1 it attempted 8 of 23 gaps and deferred
# the rest against its own deadline - so this is sized to give it room to
# choose rather than to match what it can finish.
DEFAULT_MAX_GAPS = 10

# How much sequence to report on each side of a gap. Sliced out of the
# residues already in hand, so this costs nothing extra.
DEFAULT_FLANK_BP = 50

# The shortest run of N worth reporting, matching the Reconstruction Agent's
# own `DEFAULT_MIN_GAP_LENGTH` (services/sequence/sequence_service.py).
#
# This floor was 10, on the reasoning that a stray N is an uncallable base
# rather than an assembly gap. The biology is right; the number was not.
# Handed NW_024426341.1 with no gap list at all, the Reconstruction Agent
# found 23 gaps of its own and resolved exactly two - a 5-base run and one
# other short one. Both sat under this floor, so screening at 10 hid precisely
# the gaps it had a chance with. Dropping the floor entirely is no better: the
# record's shortest runs are single Ns, and ten of those would crowd out every
# gap worth sending. Deferring to the consumer's own threshold is what keeps
# the two ends agreeing on what counts as a gap.
MIN_GAP_BP = 5

# The size band a scanned record has to fall in (see module docstring).
#
# The ceiling is a downstream constraint, not a biological one: the
# Reconstruction Agent fetches the whole record before it can work, and a
# chromosome-scale record made that request hang until the HTTP timeout. It is
# also the most consequential filter in this module - it excludes the largest
# records, which are the likeliest to carry gaps. Since it cannot be lifted
# without breaking the consumer, it is *reported* instead: `_record_census`
# counts how many records it puts out of reach and `selection_policy` carries
# the band, so a consumer that can handle a bigger record can see that one
# exists rather than being told this assembly has ten gaps.
_SCAFFOLD_MIN_BP = 50_000
_SCAFFOLD_MAX_BP = 2_000_000

# Whether to spend two extra count-only NCBI requests measuring how much of
# the assembly this module did not look at (`_record_census`). It is the only
# way the payload can say "one record of 3,899" rather than implying the
# assembly has as many gaps as were sent, but it is not free, and every call
# here shares one process-wide pacing gate with the window reads. Turn it off
# where the round-trips matter more than the provenance: `selection_policy`
# then reports those two counts as None, which already means "not measured".
REPORT_RECORD_CENSUS = True

_N_RUN_RE = re.compile(r"N+")

# A WGS project's master record: the project prefix followed by nothing but
# zeros (LVCL000000000.1, PVIE00000000.1). Real scaffolds in the same project
# carry non-zero digits (LVCL01000001.1), as do RefSeq records (NW_024426341.1).
_WGS_MASTER_RE = re.compile(r"[A-Z]+0+")


class GapFinderError(Exception):
    """Raised when gaps can't be located for an assembly."""


def _is_wgs_master(accession: str) -> bool:
    """Whether this is a WGS project's master record rather than a real sequence.

    A master record stands in for a whole WGS project. Its "sequence" is
    padding rather than assembled bases, so scanning one yields a single
    enormous run of N that is not a gap in anything - handing that on would
    send the Reconstruction Agent off to BLAST a megabase of Ns.

    This is reachable only through the `[WGS]` fallback, and for the GenBank
    assemblies that need that fallback it is often the *only* nuccore hit:
    individual WGS contigs are retrievable by accession but are not in the
    search index, so `LVCL01[WGS]` matches exactly one record, the master.
    """
    return bool(_WGS_MASTER_RE.fullmatch(accession.split(".")[0]))


def _extract_fasta_sequence(fasta_text: str) -> str:
    """Strip the `>` header line(s) out of a FASTA response, keeping only bases."""
    return "".join(
        line.strip() for line in fasta_text.splitlines() if line and not line.startswith(">")
    )


def _find_n_runs(sequence: str, min_length: int = MIN_GAP_BP) -> list[dict]:
    """Every run of N in `sequence`, as 1-based inclusive coordinates.

    Pure/offline so it can be unit-tested without hitting NCBI.

    The coordinate convention is the one the Reconstruction Agent documents
    this agent as using (`tests/unit/test_genome_agent_handoff.py`): `start`
    is the first unresolved base and `end` the last, both 1-based, and
    `length` is always `end - start + 1`. Deriving the length rather than
    reporting a separate estimate is deliberate - the two disagreeing is
    exactly what that agent's `_reconcile` raises a note about, and there is
    no reason to manufacture one here.
    """
    gaps: list[dict] = []
    for match in _N_RUN_RE.finditer(sequence):
        start = match.start() + 1
        end = match.end()
        length = end - start + 1
        if length >= min_length:
            gaps.append({"start": start, "end": end, "length": length})
    return gaps


def _attach_flanks(gap: dict, sequence: str, flank_bp: int) -> dict:
    """Slice `flank_bp` resolved bases from either side of a gap.

    Clamped to the record, so a gap at the very start or end of the sequence
    yields a short flank or an empty one rather than reaching out of bounds.
    """
    left_end = gap["start"] - 1
    left_start = max(1, left_end - flank_bp + 1)
    left_flank = sequence[left_start - 1 : left_end] if left_end >= left_start else ""

    right_start = gap["end"] + 1
    right_end = min(len(sequence), gap["end"] + flank_bp)
    right_flank = sequence[right_start - 1 : right_end] if right_end >= right_start else ""

    return {**gap, "left_flank": left_flank, "right_flank": right_flank}


async def _largest_record_uid(term: str, min_bp: int, max_bp: int) -> str | None:
    """UID of the longest nuccore record matching `term` within a length band."""
    response = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "nuccore",
            "term": f"{term} AND {min_bp}:{max_bp}[SLEN]",
            "sort": "SLEN",
            "retmax": "1",
            "retmode": "json",
        },
    )
    ids = response.json().get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


async def _wgs_term(assembly_id: str) -> str | None:
    """This assembly's WGS project as a nuccore search term, e.g. `LVCL01[WGS]`.

    Costs two requests, so it is only reached when `[Assembly]` finds nothing.
    """
    uid = await _resolve_assembly_uid(assembly_id)
    if uid is None:
        return None

    response = await asyncio.to_thread(
        ncbi_get,
        {"path": "esummary.fcgi", "db": "assembly", "id": uid, "retmode": "json"},
    )
    record = (response.json().get("result") or {}).get(uid) or {}
    prefix = (record.get("wgs") or "").strip()
    return f"{prefix}[WGS]" if prefix else None


async def _record_census(assembly_id: str) -> dict[str, int | None]:
    """How many records this assembly has, and how many the ceiling hides.

    Only one record is ever scanned (see `_select_target_record`), so without
    these two numbers `target_gaps` looks like the assembly's gaps rather than
    one record's. Reported so the consumer can see the shape of what it did
    not get.

    Two count-only searches, skipped entirely when `REPORT_RECORD_CENSUS` is
    off, and best-effort either way: any failure reports None for both rather
    than costing the caller its gaps. `[Assembly]` is the only
    term used here - a GenBank-only assembly reached through the `[WGS]`
    fallback returns an unmeasured census rather than paying for a second
    `_wgs_term` round-trip, and None means "not measured", never "zero".
    """
    async def _count(term: str) -> int | None:
        response = await asyncio.to_thread(
            ncbi_get,
            {
                "path": "esearch.fcgi",
                "db": "nuccore",
                "term": term,
                "retmax": "0",
                "retmode": "json",
            },
        )
        raw = response.json().get("esearchresult", {}).get("count")
        return int(raw) if raw is not None else None

    unmeasured: dict[str, int | None] = {
        "records_in_assembly": None,
        "records_over_size_ceiling": None,
    }
    if not REPORT_RECORD_CENSUS:
        return unmeasured

    try:
        total = await _count(f"{assembly_id}[Assembly]")
        if not total:
            return unmeasured
        over = await _count(
            f"{assembly_id}[Assembly] AND {_SCAFFOLD_MAX_BP + 1}:1000000000[SLEN]"
        )
    except Exception as exc:  # noqa: BLE001 - a census is never worth failing over
        logger.warning("[find_target_gaps] record census failed for %r: %s", assembly_id, exc)
        return unmeasured

    return {"records_in_assembly": total, "records_over_size_ceiling": over}


async def _select_target_record(assembly_id: str) -> tuple[str, int]:
    """The largest genomic record of `assembly_id`, as (accession, length).

    Both search terms below are tied to the assembly itself rather than to its
    organism, so a species with several assemblies cannot return a record from
    the wrong one. `sort=SLEN` puts the largest first.
    """
    terms = [f"{assembly_id}[Assembly]"]
    uid = await _largest_record_uid(terms[0], _SCAFFOLD_MIN_BP, _SCAFFOLD_MAX_BP)

    if uid is None:
        # Nuccore populates its `[Assembly]` field for RefSeq assemblies only:
        # `GCF_900497805.2[Assembly]` matches 15,415 records, while every
        # GenBank-only `GCA_...` accession matches nothing at all. Those are
        # disproportionately the assemblies that reach this module, since a
        # genome nobody has promoted to RefSeq is likelier to be left at
        # Scaffold or Contig level. The WGS project the assembly was built
        # from is indexed, and is just as specific to it.
        wgs = await _wgs_term(assembly_id)
        if wgs:
            logger.info(
                "[find_target_gaps] %r is not indexed under [Assembly]; using %s",
                assembly_id,
                wgs,
            )
            terms.append(wgs)
            uid = await _largest_record_uid(wgs, _SCAFFOLD_MIN_BP, _SCAFFOLD_MAX_BP)

    if uid is None:
        # A small or heavily fragmented assembly may have nothing above the
        # floor. Better to scan its biggest available record than to report
        # no gaps at all.
        logger.info(
            "[find_target_gaps] no record in %d-%d bp for %r; dropping the floor",
            _SCAFFOLD_MIN_BP,
            _SCAFFOLD_MAX_BP,
            assembly_id,
        )
        for term in terms:
            uid = await _largest_record_uid(term, 1, _SCAFFOLD_MAX_BP)
            if uid is not None:
                break

    if uid is None:
        raise GapFinderError(
            f"No genomic record under {_SCAFFOLD_MAX_BP} bp is linked to assembly "
            f"'{assembly_id}'."
        )

    response = await asyncio.to_thread(
        ncbi_get,
        {"path": "esummary.fcgi", "db": "nuccore", "id": uid, "retmode": "json"},
    )
    record = (response.json().get("result") or {}).get(uid) or {}
    accession = record.get("accessionversion") or record.get("caption")
    if not accession:
        raise GapFinderError(f"NCBI returned no accession for nuccore UID {uid}.")

    if _is_wgs_master(accession):
        raise GapFinderError(
            f"Assembly '{assembly_id}' resolves only to the WGS master record "
            f"{accession}, whose sequence is padding rather than assembled bases. "
            f"Its individual contigs are not indexed in Nuccore, so gap "
            f"coordinates cannot be read for this assembly."
        )

    try:
        length = int(record.get("slen") or 0)
    except (TypeError, ValueError):
        length = 0
    return accession, length


async def _fetch_record_sequence(accession: str, length: int) -> str:
    """The record's residues, pulled in `MAX_WINDOW_BP` slices.

    Sequential rather than concurrent: every NCBI call in this package goes
    through one process-wide pacing gate anyway (`_ncbi_client.py`), so firing
    these in parallel would buy nothing and only make the ordering harder to
    reason about.
    """
    span = min(length, _SCAFFOLD_MAX_BP) if length else _SCAFFOLD_MAX_BP
    chunks: list[str] = []
    start = 1
    while start <= span:
        stop = min(start + MAX_WINDOW_BP - 1, span)
        chunk = _extract_fasta_sequence(await fetch_window_by_accession(accession, start, stop))
        if not chunk:
            # Past the end of the record, or NCBI returned nothing for this
            # window. Either way there is no more sequence to read.
            break
        chunks.append(chunk)
        start = stop + 1
    return "".join(chunks).upper()


async def find_target_gaps(
    assembly_id: str,
    max_gaps: int = DEFAULT_MAX_GAPS,
    flank_bp: int = DEFAULT_FLANK_BP,
) -> dict:
    """Find real gap coordinates for `assembly_id` and enrich them with flanks.

    Returns::

        {
            "sequence_accession": "NW_024426341.1",
            "target_gaps": [
                {"start": ..., "end": ..., "length": ...,
                 "left_flank": "...", "right_flank": "..."},
                ...
            ],
        }

    Alongside the gaps, the return reports what was filtered out to produce
    them: `gaps_found`, `gaps_over_floor`, `gaps_selected` and a
    `selection_policy` describing every threshold applied. Without these the
    consumer cannot tell a filtered sample from an exhaustive list - ten gaps
    out of thirty, in one record out of thousands, is indistinguishable from
    an assembly that has exactly ten. Every number below is a policy decision
    made here, not a property of the genome.

    Gaps come back **shortest first**, which is also the order `max_gaps`
    truncates against.

    This is the opposite of what it looks like it should be, and the reason is
    measured rather than assumed. Sending the longest runs - the intuitive
    choice, since they are the most of the assembly left unresolved - sends
    the Reconstruction Agent exactly the gaps it cannot close: on
    NW_024426341.1 the five longest (537-2,784 bp) all came back
    INSUFFICIENT_GAP_SPANNING_HOMOLOGS, because no reference in `core_nt`
    covers both flanks of a run that size and the region is past what Evo 2
    will cover. Left to find its own gaps in the same record it resolved two,
    both only a handful of bases long. Short gaps are the ones that close, so
    those are the ones worth the payload.

    Raises GapFinderError if the assembly has no scannable genomic record. An
    assembly that resolves but has no runs of N is not an error - it returns
    an empty `target_gaps` list.
    """
    accession, length = await _select_target_record(assembly_id)
    census = await _record_census(assembly_id)
    logger.info(
        "[find_target_gaps] scanning %s (%s bp) for assembly=%r",
        accession,
        length or "unknown",
        assembly_id,
    )

    sequence = await _fetch_record_sequence(accession, length)

    # Counted before the floor is applied as well as after, so the payload can
    # say how much the floor itself removed rather than only what survived it.
    found = _find_n_runs(sequence, min_length=1)
    gaps = [gap for gap in found if gap["length"] >= MIN_GAP_BP]
    gaps.sort(key=lambda gap: gap["length"])
    selected = gaps[:max_gaps]

    policy = {
        "order": "shortest_first",
        "min_gap_bp": MIN_GAP_BP,
        "max_gaps": max_gaps,
        "flank_bp": flank_bp,
        "record_size_band_bp": [_SCAFFOLD_MIN_BP, _SCAFFOLD_MAX_BP],
        "records_scanned": 1,
        "record_length_bp": length or None,
        "bases_scanned": len(sequence),
        **census,
    }
    result = {
        "sequence_accession": accession,
        "target_gaps": [_attach_flanks(gap, sequence, flank_bp) for gap in selected],
        "gaps_found": len(found),
        "gaps_over_floor": len(gaps),
        "gaps_selected": len(selected),
        "selection_policy": policy,
    }

    if not selected:
        logger.info(
            "[find_target_gaps] no runs of N at or above %d bp in %s "
            "(%d shorter runs, %d bases scanned)",
            MIN_GAP_BP,
            accession,
            len(found),
            len(sequence),
        )
        return result

    logger.info(
        "[find_target_gaps] %d of %d gaps in %s (%d found, shortest %d bp)",
        len(selected),
        len(gaps),
        accession,
        len(found),
        selected[0]["length"],
    )
    return result


if __name__ == "__main__":

    async def _quick_test():
        print("--- Gap Finder live NCBI test ---")
        result = await find_target_gaps("GCF_017311325.1", max_gaps=3)
        print("sequence_accession:", result["sequence_accession"])
        for gap in result["target_gaps"]:
            print(
                f"  {gap['start']}-{gap['end']} ({gap['length']} bp) "
                f"L:{gap['left_flank'][:20]} R:{gap['right_flank'][:20]}"
            )

    asyncio.run(_quick_test())

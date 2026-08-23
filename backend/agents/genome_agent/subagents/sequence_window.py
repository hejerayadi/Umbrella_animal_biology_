"""
Sequence Window — real NCBI Nuccore efetch subagent (new capability).
Fetches a small DNA sequence window for a resolved assembly.

Never cached — pass-through only, and not yet wired into the LangGraph
orchestrator. It's a standalone, importable subagent for now; wiring it
into build_genome_graph() (new state fields for seq_start/seq_stop, a
new node, and a routing decision for when a query needs a sequence
window rather than metadata/annotation) is a product decision left to
whoever exposes this to users, not made here.

Assembly accessions (e.g. "GCF_018350195.1") are NOT valid Nuccore IDs —
Nuccore holds individual sequences (chromosomes, scaffolds, contigs),
each with its own accession, and an assembly is a collection of those.
Passing an assembly accession straight to efetch on db=nuccore fails
with NCBI's "Failed to understand id" error. So resolving a window
requires an extra hop first:

    1. esearch (db=assembly) the accession -> assembly UID
    2. elink (assembly -> nuccore) the UID  -> a nuccore UID for one of
       that assembly's sequences
    3. efetch (db=nuccore) that UID with seq_start/seq_stop

Step 2 just takes NCBI's first linked sequence, which is not
guaranteed to be the largest chromosome or any particular one — good
enough to prove the window-fetch mechanism works end to end, but if a
caller needs a *specific* chromosome, they should resolve and pass its
own Nuccore accession directly rather than relying on this fallback.

Safety: MAX_WINDOW_BP = 200_000 — requests larger than this raise
WindowTooLargeError before any HTTP call is made, since Nuccore records
are GB-scale and this subagent is windowed by design (see the
consolidated store-or-not matrix: "GB-scale source; windowed by
design").
"""

from __future__ import annotations

import asyncio
import logging
import re

from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)

MAX_WINDOW_BP = 200_000


class WindowTooLargeError(Exception):
    """Raised when a requested sequence window exceeds MAX_WINDOW_BP."""


class NoLinkedSequenceError(Exception):
    """Raised when an assembly accession can't be resolved to any Nuccore
    sequence to fetch a window from."""


async def _resolve_assembly_uid(assembly_id: str) -> str | None:
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": f"{assembly_id}[Assembly]",
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])
    return uid_list[0] if uid_list else None


async def _resolve_nuccore_id(assembly_id: str, assembly_uid: str) -> str | None:
    """Follow the assembly -> nuccore link to get a fetchable sequence ID.

    Tries the RefSeq linkname first, then the INSDC/GenBank one, since
    which is populated depends on whether the assembly is a GCF_ or GCA_
    accession (and RefSeq assemblies sometimes carry both).
    """
    linknames = (
        ["assembly_nuccore_refseq", "assembly_nuccore_insdc"]
        if assembly_id.startswith("GCF_")
        else ["assembly_nuccore_insdc", "assembly_nuccore_refseq"]
    )

    for linkname in linknames:
        resp = await asyncio.to_thread(
            ncbi_get,
            {
                "path": "elink.fcgi",
                "dbfrom": "assembly",
                "db": "nuccore",
                "id": assembly_uid,
                "linkname": linkname,
                "retmode": "json",
            },
        )
        data = resp.json()
        linksets = data.get("linksets", [])
        for linkset in linksets:
            for linksetdb in linkset.get("linksetdbs", []):
                links = linksetdb.get("links", [])
                if links:
                    return links[0]

    return None


async def fetch_sequence_window(
    assembly_id: str,
    seq_start: int,
    seq_stop: int,
) -> str:
    window_size = seq_stop - seq_start
    if window_size > MAX_WINDOW_BP:
        raise WindowTooLargeError(
            f"Requested window size ({window_size} bp) exceeds "
            f"MAX_WINDOW_BP ({MAX_WINDOW_BP} bp)"
        )

    assembly_uid = await _resolve_assembly_uid(assembly_id)
    if assembly_uid is None:
        raise NoLinkedSequenceError(f"Assembly '{assembly_id}' not found on NCBI.")

    nuccore_id = await _resolve_nuccore_id(assembly_id, assembly_uid)
    if nuccore_id is None:
        raise NoLinkedSequenceError(
            f"No Nuccore sequence is linked to assembly '{assembly_id}'."
        )

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "efetch.fcgi",
            "db": "nuccore",
            "id": nuccore_id,
            "seq_start": seq_start,
            "seq_stop": seq_stop,
            "rettype": "fasta",
            "retmode": "text",
        },
    )
    return resp.text


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Sequence Window live NCBI test ---")

        # Small window — should work
        seq = await fetch_sequence_window("GCF_000464555.1", 1, 1000)
        print("Window 1-1000:", seq[:100])
        assert ">" in seq or len(seq) > 0

        # Too-large window — should raise before any HTTP call
        try:
            await fetch_sequence_window("GCF_000464555.1", 1, 200_001)
            assert False, "Expected WindowTooLargeError"
        except WindowTooLargeError as exc:
            print("Correctly raised:", exc)

        print("All tests passed ✅")

    asyncio.run(_quick_test())


# ---------------------------------------------------------------------------
# Picking a sequence the Reconstruction Agent can actually work on.
#
# The Reconstruction Agent repairs ONE sequence record: it takes either pasted
# residues or a Nuccore accession, and its card is explicit that it "does not
# pick a target sequence on its own". The Genome Agent used to hand it the
# *assembly* accession (GCF_...), which is not a sequence at all - an assembly
# is a collection of thousands of them - so every handoff was rejected with
# "No sequence was given to reconstruct".
#
# This resolves an assembly's organism to the single largest RefSeq genomic
# record for that species, which is the one worth repairing: gaps (runs of N)
# are what joins contigs into scaffolds, so the big scaffolds are where the
# unresolved regions live. Measured on the polar bear's scaffold-level
# assembly, its largest scaffolds carry thousands of N bases in the first
# 200 kb, while the small unplaced fragments NCBI happens to link first carry
# none at all.
# ---------------------------------------------------------------------------


# The Reconstruction Agent downloads the whole record before it can work
# (`_fetch_by_accession` has no windowing), so the target has to be small
# enough to fetch. A chromosome-scale scaffold is not: handing over the polar
# bear's largest (125 Mb) made the request hang until the HTTP timeout.
# Capped at 2 Mb the fetch takes seconds, and the records still carry real
# gaps - the largest sub-2Mb polar bear scaffold has 17 runs of N, the
# longest 2,784 bases. The floor skips the small unplaced fragments, which
# measured gap-free.
_SCAFFOLD_MIN_BP = 50_000
_SCAFFOLD_MAX_BP = 2_000_000


def _largest_refseq_genomic_uid(organism: str) -> str | None:
    """UID of the longest fetchable RefSeq genomic record for `organism`."""
    response = ncbi_get(
        {
            "path": "esearch.fcgi",
            "db": "nuccore",
            "term": (
                f'"{organism}"[Organism] AND srcdb_refseq[PROP] '
                f"AND biomol_genomic[PROP] "
                f"AND {_SCAFFOLD_MIN_BP}:{_SCAFFOLD_MAX_BP}[SLEN]"
            ),
            "retmode": "json",
            "retmax": "1",
            "sort": "SLEN",
        }
    )
    ids = response.json().get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def find_largest_genomic_scaffold(organism: str) -> dict | None:
    """The largest RefSeq genomic sequence for a species.

    Returns ``{"accession": ..., "length_bp": ..., "title": ...}`` or None when
    the species has no RefSeq genomic record, NCBI is unreachable, or the
    response is not the shape expected. None is a normal outcome, not an error:
    the caller falls back to handing over no accession at all, which is exactly
    the behaviour that existed before this function.
    """
    if not organism or not organism.strip():
        return None

    # The species resolver reports names like "Ursus maritimus (polar bear)" -
    # scientific name with the common name in parentheses. NCBI's [Organism]
    # field takes that literally and finds nothing, so the parenthetical is
    # stripped before querying. If the full remaining name still finds nothing
    # (subspecies strings sometimes do not), fall back to the Linnaean
    # genus + species, the first two words.
    cleaned = re.sub(r"\s*\([^)]*\)", "", organism).strip()
    if not cleaned:
        return None

    try:
        uid = _largest_refseq_genomic_uid(cleaned)
        if uid is None and len(cleaned.split()) > 2:
            uid = _largest_refseq_genomic_uid(" ".join(cleaned.split()[:2]))
        if uid is None:
            return None

        summary = ncbi_get(
            {"path": "esummary.fcgi", "db": "nuccore", "id": uid, "retmode": "json"}
        ).json()
        record = summary.get("result", {}).get(uid) or {}
        accession = record.get("caption") or record.get("accessionversion")
        if not accession:
            return None

        return {
            "accession": accession,
            "length_bp": record.get("slen"),
            "title": record.get("title"),
        }
    except Exception as exc:  # noqa: BLE001 - a handoff hint must never fail the run
        logger.info("[sequence_window] scaffold lookup for %r failed: %s", organism, exc)
        return None

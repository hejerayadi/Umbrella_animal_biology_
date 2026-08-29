"""
Sequence Window — real NCBI Nuccore efetch subagent.
Fetches a small DNA sequence window for a resolved assembly.

Never cached — pass-through only. Wired into the LangGraph orchestrator via
`subagents/gap_finder.py` / `workflows/nodes/gap_finder_node.py`, which read a
whole scaffold through `fetch_window_by_accession` one window at a time,
scanning it for the runs of N that become `target_gaps` in the Reconstruction
Agent handoff. It remains a standalone, importable subagent otherwise —
nothing here assumes that caller; any other node needing an arbitrary sequence
window can call either entry point directly.

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


async def fetch_window_by_accession(
    accession: str,
    seq_start: int,
    seq_stop: int,
) -> str:
    """A window of one *named* Nuccore record, as FASTA text.

    The half of `fetch_sequence_window` that does the actual fetching, split
    out because a caller that already knows which record it wants should not
    pay for the assembly -> Nuccore resolution again. `gap_finder.py` reads a
    whole scaffold this way, one window at a time: re-resolving per window
    would cost two extra requests each, and (worse) leave the windows only
    incidentally pointing at the same record.

    An accession like "NW_024426341.1" is a valid Nuccore id, so it goes
    straight to efetch. Coordinates are 1-based inclusive, as NCBI takes them.
    """
    window_size = seq_stop - seq_start
    if window_size > MAX_WINDOW_BP:
        raise WindowTooLargeError(
            f"Requested window size ({window_size} bp) exceeds "
            f"MAX_WINDOW_BP ({MAX_WINDOW_BP} bp)"
        )

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "efetch.fcgi",
            "db": "nuccore",
            "id": accession,
            "seq_start": seq_start,
            "seq_stop": seq_stop,
            "rettype": "fasta",
            "retmode": "text",
        },
    )
    return resp.text


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

    return await fetch_window_by_accession(nuccore_id, seq_start, seq_stop)


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

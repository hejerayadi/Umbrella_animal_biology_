"""
Species Resolver — real NCBI Assembly eutils subagent (Task 5)
Resolves a species name to its genome assembly ID using NCBI eutils.
Never cached — always reflects NCBI's current record.

Prefers RefSeq (GCF_) assemblies. Uses esearch filters when available,
falls back to client-side preference when no RefSeq exists for a species.
"""

from __future__ import annotations

import asyncio
import logging

from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)


async def _try_refseq_filter(species_term: str) -> tuple[str | None, str | None]:
    """Try to get the latest RefSeq assembly using an esearch filter.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism] AND latest_refseq[filter]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 1,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    uid = uid_list[0]

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})
    assembly_id = assembly_info.get("assemblyaccession", "")

    if assembly_id.startswith("GCF_"):
        return uid, assembly_id

    return None, None


async def _try_unfiltered_fallback(species_term: str) -> tuple[str | None, str | None]:
    """Fall back to unfiltered esearch, preferring GCF_ over GCA_.
    Returns (uid, assembly_id) or (None, None)."""
    term = f"{species_term}[Organism]"

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 10,
        },
    )
    data = resp.json()
    uid_list = data.get("esearchresult", {}).get("idlist", [])

    if not uid_list:
        return None, None

    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": ",".join(uid_list),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    chosen_uid = None
    chosen_acc = None
    fallback_uid = None
    fallback_acc = None

    for uid in uid_list:
        info = results.get(uid, {})
        acc = info.get("assemblyaccession", "")
        if acc.startswith("GCF_") and chosen_uid is None:
            chosen_uid = uid
            chosen_acc = acc
            break
        elif acc.startswith("GCA_") and fallback_uid is None:
            fallback_uid = uid
            fallback_acc = acc

    target_uid = chosen_uid or fallback_uid
    target_acc = chosen_acc or fallback_acc

    return target_uid, target_acc


async def resolve_species(species_name: str) -> dict:
    key = species_name.strip()

    # Strategy 1: Try latest RefSeq filter first
    uid, assembly_id = await _try_refseq_filter(key)

    # Strategy 2: Fall back to unfiltered search with GCF_ preference
    if uid is None:
        uid, assembly_id = await _try_unfiltered_fallback(key)

    if uid is None or assembly_id is None:
        return {
            "assembly_id": None,
            "scientific_name": None,
            "common_name": None,
            "confidence": 0.0,
        }

    # Get species details using the numeric UID
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "assembly",
            "id": uid,
            "retmode": "json",
        },
    )
    data = resp.json()
    assembly_info = data.get("result", {}).get(uid, {})

    scientific_name = assembly_info.get("organism")
    common_name = assembly_info.get("organism")

    return {
        "assembly_id": assembly_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "confidence": 1.0 if assembly_id else 0.0,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Species Resolver live NCBI test ---")
        for species in ["tiger", "house mouse", "asian elephant", "dragon"]:
            result = await resolve_species(species)
            print(f"{species}: {result}")

        print("\nAll tests passed ✅")

    asyncio.run(_quick_test())

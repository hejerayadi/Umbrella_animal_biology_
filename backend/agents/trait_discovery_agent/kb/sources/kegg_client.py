import asyncio

import httpx
from langchain_core.tools import tool

from schemas.outputs import PathwayEntry
from kb.sources._http_retry import request_with_retry

KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/{kegg_gene_id}"
KEGG_GET_URL = "https://rest.kegg.jp/get/{pathway_id}"


# --------------------------------------------------------------------------- #
#  Raw implementations — called directly by the node for branching (§8) and
#  as the deterministic fallback (§9); also what tests monkeypatch.
# --------------------------------------------------------------------------- #

async def _list_pathway_candidates_raw(kegg_gene_id: str) -> list[dict]:
    """All pathways KEGG's link endpoint returns for this gene, not just the first."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await request_with_retry(
            client, "GET", KEGG_LINK_URL.format(kegg_gene_id=kegg_gene_id)
        )
        candidates = []
        for line in resp.text.strip().splitlines():
            if "\t" in line:
                _gene_part, path_part = line.split("\t", 1)
                pathway_id = path_part.replace("path:", "")
                candidates.append({"pathway_id": pathway_id})
        return candidates


async def _fetch_pathway_name_raw(pathway_id: str) -> str:
    """Resolve a KEGG pathway ID to its human-readable NAME field."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await request_with_retry(
            client, "GET", KEGG_GET_URL.format(pathway_id=pathway_id)
        )
        for text_line in resp.text.splitlines():
            if text_line.startswith("NAME"):
                return text_line.replace("NAME", "").strip()
        return ""


async def _fetch_pathway_names_raw(pathway_ids: list[str]) -> dict[str, str]:
    """Resolve MULTIPLE KEGG pathway IDs concurrently. A gene can legitimately
    have a dozen+ real KEGG links (broadly-connected genes especially), and
    resolving them one per LLM turn means paying a full NIM round-trip (which
    can itself take up to ~60s+ on a cold/queued request) per candidate. This
    collapses that into a single tool call/turn regardless of candidate count."""
    results = await asyncio.gather(
        *[_fetch_pathway_name_raw(pid) for pid in pathway_ids],
        return_exceptions=True,
    )
    return {
        pid: (name if isinstance(name, str) else "")
        for pid, name in zip(pathway_ids, results)
    }


# --------------------------------------------------------------------------- #
#  bind_tools-facing wrappers (guide §5 / §2)
# --------------------------------------------------------------------------- #

@tool
async def list_pathway_candidates(kegg_gene_id: str) -> list[dict]:
    """All pathways KEGG's link endpoint returns for this gene, not just the first."""
    return await _list_pathway_candidates_raw(kegg_gene_id)


@tool
async def fetch_pathway_name(pathway_id: str) -> str:
    """Resolve a KEGG pathway ID to its human-readable NAME field."""
    return await _fetch_pathway_name_raw(pathway_id)


@tool
async def fetch_pathway_names(pathway_ids: list[str]) -> dict[str, str]:
    """Resolve MULTIPLE KEGG pathway IDs to their names in a single call.
    Prefer this over calling fetch_pathway_name repeatedly — pass every
    candidate id you need resolved at once, not one at a time."""
    return await _fetch_pathway_names_raw(pathway_ids)


# --------------------------------------------------------------------------- #
#  Deterministic fallback (§9) — plain function, no bind_tools machinery.
# --------------------------------------------------------------------------- #

async def fetch_pathway(kegg_gene_id: str) -> PathwayEntry | None:
    """Deterministic fallback: first link only. Kept for LLM-outage degradation."""
    candidates = await _list_pathway_candidates_raw(kegg_gene_id)
    if not candidates:
        return None

    pathway_id = candidates[0]["pathway_id"]
    pathway_name = await _fetch_pathway_name_raw(pathway_id)

    return PathwayEntry(pathway_id=pathway_id, pathway_name=pathway_name)
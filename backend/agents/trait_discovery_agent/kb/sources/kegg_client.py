import asyncio

import httpx
from langchain_core.tools import tool

from schemas.outputs import PathwayEntry

KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/{kegg_gene_id}"
KEGG_GET_URL = "https://rest.kegg.jp/get/{pathway_id}"


# --------------------------------------------------------------------------- #
#  Raw implementations — called directly by the node for branching (§8) and
#  as the deterministic fallback (§9); also what tests monkeypatch.
# --------------------------------------------------------------------------- #

async def _list_pathway_candidates_raw(kegg_gene_id: str) -> list[dict]:
    """All pathways KEGG's link endpoint returns for this gene, not just the first."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(2):
            try:
                resp = await client.get(KEGG_LINK_URL.format(kegg_gene_id=kegg_gene_id))
                resp.raise_for_status()
                candidates = []
                for line in resp.text.strip().splitlines():
                    if "\t" in line:
                        _gene_part, path_part = line.split("\t", 1)
                        pathway_id = path_part.replace("path:", "")
                        candidates.append({"pathway_id": pathway_id})
                return candidates
            except httpx.TimeoutException:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise
    return []


async def _fetch_pathway_name_raw(pathway_id: str) -> str:
    """Resolve a KEGG pathway ID to its human-readable NAME field."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(2):
            try:
                resp = await client.get(KEGG_GET_URL.format(pathway_id=pathway_id))
                resp.raise_for_status()
                for text_line in resp.text.splitlines():
                    if text_line.startswith("NAME"):
                        return text_line.replace("NAME", "").strip()
                return ""
            except httpx.TimeoutException:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                raise
    return ""


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
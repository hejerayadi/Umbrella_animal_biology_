"""
KEGG REST API client.

LICENSE NOTICE — KEGG's REST API (rest.kegg.jp) is free for academic use by
individual, non-commercial, non-profit users only; any commercial use
requires a separate paid KEGG FTP/API subscription
(https://www.kegg.jp/kegg/legal.html). This is enforced below via
_assert_kegg_academic_use_only(), not just documented here, because a
docstring alone is easy to miss when this client gets reused/deployed
elsewhere. Deployments with a commercial license must set
KEGG_COMMERCIAL_USE_LICENSED=true to opt out of the guard.
"""
import asyncio
import os

import httpx
from langchain_core.tools import tool

from schemas.outputs import PathwayEntry
from kb.sources._http_retry import request_with_retry

KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/{kegg_gene_id}"
KEGG_GET_URL = "https://rest.kegg.jp/get/{pathway_id}"
KEGG_FIND_URL = "https://rest.kegg.jp/find/{organism}/{gene_symbol}"


def _assert_kegg_academic_use_only() -> None:
    """
    Refuses to call the KEGG REST API unless either (a) commercial use is
    explicitly declared unlicensed/not-applicable (default), or (b) the
    deployment has explicitly declared it holds a commercial KEGG license.

    KEGG_COMMERCIAL_USE=true with no accompanying
    KEGG_COMMERCIAL_USE_LICENSED=true is the one combination this blocks:
    it means someone has flagged commercial use is happening but hasn't
    confirmed a license covers it, which is exactly the unenforced gap the
    checklist flagged.
    """
    commercial_use = os.environ.get("KEGG_COMMERCIAL_USE", "false").lower() == "true"
    licensed = os.environ.get("KEGG_COMMERCIAL_USE_LICENSED", "false").lower() == "true"
    if commercial_use and not licensed:
        raise RuntimeError(
            "KEGG_COMMERCIAL_USE=true but KEGG_COMMERCIAL_USE_LICENSED is not "
            "set. KEGG's REST API is academic/non-commercial use only "
            "(https://www.kegg.jp/kegg/legal.html) — obtain a commercial "
            "license and set KEGG_COMMERCIAL_USE_LICENSED=true, or unset "
            "KEGG_COMMERCIAL_USE, before calling this client."
        )


# --------------------------------------------------------------------------- #
#  Raw implementations — called directly by the node for branching (§8) and
#  as the deterministic fallback (§9); also what tests monkeypatch.
# --------------------------------------------------------------------------- #

async def _find_kegg_gene_id_raw(organism: str, gene_symbol: str) -> str | None:
    """The KEGG gene id ("mmu:14176") for a gene symbol, or None.

    KEGG keys genes by Entrez id, not by symbol: `link/pathway/mmu:FGF5`
    answers 200 with an empty body, which reads downstream as "this gene is in
    no pathway" rather than as a lookup that never happened.

    `find` is a substring search over every name and alias KEGG holds, and it
    is not ranked usefully - searching mmu for FGF5 returns Fgf7 first, because
    one of Fgf7's aliases is "Fgf5b". So the reply is matched against the
    alias list of each row and only an exact symbol match is accepted.
    """
    _assert_kegg_academic_use_only()
    wanted = gene_symbol.strip().lower()
    if not wanted or not organism:
        return None

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await request_with_retry(
            client,
            "GET",
            KEGG_FIND_URL.format(organism=organism, gene_symbol=gene_symbol),
        )

    for line in resp.text.strip().splitlines():
        if "\t" not in line:
            continue
        kegg_gene_id, description = line.split("\t", 1)
        # "Fgf5, Fgf-5, Fgf3a, HBGF-5; fibroblast growth factor 5 precursor"
        # - the aliases are everything before the first semicolon.
        aliases = [alias.strip().lower() for alias in description.split(";", 1)[0].split(",")]
        if wanted in aliases:
            return kegg_gene_id.strip()
    return None


async def _list_pathway_candidates_raw(kegg_gene_id: str) -> list[dict]:
    """All pathways KEGG's link endpoint returns for this gene, not just the first."""
    _assert_kegg_academic_use_only()
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
    _assert_kegg_academic_use_only()
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
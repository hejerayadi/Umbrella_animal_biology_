import asyncio

import httpx
from langchain_core.tools import tool

from schemas.outputs import GOAnnotation
from kb.sources._http_retry import request_with_retry

QUICKGO_SEARCH_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
QUICKGO_TERMS_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"


# --------------------------------------------------------------------------- #
#  Raw implementations.
#
#  These are what the node calls directly (to decide single- vs multi-candidate
#  per §8, and as the deterministic fallback per §9) and what tests monkeypatch.
#  The @tool-wrapped versions below are thin adapters over these so the *same*
#  QuickGO calls are also exposed to the LLM via bind_tools (§5).
# --------------------------------------------------------------------------- #

async def _list_go_candidates_raw(gene_symbol: str, uniprot_accession: str) -> list[dict]:
    """Return every biological_process GO annotation QuickGO has for this accession."""
    params = {
        "geneProductId": f"UniProtKB:{uniprot_accession}",
        "aspect": "biological_process",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await request_with_retry(client, "GET", QUICKGO_SEARCH_URL, params=params)
        raw = resp.json().get("results", [])

        # Deduplicate by goId while preserving QuickGO order
        seen = set()
        candidates = []
        for r in raw:
            go_id = r.get("goId")
            if go_id and go_id not in seen:
                seen.add(go_id)
                candidates.append({"go_id": go_id, "go_name": None})
        return candidates


async def _resolve_go_term_name_raw(go_id: str) -> str | None:
    """Resolve a GO id to its human-readable name via the QuickGO ontology endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        term_resp = await request_with_retry(
            client, "GET", QUICKGO_TERMS_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        term_results = term_resp.json().get("results", [])
        if not term_results or not term_results[0].get("name"):
            return None
        return term_results[0]["name"]


async def _resolve_go_term_names_raw(go_ids: list[str]) -> dict[str, str | None]:
    """Resolve MULTIPLE GO ids concurrently. A gene can have a dozen+ real
    biological_process annotations, and resolving them one per LLM turn means
    paying a full NIM round-trip (which can itself take up to ~60s+ on a
    cold/queued request) per candidate. This collapses that into a single
    tool call/turn regardless of candidate count."""
    results = await asyncio.gather(
        *[_resolve_go_term_name_raw(gid) for gid in go_ids],
        return_exceptions=True,
    )
    return {
        gid: (name if isinstance(name, str) else None)
        for gid, name in zip(go_ids, results)
    }


# --------------------------------------------------------------------------- #
#  bind_tools-facing wrappers (guide §5 / §2: model gets direct, typed access)
# --------------------------------------------------------------------------- #

@tool
async def list_go_candidates(gene_symbol: str, uniprot_accession: str) -> list[dict]:
    """All biological_process GO annotations QuickGO has for this UniProt accession."""
    return await _list_go_candidates_raw(gene_symbol, uniprot_accession)


@tool
async def resolve_go_term_name(go_id: str) -> str | None:
    """Resolve a GO id to its human-readable name via the QuickGO ontology endpoint."""
    return await _resolve_go_term_name_raw(go_id)


@tool
async def resolve_go_term_names(go_ids: list[str]) -> dict[str, str | None]:
    """Resolve MULTIPLE GO ids to their names in a single call. Prefer this
    over calling resolve_go_term_name repeatedly — pass every candidate id
    you need resolved at once, not one at a time."""
    return await _resolve_go_term_names_raw(go_ids)


# --------------------------------------------------------------------------- #
#  Deterministic fallback (§9) — plain function, no bind_tools machinery.
# --------------------------------------------------------------------------- #

async def fetch_go_annotation(gene_symbol: str, uniprot_accession: str) -> GOAnnotation | None:
    """Deterministic fallback: take the first candidate, no ranking (§9)."""
    candidates = await _list_go_candidates_raw(gene_symbol, uniprot_accession)
    if not candidates:
        return None

    go_id = candidates[0]["go_id"]
    go_name = await _resolve_go_term_name_raw(go_id)
    if not go_name:
        return None

    return GOAnnotation(gene_symbol=gene_symbol, go_id=go_id, go_name=go_name)
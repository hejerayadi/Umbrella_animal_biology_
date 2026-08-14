import httpx
from langchain_core.tools import tool

from schemas.outputs import GOAnnotation

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
        resp = await client.get(QUICKGO_SEARCH_URL, params=params)
        resp.raise_for_status()
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
        term_resp = await client.get(
            QUICKGO_TERMS_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        term_resp.raise_for_status()
        term_results = term_resp.json().get("results", [])
        if not term_results or not term_results[0].get("name"):
            return None
        return term_results[0]["name"]


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
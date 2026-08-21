import httpx
from langchain_core.tools import tool

from schemas.outputs import ProteinEntry
from kb.sources._http_retry import request_with_retry

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def _parse_result(gene_symbol: str, entry: dict) -> dict:
    protein_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )
    comments = entry.get("comments", [])
    function_summary = next(
        (c["texts"][0]["value"] for c in comments if c.get("commentType") == "FUNCTION"),
        "",
    )
    return {
        "source_accession": entry.get("primaryAccession", ""),
        "protein_name": protein_name,
        "function_summary": function_summary,
    }


# --------------------------------------------------------------------------- #
#  Raw implementation — called directly by the node for branching (§8) and
#  also what tests monkeypatch.
# --------------------------------------------------------------------------- #

async def _list_uniprot_candidates_raw(gene_symbol: str, tax_id: int) -> list[dict]:
    """Every reviewed UniProt hit for gene+species, not just the first."""
    params = {
        "query": f"gene:{gene_symbol} AND organism_id:{tax_id} AND reviewed:true",
        "fields": "accession,protein_name,cc_function",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await request_with_retry(client, "GET", UNIPROT_URL, params=params)
        results = resp.json().get("results", [])

    return [_parse_result(gene_symbol, entry) for entry in results]


# --------------------------------------------------------------------------- #
#  bind_tools-facing wrapper (guide §5 / §2)
# --------------------------------------------------------------------------- #

@tool
async def list_uniprot_candidates(gene_symbol: str, tax_id: int) -> list[dict]:
    """Every reviewed UniProt hit for gene+species, not just the first."""
    return await _list_uniprot_candidates_raw(gene_symbol, tax_id)


# --------------------------------------------------------------------------- #
#  Deterministic fallback (§9) — plain function, no bind_tools machinery.
# --------------------------------------------------------------------------- #

async def fetch_uniprot(gene_symbol: str, tax_id: int) -> ProteinEntry | None:
    """Deterministic fallback: first reviewed hit only. Kept for LLM-outage
    degradation, matching the real protein_data_agent() prior to §8/§9."""
    candidates = await _list_uniprot_candidates_raw(gene_symbol, tax_id)
    if not candidates:
        return None

    first = candidates[0]
    return ProteinEntry(
        gene_symbol=gene_symbol,
        protein_name=first["protein_name"],
        function_summary=first["function_summary"],
        source_accession=first["source_accession"],
    )

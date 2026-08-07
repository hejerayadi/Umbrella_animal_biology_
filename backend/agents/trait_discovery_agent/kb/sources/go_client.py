import httpx
from schemas.outputs import GOAnnotation

QUICKGO_SEARCH_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
QUICKGO_TERMS_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"


async def fetch_go_annotation(gene_symbol: str, uniprot_accession: str) -> GOAnnotation | None:
    params = {
        "geneProductId": f"UniProtKB:{uniprot_accession}",
        "aspect": "biological_process",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(QUICKGO_SEARCH_URL, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        go_id = results[0]["goId"]

        # annotation/search doesn't return the human-readable term name — resolve it
        # separately from the ontology endpoint.
        term_resp = await client.get(
            QUICKGO_TERMS_URL.format(go_id=go_id),
            headers={"Accept": "application/json"},
        )
        term_resp.raise_for_status()
        term_results = term_resp.json().get("results", [])

    if not term_results or not term_results[0].get("name"):
        return None  # got an ID but couldn't resolve a name — treat as not found

    return GOAnnotation(
        gene_symbol=gene_symbol,
        go_id=go_id,
        go_name=term_results[0]["name"],
    )
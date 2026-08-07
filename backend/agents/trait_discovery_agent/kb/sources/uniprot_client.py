import httpx
from schemas.outputs import ProteinEntry

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


async def fetch_uniprot(gene_symbol: str, tax_id: int) -> ProteinEntry | None:
    params = {
        "query": f"gene:{gene_symbol} AND organism_id:{tax_id} AND reviewed:true",
        "fields": "accession,protein_name,cc_function",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(UNIPROT_URL, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])

    if not results:
        return None

    entry = results[0]
    protein_name = entry["proteinDescription"]["recommendedName"]["fullName"]["value"]
    comments = entry.get("comments", [])
    function_summary = next(
        (c["texts"][0]["value"] for c in comments if c.get("commentType") == "FUNCTION"),
        "",
    )
    accession = entry.get("primaryAccession", "")
    return ProteinEntry(
        gene_symbol=gene_symbol,
        protein_name=protein_name,
        function_summary=function_summary,
        source_accession=accession,
    )
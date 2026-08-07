import httpx
from schemas.outputs import PathwayEntry

KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/{kegg_gene_id}"
KEGG_GET_URL = "https://rest.kegg.jp/get/{pathway_id}"


async def fetch_pathway(kegg_gene_id: str) -> PathwayEntry | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        link_resp = await client.get(KEGG_LINK_URL.format(kegg_gene_id=kegg_gene_id))
        link_resp.raise_for_status()
        line = link_resp.text.strip().splitlines()
        if not line:
            return None
        pathway_id = line[0].split("\t")[1].replace("path:", "")

        get_resp = await client.get(KEGG_GET_URL.format(pathway_id=pathway_id))
        get_resp.raise_for_status()
        pathway_name = ""
        for text_line in get_resp.text.splitlines():
            if text_line.startswith("NAME"):
                pathway_name = text_line.replace("NAME", "").strip()
                break

    return PathwayEntry(pathway_id=pathway_id, pathway_name=pathway_name)
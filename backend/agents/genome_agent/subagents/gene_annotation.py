"""
Gene Annotation — real NCBI Gene eutils subagent (Task 5)
Retrieves gene/feature annotation data for a resolved assembly.
Never cached — pass-through only, rebuilt from NCBI each request.
"""

from __future__ import annotations

import asyncio
import logging

from ._ncbi_client import ncbi_get

logger = logging.getLogger(__name__)


async def get_gene_annotation(assembly_id: str) -> dict:
    # Step 1: esearch for gene IDs associated with this assembly
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esearch.fcgi",
            "db": "gene",
            "term": f"{assembly_id}[Assembly]",
            "retmode": "json",
            "retmax": 50,
        },
    )
    data = resp.json()
    gene_ids = data.get("esearchresult", {}).get("idlist", [])

    if not gene_ids:
        return {
            "gene_table": [],
            "gene_list": [],
        }

    # Step 2: esummary for gene details (batch up to 50 IDs)
    resp = await asyncio.to_thread(
        ncbi_get,
        {
            "path": "esummary.fcgi",
            "db": "gene",
            "id": ",".join(gene_ids[:50]),
            "retmode": "json",
        },
    )
    data = resp.json()
    results = data.get("result", {})

    gene_table = []
    gene_list = []
    for gene_id in gene_ids[:50]:
        gene_info = results.get(gene_id, {})
        gene_name = gene_info.get("name") or gene_info.get("Name") or gene_id
        description = gene_info.get("description") or gene_info.get("Description") or ""
        chromosome = gene_info.get("chromosome") or gene_info.get("chromosomes") or ""
        location = f"{chromosome}" if chromosome else ""
        
        gene_table.append({
            "gene_name": gene_name,
            "location": location,
            "function": description,
        })
        gene_list.append(gene_name)

    return {
        "gene_table": gene_table,
        "gene_list": gene_list,
    }


if __name__ == "__main__":
    import asyncio

    async def _quick_test():
        print("--- Gene Annotation live NCBI test ---")
        result = await get_gene_annotation("GCF_000464555.1")
        print("Tiger genes:", result)
        assert len(result["gene_list"]) > 0, "Expected at least one gene from NCBI"

        print("All tests passed ✅")

    asyncio.run(_quick_test())

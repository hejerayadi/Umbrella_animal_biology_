from schemas.inputs import PathwaysInput
from schemas.outputs import PathwaysOutput, PathwayEntry
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.kegg_client import fetch_pathway


_MOCK_KEGG_DB = {
    "UCP1": PathwayEntry(pathway_id="ko00071", pathway_name="Fatty acid degradation"),
    "PRDM16": PathwayEntry(pathway_id="ko04928", pathway_name="Thermogenesis"),
    "FGF5": PathwayEntry(pathway_id="ko04010", pathway_name="MAPK signaling pathway"),
}


async def mock_pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways, malformed = [], []
    for gene in input.gene_list:
        if gene in _MOCK_KEGG_DB:
            pathways.append(_MOCK_KEGG_DB[gene])
        else:
            malformed.append(gene)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)






async def pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways, malformed = [], []

    for gene in input.gene_list:
        kegg_gene_id = input.context.get("kegg_gene_ids", {}).get(gene)
        dedup_key = f"kegg:{kegg_gene_id}:{gene}"
        cached = await get_cached("kegg_pathways", dedup_key)

        if cached:
            pathways.append(PathwayEntry(
                pathway_id=cached["pathway_id"], pathway_name=cached["pathway_name"],
            ))
            continue

        entry = await fetch_pathway(kegg_gene_id) if kegg_gene_id else None
        if entry is None:
            malformed.append(gene)
            continue

        await upsert_point(
            "kegg_pathways",
            dedup_key,
            text_to_embed=entry.pathway_name,
            payload={
                "gene_symbol": gene,
                "pathway_id": entry.pathway_id,
                "pathway_name": entry.pathway_name,
                "source": "KEGG REST API",
            },
        )
        pathways.append(entry)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)
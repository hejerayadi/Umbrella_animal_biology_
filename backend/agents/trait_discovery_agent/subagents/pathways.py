from datetime import datetime, timezone
from schemas.inputs import PathwaysInput
from schemas.outputs import PathwaysOutput, PathwayEntry
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.kegg_client import fetch_pathway

SCHEMA_VERSION = 1


async def pathways_agent(input: PathwaysInput) -> PathwaysOutput:
    pathways, malformed = [], []

    for gene in input.gene_list:
        kegg_gene_id = input.context.get("kegg_gene_ids", {}).get(gene)
        if not kegg_gene_id:
            malformed.append(gene)
            continue

        entry = await fetch_pathway(kegg_gene_id)
        if entry is None:
            malformed.append(gene)
            continue

        dedup_key = f"kegg:{entry.pathway_id}:{gene}"
        cached = await get_cached("kegg_pathways", dedup_key)
        if cached:
            pathways.append(entry)
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
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        pathways.append(entry)

    status = AgentStatus.COMPLETED if pathways else AgentStatus.FAILED
    return PathwaysOutput(status=status, pathways=pathways, malformed_ids=malformed)


# kept for offline/CI tests — unchanged
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
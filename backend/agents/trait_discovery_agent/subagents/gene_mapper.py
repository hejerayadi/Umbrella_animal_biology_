from datetime import datetime, timezone
from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.go_client import fetch_go_annotation

SCHEMA_VERSION = 1


async def gene_mapper_agent(input: GeneMapperInput) -> GeneMapperOutput:
    annotations, unmatched = [], []

    for gene in input.gene_list:
        uniprot_accession = input.context.get("uniprot_accessions", {}).get(gene)
        if not uniprot_accession:
            unmatched.append(gene)
            continue

        entry = await fetch_go_annotation(gene, uniprot_accession)
        if entry is None:
            unmatched.append(gene)
            continue

        dedup_key = f"go:{entry.go_id}:{entry.gene_symbol}"
        cached = await get_cached("go_annotations", dedup_key)
        if cached:
            annotations.append(entry)
            continue

        await upsert_point(
            "go_annotations",
            dedup_key,
            text_to_embed=entry.go_name,
            payload={
                "gene_symbol": entry.gene_symbol,
                "go_id": entry.go_id,
                "go_name": entry.go_name,
                "source": "GO REST API (QuickGO)",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        annotations.append(entry)

    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(status=status, go_annotations=annotations, unmatched_genes=unmatched)


# kept for offline/CI tests — unchanged
_MOCK_GO_DB = {
    "FGF5": GOAnnotation(gene_symbol="FGF5", go_id="GO:0031069", go_name="hair follicle development"),
    "KRT71": GOAnnotation(gene_symbol="KRT71", go_id="GO:0031069", go_name="hair follicle development"),
    "HR": GOAnnotation(gene_symbol="HR", go_id="GO:0042633", go_name="hair cycle"),
    "TRPV3": GOAnnotation(gene_symbol="TRPV3", go_id="GO:0050977", go_name="sensory perception of touch"),
    "UCP1": GOAnnotation(gene_symbol="UCP1", go_id="GO:0009408", go_name="response to heat"),
}


async def mock_gene_mapper(input: GeneMapperInput) -> GeneMapperOutput:
    annotations, unmatched = [], []
    for gene in input.gene_list:
        if gene in _MOCK_GO_DB:
            annotations.append(_MOCK_GO_DB[gene])
        else:
            unmatched.append(gene)

    status = AgentStatus.FAILED if (not annotations or unmatched) else AgentStatus.COMPLETED
    return GeneMapperOutput(status=status, go_annotations=annotations, unmatched_genes=unmatched)
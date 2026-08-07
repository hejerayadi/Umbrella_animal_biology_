from schemas.inputs import ProteinDataInput
from schemas.outputs import ProteinDataOutput, ProteinEntry
from schemas.common import AgentStatus
from schemas.inputs import ProteinDataInput
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.uniprot_client import fetch_uniprot

_MOCK_UNIPROT_DB = {
    "UCP1": ProteinEntry(gene_symbol="UCP1", protein_name="Uncoupling protein 1",
                          function_summary="Mitochondrial proton channel, generates heat."),
    "FGF5": ProteinEntry(gene_symbol="FGF5", protein_name="Fibroblast growth factor 5",
                          function_summary="Regulates hair follicle growth cycle."),
    "KRT71": ProteinEntry(gene_symbol="KRT71", protein_name="Keratin, type II cytoskeletal 71",
                           function_summary="Structural protein of the hair shaft."),
}


async def mock_protein_data_agent(input: ProteinDataInput) -> ProteinDataOutput:
    proteins, missing = [], []
    for gene in input.gene_list:
        if gene in _MOCK_UNIPROT_DB:
            proteins.append(_MOCK_UNIPROT_DB[gene])
        else:
            missing.append(gene)

    status = AgentStatus.COMPLETED if proteins else AgentStatus.FAILED
    return ProteinDataOutput(status=status, proteins=proteins, missing_genes=missing)





from datetime import datetime, timezone
from schemas.inputs import ProteinDataInput
from schemas.outputs import ProteinDataOutput, ProteinEntry
from schemas.common import AgentStatus
from kb.qdrant_store import get_cached, upsert_point
from kb.sources.uniprot_client import fetch_uniprot

SCHEMA_VERSION = 1

async def protein_data_agent(input: ProteinDataInput) -> ProteinDataOutput:
    tax_id = input.context.get("tax_id")
    proteins, missing = [], []

    for gene in input.gene_list:
        entry = await fetch_uniprot(gene, tax_id)
        if entry is None:
            missing.append(gene)
            continue

        dedup_key = f"uniprot:{entry.source_accession}:{tax_id}"
        cached = await get_cached("uniprot_proteins", dedup_key)
        if cached:
            proteins.append(entry)
            continue

        await upsert_point(
            "uniprot_proteins",
            dedup_key,
            text_to_embed=entry.function_summary,
            payload={
                "gene_symbol": entry.gene_symbol,
                "protein_name": entry.protein_name,
                "function_summary": entry.function_summary,
                "species_tax_id": tax_id,
                "source": "UniProt REST API",
                "source_accession": entry.source_accession,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
            },
        )
        proteins.append(entry)

    status = AgentStatus.COMPLETED if proteins else AgentStatus.FAILED
    return ProteinDataOutput(status=status, proteins=proteins, missing_genes=missing)
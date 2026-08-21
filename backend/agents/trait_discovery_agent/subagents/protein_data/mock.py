"""Mock Protein Data agent kept for offline / CI tests (unchanged contract)."""
from schemas.inputs import ProteinDataInput
from schemas.outputs import ProteinDataOutput, ProteinEntry
from schemas.common import AgentStatus

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

"""Mock Gene Mapper agent kept for offline / CI tests (unchanged contract)."""
from schemas.inputs import GeneMapperInput
from schemas.outputs import GeneMapperOutput, GOAnnotation
from schemas.common import AgentStatus

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

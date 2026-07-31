from schemas.inputs import PathwaysInput
from schemas.outputs import PathwaysOutput, PathwayEntry
from schemas.common import AgentStatus

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
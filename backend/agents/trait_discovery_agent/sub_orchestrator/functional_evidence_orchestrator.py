import asyncio

from backend.agents.Literature_Agent.trait_discovery_agent.schemas.inputs import FunctionalEvidenceInput, PathwaysInput, ProteinDataInput
from backend.agents.Literature_Agent.trait_discovery_agent.schemas.outputs import FunctionalEvidenceOutput
from backend.agents.Literature_Agent.trait_discovery_agent.schemas.common import AgentStatus


class FunctionalEvidenceOrchestrator:
    def __init__(self, pathways_agent, protein_data_agent):
        self.pathways_agent = pathways_agent
        self.protein_data_agent = protein_data_agent

    async def run(self, input: FunctionalEvidenceInput) -> FunctionalEvidenceOutput:
        pathways_in = PathwaysInput(
            gene_list=input.gene_list,
            instruction=input.instruction,
            context=input.context,
        )
        protein_in = ProteinDataInput(
            gene_list=input.gene_list,
            instruction=input.instruction,
            context=input.context,
        )

        # parallel — neither depends on the other, both read the same gene_list
        pathways_out, protein_out = await asyncio.gather(
            self.pathways_agent(pathways_in),
            self.protein_data_agent(protein_in),
        )

        status = (
            AgentStatus.FAILED
            if pathways_out.status == AgentStatus.FAILED or protein_out.status == AgentStatus.FAILED
            else AgentStatus.COMPLETED
        )

        return FunctionalEvidenceOutput(
            status=status,
            pathway_data=pathways_out.pathways,
            protein_data=protein_out.proteins,
        )
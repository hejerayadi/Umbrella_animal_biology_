from workflows.state import FunctionalEvidenceState
from schemas.inputs import PathwaysInput, ProteinDataInput
from schemas.common import AgentStatus
from subagents.pathways import mock_pathways_agent
from subagents.protein_data import mock_protein_data_agent


async def pathways_node(state: FunctionalEvidenceState) -> dict:
    out = await mock_pathways_agent(PathwaysInput(
        gene_list=state.gene_list, instruction=state.instruction, context=state.context,
    ))
    return {"pathway_data": out.pathways, "pathways_status": out.status}


async def protein_data_node(state: FunctionalEvidenceState) -> dict:
    out = await mock_protein_data_agent(ProteinDataInput(
        gene_list=state.gene_list, instruction=state.instruction, context=state.context,
    ))
    return {"protein_data": out.proteins, "protein_data_status": out.status}


async def merge_node(state: FunctionalEvidenceState) -> dict:
    """Same rule as Task 1: FAILED if either child resolved nothing."""
    status = (
        AgentStatus.FAILED
        if state.pathways_status == AgentStatus.FAILED
        or state.protein_data_status == AgentStatus.FAILED
        else AgentStatus.COMPLETED
    )
    return {"status": status}
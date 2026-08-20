import logging

from workflows.state import FunctionalEvidenceState
from schemas.inputs import PathwaysInput, ProteinDataInput
from schemas.common import AgentStatus
from subagents.pathways import mock_pathways_agent
from subagents.protein_data import mock_protein_data_agent

logger = logging.getLogger(__name__)


async def pathways_node(state: FunctionalEvidenceState) -> dict:
    logger.info("pathways input gene list=%s", state.gene_list)
    out = await mock_pathways_agent(PathwaysInput(
        gene_list=state.gene_list, instruction=state.instruction, context=state.context,
    ))
    logger.info("pathways output=%s", out)
    return {"pathway_data": out.pathways, "pathways_status": out.status}


async def protein_data_node(state: FunctionalEvidenceState) -> dict:
    logger.info("protein_data input gene list=%s", state.gene_list)
    out = await mock_protein_data_agent(ProteinDataInput(
        gene_list=state.gene_list, instruction=state.instruction, context=state.context,
    ))
    logger.info("protein_data output=%s", out)
    return {"protein_data": out.proteins, "protein_data_status": out.status}


async def merge_node(state: FunctionalEvidenceState) -> dict:
    """Same rule as Task 1: FAILED if either child resolved nothing."""
    status = (
        AgentStatus.FAILED
        if state.pathways_status == AgentStatus.FAILED
        or state.protein_data_status == AgentStatus.FAILED
        else AgentStatus.COMPLETED
    )
    logger.info("functional evidence merge status=%s", status)
    return {"status": status}
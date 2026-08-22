import logging
from subagents.protein_data import protein_data_agent
from subagents.pathways import pathways_agent
from workflows.state import FunctionalEvidenceState
from schemas.inputs import PathwaysInput, ProteinDataInput
from schemas.common import AgentStatus

logger = logging.getLogger(__name__)

async def pathways_node(state: FunctionalEvidenceState) -> dict:
    logger.info("pathways input gene list=%s trait=%s", state.gene_list, state.trait_name)
    out = await pathways_agent(PathwaysInput(
        gene_list=state.gene_list,
        trait_name=state.trait_name,      # NEW
        instruction=state.instruction,
        context=state.context,
    ))
    logger.info("pathways output=%s", out)
    # §8: malformed_ids is Pathways-specific detail — returned under its own
    # key so it never collides with protein_data_node's missing_genes when
    # the two parallel branches fan back in at merge_node.
    return {
        "pathway_data": out.pathways,
        "pathways_status": out.status,
        "malformed_ids": out.malformed_ids,
    }

async def protein_data_node(state: FunctionalEvidenceState) -> dict:
    logger.info("protein_data input gene list=%s trait=%s", state.gene_list, state.trait_name)
    out = await protein_data_agent(ProteinDataInput(
        gene_list=state.gene_list,
        trait_name=state.trait_name,      # NEW
        instruction=state.instruction,
        context=state.context,
    ))
    logger.info("protein_data output=%s", out)
    return {
        "protein_data": out.proteins,
        "protein_data_status": out.status,
        "missing_genes": out.missing_genes,
    }

async def merge_node(state: FunctionalEvidenceState) -> dict:
    """§0.2: Pathways and Protein Data are independent, best-effort evidence
    sources. Either one failing on its own is non-critical — the other still
    has standalone value, so the sub-orchestrator only fails when BOTH
    children came back FAILED (i.e. there is no functional evidence left at
    all).

    §8: malformed_ids (from pathways_node) and missing_genes (from
    protein_data_node) don't need to be re-returned here — StateGraph.ainvoke
    returns the full accumulated state, not just this node's output, so
    whatever the two fan-out branches wrote under their own keys is already
    present on the way out. merge_node only ever decides `status`.
    """
    status = (
        AgentStatus.FAILED
        if state.pathways_status == AgentStatus.FAILED
        and state.protein_data_status == AgentStatus.FAILED
        else AgentStatus.COMPLETED
    )
    logger.info("functional evidence merge status=%s", status)
    return {"status": status}
import logging

from workflows.state import TraitDiscoveryState
from schemas.inputs import LiteratureSupportInput
from subagents.literature_support import literature_support_agent  # real agent

logger = logging.getLogger(__name__)


async def literature_support_node(state: TraitDiscoveryState) -> dict:
    logger.info("literature_support input gene list=%s", state.gene_list)
    out = await literature_support_agent(LiteratureSupportInput(
        trait_name=state.trait_name,
        gene_list=state.gene_list,
        instruction=state.instruction,
        context=state.context,
    ))
    logger.info("literature_support output=%s", out)
    return {
        "evidence": out.evidence,
        "literature_status": out.status,
        "_literature_target_agent": out.target_agent,
        "_literature_prompt": out.prompt_to_target_agent,
    }
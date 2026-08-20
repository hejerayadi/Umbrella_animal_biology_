from workflows.state import TraitDiscoveryState
from schemas.inputs import LiteratureSupportInput
from subagents.literature_support import mock_literature_support


async def literature_support_node(state: TraitDiscoveryState) -> dict:
    out = await mock_literature_support(LiteratureSupportInput(
        trait_name=state.trait_name,
        gene_list=state.gene_list,
        instruction=state.instruction,
        context=state.context,
    ))
    return {
        "evidence": out.evidence,
        "literature_status": out.status,
        "_literature_target_agent": out.target_agent,
        "_literature_prompt": out.prompt_to_target_agent,
    }
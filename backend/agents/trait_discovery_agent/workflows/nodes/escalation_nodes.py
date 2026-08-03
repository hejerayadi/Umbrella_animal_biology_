from workflows.state import TraitDiscoveryState
from schemas.common import AgentStatus


async def escalate_genome_agent_node(state: TraitDiscoveryState) -> dict:
    """Dependency is outside this agent -> hand off to the Platform Orchestrator, mocked for now."""
    return {
        "status": AgentStatus.NEEDS_AGENT,
        "target_agent": "Genome Agent",
        "prompt_to_target_agent": (
            f"Resolve candidate genes for trait '{state.trait_name}' "
            f"in species '{state.species_name}'."
        ),
    }


async def escalate_literature_agent_node(state: TraitDiscoveryState) -> dict:
    """Bubbles Literature Support's own NEEDS_AGENT upward untouched, same rule as Task 1 §7."""
    return {
        "status": AgentStatus.NEEDS_AGENT,
        "target_agent": state._literature_target_agent,
        "prompt_to_target_agent": state._literature_prompt,
    }
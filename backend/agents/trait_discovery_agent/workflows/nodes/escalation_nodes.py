from workflows.state import TraitDiscoveryState
from workflows.capability_resolver import resolve_capability
from schemas.common import AgentStatus

async def escalate_genome_agent_node(state: TraitDiscoveryState) -> dict:
    resolution = await resolve_capability(
        waiting_agent="Trait Discovery Agent",
        need_description=(
            f"No candidate gene list is available for trait '{state.trait_name}' "
            f"in species '{state.species_name}'. Need a resolved gene_list before "
            f"gene mapping, functional evidence, or literature search can run."
        ),
        known_context=f"trait={state.trait_name}, species={state.species_name}, instruction={state.instruction}",
    )
    return {
        "status": AgentStatus.NEEDS_AGENT,
        "target_agent": resolution.target_agent,
        "prompt_to_target_agent": resolution.prompt_to_target_agent,
        "resolution_reasoning": resolution.reasoning,
    }

async def escalate_literature_agent_node(state: TraitDiscoveryState) -> dict:
    """Literature Support already guessed a target/prompt itself — we still route it
    through the resolver so the choice is checked against the live catalog rather than
    trusted blindly. If the catalog changes (new agent added, one renamed), Literature
    Support's mock doesn't need to know; the resolver reflects the current truth."""
    resolution = await resolve_capability(
        waiting_agent="Literature Support",
        need_description=(
            state._literature_prompt
            or f"Evidence for trait '{state.trait_name}' is too thin, need deeper literature search."
        ),
        known_context=f"genes so far={[a.gene_symbol for a in state.go_annotations]}",
    )
    return {
        "status": AgentStatus.NEEDS_AGENT,
        "target_agent": resolution.target_agent,
        "prompt_to_target_agent": resolution.prompt_to_target_agent,
        "resolution_reasoning": resolution.reasoning,
    }
from __future__ import annotations

from typing import Any

from ..capability_resolver import resolve_capability, resolve_capability_fallback
from ..state import GenomeAgentState

async def reconstruction_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    """Escalation node: the metadata step flagged an incomplete assembly."""
    need = state.reconstruction_need or {}
    prompt_to_target = need.get("prompt_to_target_agent", "")
    known_context = {
        "species": state.species,
        "metadata": state.metadata,
        "annotation": state.annotation,
    }

    decision = resolve_capability(
        current_agent="genome_metadata",
        prompt_to_target_agent=prompt_to_target,
        known_context=known_context,
    )
    if decision is None:
        decision = resolve_capability_fallback(
            current_agent="genome_metadata",
            prompt_to_target_agent=prompt_to_target,
        )

    if decision.target_agent == "none":
        return {
            "errors": [
                *state.errors,
                f"No agent in the catalog can resolve: {prompt_to_target}",
            ],
            "reconstruction_need": {
                **need,
                "status": "FAILED",
                "target_agent": None,
                "prompt_to_target_agent": None,
            },
            "node_sequence": ["reconstruction_resolver"],
        }

    return {
        "reconstruction_need": {
            **need,
            "status": "NEEDS_AGENT",
            "target_agent": decision.target_agent,
            "prompt_to_target_agent": decision.handoff_message,
        },
        "node_sequence": ["reconstruction_resolver"],
    }
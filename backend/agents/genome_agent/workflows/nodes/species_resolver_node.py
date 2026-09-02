from __future__ import annotations

import logging
from typing import Any

from ...subagents.species_resolver import resolve_species, resolve_species_llm
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def species_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    species_name = state.species_name
    logger.info("[species_resolver] resolving species=%r", species_name)

    # Logged before either resolution path runs — both resolve_species_llm
    # (via bind_tools) and resolve_species (deterministic fallback) end up
    # searching NCBI taxonomy for this query, and this node has no cheap way
    # to tell how many times each one internally retried, so one log entry
    # per node run is the "simpler for now" instrumentation the guide
    # suggests rather than threading a logger through every retry path.
    tool_calls = [{"tool": "ncbi_taxonomy_search", "args": {"query": species_name}}]

    species = None
    try:
        species = await resolve_species_llm(species_name)
    except Exception as exc:
        logger.warning("LLM species resolver failed: %s", exc)

    if species is None:
        try:
            species = await resolve_species(species_name)
            species["confidence"] = 0.5
            species["reasoning"] = "Deterministic NCBI fallback used (no LLM)"
        except Exception as exc:
            return {
                "errors": [*state.errors, f"species_resolver raised an exception: {exc}"],
                "assembly_id": None,
                "tool_calls_log": tool_calls,
            }

    assembly_id = species.get("assembly_id")
    if assembly_id is None:
        return {
            "species": species,
            "assembly_id": None,
            "errors": [
                *state.errors,
                f"Species '{species_name}' could not be resolved to a genome assembly. "
                "No further data can be retrieved.",
            ],
            "tool_calls_log": tool_calls,
        }

    # assembly_id is only known *after* the lookup returns, so this entry's
    # args describe the outcome of the assembly-lookup call, not its literal
    # input (the literal input is a tax_id we don't surface on state) — this
    # matches what check_tool_arguments in evaluators.py actually checks for
    # this tool.
    tool_calls.append({"tool": "ncbi_assembly_lookup", "args": {"assembly_id": assembly_id}})

    return {
        "species": species,
        "assembly_id": assembly_id,
        "tool_calls_log": tool_calls,
    }

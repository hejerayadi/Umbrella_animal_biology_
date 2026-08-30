from __future__ import annotations

import logging
from typing import Any

from ...subagents.species_resolver import resolve_species, resolve_species_llm
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def species_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    species_name = state.species_name
    logger.info("[species_resolver] resolving species=%r", species_name)

    # Always log that we attempted NCBI taxonomy search — regardless of
    # which path (LLM or fallback) runs, both call NCBI taxonomy under the hood.
    base_tool_log = [{"tool": "ncbi_taxonomy_search", "args": {"query": species_name}}]

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
                "node_sequence": ["species_resolver"],
                "tool_calls_log": base_tool_log,
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
            "node_sequence": ["species_resolver"],
            "tool_calls_log": base_tool_log,
        }

    return {
        "species": species,
        "assembly_id": assembly_id,
        "node_sequence": ["species_resolver"],
        "tool_calls_log": [
            {"tool": "ncbi_taxonomy_search", "args": {"query": species_name}},
            {"tool": "ncbi_assembly_lookup",  "args": {"assembly_id": assembly_id}},
        ],
    }

from __future__ import annotations

import logging
from typing import Any

from ...subagents.species_resolver import resolve_species, resolve_species_llm
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def species_resolver_node(state: GenomeAgentState) -> dict[str, Any]:
    species_name = state.species_name
    logger.info("[species_resolver] resolving species=%r", species_name)

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
        }

    return {
        "species": species,
        "assembly_id": assembly_id,
    }

"""
Genome Metadata node — LangGraph node for genome metadata resolution.

Tries the LLM tool-calling path first (resolve_metadata_llm), falls back
to a deterministic single-fetch (fetch_metadata_fallback) on failure or
when no LLM is available. The fallback is NON-FATAL per the mission spec:
record the gap in reasoning, don't stop the whole request.

Follows the exact same shape as species_resolver_node.py.
"""

from __future__ import annotations

import logging
from typing import Any

from ...subagents.genome_metadata import fetch_metadata_fallback, resolve_metadata_llm
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def genome_metadata_node(state: GenomeAgentState) -> dict[str, Any]:
    assembly_id = state.assembly_id
    if assembly_id is None:
        species = state.species
        if isinstance(species, dict):
            assembly_id = species.get("assembly_id")

    if assembly_id is None:
        return {
            "errors": [
                *state.errors,
                "No assembly_id available for genome metadata lookup.",
            ],
            "metadata": None,
        }

    metadata = None
    try:
        metadata = await resolve_metadata_llm(state.species_name, assembly_id)
    except Exception as exc:
        logger.warning("LLM genome metadata failed: %s", exc)

    if metadata is None:
        try:
            metadata = await fetch_metadata_fallback(assembly_id)
        except Exception as exc:
            return {
                "errors": [*state.errors, f"genome_metadata raised an exception: {exc}"],
                "metadata": None,
            }

    return {"metadata": metadata}

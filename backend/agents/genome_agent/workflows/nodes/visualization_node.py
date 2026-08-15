from __future__ import annotations

import logging
from typing import Any

from ...subagents.visualization import generate_visualization
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def generate_visualization_node(state: GenomeAgentState) -> dict[str, Any]:
    if state.visualization is not None:
        return {"errors": []}

    scope = state.visualization_scope
    if scope == "none":
        return {"errors": []}

    genome_size = state.metadata["genome_size_bp"] if state.metadata else None
    gene_table = state.annotation["gene_table"] if state.annotation else None
    species = state.species or {}
    user_question = state.user_question
    logger.info("[generate_visualization] scope=%r, user_question=%r", scope, user_question)

    try:
        result = await generate_visualization(
            scope=scope,
            genome_size_bp=genome_size,
            gene_table=gene_table,
            assembly_id=state.assembly_id,
            common_name=species.get("common_name"),
            scientific_name=species.get("scientific_name"),
            user_question=user_question,
        )
    except Exception as exc:
        return {
            "errors": [
                *state.errors,
                f"generate_visualization raised an exception: {exc}",
            ],
            "visualization": None,
        }

    if result.get("status") == "FAILED":
        return {
            "visualization": result,
            "errors": [
                *state.errors,
                f"Visualization failed with status FAILED for scope '{scope}'.",
            ],
        }

    return {"visualization": result}
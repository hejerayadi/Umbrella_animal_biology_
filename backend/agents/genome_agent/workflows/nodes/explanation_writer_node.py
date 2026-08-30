from __future__ import annotations

import logging
from typing import Any

from ..explanation_writer import write_explanation
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def explanation_writer_node(state: GenomeAgentState) -> dict[str, Any]:
    explanation = await write_explanation(
        user_question=state.user_question,
        species=state.species,
        metadata=state.metadata,
        annotation=state.annotation,
        visualization=state.visualization,
        needs_metadata=state.needs_metadata,
        needs_annotation=state.needs_annotation,
        visualization_scope=state.visualization_scope,
    )
    return {"explanation": explanation, "node_sequence": ["explanation_writer"]}


def error_end_node(state: GenomeAgentState) -> dict[str, Any]:
    logger.info("[error_end] species resolver failed, stopping")
    return {"errors": [], "node_sequence": ["error_end"]}
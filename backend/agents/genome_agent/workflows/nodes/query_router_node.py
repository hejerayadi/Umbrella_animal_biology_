from __future__ import annotations

import logging
from typing import Any

from ..query_router import route_query, route_query_fallback
from ..state import GenomeAgentState

logger = logging.getLogger(__name__)


async def query_router_node(state: GenomeAgentState) -> dict[str, Any]:
    user_question = state.user_question
    logger.info("[query_router] routing user_question=%r", user_question)

    decision = route_query(user_question)
    if decision is None:
        decision = route_query_fallback(user_question)

    current_scope = state.visualization_scope
    new_scope = current_scope if current_scope != "" else decision.visualization_scope

    return {
        "needs_metadata": decision.needs_metadata,
        "needs_annotation": decision.needs_annotation,
        "visualization_scope": new_scope,
        "node_sequence": ["query_router"],
    }
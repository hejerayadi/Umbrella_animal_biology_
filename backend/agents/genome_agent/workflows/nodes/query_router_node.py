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

    # Only fall back to the router's own guess when no scope was requested
    # at all. Any explicit value the caller passed in — including
    # "chromosome_map" — must be respected exactly as given.
    current_scope = state.visualization_scope
    new_scope = current_scope if current_scope != "" else decision.visualization_scope

    # A chart the graph cannot draw is worse than no chart: the run reports
    # COMPLETED, the note says "No gene data available to render a chromosome
    # map", and the user is told their species has no data when what actually
    # happened is that nobody fetched it.
    #
    # Nothing else enforces this. `generate_visualization` reads `gene_table`
    # from `state.annotation`, `get_gene_annotation_node` returns early unless
    # `needs_annotation` is set, and both routers can pick a scope without
    # setting the flag that scope depends on:
    #
    #   * the keyword fallback matches "chromosome" for the scope but looks for
    #     "gene"/"annotation"/"feature"/"protein" for the flag, so "show me the
    #     chromosome map of the tiger" sets one and not the other - and its
    #     final `elif needs_metadata` branch defaults *any* metadata question to
    #     chromosome_map, with the same result;
    #   * the LLM router is never told the two are connected, so it is free to
    #     return the same pair.
    #
    # The dependency belongs here rather than in either router: this is the one
    # place where a scope - LLM, fallback, or passed in by the caller - is
    # finalised, so fixing it once covers all three.
    needs_metadata = decision.needs_metadata
    needs_annotation = decision.needs_annotation

    if new_scope == "chromosome_map" and not needs_annotation:
        logger.info(
            "[query_router] scope=chromosome_map needs the gene table; "
            "overriding needs_annotation=False"
        )
        needs_annotation = True

    # Same invariant, one chart along: the size comparison plots the queried
    # species' own `genome_size_bp`, which only `get_genome_metadata_node`
    # writes. Without it the species is missing from its own comparison.
    if new_scope == "size_comparison" and not needs_metadata:
        logger.info(
            "[query_router] scope=size_comparison needs genome size; "
            "overriding needs_metadata=False"
        )
        needs_metadata = True

    return {
        "needs_metadata": needs_metadata,
        "needs_annotation": needs_annotation,
        "visualization_scope": new_scope,
    }
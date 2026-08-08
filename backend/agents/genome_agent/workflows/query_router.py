from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .agent_catalog import load_agent_catalog
from .llm import get_llm_client

logger = logging.getLogger(__name__)


class QueryRouterDecision(BaseModel):
    needs_species_resolution: bool = Field(
        description="Whether the query needs species resolution. Usually True unless the species is already known from context."
    )
    needs_metadata: bool = Field(
        description="Whether genome metadata (size, chromosomes) is needed for this query."
    )
    needs_annotation: bool = Field(
        description="Whether gene annotation is needed for this query."
    )
    visualization_scope: str = Field(
        description="One of: chromosome_map, size_comparison, protein_structure, or none if no visualization is needed."
    )
    reasoning: str = Field(
        description="One short sentence explaining why each subagent was selected or skipped."
    )


_ROUTER_SYSTEM_PROMPT = (
    "You are the query router for the Genome Agent. "
    "Given a user's question about a species' genome, decide which subagents "
    "are needed and what kind of visualization (if any) to produce.\n\n"
    "Available agents:\n{agent_catalog}\n\n"
    "Rules:\n"
    "- species_resolution is almost always needed unless the species is already known.\n"
    "- metadata is needed for questions about genome size, chromosome count, karyotype.\n"
    "- annotation is needed for questions about genes, features, functional elements.\n"
    "- visualization_scope should be one of: chromosome_map, size_comparison, protein_structure, none.\n"
    "- If the query is about protein structure, set visualization_scope to protein_structure.\n"
)


def route_query(user_question: str) -> QueryRouterDecision | None:
    """Use the LLM to decide which subagents are needed and what chart to show."""
    catalog = load_agent_catalog()
    prompt = _ROUTER_SYSTEM_PROMPT.format(agent_catalog=catalog)

    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools([QueryRouterDecision], tool_choice=QueryRouterDecision.__name__)

    try:
        response = bound.invoke([SystemMessage(content=prompt), HumanMessage(content=user_question)])
        tool_calls = response.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            return QueryRouterDecision(**call["args"])
    except Exception as exc:
        logger.warning("LLM query router failed: %s", exc)

    return None


def route_query_fallback(user_question: str) -> QueryRouterDecision:
    """Keyword-based fallback when the LLM tool call fails."""
    question_lower = user_question.lower()

    needs_metadata = any(
        k in question_lower
        for k in ["genome", "genome size", "chromosome", "karyotype", "size", "dna"]
    )
    needs_annotation = any(
        k in question_lower
        for k in ["gene", "annotation", "annotate", "feature", "protein"]
    )
    vis_scope = "none"
    if "protein structure" in question_lower or "3d" in question_lower:
        vis_scope = "protein_structure"
    elif "chromosome" in question_lower:
        vis_scope = "chromosome_map"
    elif "size comparison" in question_lower or "compare" in question_lower:
        vis_scope = "size_comparison"
    elif needs_metadata:
        vis_scope = "chromosome_map"

    return QueryRouterDecision(
        needs_species_resolution=True,
        needs_metadata=needs_metadata,
        needs_annotation=needs_annotation,
        visualization_scope=vis_scope,
        reasoning="Fallback keyword match",
    )

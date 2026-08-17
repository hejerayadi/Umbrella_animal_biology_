"""
Gene Annotation Resolver — LLM wrapper for intelligent gene search.

Given a user's question and an assembly ID, the LLM decides:
1. Should we search broadly or target a specific trait/keyword?
2. What keyword to use (if any)?
3. How to rank/filter the results?

This keeps the LLM's reasoning separate from the deterministic NCBI calls,
and allows fallback to a simple broad search when the LLM is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


class GeneAnnotationStrategy(BaseModel):
    """LLM's decision on how to search for genes."""
    search_keyword: str | None = Field(
        default=None,
        description="Optional NCBI search keyword (e.g., 'color', 'behavior', 'metabolism'). "
        "If None, perform a broad search without filtering by trait."
    )
    ranking_criteria: str | None = Field(
        default=None,
        description="Brief explanation of how to rank results (e.g., 'prefer genes with detailed descriptions', "
        "'prioritize genes related to skin/coat characteristics')."
    )
    reasoning: str = Field(
        description="One sentence explaining the search strategy."
    )


_RESOLVER_SYSTEM_PROMPT = (
    "You are the gene annotation resolver for the Genome Agent. "
    "Given a user's question about genes in a species, decide whether to search broadly "
    "or target a specific trait/keyword.\n\n"
    "Rules:\n"
    "- If the question asks about a general topic ('show me genes for X'), return search_keyword=None for broad search.\n"
    "- If the question is about a specific trait ('genes related to Y'), extract a keyword like 'Y', 'color', 'behavior', etc.\n"
    "- search_keyword should be a single, simple word or short phrase that NCBI will understand.\n"
    "- ranking_criteria should guide how to order results (e.g., 'prefer well-annotated genes').\n"
    "- Be conservative: prefer broader searches when unsure.\n"
)


def resolve_gene_annotation_strategy(
    user_question: str,
    assembly_id: str,
) -> GeneAnnotationStrategy | None:
    """Use the LLM to decide how to search for genes.
    
    Returns:
        GeneAnnotationStrategy if successful, None if LLM unavailable.
    """
    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [GeneAnnotationStrategy],
        tool_choice=GeneAnnotationStrategy.__name__,
    )

    try:
        response = invoke_with_retry(
            lambda: bound.invoke([
                SystemMessage(content=_RESOLVER_SYSTEM_PROMPT),
                HumanMessage(content=f"Question: {user_question}\nAssembly: {assembly_id}"),
            ])
        )
        tool_calls = response.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            return GeneAnnotationStrategy(**call["args"])
    except Exception as exc:
        logger.info(
            "LLM gene annotation resolver unavailable (%s) — using fallback",
            summarize_llm_error(exc),
        )

    return None


def resolve_gene_annotation_strategy_fallback(
    user_question: str,
) -> GeneAnnotationStrategy:
    """Fallback when LLM is unavailable: always do a broad search."""
    question_lower = user_question.lower()

    # Check for trait-specific keywords
    trait_keywords = {
        "color": ["color", "colour", "pigment", "coat", "fur", "feather", "eye"],
        "behavior": ["behavior", "behaviour", "aggression", "social", "movement"],
        "metabolism": ["metabolism", "energy", "fat", "glucose", "sugar"],
        "size": ["size", "growth", "height", "weight", "large", "small"],
        "reproduction": ["reproduction", "fertility", "mating", "sex", "reproduction"],
        "immune": ["immune", "immunity", "infection", "disease", "antibody"],
        "development": ["development", "embryo", "growth", "fetal", "postnatal"],
    }

    extracted_keyword = None
    for keyword, synonyms in trait_keywords.items():
        if any(syn in question_lower for syn in synonyms):
            extracted_keyword = keyword
            break

    return GeneAnnotationStrategy(
        search_keyword=extracted_keyword,
        ranking_criteria="No specific ranking preference" if extracted_keyword is None else f"Prioritize genes related to {extracted_keyword}",
        reasoning="Keyword fallback",
    )
"""
Visualization Resolver — LLM wrapper for intelligent reference species selection.

For size_comparison visualizations, the LLM selects 2-4 sensible reference species
to compare against the queried species. This keeps the LLM reasoning separate from
the deterministic NCBI calls and rendering.

Falls back to a fixed well-known set if LLM unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


class VisualizationResolverOutput(BaseModel):
    """LLM's decision on reference species and visualization parameters."""
    reference_species: list[str] = Field(
        description="List of 2-4 sensible reference species names to compare against the queried species."
    )
    reasoning: str = Field(
        description="Brief explanation of why these reference species are good comparisons."
    )


_RESOLVER_SYSTEM_PROMPT = (
    "You are the visualization resolver for the Genome Agent. "
    "Given a species that the user asked about, suggest 2-4 good reference species "
    "to compare it against in a size_comparison visualization.\n\n"
    "Rules:\n"
    "- Choose species that are likely to have genome assemblies in NCBI (common model organisms or well-studied animals).\n"
    "- Prefer species that are phylogenetically related or biologically interesting comparisons.\n"
    "- Use common names (e.g., 'human', 'house mouse', 'dog') not scientific names.\n"
    "- Return exactly 2-4 species, not more, not fewer.\n"
    "- Do not include the queried species itself in the list.\n"
)


def resolve_visualization_references(
    queried_species: str,
    current_common_name: str | None = None,
) -> VisualizationResolverOutput | None:
    """Use the LLM to choose reference species for visualization.
    
    Args:
        queried_species: The user's original species query
        current_common_name: The common name of the resolved species (if known)
    
    Returns:
        VisualizationResolverOutput if successful, None if LLM unavailable.
    """
    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [VisualizationResolverOutput],
        tool_choice=VisualizationResolverOutput.__name__,
    )

    species_context = queried_species
    if current_common_name and current_common_name.lower() != queried_species.lower():
        species_context = f"{queried_species} (resolved to {current_common_name})"

    try:
        response = invoke_with_retry(
            lambda: bound.invoke([
                SystemMessage(content=_RESOLVER_SYSTEM_PROMPT),
                HumanMessage(content=f"Queried species: {species_context}"),
            ])
        )
        tool_calls = response.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            return VisualizationResolverOutput(**call["args"])
    except Exception as exc:
        logger.info(
            "LLM visualization resolver unavailable (%s) — using fallback",
            summarize_llm_error(exc),
        )

    return None


def resolve_visualization_references_fallback(
    queried_species: str = "",
) -> VisualizationResolverOutput:
    """Fallback when LLM unavailable: use a fixed set of well-known species.
    
    This is deterministic and works without network access.
    """
    # Fixed reference species - deliberately chosen, easy to resolve, informative comparisons
    _DEFAULT_REFERENCES = ["human", "house mouse", "chicken", "zebrafish"]
    
    return VisualizationResolverOutput(
        reference_species=_DEFAULT_REFERENCES,
        reasoning="Fallback: using well-known model organisms",
    )


class ChromosomeHighlightOutput(BaseModel):
    """LLM's decision on which gene (if any) to highlight on a chromosome map."""
    highlight_gene: str | None = Field(
        default=None,
        description="The exact gene_name from the candidate list that the user's question "
        "is asking about, or None if the question doesn't clearly name/imply one.",
    )
    reasoning: str = Field(
        description="One sentence explaining why this gene was (or wasn't) picked."
    )


_HIGHLIGHT_SYSTEM_PROMPT = (
    "You are the chromosome-map highlight resolver for the Genome Agent. "
    "Given a user's question and a list of candidate gene names already "
    "present in the gene table, decide which single gene (if any) the user "
    "wants highlighted on the chromosome map.\n\n"
    "Rules:\n"
    "- highlight_gene MUST be copied exactly from the candidate list, or be None.\n"
    "- Never invent a gene name that isn't in the candidate list.\n"
    "- If the question doesn't clearly name or imply one specific gene from the "
    "list, return highlight_gene=None rather than guessing.\n"
)


def resolve_chromosome_highlight(
    user_question: str,
    candidate_gene_names: list[str],
) -> ChromosomeHighlightOutput | None:
    """Use the LLM to pick which gene (if any) to highlight on a chromosome map.

    Returns:
        ChromosomeHighlightOutput if successful, None if LLM unavailable.
    """
    if not candidate_gene_names:
        return ChromosomeHighlightOutput(highlight_gene=None, reasoning="No genes to highlight.")

    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools(
        [ChromosomeHighlightOutput],
        tool_choice=ChromosomeHighlightOutput.__name__,
    )

    try:
        response = invoke_with_retry(
            lambda: bound.invoke([
                SystemMessage(content=_HIGHLIGHT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {user_question}\n"
                        f"Candidate genes: {', '.join(candidate_gene_names)}"
                    )
                ),
            ])
        )
        tool_calls = response.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            result = ChromosomeHighlightOutput(**call["args"])
            # Never trust an invented gene name, even from the LLM — but
            # match case-insensitively first (the LLM may return different
            # casing, e.g. "TRP53" vs "Trp53") and snap to the candidate
            # list's canonical casing, the same way the fallback heuristic
            # and the renderer already do. Only drop it if there's truly no
            # match by name, ignoring case.
            if result.highlight_gene:
                canonical_by_lower = {name.lower(): name for name in candidate_gene_names}
                canonical = canonical_by_lower.get(result.highlight_gene.lower())
                if canonical is None:
                    logger.warning(
                        "LLM proposed highlight_gene %r not in candidate list (even "
                        "case-insensitively) — dropping",
                        result.highlight_gene,
                    )
                    result.highlight_gene = None
                else:
                    result.highlight_gene = canonical
            return result
    except Exception as exc:
        logger.info(
            "LLM chromosome highlight resolver unavailable (%s) — using fallback",
            summarize_llm_error(exc),
        )

    return None


def resolve_chromosome_highlight_fallback(
    user_question: str,
    candidate_gene_names: list[str],
) -> ChromosomeHighlightOutput:
    """Fallback when LLM unavailable: simple heuristic string match.

    Looks for a capitalized, reasonably short token in the question that
    matches one of the candidate gene names exactly (case-insensitive).
    Never invents a gene name outside the candidate list.
    """
    highlight_gene = None
    if user_question:
        gene_lookup = {name.lower(): name for name in candidate_gene_names}
        for word in user_question.split():
            cleaned = word.strip(".,!?;:").rstrip("'s")
            if len(cleaned) <= 10 and cleaned[:1].isupper():
                match = gene_lookup.get(cleaned.lower())
                if match:
                    highlight_gene = match
                    break

    return ChromosomeHighlightOutput(
        highlight_gene=highlight_gene,
        reasoning="Fallback: heuristic keyword match against candidate genes.",
    )
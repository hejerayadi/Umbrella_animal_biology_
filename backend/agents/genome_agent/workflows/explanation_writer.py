from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm_client, invoke_with_retry, summarize_llm_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt engineering: the LLM is handed the raw structured facts and is
# responsible for the actual writing. No English sentences are pre-built in
# Python — that used to happen here and made the "explanation" mostly a
# hardcoded template with the LLM only polishing it. Mirrors the
# trait_discovery_agent explanation_writer: a ChatPromptTemplate + LCEL
# chain (`prompt | llm`) that turns structured data into prose.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are the explanation writer for the Genome Agent, a genomics research "
    "assistant summarizing findings for a scientist.\n"
    "Reason privately about how the species, genome metadata, gene annotation, "
    "and visualization results relate to the user's question, then write ONLY "
    "the final summary — a few short plain-language sentences or a short list.\n\n"
    "Rules:\n"
    "- Only use the facts given below. Never invent data.\n"
    "- If a section says data is unavailable or missing, say so plainly instead "
    "of skipping it.\n"
    "- Do not mention internal agent machinery (node names, statuses like "
    "NEEDS_AGENT) unless it genuinely helps the user understand what happened.\n"
    "- No headers, no markdown, no restating these instructions."
)

USER_PROMPT = (
    "User's question: {user_question}\n\n"
    "Species: {species}\n"
    "Genome metadata: {metadata}\n"
    "Gene annotation: {annotation}\n"
    "Visualization: {visualization}\n"
)


def _describe_species(species: dict[str, Any] | None) -> str:
    if not species:
        return "not resolved"
    if species.get("assembly_id") is None:
        return "could not be resolved to a genome assembly"
    return (
        f"{species.get('common_name')} ({species.get('scientific_name')}), "
        f"assembly {species.get('assembly_id')}"
    )


def _describe_metadata(metadata: dict[str, Any] | None, needs_metadata: bool) -> str:
    if not needs_metadata:
        return "not requested for this query"
    if not metadata:
        return "requested but unavailable (lookup failed or returned empty)"
    return (
        f"genome size {metadata.get('genome_size_bp')} bp, "
        f"{metadata.get('chromosome_count')} chromosomes, "
        f"karyotype {metadata.get('karyotype')}"
    )


def _describe_annotation(annotation: dict[str, Any] | None, needs_annotation: bool) -> str:
    if not needs_annotation:
        return "not requested for this query"
    if not annotation:
        return "requested but unavailable (lookup failed)"
    gene_list = annotation.get("gene_list", [])
    return f"genes: {', '.join(gene_list)}" if gene_list else "no genes annotated"


def _describe_visualization(
    visualization: dict[str, Any] | None, visualization_scope: str
) -> str:
    if visualization_scope in ("", "none"):
        return "not requested for this query"
    if not visualization:
        return "requested but unavailable (generation failed)"

    comparisons = visualization.get("comparisons")
    if comparisons:
        comp_text = "; ".join(
            f"{c['common_name']} ({c['scientific_name']}): "
            f"{c['genome_size_bp'] / 1_000_000_000:.2f} Gb"
            + (" [queried species]" if c.get("is_queried_species") else "")
            for c in comparisons
        )
        note = visualization.get("note")
        return f"{note + '. ' if note else ''}cross-species comparison — {comp_text}"

    status = visualization.get("status", "unknown")
    if status == "needs_agent":
        return f"requires an external agent ({visualization.get('target_agent')})"
    return f"status: {status}"


def _facts_fallback(
    species: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    annotation: dict[str, Any] | None,
    visualization: dict[str, Any] | None,
    needs_metadata: bool,
    needs_annotation: bool,
    visualization_scope: str,
) -> str:
    """Deterministic, non-LLM summary used only when the model is unreachable."""
    lines = [
        f"Species: {_describe_species(species)}",
        f"Genome metadata: {_describe_metadata(metadata, needs_metadata)}",
        f"Gene annotation: {_describe_annotation(annotation, needs_annotation)}",
        f"Visualization: {_describe_visualization(visualization, visualization_scope)}",
    ]
    return "\n".join(lines)


async def write_explanation(
    user_question: str,
    species: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    annotation: dict[str, Any] | None,
    visualization: dict[str, Any] | None,
    needs_metadata: bool = True,
    needs_annotation: bool = True,
    visualization_scope: str = "",
) -> str:
    """Write a plain-language summary of the gathered genomic data.

    The facts are handed to the LLM as structured data (species/metadata/
    annotation/visualization) via a ChatPromptTemplate; the LLM does the
    actual writing. Falls back to a compact, deterministic facts listing
    only when the LLM is unreachable, so the workflow keeps working without
    an API key.

    needs_metadata / needs_annotation / visualization_scope disambiguate
    "never requested" from "requested but came back empty/failed" — without
    them, both collapse to the same wording and the writer (LLM or
    fallback) can't tell a skipped step from a broken one.
    """
    payload = {
        "user_question": user_question,
        "species": _describe_species(species),
        "metadata": _describe_metadata(metadata, needs_metadata),
        "annotation": _describe_annotation(annotation, needs_annotation),
        "visualization": _describe_visualization(visualization, visualization_scope),
    }

    def _fallback() -> str:
        return _facts_fallback(
            species, metadata, annotation, visualization,
            needs_metadata, needs_annotation, visualization_scope,
        )

    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return _fallback()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    chain = prompt | client

    try:
        # invoke_with_retry owns pacing/backoff against the shared NVIDIA
        # endpoint (see llm.py) and is synchronous by design, so it runs in
        # a worker thread to keep this function itself async — matching the
        # rest of the async node graph — without bypassing that gate.
        response = await asyncio.to_thread(
            invoke_with_retry, lambda: chain.invoke(payload)
        )
        return response.content.strip()
    except Exception as exc:
        logger.info(
            "LLM explanation writer unavailable (%s) — using deterministic fallback",
            summarize_llm_error(exc),
        )
        return _fallback()
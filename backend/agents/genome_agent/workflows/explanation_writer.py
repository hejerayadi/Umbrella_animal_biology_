from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from .llm import get_llm_client

logger = logging.getLogger(__name__)


_EXPLANATION_SYSTEM_PROMPT = (
    "You are the explanation writer for the Genome Agent. "
    "Given the data that was successfully gathered for a user's question, "
    "write a clear, plain-language summary of the findings.\n\n"
    "Rules:\n"
    "- Only use the data provided. Do not invent information.\n"
    "- If some data is missing, say so plainly.\n"
    "- Be concise: a few short paragraphs or a short list.\n"
    "- Do not mention internal agent machinery unless it genuinely helps the user.\n"
)


def write_explanation(
    user_question: str,
    species: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    annotation: dict[str, Any] | None,
    visualization: dict[str, Any] | None,
) -> str:
    """Write a plain-language summary of the gathered genomic data."""
    findings: list[str] = []

    if species:
        findings.append(f"Species resolved: {species.get('common_name')} ({species.get('scientific_name')})")
        findings.append(f"Assembly ID: {species.get('assembly_id')}")
    else:
        findings.append("Species could not be resolved.")

    if metadata:
        findings.append(
            f"Genome size: {metadata.get('genome_size_bp')} bp"
        )
        findings.append(
            f"Chromosome count: {metadata.get('chromosome_count')}"
        )
        findings.append(f"Karyotype: {metadata.get('karyotype')}")
    else:
        findings.append("Genome metadata is unavailable.")

    if annotation:
        gene_list = annotation.get("gene_list", [])
        if gene_list:
            findings.append(f"Genes found: {', '.join(gene_list)}")
        else:
            findings.append("No genes were annotated for this assembly.")
    else:
        findings.append("Gene annotation is unavailable.")

    if visualization:
        status = visualization.get("status", "unknown")
        if status == "completed":
            findings.append("Visualization was generated successfully.")
        elif status == "needs_agent":
            findings.append(
                "Visualization requires an external agent: "
                f"{visualization.get('target_agent')}"
            )
        else:
            findings.append(f"Visualization status: {status}")
    else:
        findings.append("Visualization is unavailable.")

    findings_text = "\n".join(findings)

    prompt = (
        f"User's question: {user_question}\n\n"
        f"Findings:\n{findings_text}\n\n"
        "Write a concise plain-language summary of these findings for the user."
    )

    try:
        client = get_llm_client()
        response = client.invoke(
            [SystemMessage(content=_EXPLANATION_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM explanation writer failed: %s", exc)
        return findings_text

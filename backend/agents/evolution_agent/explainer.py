"""Explainer (LLM #2) for the Evolution Agent.

Generates a human-readable explanation from a structured EvolutionAnalysisResult.
This is the ONLY second LLM call — maximum 2 LLM calls per request (Planner + Explainer).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .schema import EvolutionAnalysisResult, PlannedFeature

_logger = logging.getLogger(__name__)

EXPLAINER_SYSTEM_PROMPT = """\
You are the Evolution Agent explainer. Given structured analysis results,
write a clear, concise explanation in 2-4 sentences.

Rules:
- Use plain scientific language (no jargon without explanation).
- Mention the model used for phylogenetic reconstruction if available.
- Note the closest species pair and their similarity score.
- Mention bootstrap support if available.
- Do NOT fabricate information not present in the input.
- Output ONLY the explanation text. No JSON, no markdown, no code fences.
"""


async def explain(
    analysis: EvolutionAnalysisResult,
    feature: str,
    llm=None,
) -> str:
    """Generate a human-readable explanation. Never raises — returns a fallback on error.

    Parameters
    ----------
    analysis : EvolutionAnalysisResult
        The structured result to explain.
    feature : str
        The PlannedFeature value (e.g. "molecular_comparison").
    llm : optional
        Injectable LLM for tests.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        try:
            from .framework.llm_client import get_llm
            llm = get_llm()
        except Exception as exc:
            _logger.warning("[Explainer] no LLM backend configured: %s", exc)
            return _fallback_explanation(analysis, feature)

    # Build a compact input for the LLM
    input_data: dict = {
        "feature": feature,
        "species_list": analysis.species_list,
        "overall_confidence": analysis.overall_confidence,
    }
    if analysis.molecular and feature in (PlannedFeature.MOLECULAR_COMPARISON.value, PlannedFeature.FULL_ANALYSIS.value):
        mc = analysis.molecular
        input_data["similarity_scores"] = [
            {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
            for e in mc.similarity_scores[:5]  # top 5
        ]
        input_data["species_groups"] = [
            {"group_id": g.group_id, "species": g.species, "mean_score": g.mean_score}
            for g in mc.species_groups
        ]
    if analysis.phylogenetic and feature in (PlannedFeature.PHYLOGENETIC_TREE.value, PlannedFeature.FULL_ANALYSIS.value):
        phylo = analysis.phylogenetic
        input_data["model"] = phylo.model
        input_data["bootstrap_support"] = phylo.bootstrap_support
        input_data["confidence_values"] = phylo.confidence_values

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(input_data, indent=2)),
            ]
        )
        text = getattr(response, "content", str(response)).strip()
        if text:
            return text
    except Exception as exc:
        _logger.warning("[Explainer] LLM call failed (%s)", type(exc).__name__)

    return _fallback_explanation(analysis, feature)


def _fallback_explanation(analysis: EvolutionAnalysisResult, feature: str) -> str:
    """Deterministic fallback when LLM is unavailable."""
    species = analysis.species_list
    n = len(species)
    conf = f"{analysis.overall_confidence * 100:.0f}%"

    if feature == PlannedFeature.PHYLOGENETIC_TREE.value and analysis.phylogenetic:
        return (
            f"Phylogenetic tree built for {n} species using "
            f"{analysis.phylogenetic.model} model ({conf} confidence)."
        )
    if feature == PlannedFeature.MOLECULAR_COMPARISON.value and analysis.molecular:
        mc = analysis.molecular
        if mc.similarity_scores:
            top = max(mc.similarity_scores, key=lambda e: e.score)
            return (
                f"Molecular comparison of {n} species: "
                f"{top.species_a} and {top.species_b} are most similar "
                f"(score {top.score:.2f}), {conf} confidence."
            )
        return f"Molecular comparison of {n} species ({conf} confidence)."

    # full_analysis
    parts = [f"Analysed {n} species ({conf} confidence)."]
    if analysis.molecular and analysis.molecular.similarity_scores:
        top = max(analysis.molecular.similarity_scores, key=lambda e: e.score)
        parts.append(
            f"Closest pair: {top.species_a} and {top.species_b} "
            f"(similarity {top.score:.2f})."
        )
    if analysis.phylogenetic:
        parts.append(f"Tree model: {analysis.phylogenetic.model}.")
    return " ".join(parts)

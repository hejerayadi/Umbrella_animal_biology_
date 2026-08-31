"""Planner (LLM #1) for the Evolution Agent.

Decides what the user is asking for and extracts parameters.
Maximum 2 LLM calls per request (Planner + Explainer).

Two behaviours are guaranteed:
- Never raises.  LLM unavailable / network error / unparsable output →
  PlannerDecision with feature=CLARIFICATION_REQUIRED.
- Never invents a feature outside the enum.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from langsmith import traceable

from .schema import PlannerDecision, PlannedFeature

_logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are the planner of the Evolution Agent (Umbrella BioHub).
Decide what the user is asking for and extract the parameters.

Respond with strict JSON — no prose, no code fences, no extra keys:

{
  "feature": "molecular_comparison" | "phylogenetic_tree"
             | "full_analysis" | "clarification_required",
  "species_list": ["<scientific name>", ...],
  "reference_species": "<outgroup>" | null,
  "explicitly_requested_both": true | false,
  "clarification_question": "<one question>" | null
}

Rules:
- molecular_comparison : sequence similarity, embeddings, similarity network,
                         species grouping/clustering. NOT a tree.
- phylogenetic_tree    : a tree, a phylogeny, branch support, topology.
                         NOT pairwise similarity scores.
- full_analysis        : ONLY when the user explicitly asks for BOTH a
                         similarity network AND a phylogenetic tree in the
                         same request. Set "explicitly_requested_both": true.
- clarification_required : the request is vague, names no species, is not
                         about evolutionary analysis, or you are unsure which
                         of the two results is wanted. Fill
                         "clarification_question" with ONE short question.
                         NEVER guess. NEVER default to full_analysis.
- species_list : every species mentioned, scientific names where possible.
- Output ONLY the JSON.
"""


# ---------------------------------------------------------------------------
# JSON parsing helpers (kept from intent.py — robust and tested)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Best-effort dict from model output that may be wrapped in fences."""
    stripped = text.strip()
    candidates: list[str] = [stripped] if stripped else []

    candidates.extend(
        b.strip()
        for b in re.findall(
            r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE
        )
        if b.strip()
    )
    candidates.extend(
        s.strip()
        for s in re.findall(r"\{[\s\S]*?\}", stripped)
        if s.strip()
    )

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown", ""}:
        return None
    return text


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s for s in (_clean_str(item) for item in value) if s]


# ---------------------------------------------------------------------------
# Deterministic guards (post-LLM, no network)
# ---------------------------------------------------------------------------

_VALID_FEATURES = {e.value for e in PlannedFeature}


def _apply_guards(payload: dict) -> PlannerDecision:
    """Apply deterministic rules to LLM output. Never raises."""
    feature_str = _clean_str(payload.get("feature"))
    species_list = _clean_list(payload.get("species_list"))
    reference_species = _clean_str(payload.get("reference_species"))
    explicitly_both = bool(payload.get("explicitly_requested_both"))
    clarification_q = _clean_str(payload.get("clarification_question"))

    # Guard: unknown feature
    if feature_str is None or feature_str not in _VALID_FEATURES:
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            species_list=species_list,
            reference_species=reference_species,
            clarification_question=clarification_q or "Would you like a similarity network or a phylogenetic tree?",
            source="llm",
        )

    feature = PlannedFeature(feature_str)

    # Guard: full_analysis without explicit request for both
    if feature == PlannedFeature.FULL_ANALYSIS and not explicitly_both:
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            species_list=species_list,
            reference_species=reference_species,
            clarification_question="Would you like a similarity network, a phylogenetic tree, or both?",
            source="llm",
        )

    # Guard: no species
    if not species_list:
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            species_list=[],
            reference_species=reference_species,
            clarification_question="Which species would you like to analyse?",
            source="llm",
        )

    # Guard: molecular_comparison with < 2 species
    if feature == PlannedFeature.MOLECULAR_COMPARISON and len(species_list) < 2:
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            species_list=species_list,
            reference_species=reference_species,
            clarification_question="Molecular comparison requires at least 2 species. Which others would you like to include?",
            source="llm",
        )

    # Guard: phylogenetic_tree with < 3 species
    if feature == PlannedFeature.PHYLOGENETIC_TREE and len(species_list) < 3:
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            species_list=species_list,
            reference_species=reference_species,
            clarification_question="Phylogenetic tree requires at least 3 species. Would you like to add more, or switch to a similarity comparison?",
            source="llm",
        )

    return PlannerDecision(
        feature=feature,
        species_list=species_list,
        reference_species=reference_species,
        clarification_question=clarification_q,
        explicitly_requested_both=explicitly_both,
        source="llm",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="Planner (LLM #1)", run_type="chain")
async def plan(prompt: str, llm=None) -> PlannerDecision:
    """Plan what the user wants. Never raises.

    ``llm`` is injectable for tests — pass a stub that implements
    ``ainvoke([SystemMessage, HumanMessage]) -> response`` to avoid
    any network call or API key requirement.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        try:
            from .framework.llm_client import LLMUnavailable, get_llm
            llm = get_llm()
        except Exception as exc:
            _logger.warning("[Planner] no LLM backend configured: %s", exc)
            return PlannerDecision(
                feature=PlannedFeature.CLARIFICATION_REQUIRED,
                clarification_question="Could you rephrase your question about evolutionary analysis?",
                source="llm_unavailable",
            )

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        _logger.warning("[Planner] classification call failed (%s)", type(exc).__name__)
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            clarification_question="Could you rephrase your question about evolutionary analysis?",
            source="error",
        )

    text = getattr(response, "content", str(response))
    payload = _extract_json(text)
    if payload is None:
        _logger.info("[Planner] could not parse model output as JSON: %r", text[:200])
        return PlannerDecision(
            feature=PlannedFeature.CLARIFICATION_REQUIRED,
            clarification_question="Could you rephrase your question about evolutionary analysis?",
            source="unparsable",
        )

    decision = _apply_guards(payload)
    _logger.info(
        "[Planner] feature=%s species=%r source=%s",
        decision.feature.value, decision.species_list, decision.source,
    )
    return decision

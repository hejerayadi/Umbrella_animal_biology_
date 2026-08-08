"""Intent classification for the Evolution Agent — Sprint 2.

Sprint 2 simplification: there are only two skills (molecular_comparison,
phylogenetic_tree) but the pipeline always runs both in sequence.  The
classifier therefore only needs to decide:

    1. Is this question about evolutionary analysis at all?
    2. If yes — extract the species list and optional reference species.

The feature is always "full_analysis" for Sprint 2.  The individual
feature values (molecular_comparison, phylogenetic_tree) are kept in the
enum for direct single-step calls from tests or future callers.

Two behaviours are guaranteed:
- Never raises.  LLM unavailable / network error / unparsable output →
  RecognizedIntent(feature=None), caller decides what to do.
- Never invents a feature outside the enum.  A model that responds with
  "protein_folding" gets feature=None, not a guess.

Plain JSON prompting is used throughout — Azure, Groq and GitHub Models
all handle it reliably; tool-calling support is uneven across the three.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

_VALID_FEATURES = {
    "molecular_comparison",
    "phylogenetic_tree",
    "full_analysis",
}

INTENT_SYSTEM_PROMPT = """\
You are the Global Scientific Orchestrator for the Umbrella BioHub.
Given a user's free-form question, decide whether it is asking for an
EVOLUTIONARY ANALYSIS and extract the parameters.

Respond with a strict JSON object — no prose, no code fences, no extra keys:

{
  "feature": "full_analysis" | "molecular_comparison" | "phylogenetic_tree" | null,
  "species_list": ["<species 1>", "<species 2>", ...],
  "reference_species": "<outgroup or root species, or null>"
}

Rules:
- full_analysis      : the user wants a complete evolutionary comparison —
                       sequence similarity AND a phylogenetic tree.
                       Use this when in doubt about which sub-task to run.
- molecular_comparison: the user asks only about sequence similarity,
                       embeddings, or species grouping — NOT a tree.
- phylogenetic_tree  : the user asks only for a tree or phylogeny — NOT
                       pairwise scores.
- species_list       : every species mentioned. Use scientific names where
                       possible (e.g. "Homo sapiens"). Include at least 2.
- reference_species  : the outgroup or root species if the user names one,
                       else null.
- If the question has nothing to do with evolutionary biology or species
  comparison, set "feature" to null and "species_list" to [].
- Output ONLY the JSON. No markdown, no explanation.\
"""


@dataclass(frozen=True)
class RecognizedIntent:
    """What the classifier managed to work out.

    ``feature=None`` means the question was not about evolutionary analysis
    (or the classifier could not parse a valid answer).
    """

    feature:          str | None       = None
    species_list:     list[str]        = field(default_factory=list)
    reference_species: str | None      = None
    # Diagnostic — "llm", "llm_unavailable", "unparsable", "error"
    source:           str              = "llm"

    @property
    def is_usable(self) -> bool:
        return self.feature is not None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Best-effort dict from model output that may be wrapped in fences."""
    stripped = text.strip()
    candidates: list[str] = [stripped] if stripped else []

    # ```json ... ``` blocks despite instructions
    candidates.extend(
        b.strip()
        for b in re.findall(
            r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE
        )
        if b.strip()
    )
    # JSON object embedded in prose
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


def _to_intent(payload: dict) -> RecognizedIntent:
    feature = _clean_str(payload.get("feature"))
    if feature is not None and feature not in _VALID_FEATURES:
        _logger.info(
            "[Intent] model returned unknown feature %r; treating as none", feature
        )
        feature = None

    species_list      = _clean_list(payload.get("species_list"))
    reference_species = _clean_str(payload.get("reference_species"))

    return RecognizedIntent(
        feature=feature,
        species_list=species_list,
        reference_species=reference_species,
        source="llm" if feature else "unparsable",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def classify_intent(prompt: str, llm=None) -> RecognizedIntent:
    """Classify ``prompt`` for evolutionary analysis.  Never raises.

    ``llm`` is injectable for tests — pass a stub that implements
    ``ainvoke([SystemMessage, HumanMessage]) -> response`` to avoid
    any network call or API key requirement.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        try:
            # Re-use the shared LLM client (Azure → Groq → GitHub Models).
            from ..biodiversity_agent.framework.llm_client import (
                LLMUnavailable,
                get_llm,
            )
            llm = get_llm()
        except Exception:
            try:
                from .framework.llm_client import LLMUnavailable, get_llm  # type: ignore
                llm = get_llm()
            except Exception as exc:
                _logger.warning("[Intent] no LLM backend configured: %s", exc)
                return RecognizedIntent(source="llm_unavailable")

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "[Intent] classification call failed (%s)", type(exc).__name__
        )
        return RecognizedIntent(source="error")

    text    = getattr(response, "content", str(response))
    payload = _extract_json(text)
    if payload is None:
        _logger.info(
            "[Intent] could not parse model output as JSON: %r", text[:200]
        )
        return RecognizedIntent(source="unparsable")

    intent = _to_intent(payload)
    _logger.info(
        "[Intent] feature=%s species=%r reference=%r",
        intent.feature, intent.species_list, intent.reference_species,
    )
    return intent

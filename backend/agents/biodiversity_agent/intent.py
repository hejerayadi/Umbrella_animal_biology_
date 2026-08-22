"""Intent classification: a free-form question -> a biodiversity feature.

The Biodiversity Orchestrator dispatches on `AgentRequest.feature`, but the
Global Orchestrator only ever sends `instruction` and `context` - the shared
contract has no `feature` field. Something has to read the sentence and decide
which of the four skills it is asking for, or every request from the frontend
lands in `fail_no_feature`.

This module is that step. It was lifted out of `dashboard.py`, which had been
doing it inline to stand in for the Global Orchestrator; the dashboard now
imports it back, so there is one prompt and one parser rather than two copies
drifting apart.

Two behaviours matter now that this runs behind an HTTP endpoint rather than in
a Streamlit page, where a crash was just a red box to retry:

- It never raises. A missing LLM, a network error, or a model that answers with
  prose all produce a `RecognizedIntent` with `feature=None`, and the caller
  decides what that means.
- It never guesses a feature. Returning `species_distribution_map` because it
  is the most common would produce a confident map for a question about
  migration. `feature=None` is the honest answer, and the adapter turns it into
  a message naming what this agent can actually do.

Deliberately parses JSON by hand rather than using `with_structured_output`.
`get_llm()` can return Azure, Groq or GitHub Models depending on which keys are
present, and their tool-calling support is not uniform - plain JSON is the one
thing all three do reliably. The parser below is written to survive the ways
they each get it slightly wrong.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .framework.llm_client import LLMUnavailable, get_llm
from .schema import BiodiversityFeature

_logger = logging.getLogger(__name__)

_VALID_FEATURES = {feature.value for feature in BiodiversityFeature}

# Moved verbatim from dashboard.py - the wording is already tuned against the
# four features and their edge cases, so it is kept as-is rather than rewritten.
INTENT_SYSTEM_PROMPT = """You are the Global Scientific Orchestrator for the Umbrella BioHub.
Given a user's free-form question, decide which BIODIVERSITY feature to run and
extract the parameters. Respond with a strict JSON object, no prose, no code
fences, no extra keys:

{
  "feature": "species_distribution_map" | "habitat_visualization" | "biodiversity_hotspots" | "migration_analysis",
  "species_name": "<scientific Latin binomial only, or null>",
  "region": "<continent or country, default 'global'>"
}

Rules:
- species_distribution_map: user asks WHERE a species is observed (point map).
- habitat_visualization: user asks about a species HABITAT / conservation status.
- biodiversity_hotspots: user asks about REGIONS with many species (no single species).
- migration_analysis: user asks about MIGRATION / seasonal movement of a species.
- species_name MUST be the scientific Latin binomial only (e.g. "Loxodonta africana",
  "Panthera tigris", "Ursus maritimus", "Sterna paradisaea", "Ciconia ciconia",
  "Megaptera novaeangliae", "Danaus plexippus"). Never a common name,
  never a plural, never a parenthetical, never a disjunction "A / B". If the user
  gave a common name in any language (French: "cigogne blanche" -> "Ciconia ciconia",
  "baleine à bosse" -> "Megaptera novaeangliae", "papillon monarque" ->
  "Danaus plexippus"; English: "white stork" -> "Ciconia ciconia"), translate it.
  If the user named a broader group with no single accepted binomial (e.g.
  "elephants", "tigers", "bears"), pick the most representative canonical species
  (savanna elephant -> Loxodonta africana, tiger -> Panthera tigris, brown bear
  -> Ursus arctos).
- If the user asks about hotspots, species_name is null.
- If the question is not about any of these four, set "feature" to null.
- Output ONLY the JSON. No markdown."""


@dataclass(frozen=True)
class RecognizedIntent:
    """What the classifier managed to work out. `feature` None means it could not."""

    feature: str | None = None
    species_name: str | None = None
    region: str = "global"
    # How this was arrived at, carried through to the caller for logging:
    # "llm", "llm_unavailable", "unparsable", or "error".
    source: str = "llm"

    @property
    def is_usable(self) -> bool:
        """True when there is a real feature to dispatch on."""
        return self.feature is not None


def _extract_json(text: str) -> dict | None:
    """Best-effort dict from model output that may be wrapped in prose or fences."""

    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    # ```json ... ``` despite being told not to.
    candidates.extend(
        block.strip()
        for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if block.strip()
    )
    # A JSON object embedded in a sentence.
    candidates.extend(
        span.strip() for span in re.findall(r"\{[\s\S]*\}", stripped) if span.strip()
    )

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clean(value: object) -> str | None:
    """Normalise a model-supplied string, treating its ways of saying 'nothing'."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown", ""}:
        return None
    return text


def _to_intent(payload: dict) -> RecognizedIntent:
    """Validate a parsed payload into an intent, rejecting invented features."""

    feature = _clean(payload.get("feature"))
    if feature is not None and feature not in _VALID_FEATURES:
        # A feature name outside the enum is not repaired or guessed at - the
        # router would raise on it anyway, and a wrong skill is worse than none.
        _logger.info("[Intent] model returned unknown feature %r; treating as none", feature)
        feature = None

    return RecognizedIntent(
        feature=feature,
        species_name=_clean(payload.get("species_name")),
        region=_clean(payload.get("region")) or "global",
        source="llm" if feature else "unparsable",
    )


async def classify_intent(prompt: str, llm=None) -> RecognizedIntent:
    """Classify `prompt` into one of the four features. Never raises.

    `llm` is injectable so the mapping can be tested without a network call or
    an API key.
    """

    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        try:
            llm = get_llm()
        except LLMUnavailable as exc:
            _logger.warning("[Intent] no LLM backend configured: %s", exc)
            return RecognizedIntent(source="llm_unavailable")

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=INTENT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
    except Exception as exc:  # noqa: BLE001 - a backend may fail any number of ways
        _logger.warning("[Intent] classification call failed (%s)", type(exc).__name__)
        return RecognizedIntent(source="error")

    text = getattr(response, "content", str(response))
    payload = _extract_json(text)
    if payload is None:
        _logger.info("[Intent] could not parse model output as JSON: %r", text[:200])
        return RecognizedIntent(source="unparsable")

    intent = _to_intent(payload)
    _logger.info(
        "[Intent] feature=%s species=%r region=%r",
        intent.feature, intent.species_name, intent.region,
    )
    return intent

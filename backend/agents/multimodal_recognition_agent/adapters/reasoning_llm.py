"""The optional reasoning-LLM boundary.

GPT-5 mini is the approved reasoning model for this agent. It is **disabled by
default** and Recognition works completely without it: deterministic rules are
the first and default path, and they stay the fallback whenever the model is
absent, misconfigured, slow, or wrong.

What the model is allowed to do is deliberately tiny. It may refine an ambiguous
intent and fill in hints the rules left empty. That is all. It cannot see an
image, cannot score anything, cannot name a species that retrieval did not
return, and cannot reach another agent - not by policy, but because the types
below give it no way to.

Three structural guarantees, each enforced by construction rather than by care:

1. `ReasoningRequest` is a frozen dataclass with four fields, all text. There is
   no field for image bytes, base64, a data URL, a vector, the request context,
   or a credential, so none can be passed even by mistake.
2. `ReasoningResult` is a frozen dataclass with five optional fields. Anything
   the model returns outside that shape is discarded and the rules stand.
   `intent` is validated against the fixed enum before it is accepted.
3. The call budget lives on a `ReasoningBudget` created fresh per request, not
   on the adapter. A leaked adapter cannot accumulate calls across requests.

Nothing here opens a connection at import time or at agent construction. The
client is built lazily, on the first call that is actually permitted to happen -
so an enabled-but-never-needed adapter still performs no network operation.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain.models import Intent

_logger = logging.getLogger(__name__)

# The only intents the model may choose from. Anything else is malformed output.
_ALLOWED_INTENTS: frozenset[str] = frozenset(("recognition", "similarity", "scientific_follow_up"))

# Hints are short by nature; a long one is a sign the model is writing prose (or
# echoing its input) rather than extracting, so it is dropped.
_MAX_HINT_LENGTH = 120

# The approved environment-variable contract. Read at lazy-construction time and
# never copied into configuration objects, logs, fixtures or this source.
_ENV_BASE_URL = "AZURE_OPENAI_BASE_URL"
_ENV_API_KEY = "AZURE_OPENAI_API_KEY"
_ENV_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
_ENV_REASONING_EFFORT = "AZURE_OPENAI_REASONING_EFFORT"
_ENV_MAX_OUTPUT_TOKENS = "AZURE_OPENAI_MAX_OUTPUT_TOKENS"


@dataclass(frozen=True)
class ReasoningRequest:
    """Everything the model is ever given. Four text fields, and no more.

    `candidate_names` and `candidate_species_ids` are the species retrieval
    already returned. They are supplied so the model can read the situation, and
    they are the reason it can never invent one: the workflow only ever matches
    its answer back against this list.
    """

    instruction: str
    rule_intent: str
    candidate_names: tuple[str, ...] = ()
    candidate_species_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningResult:
    """The strict schema. Anything outside it never reaches the workflow."""

    intent: Intent | None = None
    taxon_hint: str | None = None
    location_hint: str | None = None
    habitat_hint: str | None = None
    language: str | None = None


@dataclass
class ReasoningBudget:
    """One request's allowance. Created per request, never shared.

    Making this an argument rather than adapter state is what stops a
    long-lived adapter from drifting over its per-request limit.
    """

    max_calls: int = 2
    calls_made: int = 0
    accepted: bool = False

    def consume(self) -> bool:
        """Take one call if any remain. False means: use the rules."""
        if self.calls_made >= self.max_calls:
            return False
        self.calls_made += 1
        return True


class ReasoningLLM(Protocol):
    """What the text analyser needs from a reasoning model."""

    enabled: bool
    name: str

    def analyze(self, request: ReasoningRequest) -> ReasoningResult | None:
        ...


class NullReasoningLLM:
    """The default. Costs nothing, calls nothing, returns nothing.

    Recognition runs on deterministic rules with this in place, which is why
    the LLM being unavailable is not a failure mode - it is the normal state.
    """

    enabled = False
    name = "disabled"

    def analyze(self, request: ReasoningRequest) -> ReasoningResult | None:
        return None


def sanitize_result(raw: Any) -> ReasoningResult | None:
    """Turn whatever came back into the fixed schema, or into nothing.

    Rejects rather than repairs: a model that answers off-contract is a model
    whose answer we should not be using.
    """
    if not isinstance(raw, dict):
        return None

    intent = raw.get("intent")
    if intent is not None and intent not in _ALLOWED_INTENTS:
        # An unknown intent means the whole response is untrustworthy, not just
        # that one field.
        return None

    def _hint(key: str) -> str | None:
        value = raw.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or len(value) > _MAX_HINT_LENGTH:
            return None
        return value

    language = _hint("language")
    if language is not None and not (2 <= len(language) <= 5):
        language = None

    return ReasoningResult(
        intent=intent,
        taxon_hint=_hint("taxon_hint"),
        location_hint=_hint("location_hint"),
        habitat_hint=_hint("habitat_hint"),
        language=language,
    )


class BoundedAzureReasoningLLM:
    """The approved deployment, kept on a very short leash.

    Constructed only when explicitly enabled and fully configured. Even then it
    builds no client until a call is actually permitted, so importing this
    module or starting the agent performs no network operation.
    """

    name = "azure-reasoning"

    _SYSTEM_PROMPT = (
        "You classify a short user instruction for a species-recognition agent.\n"
        "Return ONLY a JSON object with these keys: intent, taxon_hint, "
        "location_hint, habitat_hint, language.\n"
        "intent must be exactly one of: recognition, similarity, scientific_follow_up.\n"
        "Use null for any hint the instruction does not state. Never guess a species. "
        "Never invent a location. You are not identifying anything in an image - you "
        "are only reading the sentence."
    )

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self._timeout = timeout_seconds
        self._client: Any = None
        # Presence only. The values are read at call time and never stored.
        self.enabled = bool(os.getenv(_ENV_BASE_URL) and os.getenv(_ENV_API_KEY)
                            and os.getenv(_ENV_DEPLOYMENT))
        if not self.enabled:
            _logger.info(
                "[Recognition] reasoning LLM enabled but not configured; "
                "deterministic rules will be used."
            )

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # imported lazily, never at module scope

            self._client = OpenAI(
                base_url=os.environ[_ENV_BASE_URL],
                api_key=os.environ[_ENV_API_KEY],
                timeout=self._timeout,
            )
        return self._client

    def analyze(self, request: ReasoningRequest) -> ReasoningResult | None:
        if not self.enabled:
            return None

        payload = (
            f"instruction: {request.instruction}\n"
            f"rule_intent: {request.rule_intent}\n"
            f"retrieved_species: {', '.join(request.candidate_names) or 'none'}"
        )

        try:
            import json

            response = self._ensure_client().responses.create(
                model=os.environ[_ENV_DEPLOYMENT],
                input=f"{self._SYSTEM_PROMPT}\n\n{payload}",
                reasoning={"effort": os.getenv(_ENV_REASONING_EFFORT, "low")},
                max_output_tokens=int(os.getenv(_ENV_MAX_OUTPUT_TOKENS, "400")),
                store=False,
            )
            return sanitize_result(json.loads(response.output_text))
        except Exception as exc:  # noqa: BLE001 - any failure means "use the rules"
            # The error type is logged, never the payload or the response.
            _logger.info(
                "[Recognition] reasoning LLM unavailable (%s); falling back to rules.",
                type(exc).__name__,
            )
            return None


def build_reasoning_llm(enabled: bool, timeout_seconds: float) -> ReasoningLLM:
    """The one place an adapter is chosen. Disabled unless explicitly turned on."""
    if not enabled:
        # The null brain still exposes plan/explain, so the workflow can call
        # the same contract whether a model is present or not.
        return NullRecognitionLLM()
    return BoundedAzureReasoningLLM(timeout_seconds=timeout_seconds)


# ===========================================================================
# Phase 4 - the planning and explanation contract
#
# The validated decisions make GPT-5 mini the agent's reasoning brain: it plans
# the internal execution and it writes the final explanation. Exactly two calls
# per request, and not one more.
#
# It still decides nothing scientific. The plan it produces is a *request* to
# run steps from a fixed vocabulary; anything outside that vocabulary is
# rejected wholesale and the deterministic plan is used instead. The explanation
# it writes is checked against the evidence before it is accepted, so it cannot
# name a species retrieval did not return.
# ===========================================================================

# The complete set of steps a plan may ask for. A plan naming anything else -
# "compare", "clarify", "call_agent", "fetch_url" - is rejected entirely.
ALLOWED_PLAN_STEPS: tuple[str, ...] = (
    "embed_image",
    "retrieve_candidates",
    "validate_taxonomy",
    "score_confidence",
    "explain",
)

# Steps that must run whatever the plan says. The model cannot skip the evidence
# and jump to a conclusion.
MANDATORY_PLAN_STEPS: tuple[str, ...] = (
    "embed_image",
    "retrieve_candidates",
    "score_confidence",
)

_MAX_TOP_K = 50


@dataclass(frozen=True)
class PlanRequest:
    """What the planner is given. Text and image *metadata* only.

    Note what is absent: no bytes, no base64, no data URL, no vector. The
    planner is told an image exists and what type it is - never what it
    contains. That is BioCLIP-2's job, not the model's.
    """

    instruction: str
    has_image: bool
    image_media_type: str
    rule_intent: str


@dataclass(frozen=True)
class RecognitionPlan:
    """A validated internal plan. Only ever built by `sanitize_plan`."""

    steps: tuple[str, ...]
    intent: Intent
    top_k: int
    taxon_hint: str | None = None
    location_hint: str | None = None
    habitat_hint: str | None = None
    language: str | None = None
    requested_capability: str | None = None
    source: str = "llm"


@dataclass(frozen=True)
class ExplainRequest:
    """The structured evidence the explanation must be grounded in.

    Every field here was computed by deterministic code. The model is being
    asked to phrase this, not to add to it.
    """

    decision: str
    text_alignment: str
    primary_species: str | None
    candidate_names: tuple[str, ...]
    top_score: float | None
    margin: float | None
    taxonomy_status: str | None
    retrieval_mode: str
    embedding_mode: str
    visual_evidence_sufficient: bool


def sanitize_plan(raw: Any, *, default_top_k: int, rule_intent: str) -> RecognitionPlan | None:
    """Validate a proposed plan strictly. Reject rather than repair.

    A malformed plan is not a small problem to be patched up - it means the
    model is not behaving to contract, so nothing it said is used.
    """
    if not isinstance(raw, dict):
        return None

    steps = raw.get("steps")
    if not isinstance(steps, (list, tuple)) or not steps:
        return None
    if any(not isinstance(step, str) for step in steps):
        return None
    # One forbidden step invalidates the whole plan.
    if any(step not in ALLOWED_PLAN_STEPS for step in steps):
        return None
    if any(required not in steps for required in MANDATORY_PLAN_STEPS):
        return None

    intent = raw.get("intent", rule_intent)
    if intent not in _ALLOWED_INTENTS:
        return None

    top_k = raw.get("top_k", default_top_k)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= _MAX_TOP_K:
        return None

    def _hint(key: str) -> str | None:
        value = raw.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and len(value) <= _MAX_HINT_LENGTH else None

    language = _hint("language")
    if language is not None and not (2 <= len(language) <= 5):
        language = None

    capability = _hint("requested_capability")

    return RecognitionPlan(
        # Order is ours, not the model's: it may choose *what* runs, never the
        # sequence the evidence is built in.
        steps=tuple(step for step in ALLOWED_PLAN_STEPS if step in steps),
        intent=intent,
        top_k=top_k,
        taxon_hint=_hint("taxon_hint"),
        location_hint=_hint("location_hint"),
        habitat_hint=_hint("habitat_hint"),
        language=language,
        requested_capability=capability,
        source="llm",
    )


def deterministic_plan(*, intent: str, top_k: int, evidence: Any = None) -> RecognitionPlan:
    """The plan used when there is no model, or when its plan was rejected."""
    return RecognitionPlan(
        steps=ALLOWED_PLAN_STEPS,
        intent=intent,  # type: ignore[arg-type]
        top_k=top_k,
        taxon_hint=getattr(evidence, "taxon_hint", None),
        location_hint=getattr(evidence, "location_hint", None),
        habitat_hint=getattr(evidence, "habitat_hint", None),
        language=getattr(evidence, "language", None),
        requested_capability=getattr(evidence, "requested_capability", None),
        source="deterministic",
    )


def explanation_is_grounded(text: Any, request: ExplainRequest) -> bool:
    """Would accepting this explanation put an unsupported claim in the output?

    The check that matters: a species name may appear only if retrieval
    returned it. Everything else the model writes is prose about evidence we
    computed; a species it introduces would be a fabrication.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 2000:
        return False

    lowered = stripped.lower()
    allowed = {name.lower() for name in request.candidate_names}
    if request.primary_species:
        allowed.add(request.primary_species.lower())

    # Any binomial-looking name in the text must be one we actually retrieved.
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b", stripped):
        candidate = f"{match.group(1)} {match.group(2)}".lower()
        if candidate in allowed:
            continue
        if any(candidate in name or name in candidate for name in allowed):
            continue
        if match.group(1).lower() in _EXPLANATION_SAFE_WORDS:
            continue
        return False

    # A similarity score is not a probability, and the explanation may not imply
    # otherwise.
    for forbidden in ("% probability", "percent probability", "probability of",
                      "certainty of", "% confident", "% certain"):
        if forbidden in lowered:
            return False
    return True


# Ordinary sentence openings that the binomial pattern would otherwise flag.
_EXPLANATION_SAFE_WORDS = frozenset({
    "the", "this", "that", "retrieval", "recognition", "taxonomy", "similarity",
    "confidence", "evidence", "image", "qdrant", "candidates", "based", "no",
    "none", "there", "these", "those", "results", "scores", "margin", "local",
    "mock", "insufficient", "visual", "species", "identification", "reference",
})


class NullRecognitionLLM(NullReasoningLLM):
    """The default brain: absent. Recognition runs deterministically without it."""

    def plan(self, request: PlanRequest) -> RecognitionPlan | None:
        return None

    def explain(self, request: ExplainRequest) -> str | None:
        return None


class FakeGPT5MiniProvider:
    """A deterministic stand-in for GPT-5 mini, for offline testing.

    It exercises the real contract - two calls, strict schemas, rejectable
    output - without a network, a credential, or a token spent. The real
    provider replaces this class and nothing else.
    """

    name = "fake-gpt-5-mini"
    enabled = True

    def __init__(
        self,
        *,
        plan_override: Any = None,
        explanation_override: Any = None,
        fail_plan: BaseException | None = None,
        fail_explain: BaseException | None = None,
    ) -> None:
        # The overrides exist so tests can drive malformed output, forbidden
        # steps and provider failures without a network.
        self._plan_override = plan_override
        self._explanation_override = explanation_override
        self._fail_plan = fail_plan
        self._fail_explain = fail_explain
        self.plan_calls = 0
        self.explain_calls = 0
        self.seen_plan_requests: list[PlanRequest] = []
        self.seen_explain_requests: list[ExplainRequest] = []

    # -- call 1 of 2 --------------------------------------------------------

    def plan(self, request: PlanRequest) -> Any:
        self.plan_calls += 1
        self.seen_plan_requests.append(request)
        if self._fail_plan is not None:
            raise self._fail_plan
        if self._plan_override is not None:
            return self._plan_override

        # A plausible, well-formed plan derived only from the instruction.
        lowered = request.instruction.lower()
        intent = request.rule_intent
        steps = list(ALLOWED_PLAN_STEPS)
        return {
            "steps": steps,
            "intent": intent,
            "top_k": 30,
            "taxon_hint": None,
            "location_hint": None,
            "habitat_hint": next(
                (word for word in ("savanna", "forest", "arctic", "desert") if word in lowered),
                None,
            ),
            "language": None,
            "requested_capability": None,
        }

    # -- call 2 of 2 --------------------------------------------------------

    def explain(self, request: ExplainRequest) -> Any:
        self.explain_calls += 1
        self.seen_explain_requests.append(request)
        if self._fail_explain is not None:
            raise self._fail_explain
        if self._explanation_override is not None:
            return self._explanation_override

        if request.decision == "not_identified" or request.primary_species is None:
            return (
                "The retrieved references were not close enough to support naming a "
                "species, so no identification is claimed."
            )
        score = "unknown" if request.top_score is None else f"{request.top_score:.4f}"
        return (
            f"{request.primary_species} is the closest retrieved match "
            f"(aggregated similarity {score}). The decision is {request.decision} and the "
            f"text alignment is {request.text_alignment}."
        )

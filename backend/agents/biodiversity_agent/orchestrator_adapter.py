"""Adapter between the HTTP boundary and the real Biodiversity Orchestrator.

`api.py` receives the platform-wide request - `instruction` plus `context` -
and nothing else. The orchestrator dispatches on `AgentRequest.feature`, and
returns FAILED when it is absent. This module bridges that gap: it works out
the feature, fills in the extended request the orchestrator wants, runs it, and
shapes the result back into what the Global Orchestrator merges into its shared
context.

Unlike the Trait agent's adapter there is no schema translation here. Both
sides already import `AgentRequest`/`AgentResult`/`AgentStatus` from
`schema.py`, which was written as a superset of the platform contract - so
statuses, escalation fields and the free-form `output` all pass straight
through. What this layer adds is the missing `feature`, and the output shape
the rest of the platform expects.

Nothing under `orchestrator/`, `workers/` or `services/` is imported *into* -
only from. That is what keeps the 18 existing tests, and the Streamlit
dashboard, working exactly as they do today.
"""

from __future__ import annotations

import logging
from typing import Any

from .intent import RecognizedIntent, classify_intent
from .orchestrator import BiodiversityOrchestrator
from .schema import AgentRequest, AgentResult, AgentStatus, BiodiversityFeature

_logger = logging.getLogger(__name__)

# Returned when the classifier could not name a feature. Deliberately lists the
# four skills: the user asked this agent something it may simply not do, and
# saying so is more use than a map of the wrong thing.
_NO_FEATURE_MESSAGE = (
    "The Biodiversity Agent could not tell which of its skills this question needs. "
    "It can map where a species is observed (species distribution), show a species' "
    "habitat and conservation status, find biodiversity hotspots in a region, or "
    "analyse a species' migration. Could you rephrase the question towards one of those?"
)

# Same idea, for the one gate inside the orchestrator this adapter can see coming:
# distribution, habitat and migration all require a species; hotspots does not.
_SPECIES_REQUIRED = {
    BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
    BiodiversityFeature.HABITAT_VISUALIZATION.value,
    BiodiversityFeature.MIGRATION_ANALYSIS.value,
}

_NO_SPECIES_MESSAGE = (
    "That question needs a species before the Biodiversity Agent can answer it - "
    "'{feature}' is species-scoped, and no species was named in the request. "
    "Which animal did you mean?"
)


def resolve_species(context: dict[str, Any], intent: RecognizedIntent) -> str | None:
    """Pick the species name, preferring what the platform already established.

    The Global Orchestrator runs a dedicated extraction step before any agent
    is called and writes `context["species"]`. That is a purpose-built reading
    of the question, so it wins over the classifier's incidental guess; the
    classifier only fills in when the orchestrator supplied nothing.
    """
    return (
        context.get("species")
        or context.get("species_name")
        or intent.species_name
    )


def to_orchestrator_request(
    request: AgentRequest, intent: RecognizedIntent
) -> AgentRequest:
    """Build the extended request the orchestrator dispatches on."""

    context = request.context or {}
    return AgentRequest(
        instruction=request.instruction,
        context=dict(context),
        feature=intent.feature,
        species_name=resolve_species(context, intent),
        region=intent.region or "global",
    )


def to_platform_result(result: AgentResult) -> AgentResult:
    """Reshape the orchestrator's result into the platform's output contract.

    `card.json` declares this agent's output as `biodiversity_report` and
    `map_url`, and the Image Generation agent branches on
    `"biodiversity_report" in context`. The orchestrator's aggregator hands back
    the raw worker payload when only one worker ran, so without this wrapping
    that key would never appear and Image Generation would sit waiting for
    something no one publishes.
    """

    if result.status is AgentStatus.FAILED:
        return result

    report: dict[str, Any] = {"findings": result.output}
    for key, value in (
        ("hotspots", result.hotspots),
        ("migration_route", result.migration_route),
        ("observation_count", result.observation_count),
        ("confidence", result.confidence),
    ):
        if value is not None:
            report[key] = value
    if result.source_agents:
        report["source_agents"] = list(result.source_agents)

    output: dict[str, Any] = {"biodiversity_report": report}
    if result.map_url:
        output["map_url"] = result.map_url

    # Escalations keep their target and prompt: the Global Orchestrator reads
    # those to decide who runs next, and the partial findings still travel.
    return AgentResult(
        status=result.status,
        target_agent=result.target_agent,
        prompt_to_target_agent=result.prompt_to_target_agent,
        output=output,
        map_url=result.map_url,
        hotspots=result.hotspots,
        migration_route=result.migration_route,
        observation_count=result.observation_count,
        confidence=result.confidence,
        source_agents=list(result.source_agents),
    )


def _failed(message: str) -> AgentResult:
    return AgentResult(
        status=AgentStatus.FAILED,
        output=message,
        source_agents=["Biodiversity Agent Orchestrator"],
    )


class OrchestratorBiodiversityAgent:
    """Serves the real orchestrator behind the agent's HTTP endpoint.

    Same `run(request) -> AgentResult` shape as `BiodiversityMock`, except it
    is async, because the orchestrator's graph is.
    """

    def __init__(self, orchestrator: BiodiversityOrchestrator | None = None) -> None:
        # Built once: constructing it compiles a LangGraph state machine and
        # opens the taxonomy backend.
        self._orchestrator = orchestrator or BiodiversityOrchestrator()

    async def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}

        # A caller that already knows what it wants skips classification
        # entirely - `context["features"]` is the orchestrator's own parallel
        # path, and re-deriving it from prose could only lose information.
        if context.get("features"):
            _logger.info("[Biodiversity] features supplied by caller; skipping classification")
            result = await self._orchestrator.run(request)
            return to_platform_result(result)

        intent = await classify_intent(request.instruction)
        if not intent.is_usable:
            _logger.info("[Biodiversity] no feature resolved (%s)", intent.source)
            return _failed(_NO_FEATURE_MESSAGE)

        orchestrator_request = to_orchestrator_request(request, intent)

        # The orchestrator would fail this itself, but its message names its own
        # internal fields. Answering here lets the user be told what to do.
        if (
            orchestrator_request.feature in _SPECIES_REQUIRED
            and not orchestrator_request.species_name
        ):
            _logger.info("[Biodiversity] %s needs a species, none given", intent.feature)
            return _failed(_NO_SPECIES_MESSAGE.format(feature=intent.feature))

        result = await self._orchestrator.run(orchestrator_request)
        return to_platform_result(result)

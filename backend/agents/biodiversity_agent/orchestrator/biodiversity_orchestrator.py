"""Biodiversity Agent Orchestrator - the entry point for the whole domain.

Design decisions (see ``docs/framework_benchmark_results.md`` for the
full comparison):

- **Framework**: LangGraph. The Biodiversity Agent is naturally a small
  state machine with a routing decision, an optional species-name
  normalization step, a fan-out to workers, and an aggregation node.
  LangGraph's ``StateGraph`` maps 1-to-1 to this shape; CrewAI's
  role-based abstraction adds indirection we do not need.
- **Concurrency**: ``asyncio.gather`` inside a LangGraph node. Workers
  are synchronous today (mocks) - we wrap them in
  ``asyncio.to_thread`` so the fan-out node stays truly parallel. When
  we replace mocks with async I/O, the node becomes a straight
  ``asyncio.gather`` of coroutines.
- **Escalation**: any child ``NEEDS_AGENT`` bubbles up unmodified.
  The orchestrator never resolves a cross-agent dependency itself.

Execution paths this orchestrator supports (all covered by tests):

1. **Sequential** - single feature -> single worker.
2. **Parallel** - multiple features supplied via
   ``context["features"]`` -> fan-out with ``asyncio.gather``.
3. **Conditional** - species-name normalization can short-circuit into
   FAILED without dispatching any worker if the taxonomy lookup fails
   *and* no explicit ``species_name`` was passed.
4. **Escalation** - any worker returning ``NEEDS_AGENT`` is bubbled up.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    BiodiversityFeature,
)
from ..workers.habitat.mock import HabitatMock
from ..workers.hotspots.mock import HotspotsMock
from ..workers.hotspots.worker import HotspotsWorker  # real M3 pipeline
from ..workers.migration.mock import MigrationMock
from ..workers.migration.worker import MigrationWorker  # real Random Forest (Miriam)
from ..workers.species_distribution.worker import SpeciesDistributionWorker
from .aggregator import aggregate
from .router import Router
from .services.qdrant_client import SpeciesTaxonomyService


# ---------- graph state ----------


@dataclass
class BiodiversityState:
    """Object passed through every LangGraph node."""

    request: AgentRequest
    # populated by nodes as the graph advances
    features_to_run: list[BiodiversityFeature] = field(default_factory=list)
    worker_results: list[tuple[BiodiversityFeature, AgentResult]] = field(
        default_factory=list
    )
    result: AgentResult | None = None
    # normalized species name after taxonomy lookup, if any
    normalized_species: str | None = None


# ---------- orchestrator ----------


class BiodiversityOrchestrator:
    """Domain orchestrator - callers hand it an ``AgentRequest`` and get an
    ``AgentResult`` back. Nothing else.
    """

    def __init__(
        self,
        workers: dict[BiodiversityFeature, Any] | None = None,
        taxonomy: SpeciesTaxonomyService | None = None,
    ) -> None:
        # Sprint 3 - two real workers plugged in:
        #   * Species Distribution -> real GBIF Occurrence Search
        #     (SpeciesDistributionWorker)
        #   * Biodiversity Hotspots -> real M3 pipeline: GBIF -> clean ->
        #     equal-area grid -> Shannon/Simpson/Chao1 -> effort correction
        #     -> DBSCAN (haversine, silhouette-tuned) -> ranked hotspots
        #     -> folium heatmap (HotspotsWorker, from Ouissale's branch)
        # Habitat and Migration stay on their mocks until their Sprint 3
        # implementations land. The orchestrator and its tests do not care
        # which flavor is plugged in as long as the
        # ``run(AgentRequest) -> AgentResult`` contract holds - HotspotsWorker
        # keeps ``progress`` as a keyword-only optional kwarg so the domain
        # dispatch (progress-free) and Ouissale's dashboard (progress-aware)
        # can both call it without changes.
        self._workers = workers or {
            BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionWorker(),
            BiodiversityFeature.HABITAT_VISUALIZATION:    HabitatMock(),
            BiodiversityFeature.BIODIVERSITY_HOTSPOTS:    HotspotsWorker(),
            BiodiversityFeature.MIGRATION_ANALYSIS:       MigrationWorker(),
        }
        self._router = Router(self._workers)
        self._taxonomy = taxonomy or SpeciesTaxonomyService.from_env()
        self._graph = self._build_graph()

    # ---------- public API ----------

    async def run(self, request: AgentRequest) -> AgentResult:
        state = BiodiversityState(request=request)
        final = await self._graph.ainvoke(state)
        # LangGraph returns the raw state dict, not the dataclass instance.
        # ``result`` is what our aggregator node set.
        if isinstance(final, dict):
            return final["result"]
        return final.result  # pragma: no cover

    # ---------- LangGraph wiring ----------

    def _build_graph(self):
        graph = StateGraph(BiodiversityState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("normalize_species", self._normalize_species_node)
        graph.add_node("dispatch", self._dispatch_node)
        graph.add_node("aggregate", self._aggregate_node)
        graph.add_node("fail_no_feature", self._fail_no_feature_node)

        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "normalize": "normalize_species",
                "no_feature": "fail_no_feature",
            },
        )
        graph.add_conditional_edges(
            "normalize_species",
            self._route_after_normalize,
            {
                "dispatch": "dispatch",
                "no_species": "fail_no_feature",
            },
        )
        graph.add_edge("dispatch", "aggregate")
        graph.add_edge("aggregate", END)
        graph.add_edge("fail_no_feature", END)

        return graph.compile()

    # ---------- nodes ----------

    def _plan_node(self, state: BiodiversityState) -> dict:
        """Decide which features to run.

        Priority:
        1. Explicit ``context["features"]`` list - parallel path.
        2. Single ``request.feature`` - sequential path.
        3. Neither -> we escalate to the fail_no_feature node.
        """

        req = state.request
        multi = req.context.get("features")
        features: list[BiodiversityFeature] = []
        if multi:
            for f in multi:
                try:
                    features.append(BiodiversityFeature(f))
                except ValueError:
                    continue
        elif req.feature:
            try:
                features.append(BiodiversityFeature(req.feature))
            except ValueError:
                features = []

        return {"features_to_run": features}

    def _route_after_plan(self, state: BiodiversityState) -> str:
        return "normalize" if state.features_to_run else "no_feature"

    def _normalize_species_node(self, state: BiodiversityState) -> dict:
        req = state.request
        # Features that need a species: distribution, habitat, migration.
        # Hotspots is region-scoped and does not require a species.
        species_required = any(
            f
            in {
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP,
                BiodiversityFeature.HABITAT_VISUALIZATION,
                BiodiversityFeature.MIGRATION_ANALYSIS,
            }
            for f in state.features_to_run
        )
        if not species_required:
            return {"normalized_species": None}

        raw = (
            req.species_name
            or req.context.get("species")
            or req.context.get("species_name")
        )
        if not raw:
            return {"normalized_species": None}

        normalized = self._taxonomy.normalize(raw) or raw
        # Mutate the request in-place so workers see the canonical name.
        req.species_name = normalized
        return {"normalized_species": normalized}

    def _route_after_normalize(self, state: BiodiversityState) -> str:
        req = state.request
        needs_species = any(
            f
            in {
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP,
                BiodiversityFeature.HABITAT_VISUALIZATION,
                BiodiversityFeature.MIGRATION_ANALYSIS,
            }
            for f in state.features_to_run
        )
        if needs_species and not req.species_name:
            return "no_species"
        return "dispatch"

    async def _dispatch_node(self, state: BiodiversityState) -> dict:
        """Fan-out to every selected worker, in parallel via ``asyncio.gather``."""

        async def call(feature: BiodiversityFeature) -> tuple[BiodiversityFeature, AgentResult]:
            worker = self._workers[feature]
            # Mock workers are sync - offload to a thread so gather is truly parallel.
            result = await asyncio.to_thread(worker.run, state.request)
            return feature, result

        results = await asyncio.gather(*(call(f) for f in state.features_to_run))
        return {"worker_results": list(results)}

    def _aggregate_node(self, state: BiodiversityState) -> dict:
        return {"result": aggregate(state.worker_results)}

    def _fail_no_feature_node(self, state: BiodiversityState) -> dict:
        return {
            "result": AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Biodiversity Orchestrator: no valid feature to run. "
                    "Set AgentRequest.feature or context['features']."
                ),
                source_agents=["Biodiversity Agent Orchestrator"],
            )
        }

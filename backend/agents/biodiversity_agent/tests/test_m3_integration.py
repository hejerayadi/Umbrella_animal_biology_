"""M3 integration tests - Sprint 3 tasks 4 and 5.

Task 4 asks for the module to be integrated with the sub-orchestrator; task 5
asks for the whole chain to be validated:

    Sub-Orchestrator -> Agent -> Result -> Sub-Orchestrator

M3 is the single-agent case the brief describes, so there is deliberately no
multi-agent logic here: the sub-orchestrator delegates, the worker answers, the
aggregator returns.

The GBIF call is stubbed throughout, so these run offline in milliseconds. The
arithmetic behind the numbers is covered in ``test_m3_pipeline.py``.
"""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.orchestrator import BiodiversityOrchestrator
from backend.agents.biodiversity_agent.orchestrator.services.qdrant_client import (
    SpeciesTaxonomyService,
    _OfflineBackend,
)
from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    BiodiversityFeature,
)
from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
from backend.agents.biodiversity_agent.workers.hotspots.schema import (
    BiodiversityHotspotOutput,
    ClusteringQuality,
    EffortCorrection,
    HotspotCluster,
    HotspotValidation,
    MapRenderSpec,
)
from backend.agents.biodiversity_agent.workers.hotspots.status import (
    M3Outcome,
    build_result,
    outcome_of,
)
from backend.agents.biodiversity_agent.workers.hotspots.worker import HotspotsWorker
from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock
from backend.agents.biodiversity_agent.workers.species_distribution.mock import (
    SpeciesDistributionMock,
)
from backend.agents.biodiversity_agent.tests.m3_gazetteer import (
    patch_gazetteer,
)


@pytest.fixture(autouse=True)
def _offline_gazetteer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in the suite: place lookups use the fixed table."""

    patch_gazetteer(monkeypatch)


# ----------------------------------------------------------------- doubles


def build_payload(clusters: int = 2) -> BiodiversityHotspotOutput:
    """A structurally real M3 payload, without touching GBIF."""

    ranked = [
        HotspotCluster(cluster_id=i, label=f"Hotspot {i} (KE)",
                       centroid=(1.0, 30.0), area_km2=10_000.0, cell_count=6,
                       species_count=400 - i * 50, record_count=1_000,
                       score=1.0 - i * 0.1, rank=i + 1)
        for i in range(clusters)
    ]
    return BiodiversityHotspotOutput(
        region="kenya", study_area_km2=580_000.0, records_retrieved=9_000,
        records_analysed=7_000, species_analysed=1_200, cells_analysed=80,
        noise_cells=55, clusters=ranked, ranked_hotspots=ranked,
        effort_correction=EffortCorrection(method="test", applied=True,
                                           median_effort=30.0),
        quality=ClusteringQuality(n_clusters=clusters, noise_share=0.69,
                                  silhouette=0.91, eps_km=100.0, min_samples=8),
        validation=HotspotValidation(),
        parameters_used={"taxon_filter": "Animalia", "eps_km": 100.0},
        render_spec=MapRenderSpec(html_path="/tmp/map.html"), citations=[],
        summary="Two hotspots were found in Kenya.", confidence=0.9)


class StubHotspots(HotspotsWorker):
    """The real worker with only the GBIF-facing analysis replaced.

    Subclassing rather than duplicating keeps the request parsing, the region
    resolution and the status handling under test - only the network call and the
    clustering are stubbed.
    """

    def __init__(self, clusters: int = 2, fails: bool = False) -> None:
        self.clusters = clusters
        self.fails = fails
        self.calls: list[str] = []

    def analyse(self, params, **kwargs) -> AgentResult:
        self.calls.append(params.region_name)
        if self.fails:
            # What the real worker returns when GBIF gives back nothing usable:
            # the API layer degrades to zero records rather than raising, so the
            # failure arrives as a status, never as an exception.
            return build_result(
                M3Outcome.FAILED,
                output="GBIF returned no usable records for this region.",
                source_agents=["Biodiversity Hotspots Agent"])
        payload = build_payload(self.clusters)
        payload.region = params.region_name
        return AgentResult(
            status=AgentStatus.COMPLETED, output=payload,
            map_url=payload.render_spec.html_path,
            hotspots=[{"label": h.label, "species_count": h.species_count}
                      for h in payload.ranked_hotspots],
            confidence=payload.confidence,
            source_agents=["Biodiversity Hotspots Agent"])


def make_orchestrator(hotspots) -> BiodiversityOrchestrator:
    """The sub-orchestrator with M3 injected and the other three left as mocks.

    This is the integration point, and it needs no change to the orchestrator:
    its constructor already accepts a worker map.
    """

    return BiodiversityOrchestrator(
        workers={
            BiodiversityFeature.BIODIVERSITY_HOTSPOTS: hotspots,
            BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
            BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
            BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
        },
        taxonomy=SpeciesTaxonomyService(_OfflineBackend()))


# ------------------------------------------------- worker-level contract


def test_the_region_is_read_from_the_request() -> None:
    worker = StubHotspots()
    result = worker.run(AgentRequest(
        instruction="Where are the biodiversity hotspots in Kenya?",
        context={}, region="kenya"))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["kenya"]


def test_context_parameters_reach_the_analysis() -> None:
    """The orchestrator passes settings through ``context``, per the contract."""

    class _Capture(StubHotspots):
        captured = None

        def analyse(self, params, **kwargs):
            _Capture.captured = params
            return super().analyse(params, **kwargs)

    worker = _Capture()
    worker.run(AgentRequest(
        instruction="hotspots", context={"hotspot_params": {
            "cell_size_km": 60.0, "min_records_per_cell": 20, "top_n": 3}},
        region="kenya"))

    assert _Capture.captured.cell_size_km == 60.0
    assert _Capture.captured.min_records_per_cell == 20
    assert _Capture.captured.top_n == 3


def test_an_unknown_region_asks_instead_of_guessing() -> None:
    result = StubHotspots().run(AgentRequest(
        instruction="hotspots in Atlantis", context={}, region="atlantis"))

    # NEEDS_CLARIFICATION narrows to FAILED for the platform; M3's own outcome
    # stays readable in the payload.
    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert result.status is AgentStatus.FAILED
    assert "congo basin" in result.output["known_regions"]


def test_the_region_is_read_from_the_question_alone() -> None:
    """A chat client sends only a sentence - no ``region`` field.

    This is the path the dashboard's quick-buttons use, and it must not need an
    LLM: the sentence is matched against the known study areas.
    """

    worker = StubHotspots()
    result = worker.run(AgentRequest(
        instruction="Where are the biodiversity hotspots in Madagascar?",
        context={}))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["madagascar"]


def test_the_explicit_region_wins_over_the_sentence() -> None:
    worker = StubHotspots()
    worker.run(AgentRequest(instruction="hotspots in Madagascar please",
                            context={}, region="kenya"))

    assert worker.calls == ["kenya"]


def test_a_sentence_naming_no_known_area_still_asks() -> None:
    result = StubHotspots().run(AgentRequest(
        instruction="Which part of Africa has the most species?", context={}))

    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert "congo basin" in result.output["known_regions"]


def test_the_aliases_resolve() -> None:
    for sentence, expected in [
        ("hotspots in the Congo", "congo basin"),
        ("biodiversity of SE Asia", "southeast asia"),
        ("show me the Amazon basin", "amazon"),
        ("richest cells in the Sahara desert", "sahara"),
    ]:
        worker = StubHotspots()
        worker.run(AgentRequest(instruction=sentence, context={}))
        assert worker.calls == [expected], sentence


def test_two_regions_in_one_question_asks_which() -> None:
    """"Compare the Amazon and the Congo Basin" is two analyses, not one."""

    worker = StubHotspots()
    result = worker.run(AgentRequest(
        instruction="Compare biodiversity between the Amazon and the Congo Basin",
        context={}))

    assert outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION
    assert worker.calls == [], "nothing should be analysed before the answer"
    assert result.output["options"] == ["amazon", "congo basin"]


def test_answering_that_question_completes_the_run() -> None:
    """The follow-up answer arrives as an explicit region, and finishes the job."""

    worker = StubHotspots()
    result = worker.run(AgentRequest(
        instruction="Compare biodiversity between the Amazon and the Congo Basin",
        context={}, region="amazon"))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["amazon"]


def test_one_region_named_twice_is_not_ambiguous() -> None:
    worker = StubHotspots()
    result = worker.run(AgentRequest(
        instruction="hotspots in the Amazon - I mean the Amazon basin", context={}))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["amazon"]


# ------------------------------------------- sub-orchestrator integration


@pytest.mark.asyncio
async def test_the_chain_delegates_and_returns() -> None:
    """Sub-Orchestrator -> Agent -> Result -> Sub-Orchestrator."""

    worker = StubHotspots()
    result = await make_orchestrator(worker).run(AgentRequest(
        instruction="Where are the biodiversity hotspots in Kenya?",
        context={},
        feature=BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        region="kenya"))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["kenya"], "the task must be delegated to M3"
    assert result.hotspots, "the aggregator must surface the ranked hotspots"
    assert result.map_url, "the map must reach the caller"
    assert "Biodiversity Agent Orchestrator" in result.source_agents
    assert "Biodiversity Hotspots Agent" in result.source_agents


@pytest.mark.asyncio
async def test_hotspots_needs_no_species_name() -> None:
    """M3 is region-scoped, so the species-normalisation branch must be skipped."""

    worker = StubHotspots()
    result = await make_orchestrator(worker).run(AgentRequest(
        instruction="hotspots in Madagascar", context={},
        feature=BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        region="madagascar"))

    assert result.status is AgentStatus.COMPLETED
    assert worker.calls == ["madagascar"]


@pytest.mark.asyncio
async def test_context_is_passed_through_the_orchestrator() -> None:
    class _Capture(StubHotspots):
        captured = None

        def analyse(self, params, **kwargs):
            _Capture.captured = params
            return super().analyse(params, **kwargs)

    await make_orchestrator(_Capture()).run(AgentRequest(
        instruction="hotspots", context={"hotspot_params": {"top_n": 7}},
        feature=BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        region="kenya"))

    assert _Capture.captured.top_n == 7


@pytest.mark.asyncio
async def test_a_failure_inside_m3_is_reported_not_swallowed() -> None:
    result = await make_orchestrator(StubHotspots(fails=True)).run(AgentRequest(
        instruction="hotspots in Kenya", context={},
        feature=BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        region="kenya"))

    # The orchestrator's aggregator reports a single failed child as FAILED.
    assert result.status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_the_other_three_workers_are_untouched() -> None:
    """Injecting M3 must not disturb the features it does not own."""

    result = await make_orchestrator(StubHotspots()).run(AgentRequest(
        instruction="Where do African elephants live?", context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="Loxodonta africana"))

    assert result.status is AgentStatus.COMPLETED
    assert "Species Distribution Agent" in result.source_agents

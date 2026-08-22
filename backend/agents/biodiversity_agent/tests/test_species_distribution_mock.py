"""Species Distribution mock behavior - the worker contract in isolation."""

from __future__ import annotations

from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentStatus,
    BiodiversityFeature,
)
from backend.agents.biodiversity_agent.workers.species_distribution.mock import (
    SpeciesDistributionMock,
)
from backend.agents.biodiversity_agent.workers.species_distribution.schema import (
    SpeciesDistributionOutput,
)


def _request(**kwargs) -> AgentRequest:
    defaults = dict(
        instruction="Where do X live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
    )
    defaults.update(kwargs)
    return AgentRequest(**defaults)


def test_known_species_returns_completed_with_points() -> None:
    worker = SpeciesDistributionMock()
    result = worker.run(_request(species_name="Loxodonta africana"))

    assert result.status is AgentStatus.COMPLETED
    assert isinstance(result.output, SpeciesDistributionOutput)
    assert result.output.coordinates, "coordinates must not be empty"
    assert result.observation_count and result.observation_count > 0
    # ``map_url`` is either a real folium file:// URL (folium installed)
    # or the placeholder https://maps.umbrella.local/... URL (offline).
    assert result.map_url and (
        result.map_url.startswith("file://")
        or result.map_url.startswith("https://maps.umbrella.local/")
    )
    assert result.confidence is not None
    assert "Species Distribution Agent" in result.source_agents


def test_unknown_species_returns_failed() -> None:
    worker = SpeciesDistributionMock()
    result = worker.run(_request(species_name="Draco magicus"))
    assert result.status is AgentStatus.FAILED


def test_missing_species_returns_failed() -> None:
    worker = SpeciesDistributionMock()
    result = worker.run(_request())
    assert result.status is AgentStatus.FAILED


def test_confidence_scales_with_observation_count() -> None:
    """The mapping is coarse but monotonic - larger counts must never
    produce a lower confidence."""

    worker = SpeciesDistributionMock()
    results = [
        worker.run(_request(species_name=n))
        for n in ("Loxodonta africana", "Ursus maritimus", "Panthera tigris")
    ]
    confidences = sorted(
        (r.observation_count or 0, r.confidence or 0.0) for r in results
    )
    for i in range(1, len(confidences)):
        assert confidences[i][1] >= confidences[i - 1][1]

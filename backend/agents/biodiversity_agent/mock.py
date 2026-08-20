"""Legacy top-level mock kept for backwards compatibility.

The real Sprint 2 mocks live per-worker under ``workers/<skill>/mock.py``.
This module simply forwards to the Species Distribution mock when called
without a ``feature`` field, so any code path that instantiated the old
``BiodiversityMock`` keeps working.
"""

from __future__ import annotations

from .schema import AgentRequest, AgentResult, AgentStatus, BiodiversityFeature
from .workers.species_distribution.mock import SpeciesDistributionMock
from .workers.habitat.mock import HabitatMock
from .workers.hotspots.mock import HotspotsMock
from .workers.migration.mock import MigrationMock


class BiodiversityMock:
    """Backwards-compatible entrypoint.

    Prefer using ``orchestrator.biodiversity_orchestrator.BiodiversityOrchestrator``
    directly. This class exists so the pre-Sprint 2 call sites that did
    ``BiodiversityMock().run(request)`` still work.
    """

    def __init__(self) -> None:
        self._workers = {
            BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
            BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
            BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsMock(),
            BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
        }

    def run(self, request: AgentRequest) -> AgentResult:
        species = request.species_name or request.context.get("species")
        if species is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="Species not specified.",
            )

        feature = request.feature or BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value
        try:
            feature_enum = BiodiversityFeature(feature)
        except ValueError:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"Unknown biodiversity feature: {feature}",
            )

        request.species_name = species
        return self._workers[feature_enum].run(request)

"""Stub for the real GBIF-backed Species Distribution worker.

This file is intentionally not called during Sprint 2 - the orchestrator
depends on the ``SpeciesDistributionMock``. The class below documents
the contract and the intended GBIF pipeline so the Sprint 3 hand-over
is unambiguous.
"""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


class SpeciesDistributionWorker:
    """Real worker - to be implemented in Sprint 3.

    Steps:
    1. Normalize ``request.species_name`` via the shared Qdrant taxonomy
       lookup service.
    2. Call
       ``https://api.gbif.org/v1/occurrence/search?scientificName={name}&hasCoordinate=true``
       with pagination.
    3. Deduplicate on ``(lat, lon)`` rounded to 4 decimals; drop points
       where either coordinate is out of range or equal to zero.
    4. Persist raw records into the ``occurrences`` PostgreSQL table
       (keyed on ``gbif_id``) so downstream skills can reuse them.
    5. Render a folium map, upload the HTML to the static bucket, return
       the URL.
    6. Compute confidence from ``observation_count``.
    """

    def __init__(self, gbif_client=None, qdrant_service=None, map_service=None):
        self._gbif = gbif_client
        self._qdrant = qdrant_service
        self._map = map_service

    def run(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError(
            "SpeciesDistributionWorker is a Sprint 3 deliverable. "
            "For Sprint 2, use SpeciesDistributionMock."
        )

"""Deterministic mock for the Species Distribution worker.

Sprint 2 does NOT require a real GBIF integration - the orchestrator
must be exercised against a well-behaved fake first. This mock:

- Returns FAILED for a species not in its small catalogue, mirroring
  the real "species absent from GBIF" failure mode.
- Returns a small hand-curated point map for known species, sized by
  a fake observation count so confidence math can be exercised.
- Filters out ``(0, 0)`` points and duplicates, so the tests can prove
  the cleaning contract without a real API.

Swap this class out for ``worker.SpeciesDistributionWorker`` (Sprint 3+)
without changing the orchestrator - the ``run`` signature is stable.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ...schema import AgentRequest, AgentResult, AgentStatus
from .schema import SpeciesDistributionOutput

# NOTE: ``map_renderer`` is imported lazily inside ``run()`` to avoid a
# circular import - ``orchestrator/__init__.py`` re-exports the
# orchestrator, which itself imports this mock, and eager-importing the
# renderer here would close the loop.


# A minimal fixture catalogue - just enough species to write meaningful
# tests without pretending to be GBIF. Coordinates are hand-picked to be
# on-land and roughly in-range for each species.
_FIXTURES: dict[str, list[tuple[float, float]]] = {
    "loxodonta africana": [  # African elephant
        (-1.29, 36.82),   # Nairobi NP, KE
        (-2.65, 34.83),   # Serengeti, TZ
        (-19.02, 23.44),  # Chobe, BW
        (-13.19, 27.15),  # Kafue, ZM
        (5.32, 20.06),    # Central African Republic
    ],
    "ursus maritimus": [   # Polar bear
        (78.22, 15.65),   # Svalbard
        (74.75, -94.99),  # Nunavut, CA
        (71.29, -156.79),  # Alaska
        (80.42, 58.05),   # Franz Josef Land, RU
    ],
    "panthera tigris": [   # Tiger
        (28.02, 79.71),   # Corbett, IN
        (21.32, 79.65),   # Pench, IN
        (13.35, 100.99),  # Thailand
        (11.02, 76.75),   # Nilgiri Biosphere, IN
    ],
    "canis lupus": [       # Wolf
        (46.83, -110.72),
        (60.13, 25.05),
        (48.95, 15.10),
        (52.13, -117.65),
    ],
}


class SpeciesDistributionMock:
    """Deterministic stand-in for the real GBIF-backed worker."""

    def run(self, request: AgentRequest) -> AgentResult:
        species = self._resolve_species_name(request)
        if not species:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_name is required",
                source_agents=["Species Distribution Agent"],
            )

        raw = _FIXTURES.get(species.lower())
        if raw is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"No occurrences found for '{species}' in the mock catalogue.",
                source_agents=["Species Distribution Agent"],
            )

        cleaned = self._clean_coordinates(raw)
        if not cleaned:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"All occurrences of '{species}' had invalid coordinates.",
                source_agents=["Species Distribution Agent"],
            )

        # Multiply by a stable per-species factor so tests get deterministic
        # observation counts without hard-coding numbers in the fixtures.
        multiplier = self._deterministic_multiplier(species)
        observation_count = len(cleaned) * multiplier

        # Try to render a real folium map. Falls back to the placeholder
        # URL if folium is missing or rendering fails - the contract stays
        # the same either way. Lazy import breaks a circular dependency
        # (see module docstring at top of file).
        try:
            from ...orchestrator.services.map_renderer import render_point_map

            map_url = render_point_map(species, cleaned)
        except Exception:
            map_url = self._fake_map_url(species)

        payload = SpeciesDistributionOutput(
            species_name=species,
            coordinates=cleaned,
            observation_count=observation_count,
            map_url=map_url,
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=payload.map_url,
            observation_count=observation_count,
            confidence=self._confidence(observation_count),
            source_agents=["Species Distribution Agent"],
        )

    # ---------- internal helpers ----------

    @staticmethod
    def _resolve_species_name(request: AgentRequest) -> str | None:
        if request.species_name:
            return request.species_name
        return request.context.get("species") or request.context.get("species_name")

    @staticmethod
    def _clean_coordinates(
        raw: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        seen: set[tuple[float, float]] = set()
        cleaned: list[tuple[float, float]] = []
        for lat, lon in raw:
            if (lat, lon) == (0.0, 0.0):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            key = (round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append((lat, lon))
        return cleaned

    @staticmethod
    def _deterministic_multiplier(species: str) -> int:
        # Use a hash so the same species always yields the same fake count.
        digest = hashlib.md5(species.lower().encode("utf-8")).digest()
        return 20 + (digest[0] % 200)  # between 20 and 219

    @staticmethod
    def _fake_map_url(species: str) -> str:
        slug = species.lower().replace(" ", "_")
        return f"https://maps.umbrella.local/species/{slug}.html"

    @staticmethod
    def _confidence(observation_count: int) -> float:
        if observation_count >= 1000:
            return 0.95
        if observation_count >= 100:
            return 0.80
        if observation_count >= 10:
            return 0.60
        return 0.30

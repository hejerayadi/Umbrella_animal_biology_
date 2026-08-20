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
# tests without pretending to be GBIF. Each entry is a list of dicts so
# the renderer can populate rich popups (common name, country, region,
# IUCN status). Real Sprint 3 records will have the same shape once GBIF
# + IUCN are wired in.
_FIXTURES: dict[str, dict] = {
    "loxodonta africana": {
        "common_name": "African elephant",
        "conservation_status": "Endangered",
        "observations": [
            {"lat": -1.29,  "lon": 36.82, "country": "Kenya",       "region": "Nairobi National Park",  "year": 2024},
            {"lat": -2.65,  "lon": 34.83, "country": "Tanzania",    "region": "Serengeti NP",           "year": 2023},
            {"lat": -19.02, "lon": 23.44, "country": "Botswana",    "region": "Chobe NP",               "year": 2024},
            {"lat": -13.19, "lon": 27.15, "country": "Zambia",      "region": "Kafue NP",               "year": 2022},
            {"lat":  5.32,  "lon": 20.06, "country": "CAR",         "region": "Bamingui-Bangoran",      "year": 2023},
        ],
    },
    "ursus maritimus": {
        "common_name": "Polar bear",
        "conservation_status": "Vulnerable",
        "observations": [
            {"lat": 78.22, "lon":  15.65, "country": "Norway",      "region": "Svalbard",              "year": 2024},
            {"lat": 74.75, "lon": -94.99, "country": "Canada",      "region": "Nunavut",               "year": 2023},
            {"lat": 71.29, "lon": -156.79,"country": "USA",         "region": "Alaska",                "year": 2024},
            {"lat": 80.42, "lon":  58.05, "country": "Russia",      "region": "Franz Josef Land",      "year": 2022},
        ],
    },
    "panthera tigris": {
        "common_name": "Tiger",
        "conservation_status": "Endangered",
        "observations": [
            {"lat": 28.02, "lon":  79.71, "country": "India",       "region": "Corbett NP",            "year": 2024},
            {"lat": 21.32, "lon":  79.65, "country": "India",       "region": "Pench NP",              "year": 2023},
            {"lat": 13.35, "lon": 100.99, "country": "Thailand",    "region": "Khao Ang Rue Nai",      "year": 2023},
            {"lat": 11.02, "lon":  76.75, "country": "India",       "region": "Nilgiri Biosphere",     "year": 2024},
        ],
    },
    "canis lupus": {
        "common_name": "Gray wolf",
        "conservation_status": "Least Concern",
        "observations": [
            {"lat": 46.83, "lon": -110.72, "country": "USA",        "region": "Yellowstone",           "year": 2024},
            {"lat": 60.13, "lon":   25.05, "country": "Finland",    "region": "Nuuksio NP",            "year": 2023},
            {"lat": 48.95, "lon":   15.10, "country": "Austria",    "region": "Thayatal NP",           "year": 2023},
            {"lat": 52.13, "lon": -117.65, "country": "Canada",     "region": "Jasper NP",             "year": 2024},
        ],
    },
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

        record = _FIXTURES.get(species.lower())
        if record is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"No occurrences found for '{species}' in the mock catalogue.",
                source_agents=["Species Distribution Agent"],
            )

        raw_coords = [(o["lat"], o["lon"]) for o in record["observations"]]
        cleaned, kept_indices = self._clean_coordinates_indexed(raw_coords)
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
        confidence = self._confidence(observation_count)

        # Build per-point metadata (common_name, country, region, status,
        # year) so the renderer can populate rich popups. Any observation
        # dropped by the cleaner is skipped here too via ``kept_indices``.
        kept_obs = [record["observations"][i] for i in kept_indices]
        metadata = [
            {
                "common_name":         record.get("common_name", ""),
                "conservation_status": record.get("conservation_status", ""),
                "country":             o.get("country", ""),
                "region":              o.get("region", ""),
                "year":                o.get("year"),
            }
            for o in kept_obs
        ]

        # Try to render a real folium map. Falls back to the placeholder
        # URL if folium is missing or rendering fails - the contract stays
        # the same either way. Lazy import breaks a circular dependency
        # (see module docstring at top of file).
        try:
            from ...orchestrator.services.map_renderer import render_point_map

            map_url = render_point_map(
                species,
                cleaned,
                metadata=metadata,
                confidence=confidence,
            )
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
            confidence=confidence,
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
        """Backwards-compatible wrapper - returns just the cleaned coords."""

        cleaned, _ = SpeciesDistributionMock._clean_coordinates_indexed(raw)
        return cleaned

    @staticmethod
    def _clean_coordinates_indexed(
        raw: list[tuple[float, float]],
    ) -> tuple[list[tuple[float, float]], list[int]]:
        """Same cleaning as ``_clean_coordinates`` but also returns the
        index into ``raw`` for each kept point. Callers use those indices
        to line the cleaned coords up with their metadata."""

        seen: set[tuple[float, float]] = set()
        cleaned: list[tuple[float, float]] = []
        kept_indices: list[int] = []
        for idx, (lat, lon) in enumerate(raw):
            if (lat, lon) == (0.0, 0.0):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            key = (round(lat, 4), round(lon, 4))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append((lat, lon))
            kept_indices.append(idx)
        return cleaned, kept_indices

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

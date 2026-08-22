"""Deterministic mock for the Habitat Visualization worker (Sprint 2 stub)."""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


_HABITATS: dict[str, dict] = {
    "loxodonta africana": {
        "habitat_regions": ["Sub-Saharan Africa savannah", "Central African forests"],
        "conservation_status": "Endangered",
    },
    "ursus maritimus": {
        "habitat_regions": ["Arctic sea ice", "Coastal Greenland", "Northern Canada"],
        "conservation_status": "Vulnerable",
    },
    "panthera tigris": {
        "habitat_regions": ["Indian subcontinent", "Southeast Asian forests"],
        "conservation_status": "Endangered",
    },
    "canis lupus": {
        "habitat_regions": ["North American boreal", "Eurasian forests", "Tundra"],
        "conservation_status": "Least Concern",
    },
}


class HabitatMock:
    def run(self, request: AgentRequest) -> AgentResult:
        species = request.species_name or request.context.get("species")
        if not species:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_name is required",
                source_agents=["Habitat Visualization Agent"],
            )

        record = _HABITATS.get(species.lower())
        if record is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"No habitat record for '{species}'.",
                source_agents=["Habitat Visualization Agent"],
            )

        payload = {
            "species_name": species,
            **record,
            "map_url": f"https://maps.umbrella.local/habitat/{species.lower().replace(' ', '_')}.html",
            "climate_included": request.include_climate_data,
        }
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=payload["map_url"],
            source_agents=["Habitat Visualization Agent"],
        )

"""Deterministic mock for the Habitat Visualization worker (Sprint 2 stub).

Sprint 2 scope: return a hand-curated set of habitat regions per species,
rendered as a polygon overlay via the shared
``services.map_renderer.render_habitat_map``. Sprint 3 will replace the
rectangular ``bounds`` below with real GeoJSON polygons pulled from
IUCN / WorldClim.

Ouissale owns the real Sprint 3 implementation. Everything that calls
into this file today (orchestrator, dashboard, tests) will keep working
because the return shape is unchanged.
"""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


# Each region is ``[[sw_lat, sw_lon], [ne_lat, ne_lon]]`` — the renderer
# draws a rectangle overlay per region. Real Sprint 3 fixtures should
# swap these for full GeoJSON polygons.
_HABITATS: dict[str, dict] = {
    "loxodonta africana": {
        "common_name": "African elephant",
        "conservation_status": "Endangered",
        "regions": [
            {"name": "Sub-Saharan savannah", "bounds": [[-20, 10], [10, 40]],
             "description": "Grasslands and mixed woodlands across East and Southern Africa."},
            {"name": "Central African forests", "bounds": [[-5, 12], [8, 30]],
             "description": "Dense equatorial forest habitat."},
        ],
    },
    "ursus maritimus": {
        "common_name": "Polar bear",
        "conservation_status": "Vulnerable",
        "regions": [
            {"name": "Arctic sea ice",     "bounds": [[70, -30], [85, 40]],
             "description": "Pack-ice habitat critical for seal hunting."},
            {"name": "Northern Canada",     "bounds": [[60, -140], [78, -60]],
             "description": "Coastal denning grounds."},
        ],
    },
    "panthera tigris": {
        "common_name": "Tiger",
        "conservation_status": "Endangered",
        "regions": [
            {"name": "Indian subcontinent", "bounds": [[8, 68], [30, 92]],
             "description": "Reserves across India, Nepal, Bhutan."},
            {"name": "Southeast Asian forests", "bounds": [[0, 95], [22, 110]],
             "description": "Tropical forest strongholds."},
        ],
    },
    "canis lupus": {
        "common_name": "Gray wolf",
        "conservation_status": "Least Concern",
        "regions": [
            {"name": "North American boreal", "bounds": [[45, -140], [70, -55]],
             "description": "Widely distributed across Canada and Alaska."},
            {"name": "Eurasian forests",       "bounds": [[40, 5], [70, 140]],
             "description": "Recovering populations across Europe and Russia."},
        ],
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

        # Lazy import breaks the circular dep (orchestrator/__init__ -> workers).
        try:
            from ...orchestrator.services.map_renderer import render_habitat_map

            map_url = render_habitat_map(
                species,
                regions=record["regions"],
                conservation_status=record["conservation_status"],
            )
        except Exception:
            slug = species.lower().replace(" ", "_")
            map_url = f"https://maps.umbrella.local/habitat/{slug}.html"

        payload = {
            "species_name":         species,
            "common_name":          record["common_name"],
            "conservation_status":  record["conservation_status"],
            "habitat_regions":      [r["name"] for r in record["regions"]],
            "map_url":              map_url,
            "climate_included":     request.include_climate_data,
        }
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=map_url,
            source_agents=["Habitat Visualization Agent"],
        )

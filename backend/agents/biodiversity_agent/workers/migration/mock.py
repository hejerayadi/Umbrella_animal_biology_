"""Deterministic mock for the Migration Analysis worker (Sprint 2 stub)."""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


_ROUTES: dict[str, dict] = {
    "sterna paradisaea": {  # Arctic tern
        "route": [
            (71.29, -156.79),  # Alaska
            (65.00, -18.00),   # Iceland
            (0.00, -25.00),    # Mid-Atlantic
            (-45.00, -60.00),  # South Atlantic
            (-70.00, 0.00),    # Antarctic circle
        ],
        "seasonal_pattern": "Annual pole-to-pole migration",
    },
    "loxodonta africana": {
        "route": [
            (-19.02, 23.44),
            (-18.35, 24.10),
            (-17.85, 24.75),
        ],
        "seasonal_pattern": "Seasonal movement between Chobe wet and dry ranges",
    },
}


class MigrationMock:
    def run(self, request: AgentRequest) -> AgentResult:
        species = request.species_name or request.context.get("species")
        if not species:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_name is required",
                source_agents=["Migration Analysis Agent"],
            )

        record = _ROUTES.get(species.lower())
        if record is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"No migration record for '{species}'.",
                source_agents=["Migration Analysis Agent"],
            )

        payload = {
            "species_name": species,
            **record,
            "map_url": f"https://maps.umbrella.local/migration/{species.lower().replace(' ', '_')}.html",
        }
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=payload["map_url"],
            migration_route=record["route"],
            source_agents=["Migration Analysis Agent"],
        )

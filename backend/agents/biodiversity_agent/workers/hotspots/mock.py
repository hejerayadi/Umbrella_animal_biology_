"""Deterministic mock for the Biodiversity Hotspots worker (Sprint 2 stub)."""

from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


_HOTSPOTS_BY_REGION: dict[str, list[dict]] = {
    "global": [
        {"name": "Amazon Rainforest",   "center": (-3.47, -62.22), "species_richness": 987},
        {"name": "Congo Basin",         "center": (-0.72, 23.66),  "species_richness": 812},
        {"name": "Southeast Asian arc", "center": (2.16, 111.74),  "species_richness": 776},
        {"name": "Madagascar",          "center": (-19.55, 46.63), "species_richness": 634},
    ],
    "africa": [
        {"name": "Congo Basin",  "center": (-0.72, 23.66),  "species_richness": 812},
        {"name": "Madagascar",   "center": (-19.55, 46.63), "species_richness": 634},
        {"name": "Serengeti",    "center": (-2.65, 34.83),  "species_richness": 508},
    ],
    "asia": [
        {"name": "Southeast Asian arc", "center": (2.16, 111.74),  "species_richness": 776},
        {"name": "Western Ghats",       "center": (11.02, 76.75),  "species_richness": 481},
    ],
}


class HotspotsMock:
    def run(self, request: AgentRequest) -> AgentResult:
        region = (request.region or "global").lower()
        hotspots = _HOTSPOTS_BY_REGION.get(region)
        if hotspots is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"No hotspot data for region '{region}'.",
                source_agents=["Biodiversity Hotspots Agent"],
            )

        payload = {
            "region": region,
            "hotspots": hotspots,
            "heatmap_url": f"https://maps.umbrella.local/hotspots/{region}.html",
        }
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=payload["heatmap_url"],
            hotspots=hotspots,
            source_agents=["Biodiversity Hotspots Agent"],
        )

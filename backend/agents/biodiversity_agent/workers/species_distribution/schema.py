"""Typed input/output payloads for the Species Distribution worker.

These are thin wrappers over the shared ``AgentRequest`` / ``AgentResult``
so the worker can accept the standard contract but the orchestrator
still gets strongly-typed fields when it introspects the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeciesDistributionOutput:
    """Payload placed inside ``AgentResult.output`` by the worker."""

    species_name: str
    coordinates: list[tuple[float, float]] = field(default_factory=list)
    observation_count: int = 0
    map_url: str | None = None

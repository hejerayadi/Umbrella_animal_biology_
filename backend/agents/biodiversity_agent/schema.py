"""Shared data contracts for the Biodiversity Agent domain.

This module keeps compatibility with the platform-wide standard
(``AgentRequest`` / ``AgentResult`` / ``AgentStatus``) while extending
the result with biodiversity-specific fields the four worker agents
produce: ``map_url``, ``hotspots``, ``migration_route`` and
``observation_count``.

The extended fields default to ``None`` so a worker that does not use
one of them simply omits it - the contract stays backwards compatible
with the minimal Global Orchestrator contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(Enum):
    """Universal execution status used across every agent in the platform."""

    COMPLETED = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE = "continue"
    FAILED = "failed"


class BiodiversityFeature(str, Enum):
    """The four skills the Biodiversity Agent knows how to route to.

    The Global Orchestrator sets this field on the ``AgentRequest`` so the
    Biodiversity Orchestrator can dispatch without re-parsing the prompt.
    """

    SPECIES_DISTRIBUTION_MAP = "species_distribution_map"
    HABITAT_VISUALIZATION = "habitat_visualization"
    BIODIVERSITY_HOTSPOTS = "biodiversity_hotspots"
    MIGRATION_ANALYSIS = "migration_analysis"


@dataclass
class AgentRequest:
    """Standard input contract for any agent, extended for biodiversity.

    ``instruction`` and ``context`` are the minimal platform contract.
    The remaining fields are optional biodiversity hints - when absent
    the orchestrator infers them from ``context`` or falls back to
    defaults.
    """

    instruction: str
    context: dict[str, Any] = field(default_factory=dict)

    # Biodiversity-specific optional hints
    feature: str | None = None
    species_name: str | None = None
    region: str = "global"
    time_period: tuple[int, int] | None = None
    include_climate_data: bool = False
    session_id: str | None = None


@dataclass
class AgentResult:
    """Standard output contract, extended with biodiversity payload fields.

    The ``output`` field remains the general-purpose payload so this
    result is a drop-in replacement for the Global Orchestrator's
    expected shape. ``map_url``, ``hotspots``, ``migration_route`` and
    ``observation_count`` are biodiversity-specific conveniences that
    frontends can read directly without unpacking ``output``.
    """

    status: AgentStatus

    # Escalation fields — populated when status == NEEDS_AGENT
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None

    # Universal free-form payload
    output: Any | None = None

    # Biodiversity-specific structured fields
    map_url: str | None = None
    hotspots: list[Any] | None = None
    migration_route: list[tuple[float, float]] | None = None
    observation_count: int | None = None

    # Provenance / trust
    confidence: float | None = None
    source_agents: list[str] = field(default_factory=list)

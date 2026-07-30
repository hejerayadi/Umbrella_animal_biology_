"""Static catalog of worker agents.

`AGENT_CARDS` is built only from the stable `card.json` metadata files - no
Python code from `backend/agents/` is imported for that part, so it can
never be affected by in-progress changes to the agents' dataclasses. The
Planner and Capability Resolver use this to know what agents exist and pick
between them.

`AGENT_REGISTRY` maps each agent name to a real, runnable instance (the mock
worker for now) so LangGraph can actually call `agent.run(request)`. Swapping
a mock for a real implementation later only means changing the import and
instantiation below - nothing else in the orchestrator needs to change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .agent_card import AgentCard
from .agents.Literature_Agent import LiteratureMock
from .agents.Protein_visualization import ProteinMock
from .agents.biodiversity_agent import BiodiversityMock
from .agents.evolution_agent import EvolutionMock
from .agents.genome_agent import GenomeMock
from .agents.multimodal_recognition_agent import MultimodalMock
from .agents.reconstruction_agent import ReconstructionMock
from .agents.trait_discovery_agent import TraitMock

_AGENTS_DIR = Path(__file__).parent / "agents"

# Maps the orchestrator-facing agent name to its folder under backend/agents/.
_AGENT_FOLDERS: dict[str, str] = {
    "Genome": "genome_agent",
    "Evolution": "evolution_agent",
    "Biodiversity": "biodiversity_agent",
    "Literature": "Literature_Agent",
    "Multimodal": "multimodal_recognition_agent",
    "Reconstruction": "reconstruction_agent",
    "Trait": "trait_discovery_agent",
    "Protein": "Protein_visualization",
}


class WorkerAgent(Protocol):
    """Structural interface every worker agent satisfies.

    Deliberately untyped beyond `run` - each agent still defines its own
    local `AgentRequest`/`AgentResult` classes, so the orchestrator treats
    them by duck typing instead of depending on one shared contract.
    """

    def run(self, request: Any) -> Any: ...


def _load_card(folder_name: str) -> AgentCard:
    card_path = _AGENTS_DIR / folder_name / "card.json"
    data = json.loads(card_path.read_text(encoding="utf-8"))
    return AgentCard(
        name=data["name"],
        description=data["description"],
        capabilities=list(data.get("capabilities", [])),
        required_inputs=list(data.get("input", {}).keys()),
        produced_outputs=list(data.get("output", {}).keys()),
    )


AGENT_CARDS: dict[str, AgentCard] = {
    agent_name: _load_card(folder_name) for agent_name, folder_name in _AGENT_FOLDERS.items()
}

AGENT_REGISTRY: dict[str, WorkerAgent] = {
    "Genome": GenomeMock(),
    "Evolution": EvolutionMock(),
    "Biodiversity": BiodiversityMock(),
    "Literature": LiteratureMock(),
    "Multimodal": MultimodalMock(),
    "Reconstruction": ReconstructionMock(),
    "Trait": TraitMock(),
    "Protein": ProteinMock(),
}

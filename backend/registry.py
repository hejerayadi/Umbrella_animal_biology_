"""Static catalog of worker agents.

`AGENT_CARDS` is built only from the stable `card.json` metadata files. The
Planner and Capability Resolver use it to know what agents exist and to pick
between them.

`AGENT_ENDPOINTS` maps each agent name to the base URL of its HTTP service.
Agents are independent services reached over `POST /execute`, so this module
imports no code from `backend/agents/` at all - a broken or half-finished
agent package can no longer stop the orchestrator from starting.

Every URL can be overridden with an environment variable, so the same code
runs against local uvicorn processes, containers, or deployed services
without edits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .agent_card import AgentCard

_AGENTS_DIR = Path(__file__).parent / "agents"

# Maps the orchestrator-facing agent name to its folder under backend/agents/
# (where its card.json lives), plus the port its API listens on by default.
_AGENT_FOLDERS: dict[str, str] = {
    "Genome": "genome_agent",
    "Evolution": "evolution_agent",
    "Biodiversity": "biodiversity_agent",
    "Literature": "Literature_Agent",
    "Multimodal": "multimodal_recognition_agent",
    "Reconstruction": "reconstruction_agent",
    "Trait": "trait_discovery_agent",
    "Protein": "Protein_visualization",
    "ImageGeneration": "image_generation_agent",
}

# Default local port per agent. Port 8000 is reserved for backend/api.py.
_AGENT_PORTS: dict[str, int] = {
    "Genome": 8001,
    "Evolution": 8002,
    "Biodiversity": 8003,
    "Literature": 8004,
    "Multimodal": 8005,
    "Reconstruction": 8006,
    "Trait": 8007,
    "Protein": 8008,
    "ImageGeneration": 8009,
}


def _endpoint(agent_name: str, port: int) -> str:
    """Base URL for one agent, overridable per agent via the environment.

    e.g. "Genome" -> GENOME_AGENT_URL, "ImageGeneration" -> IMAGEGENERATION_AGENT_URL
    """
    return os.getenv(f"{agent_name.upper()}_AGENT_URL", f"http://localhost:{port}")


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

AGENT_ENDPOINTS: dict[str, str] = {
    agent_name: _endpoint(agent_name, port) for agent_name, port in _AGENT_PORTS.items()
}

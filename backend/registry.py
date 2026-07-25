from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agent_card import AgentCard
from .agents.Literature_Agent.card import AGENT_CARD as LITERATURE_CARD
from .agents.Literature_Agent.mock import execute as literature_execute
from .agents.Protein_visualization.card import AGENT_CARD as PROTEIN_CARD
from .agents.Protein_visualization.mock import execute as protein_execute
from .agents.biodiversity_agent.card import AGENT_CARD as BIODIVERSITY_CARD
from .agents.biodiversity_agent.mock import execute as biodiversity_execute
from .agents.evolution_agent.card import AGENT_CARD as EVOLUTION_CARD
from .agents.evolution_agent.mock import execute as evolution_execute
from .agents.genome_agent.card import AGENT_CARD as GENOME_CARD
from .agents.genome_agent.mock import execute as genome_execute
from .agents.multimodal_recognition_agent.card import AGENT_CARD as MULTIMODAL_CARD
from .agents.multimodal_recognition_agent.mock import execute as multimodal_execute
from .agents.reconstruction_agent.card import AGENT_CARD as RECONSTRUCTION_CARD
from .agents.reconstruction_agent.mock import execute as reconstruction_execute
from .agents.trait_discovery_agent.card import AGENT_CARD as TRAIT_CARD
from .agents.trait_discovery_agent.mock import execute as trait_execute


@dataclass(frozen=True)
class AgentHandle:
    card: AgentCard
    mock_execute: Callable[[object], object]
    depends_on: tuple[str, ...] = ()


AGENT_REGISTRY: dict[str, AgentHandle] = {
    "Genome": AgentHandle(card=GENOME_CARD, mock_execute=genome_execute),
    "Evolution": AgentHandle(card=EVOLUTION_CARD, mock_execute=evolution_execute, depends_on=("Genome",)),
    "Biodiversity": AgentHandle(card=BIODIVERSITY_CARD, mock_execute=biodiversity_execute),
    "Literature": AgentHandle(card=LITERATURE_CARD, mock_execute=literature_execute),
    "Multimodal": AgentHandle(card=MULTIMODAL_CARD, mock_execute=multimodal_execute),
    "Reconstruction": AgentHandle(card=RECONSTRUCTION_CARD, mock_execute=reconstruction_execute, depends_on=("Genome",)),
    "Trait": AgentHandle(card=TRAIT_CARD, mock_execute=trait_execute, depends_on=("Genome",)),
    "Protein": AgentHandle(card=PROTEIN_CARD, mock_execute=protein_execute, depends_on=("Trait",)),
}
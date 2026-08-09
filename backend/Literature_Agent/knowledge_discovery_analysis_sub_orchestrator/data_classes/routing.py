"""
Structures de données liées au routage.
Ce sont les objets échangés entre l'orchestrateur et les agents enfants
lorsqu'une décision de délégation est prise.
"""

from pydantic import BaseModel, Field


class AgentCall(BaseModel):
    """
    Représente UN appel décidé par l'orchestrateur vers un agent enfant.
    Correspond aux arguments extraits d'un function_call du LLM.
    """
    agent_name: str = Field(..., description="Nom de l'agent ciblé, ex: 'scientific_analysis'")
    capability: str = Field(..., description="Capacité précise demandée, ex: 'gap_detection'")
    query: str = Field(..., description="Requête reformulée transmise à l'agent")


class RoutingDecision(BaseModel):
    """
    Résultat complet d'une décision de routage prise par l'orchestrateur.
    Peut contenir plusieurs AgentCall si l'orchestrateur décide
    d'appeler plusieurs agents (exécution parallèle).
    """
    calls: list[AgentCall] = Field(
        default_factory=list,
        description="Liste des agents à appeler pour cette requête"
    )
    reasoning: str | None = Field(
        default=None,
        description="Raisonnement du LLM ayant mené à cette décision (debug/traçabilité)"
    )

from pydantic import BaseModel, Field


class AgentCall(BaseModel):
    """Un appel décidé par le LLM vers un agent enfant, avec sa capacité et sa requête."""
    agent_name: str = Field(..., description="Nom de l'agent ciblé, ex: 'scientific_analysis'")
    capability: str = Field(..., description="Capacité précise demandée, ex: 'gap_detection'")
    query: str = Field(..., description="Requête reformulée transmise à l'agent")


class RoutingDecision(BaseModel):
    """
    Résultat du routage. L'ORDRE de la liste `calls` est l'ordre d'exécution
    voulu par le LLM (le graphe l'exécute séquentiellement dans cet ordre).
    """
    calls: list[AgentCall] = Field(default_factory=list)

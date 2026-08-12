from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Réponse standardisée renvoyée par un agent enfant vers l'orchestrateur."""
    agent_name: str
    capability: str
    result: str
    success: bool = True
    error_message: str | None = None


class AggregatedResponse(BaseModel):
    """Réponse finale après agrégation séquentielle des AgentResponse."""
    query: str
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    final_answer: str

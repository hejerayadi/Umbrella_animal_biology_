"""
Schemas d'entree et de sortie de l'agent Scientific Analysis.
Garantit un contrat fixe, quelle que soit la maniere dont le LLM
a raisonne en interne (tools appeles, nombre d'iterations, etc.)
"""
from pydantic import BaseModel, Field, field_validator


class AgentInput(BaseModel):
    """Ce que l'agent recoit en entree."""
    query: str = Field(..., description="La question ou le claim a analyser")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("La query ne peut pas etre vide.")
        return v.strip()


class SourceRef(BaseModel):
    """Reference a une source utilisee pour construire la reponse."""
    id: str
    source: str = "unknown"
    score: float | None = None


class AgentOutput(BaseModel):
    """Ce que l'agent retourne, toujours sous cette forme."""
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    status: str = Field(default="ok", description="'ok', 'insufficient_info', ou 'error'")
    error_message: str | None = None

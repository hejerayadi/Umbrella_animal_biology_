"""
Structures de données liées aux réponses produites par les agents enfants.
Ce sont les objets que l'orchestrateur reçoit et doit agréger.
"""

from pydantic import BaseModel, Field


class ScientificClaim(BaseModel):
    """
    Une affirmation scientifique extraite d'une source,
    utilisée notamment par Scientific Analysis (QA, contradiction, gap).
    """
    text: str = Field(..., description="Le contenu de l'affirmation")
    source: str = Field(..., description="Identifiant ou titre de la source (paper, DOI...)")
    confidence: float = Field(default=1.0, description="Confiance du modèle dans cette extraction")


class AgentResponse(BaseModel):
    """
    Réponse standardisée renvoyée par n'importe quel agent enfant
    vers l'orchestrateur. Tous les agents doivent respecter ce format
    pour que l'agrégation soit fiable.
    """
    agent_name: str = Field(..., description="Nom de l'agent ayant produit la réponse")
    capability: str = Field(..., description="Capacité exécutée")
    result: str = Field(..., description="Résultat textuel principal à présenter à l'utilisateur")
    claims: list[ScientificClaim] = Field(
        default_factory=list,
        description="Affirmations/sources utilisées pour produire le résultat"
    )
    success: bool = Field(default=True, description="False si l'agent a rencontré une erreur")
    error_message: str | None = Field(default=None, description="Détail de l'erreur si success=False")


class AggregatedResponse(BaseModel):
    """
    Réponse finale renvoyée à l'utilisateur, après agrégation
    d'une ou plusieurs AgentResponse par l'orchestrateur.
    """
    query: str = Field(..., description="Question originale de l'utilisateur")
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    final_answer: str = Field(..., description="Synthèse finale présentée à l'utilisateur")

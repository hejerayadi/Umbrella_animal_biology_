"""Public API contracts."""

from app.contracts.agent_result import AgentResult, AgentStatus
from app.contracts.protein_request import AgentTask, ProteinAnalysisRequest
from app.contracts.protein_response import AnalysisAccepted, ProteinAnalysisResponse

__all__ = [
    "AgentResult",
    "AgentStatus",
    "AgentTask",
    "AnalysisAccepted",
    "ProteinAnalysisRequest",
    "ProteinAnalysisResponse",
]

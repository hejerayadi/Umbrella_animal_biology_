import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from backend.agents.Protein_visualization.app.api.v1.dependencies import (
    get_analysis_repository,
    get_orchestrator,
)
from backend.agents.Protein_visualization.app.contracts.agent_result import AgentResult
from backend.agents.Protein_visualization.app.contracts.envelope import ApiResponse, success
from backend.agents.Protein_visualization.app.contracts.protein_request import ProteinAnalysisRequest
from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus
from backend.agents.Protein_visualization.app.observability.context import log_context
from backend.agents.Protein_visualization.app.observability.logging import log_stage
from backend.agents.Protein_visualization.app.observability.metrics import metrics
from backend.agents.Protein_visualization.app.orchestrators.protein.orchestrator import ProteinOrchestrator

router = APIRouter(prefix="/protein-structure-analyses", tags=["analyses"])

logger = logging.getLogger("app.analyses")


@router.post(
    "",
    response_model=ApiResponse[AgentResult],
    status_code=status.HTTP_201_CREATED,
)
async def create_analysis(
    task: ProteinAnalysisRequest,
    orchestrator: Annotated[ProteinOrchestrator, Depends(get_orchestrator)],
    # ``AnalysisRepository | None``; typed loosely so SQLAlchemy stays unimported
    # while persistence is disabled.
    repository: Annotated[Any, Depends(get_analysis_repository)],
) -> ApiResponse[AgentResult]:
    metrics.increment("analyses_started")
    with log_context(task_id=str(task.task_id), trace_id=str(task.trace_id)):
        analysis_id = None
        if repository is not None:
            analysis_id, _ = await asyncio.to_thread(repository.create, task)

        with log_context(analysis_id=str(analysis_id) if analysis_id else None):
            with log_stage(logger, "protein_analysis", capability="orchestrator") as outcome:
                result = await orchestrator.execute(task)
                analysis_status = AnalysisStatus(result.output["status"])
                outcome["agent_status"] = result.status.value
                outcome["analysis_status"] = analysis_status.value
                outcome["validation_status"] = result.output["validation_status"]

            if repository is not None and analysis_id is not None:
                await asyncio.to_thread(
                    repository.update,
                    analysis_id,
                    analysis_status,
                    result.to_dict(),
                )
            metrics.increment(f"analyses_{analysis_status.value.lower()}")
            metrics.increment(f"agent_results_{result.status.value}")
            return success(
                result,
                task_id=str(task.task_id),
                analysis_id=str(result.output["analysis_id"]),
            )

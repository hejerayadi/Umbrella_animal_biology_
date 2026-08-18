from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_knowledge_base
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.contracts.envelope import ApiResponse, success
from backend.agents.Protein_visualization.app.observability.metrics import metrics
from backend.agents.Protein_visualization.app.observability.tracing import get_tracing_status

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str


class DependencyStatus(BaseModel):
    name: str
    configured: bool
    reachable: bool | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    biological_sources: Literal["real"] = "real"
    dependencies: list[DependencyStatus]


@router.get("/health", response_model=ApiResponse[HealthResponse])
async def health() -> ApiResponse[HealthResponse]:
    settings = get_settings()
    return success(
        HealthResponse(
            service=settings.app_name,
            version=settings.app_version,
            environment=settings.app_env,
        )
    )


@router.get("/ready", response_model=ApiResponse[ReadinessResponse])
async def ready() -> ApiResponse[ReadinessResponse]:
    """Reports whether the configured dependencies can serve a full analysis."""
    settings = get_settings()
    knowledge_base = get_knowledge_base()
    store = knowledge_base.store
    if store:
        qdrant_reachable, qdrant_detail = await store.readiness(knowledge_base.embedding.dimensions)
    elif settings.qdrant_url:
        qdrant_reachable = False
        qdrant_detail = knowledge_base.unavailable_reason or "Qdrant client is unavailable"
    else:
        qdrant_reachable = False
        qdrant_detail = "QDRANT_URL is not set; retrieval is disabled"
    qdrant = DependencyStatus(
        name="qdrant",
        configured=bool(settings.qdrant_url),
        reachable=qdrant_reachable if settings.qdrant_url else None,
        detail=qdrant_detail,
    )
    llm = DependencyStatus(
        name="azure_llm",
        configured=settings.llm_provider != "disabled",
        detail=None if settings.llm_provider != "disabled" else "LLM_PROVIDER=disabled",
    )
    persistence = DependencyStatus(
        name="persistence",
        configured=settings.persistence_enabled,
        detail=None if settings.persistence_enabled else "PostgreSQL is out of scope for this sprint",
    )
    # Reported, never graded: tracing is an observability aid, so a run whose
    # trace is not exported is still a complete analysis.
    status = get_tracing_status()
    tracing = DependencyStatus(
        name="langsmith",
        configured=settings.langsmith_tracing,
        detail=f"project {status.project}" if status.enabled else status.reason,
    )
    dependencies = [qdrant, llm, persistence, tracing]
    degraded = qdrant.configured and qdrant.reachable is False
    return success(
        ReadinessResponse(
            status="degraded" if degraded else "ready",
            dependencies=dependencies,
        )
    )


@router.get("/metrics", response_model=ApiResponse[dict[str, int]])
async def metric_snapshot() -> ApiResponse[dict[str, int]]:
    return success(metrics.snapshot())

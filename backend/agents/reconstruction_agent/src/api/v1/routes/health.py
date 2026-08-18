"""Liveness and readiness.

Answers whether the process is up, and - more usefully - what it is actually
configured to do. The common failure here is not a crash: it is a deployment
that starts cleanly with no EMBL-EBI contact address and then fails every
reconstruction.
"""
from __future__ import annotations

from asgi_correlation_id import correlation_id
from fastapi import APIRouter, Depends

from api.dependencies import get_app_settings, get_checkpointer, get_tool_registry
from api.v1.envelope import Envelope
from api.v1.schemas import HealthData, ServiceStatus
from configuration.settings import Settings

router = APIRouter(tags=["health"])


def _checkpoints_durable() -> bool:
    handle = get_checkpointer()
    return bool(handle and handle.durable)


@router.get("/health", response_model=Envelope[HealthData])
def health(settings: Settings = Depends(get_app_settings)) -> Envelope[HealthData]:
    """Process health plus the agent's effective capabilities."""
    return Envelope.ok(
        HealthData(
            status="ok",
            agent=settings.agent_name,
            environment=settings.app.env.value,
            tools=get_tool_registry().names,
            llm_provider=settings.llm.provider.value,
            services={
                "ncbi": ServiceStatus(
                    configured=True,
                    detail=(
                        f"{settings.ncbi.requests_per_second}/s "
                        f"({'with API key' if settings.ncbi.api_key else 'anonymous'})"
                    ),
                ),
                "embl_ebi": ServiceStatus(
                    configured=bool(settings.embl_ebi.contact_email),
                    detail=(
                        None
                        if settings.embl_ebi.contact_email
                        else "EMBL_EBI_CONTACT_EMAIL is unset; BLAST and MAFFT will refuse to run."
                    ),
                ),
                "azure": ServiceStatus(
                    configured=settings.azure.configured,
                    detail=(
                        settings.azure.openai_deployment
                        if settings.azure.configured
                        else "Azure OpenAI is not configured; planning falls back to the "
                        "deterministic pipeline."
                    ),
                ),
                "nvidia_evo2": ServiceStatus(
                    configured=settings.nvidia.configured,
                    detail=(
                        settings.nvidia.model
                        if settings.nvidia.configured
                        else "NVIDIA_API_KEY is unset; Evo 2 plausibility scoring is unavailable."
                    ),
                ),
                # The one that decides whether a long reconstruction can finish
                # at all: without durable checkpoints a CONTINUE restarts from
                # nothing and the orchestrator fails it after three retries.
                "checkpoints": ServiceStatus(
                    configured=_checkpoints_durable(),
                    detail=(
                        "Postgres (shared Umbrella database)"
                        if _checkpoints_durable()
                        else "In-memory only; CONTINUE cannot resume across calls."
                    ),
                ),
            },
        ),
        request_id=correlation_id.get(),
    )


@router.get("/health/live")
def liveness() -> dict[str, str]:
    """Bare liveness for a container probe.

    Unenveloped and dependency-free on purpose: a probe should answer even
    when the tool registry cannot be built, and it must stay cheap enough to
    poll every few seconds.
    """
    return {"status": "ok"}

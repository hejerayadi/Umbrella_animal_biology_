"""Liveness and readiness.

These answer two different questions and are kept apart deliberately.

`/health` says the process is up. `/ready` says the process can actually do its
job, which is a stronger claim. This agent needs two services for two different
jobs - NCBI for sequences, taxonomy and homology search, EMBL-EBI for alignment
- and both ask callers to identify themselves. Without a contact address for
EBI every gap reaches alignment and comes back unresolved with no obvious
cause; reporting that as healthy would hide a total loss of function behind a
green check.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, ConfigDict

from reconstruction_agent.api.dependencies import get_settings_dependency
from reconstruction_agent.api.v1.schemas.envelope import ApiResponse
from reconstruction_agent.config.settings import Settings

router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    """Whether one external service is usable, and why not if it is not."""

    model_config = ConfigDict(frozen=True)

    name: str
    configured: bool
    #: True when the agent cannot produce a reconstruction without it.
    required: bool
    detail: str = ""


class HealthData(BaseModel):
    """What the agent can currently do."""

    model_config = ConfigDict(frozen=True)

    status: str
    service: str
    environment: str
    dependencies: tuple[DependencyStatus, ...] = ()

    @property
    def ready(self) -> bool:
        return all(dep.configured for dep in self.dependencies if dep.required)


def _dependencies(settings: Settings) -> tuple[DependencyStatus, ...]:
    return (
        DependencyStatus(
            name="ncbi-blast",
            configured=settings.ncbi_blast.configured,
            required=True,
            detail=(
                ""
                if settings.ncbi_blast.configured
                else "NCBI_BLAST_CONTACT_EMAIL is unset; NCBI asks to be able to "
                "contact whoever submits searches."
            ),
        ),
        DependencyStatus(
            name="embl-ebi-mafft",
            configured=settings.embl_ebi.configured,
            required=True,
            detail=(
                ""
                if settings.embl_ebi.configured
                else "EMBL_EBI_CONTACT_EMAIL is unset; EBI rejects anonymous jobs, "
                "so alignment is unavailable and every gap comes back unresolved."
            ),
        ),
        DependencyStatus(
            name="ncbi-eutilities",
            configured=True,
            required=True,
            detail=(
                ""
                if settings.ncbi.api_key
                else "No NCBI_API_KEY: usable, but paced at 3 requests/second instead of 10."
            ),
        ),
        DependencyStatus(
            name="nvidia-evo2",
            configured=settings.nvidia.configured,
            required=False,
            detail=(
                ""
                if settings.nvidia.configured
                else "No NVIDIA_API_KEY: Evo 2 arbitration is disabled; homology "
                "and alignment evidence are unaffected."
            ),
        ),
        DependencyStatus(
            name="llm",
            configured=settings.llm_enabled,
            required=False,
            detail=(
                ""
                if settings.llm_enabled
                else "No LLM configured: the deterministic fallback plan is used."
            ),
        ),
    )


@router.get("/health", response_model=ApiResponse[HealthData])
async def health(
    settings: Settings = Depends(get_settings_dependency),
) -> ApiResponse[HealthData]:
    """Liveness. Answers 200 whenever the process is serving."""
    data = HealthData(
        status="ok",
        service=settings.app.name,
        environment=settings.app.env.value,
        dependencies=_dependencies(settings),
    )
    return ApiResponse[HealthData].ok(data)


@router.get("/ready", response_model=ApiResponse[HealthData])
async def ready(
    response: Response,
    settings: Settings = Depends(get_settings_dependency),
) -> ApiResponse[HealthData]:
    """Readiness. Answers 503 when a required dependency is missing."""
    data = HealthData(
        status="ok",
        service=settings.app.name,
        environment=settings.app.env.value,
        dependencies=_dependencies(settings),
    )
    if not data.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        data = data.model_copy(update={"status": "degraded"})
    return ApiResponse[HealthData].ok(data)

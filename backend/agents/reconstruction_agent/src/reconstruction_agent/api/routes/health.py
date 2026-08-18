"""Liveness and readiness.

`/health` answers whether the process is up. It also reports what the agent is
actually configured to do, because the common failure here is not a crash - it
is a deployment that starts cleanly with no EMBL-EBI contact address and then
fails every reconstruction.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...configuration.settings import Settings
from ..dependencies import get_app_settings, get_tool_registry
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Process health plus the agent's effective capabilities."""
    return HealthResponse(
        status="ok",
        agent=settings.agent_name,
        tools=get_tool_registry().names,
        llm_enabled=settings.llm.enabled,
        external_services_configured={
            # NCBI works anonymously at a lower rate, so a key is optional.
            "ncbi": True,
            "ncbi_api_key": bool(settings.ncbi.api_key),
            # EMBL-EBI rejects submissions without one, so BLAST and MAFFT are
            # unusable when this is False.
            "embl_ebi": bool(settings.embl_ebi.contact_email),
            "llm": settings.llm.enabled,
        },
    )

"""The FastAPI application.

`create_app()` is a factory so tests can build an isolated instance; `app` is
the module-level singleton uvicorn serves.
"""
from __future__ import annotations

from fastapi import FastAPI

from ..configuration.logging import configure_logging, get_logger
from ..configuration.settings import get_settings
from ..observability.tracing import configure_tracing
from .routes import health, reconstruction

_log = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the agent's HTTP application."""
    settings = get_settings()

    configure_logging(settings.observability)
    configure_tracing(settings.observability)

    app = FastAPI(
        title="Reconstruction Agent",
        version="0.1.0",
        description=(
            "Reconstructs unresolved regions of incomplete animal genomes from homologous "
            "reference sequence. Called by the Global Scientific Orchestrator."
        ),
    )

    app.include_router(reconstruction.router)
    app.include_router(health.router)

    if not settings.embl_ebi.contact_email:
        # Startup succeeds either way - the agent is still useful for gap
        # detection - but this is the single most common misconfiguration and
        # it otherwise only surfaces as a failed reconstruction much later.
        _log.warning(
            "EMBL_EBI_CONTACT_EMAIL is not set; BLAST and MAFFT will refuse to run. "
            "Set it in the agent's .env before expecting reconstructions."
        )

    _log.info("Reconstruction Agent ready (LLM %s).",
              "enabled" if settings.llm.enabled else "disabled")
    return app


app = create_app()

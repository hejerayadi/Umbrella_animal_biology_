"""The FastAPI application.

Two contracts are served from one process, deliberately:

- `/api/v1/...` for any client, in the `{data, meta, error}` envelope, using
  HTTP status codes properly.
- `POST /execute` at the root for the Umbrella orchestrator, which parses the
  repo-wide bare `AgentResult` and would break if it were wrapped or if it ever
  received a non-200.

Keeping both here is what lets the versioned contract evolve without touching
shared orchestrator code that eight other agents also depend on.
"""

from __future__ import annotations

from fastapi import FastAPI

from reconstruction_agent.api import exception_handlers, orchestrator
from reconstruction_agent.api.v1.router import api_router
from reconstruction_agent.config.settings import Settings, get_settings
from reconstruction_agent.observability.logger import configure_logging, get_logger

_log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level constant so tests can build an app
    from explicit settings instead of whatever `.env` happens to be on the
    developer machine.
    """
    settings = settings or get_settings()
    configure_logging(
        level=settings.observability.log_level,
        log_format=settings.effective_log_format,
    )

    app = FastAPI(
        title=settings.app.name,
        version="1.0.0",
        summary="Reconstructs unresolved regions of incomplete animal genome assemblies.",
        docs_url="/docs" if settings.app.docs_enabled else None,
        redoc_url=None,
    )

    # Endpoints read their configuration from here, so an application built
    # from explicit settings behaves the way it was built.
    app.state.settings = settings

    app.include_router(api_router)
    # Root-mounted and unenveloped: this is the orchestrator door.
    app.include_router(orchestrator.router)
    exception_handlers.register(
        app,
        request_id_header=settings.observability.request_id_header,
        trace_id_header=settings.observability.trace_id_header,
    )

    _log.info(
        "application_configured",
        environment=settings.app.env.value,
        port=settings.app.port,
        embl_ebi_configured=settings.embl_ebi.configured,
        evo2_configured=settings.nvidia.configured,
        llm_enabled=settings.llm_enabled,
    )
    return app


#: The application object `backend/run_agents.py` launches, via the `api.py`
#: shim at the agent root.
app = create_app()


def main() -> None:
    """Run the service directly, for local development."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "reconstruction_agent.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
    )


if __name__ == "__main__":
    main()

"""The FastAPI application.

`create_app()` is a factory so tests can build an isolated instance; `app` is
the module-level singleton uvicorn serves.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import shutdown as close_dependencies
from api.dependencies import startup as open_dependencies
from api.v1.envelope import Envelope, ErrorCode, ErrorDetail, error_payload
from api.v1.router import api_router, orchestrator_router
from configuration.logging import configure_logging, get_logger
from configuration.runtime import use_selector_event_loop
from configuration.settings import Settings, get_settings
from observability.tracing import configure_tracing

_log = get_logger(__name__)


# The middleware's default validator accepts only UUID4, which would discard
# any correlation id the orchestrator sets in its own format and silently break
# cross-agent tracing. This accepts an id shaped like an identifier instead -
# bounded and without control characters, so it cannot be used to forge log
# lines - and rejects anything else, which then gets a generated id.
_CORRELATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _is_safe_correlation_id(value: str) -> bool:
    return bool(_CORRELATION_ID_PATTERN.fullmatch(value))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown.

    The capability warnings live here rather than at import time so they are
    emitted once per process, after logging is configured, and appear in the
    same stream as everything else.
    """
    settings: Settings = app.state.settings

    if not settings.embl_ebi.contact_email:
        # Startup succeeds either way - the agent is still useful for gap
        # detection - but this is the single most common misconfiguration and
        # it otherwise only surfaces as a failed reconstruction much later.
        _log.warning(
            "embl_ebi_unconfigured",
            detail="EMBL_EBI_CONTACT_EMAIL is not set; BLAST and MAFFT will refuse to run.",
        )

    if settings.llm.enabled and not settings.azure.configured:
        _log.warning(
            "llm_unconfigured",
            provider=settings.llm.provider.value,
            detail="Azure credentials are incomplete; planning falls back to the "
            "deterministic pipeline.",
        )

    # Opens the checkpoint store and the audit repository. Both need a running
    # event loop, which is why they are here and not in a module-level factory.
    await open_dependencies(settings)

    if not settings.database.configured:
        # Worth saying out loud: without a checkpoint store the agent cannot
        # resume a CONTINUE, so any reconstruction too slow for one 120 s call
        # restarts from nothing on every retry and then fails.
        _log.warning(
            "checkpoints_not_durable",
            detail="No DATABASE_URL in backend/.env; CONTINUE will not resume.",
        )

    _log.info(
        "agent_ready",
        environment=settings.app.env.value,
        host=settings.app.host,
        port=settings.app.port,
        llm_provider=settings.llm.provider.value,
        log_format=settings.log_format.value,
    )
    try:
        yield
    finally:
        await close_dependencies()
        _log.info("agent_stopping")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the agent's HTTP application."""
    settings = settings or get_settings()

    # Before any loop exists: psycopg's async driver cannot run on Windows'
    # default proactor loop, and without this the checkpoint store is silently
    # unreachable.
    use_selector_event_loop()
    configure_logging(settings.observability, log_format=settings.log_format)
    configure_tracing(settings.observability)

    app = FastAPI(
        title="Reconstruction Agent",
        version="0.1.0",
        description=(
            "Reconstructs unresolved regions of incomplete animal genomes from homologous "
            "reference sequence. Called by the Global Scientific Orchestrator."
        ),
        lifespan=lifespan,
        # Interactive docs are a development convenience, not a public surface.
        docs_url="/docs" if settings.app.docs_enabled else None,
        redoc_url="/redoc" if settings.app.docs_enabled else None,
        openapi_url="/openapi.json" if settings.app.docs_enabled else None,
    )
    app.state.settings = settings

    # Accepts an inbound X-Request-ID so a correlation id set by the
    # orchestrator carries through into this agent's logs, and generates one
    # when there is none.
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name=settings.observability.correlation_id_header,
        validator=_is_safe_correlation_id,
    )

    # The orchestrator is a server-to-server caller and needs no CORS. A
    # browser does: `reconstruction agent.html` at the repository root drives
    # this agent directly, and opened from disk it sends `Origin: null`.
    # Off by default in production, where a browser should not be reaching the
    # agent at all.
    origins = settings.app.allowed_origins
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            # No cookies are involved, and credentialed requests cannot use the
            # `*` origin the development default relies on.
            allow_credentials=False,
        )

    app.include_router(orchestrator_router)
    app.include_router(api_router)

    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Make unhandled failures come back in the envelope, not FastAPI's shape.

    Only reachable for the v1 routes and for malformed requests - `/execute`
    catches everything itself, because the orchestrator needs `AgentResult`
    rather than this envelope.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            ErrorDetail(
                field=".".join(str(part) for part in error.get("loc", ())[1:]) or None,
                message=str(error.get("msg", "invalid value")),
            )
            for error in exc.errors()
        ]
        payload = Envelope[None].fail(
            ErrorCode.VALIDATION_ERROR,
            "The request body did not validate.",
            details=details,
            request_id=correlation_id.get(),
        ).model_dump(mode="json")
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(Exception)
    async def unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        _log.exception("unhandled_exception")
        return JSONResponse(
            status_code=500,
            content=error_payload(
                ErrorCode.INTERNAL_ERROR,
                "An unexpected error occurred.",
                request_id=correlation_id.get(),
            ),
        )


app = create_app()

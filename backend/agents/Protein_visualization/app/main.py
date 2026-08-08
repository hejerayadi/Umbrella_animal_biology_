import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_knowledge_base
from backend.agents.Protein_visualization.app.api.v1.routes import router as v1_router
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.contracts.envelope import ApiResponse, ErrorCode, failure
from backend.agents.Protein_visualization.app.domain.exceptions import (
    InvalidProteinRequestError,
    ProteinAgentError,
    ProteinNotFoundError,
    UpstreamServiceError,
)
from backend.agents.Protein_visualization.app.observability.logging import configure_logging, log_event
from backend.agents.Protein_visualization.app.observability.middleware import RequestContextMiddleware

PROBLEM_BASE = "https://umbrella.bio/problems"

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    knowledge_base = get_knowledge_base()
    if knowledge_base.store:
        try:
            await knowledge_base.store.ensure_collection(knowledge_base.embedding.dimensions)
        except Exception as exc:
            log_event(
                logger,
                "qdrant.initialization_failed",
                logging.ERROR,
                error_code=type(exc).__name__,
                detail=str(exc),
            )
        else:
            log_event(
                logger,
                "qdrant.collection_ready",
                collection=settings.qdrant_collection,
                dimensions=knowledge_base.embedding.dimensions,
            )
    log_event(
        logger,
        "service.started",
        app_env=settings.app_env,
        biological_sources="real",
        llm_provider=settings.llm_provider,
        qdrant_configured=bool(settings.qdrant_url),
        routes=len(application.routes),
    )
    yield
    log_event(logger, "service.stopped", app_env=settings.app_env)


def _correlation(request: Request) -> dict[str, Any]:
    return {
        "request_id": getattr(request.state, "request_id", None),
        "trace_id": getattr(request.state, "trace_id", None),
    }


def _envelope_response(response: ApiResponse[None]) -> JSONResponse:
    assert response.error is not None
    return JSONResponse(
        response.model_dump(mode="json", exclude_none=False),
        status_code=response.error.status,
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Protein identity, structure, annotation, and visualization agent.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestContextMiddleware)
    application.include_router(v1_router, prefix=settings.api_prefix)

    @application.exception_handler(ProteinNotFoundError)
    async def not_found(request: Request, exc: ProteinNotFoundError) -> JSONResponse:
        return _envelope_response(
            failure(
                code=ErrorCode.protein_not_found,
                problem_type=f"{PROBLEM_BASE}/protein-not-found",
                title="Protein not found",
                status=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
                instance=request.url.path,
                **_correlation(request),
            )
        )

    @application.exception_handler(InvalidProteinRequestError)
    async def invalid_request(request: Request, exc: InvalidProteinRequestError) -> JSONResponse:
        return _envelope_response(
            failure(
                code=ErrorCode.validation_error,
                problem_type=f"{PROBLEM_BASE}/invalid-protein-request",
                title="Invalid protein request",
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
                instance=request.url.path,
                **_correlation(request),
            )
        )

    @application.exception_handler(UpstreamServiceError)
    async def upstream_error(request: Request, exc: UpstreamServiceError) -> JSONResponse:
        return _envelope_response(
            failure(
                code=ErrorCode.upstream_unavailable,
                problem_type=f"{PROBLEM_BASE}/upstream-unavailable",
                title="Upstream biological source unavailable",
                status=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
                instance=request.url.path,
                **_correlation(request),
            )
        )

    @application.exception_handler(ProteinAgentError)
    async def agent_error(request: Request, exc: ProteinAgentError) -> JSONResponse:
        return _envelope_response(
            failure(
                code=ErrorCode.agent_error,
                problem_type=f"{PROBLEM_BASE}/protein-agent-error",
                title="Protein analysis failed",
                status=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
                instance=request.url.path,
                **_correlation(request),
            )
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope_response(
            failure(
                code=ErrorCode.validation_error,
                problem_type=f"{PROBLEM_BASE}/validation-error",
                title="Request validation failed",
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The request body is invalid.",
                instance=request.url.path,
                errors=[dict(error) for error in exc.errors()],
                **_correlation(request),
            )
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = (
            ErrorCode.unauthorized
            if exc.status_code == status.HTTP_401_UNAUTHORIZED
            else ErrorCode.not_found
            if exc.status_code == status.HTTP_404_NOT_FOUND
            else ErrorCode.agent_error
        )
        return _envelope_response(
            failure(
                code=code,
                problem_type=f"{PROBLEM_BASE}/http-error",
                title="Request could not be served",
                status=exc.status_code,
                detail=str(exc.detail),
                instance=request.url.path,
                **_correlation(request),
            )
        )

    @application.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        log_event(
            logger,
            "unhandled_error",
            logging.ERROR,
            exc_info=True,
            path=request.url.path,
            error_code=type(exc).__name__,
        )
        return _envelope_response(
            failure(
                code=ErrorCode.internal_error,
                problem_type=f"{PROBLEM_BASE}/internal-error",
                title="Internal server error",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The service failed to complete the request.",
                instance=request.url.path,
                **_correlation(request),
            )
        )

    return application


app = create_app()

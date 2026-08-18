from __future__ import annotations

import logging
import asyncio
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from .api.v1.router import router as v1_router
from .contracts import ApiProblem, ApiResponse, ProblemDetail, response_meta
from .core.config import get_settings
from .core.sessions import SessionStore
from .db.session import create_database
from .services.audit import purge_old_audit_events
from .services.email import EmailService

logger = logging.getLogger("umbrella.api")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _problem_response(request: Request, problem: ApiProblem) -> JSONResponse:
    body = ApiResponse[None](
        data=None,
        meta=response_meta(request),
        error=ProblemDetail(
            code=problem.code,
            type=f"https://umbrella.bio/problems/{problem.code.casefold().replace('_', '-')}",
            title=problem.title,
            status=problem.status,
            detail=problem.detail,
            instance=request.url.path,
            errors=problem.errors,
        ),
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=problem.status)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    engine, session_factory = create_database(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    await redis.ping()
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.session_store = SessionStore(redis, settings)
    app.state.email_service = EmailService(settings)
    if await redis.set("maintenance:audit-retention", "1", nx=True, ex=86_400):
        async with session_factory() as db:
            await purge_old_audit_events(db, settings.audit_retention_days)
    yield
    await redis.aclose()
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-CSRF-Token", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.started_at = time.perf_counter()
        candidate = request.headers.get(settings.request_id_header, "")
        request.state.request_id = candidate[:64] if candidate else str(uuid.uuid4())
        response = await call_next(request)
        response.headers[settings.request_id_header] = request.state.request_id
        return response

    @application.exception_handler(ApiProblem)
    async def api_problem(request: Request, exc: ApiProblem) -> JSONResponse:
        return _problem_response(request, exc)

    @application.exception_handler(RequestValidationError)
    async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return _problem_response(
            request,
            ApiProblem(422, "VALIDATION_ERROR", "Request validation failed", "The request is invalid.", errors=errors),
        )

    @application.exception_handler(HTTPException)
    async def http_problem(request: Request, exc: HTTPException) -> JSONResponse:
        return _problem_response(
            request, ApiProblem(exc.status_code, "HTTP_ERROR", "Request failed", str(exc.detail))
        )

    @application.exception_handler(Exception)
    async def unexpected_problem(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=exc)
        return _problem_response(
            request,
            ApiProblem(500, "INTERNAL_ERROR", "Internal server error", "An unexpected error occurred."),
        )

    application.include_router(v1_router, prefix=settings.api_prefix)
    return application


app = create_app()

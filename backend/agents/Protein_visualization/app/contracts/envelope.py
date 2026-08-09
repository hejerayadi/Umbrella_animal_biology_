"""Uniform API envelope: every `/api/v1` response is ``{data, meta, error}``.

``data`` carries the endpoint payload on success and is ``null`` on failure.
``meta`` always carries correlation ids and timing. ``error`` is ``null`` on
success and otherwise an RFC 9457 problem detail extended with a stable ``code``.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.agents.Protein_visualization.app.contracts.problem_details import ProblemDetails
from backend.agents.Protein_visualization.app.observability.context import get_log_context, request_duration_ms

API_VERSION = "v1"


class ErrorCode:
    """Stable machine-readable error codes returned in ``error.code``."""

    validation_error = "VALIDATION_ERROR"
    protein_not_found = "PROTEIN_NOT_FOUND"
    upstream_unavailable = "UPSTREAM_UNAVAILABLE"
    agent_error = "AGENT_ERROR"
    unauthorized = "UNAUTHORIZED"
    not_found = "NOT_FOUND"
    internal_error = "INTERNAL_ERROR"


class ResponseMeta(BaseModel):
    api_version: str = API_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = None
    trace_id: str | None = None
    task_id: str | None = None
    analysis_id: str | None = None
    duration_ms: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ApiError(ProblemDetails):
    code: str = ErrorCode.internal_error


class ApiResponse[T](BaseModel):
    data: T | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    error: ApiError | None = None


def build_meta(**overrides: Any) -> ResponseMeta:
    """Meta pre-filled from the request log context (request_id, trace_id, timing)."""
    context = get_log_context()
    fields: dict[str, Any] = {
        "request_id": context.get("request_id"),
        "trace_id": context.get("trace_id"),
        "task_id": context.get("task_id"),
        "analysis_id": context.get("analysis_id"),
        "duration_ms": request_duration_ms(),
    }
    fields.update({key: value for key, value in overrides.items() if value is not None})
    return ResponseMeta(**fields)


def success[T](data: T, **meta_overrides: Any) -> ApiResponse[T]:
    return ApiResponse[T](data=data, meta=build_meta(**meta_overrides))


def failure(
    *,
    code: str,
    title: str,
    status: int,
    detail: str,
    problem_type: str = "about:blank",
    instance: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    **meta_overrides: Any,
) -> ApiResponse[None]:
    meta = build_meta(**meta_overrides)
    return ApiResponse[None](
        data=None,
        meta=meta,
        error=ApiError(
            code=code,
            type=problem_type,
            title=title,
            status=status,
            detail=detail,
            instance=instance,
            trace_id=meta.trace_id,
            errors=errors or [],
        ),
    )

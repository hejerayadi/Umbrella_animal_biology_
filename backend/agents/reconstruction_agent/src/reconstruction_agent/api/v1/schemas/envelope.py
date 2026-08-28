"""The `{data, meta, error}` envelope every v1 endpoint answers with.

One shape for every response, so a client can parse the outer structure before
it knows anything about the resource, and can branch on `error is None` instead
of interpreting a status code.

Exactly one of `data` and `error` is ever populated.

`POST /execute` deliberately does NOT use this envelope. That endpoint belongs
to the Umbrella orchestrator, which parses the repo-wide `AgentResult` shape
directly and would break if it were wrapped. The two contracts live side by
side on purpose - see `api/orchestrator.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import ErrorCode

API_VERSION = "v1"
AGENT_NAME = "reconstruction-agent"


class ApiMeta(BaseModel):
    """Technical context, present on every response, success or failure.

    Never business data. A client that needs a value to answer a scientific
    question should find it in `data`; everything here is about the call.
    """

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    agent: str = AGENT_NAME
    #: Unique per HTTP attempt.
    request_id: str | None = None
    #: Stable across an entire orchestrator run.
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None

    #: Loop and budget accounting, on reconstruction responses only.
    iteration_count: int | None = None
    budget: dict[str, int | bool] | None = None


class ApiError(BaseModel):
    """A structured failure.

    `code` is the contract - clients branch on it, so the strings are stable
    and never renamed. `retryable` answers the only other question a caller
    has, and is decided where the cause is known rather than re-derived from a
    message at the boundary.
    """

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str = Field(description="Human-readable summary. Safe to show a user.")
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ApiResponse[T](BaseModel):
    """`{data, meta, error}` - the shape of every v1 response."""

    model_config = ConfigDict(frozen=True)

    data: T | None = None
    meta: ApiMeta = Field(default_factory=ApiMeta)
    error: ApiError | None = None

    @classmethod
    def ok(
        cls,
        data: T,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        duration_ms: float | None = None,
        iteration_count: int | None = None,
        budget: dict[str, int | bool] | None = None,
    ) -> ApiResponse[T]:
        return cls(
            data=data,
            meta=ApiMeta(
                request_id=request_id,
                trace_id=trace_id,
                duration_ms=duration_ms,
                iteration_count=iteration_count,
                budget=budget,
            ),
            error=None,
        )

    @classmethod
    def fail(
        cls,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> ApiResponse[T]:
        return cls(
            data=None,
            meta=ApiMeta(request_id=request_id, trace_id=trace_id),
            error=ApiError(
                code=code,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )


def error_payload(
    code: ErrorCode,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """A serialised failure envelope, for exception handlers.

    Handlers return a `JSONResponse`, which needs a plain dict rather than a
    model, and cannot know the concrete payload type at that point.
    """
    response: ApiResponse[Any] = ApiResponse[Any].fail(
        code,
        message,
        retryable=retryable,
        details=details,
        request_id=request_id,
        trace_id=trace_id,
    )
    return response.model_dump(mode="json")

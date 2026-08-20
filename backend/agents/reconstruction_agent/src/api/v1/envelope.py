"""The `{data, meta, error}` response envelope for v1 resource endpoints.

Every v1 endpoint answers with the same three top-level keys, so a client can
parse the shape before it knows anything about the resource:

- `data`  - the payload on success, `null` on failure.
- `meta`  - request context that is always present: correlation id, API
            version, timing. Never business data.
- `error` - `null` on success, a structured problem on failure.

Exactly one of `data`/`error` is non-null. That invariant is what lets a caller
branch on `error is None` instead of inspecting HTTP codes.

NOTE: `POST /execute` deliberately does NOT use this envelope. That endpoint
belongs to the orchestrator, which parses the repo-wide `AgentResult` shape
(`backend/orchestrator/schema.py`) and would break if it were wrapped. See
`api/v1/routes/reconstruction.py`.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

API_VERSION = "v1"


class ErrorCode(str, Enum):
    """Stable, machine-readable failure reasons.

    Clients branch on these, so the strings are part of the API contract -
    rename one and you break callers. The HTTP status is a coarser signal that
    these refine.
    """

    VALIDATION_ERROR = "validation_error"
    INVALID_SEQUENCE = "invalid_sequence"
    NO_GAPS_FOUND = "no_gaps_found"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(BaseModel):
    """One field-level problem, for validation failures."""

    model_config = ConfigDict(frozen=True)

    field: str | None = Field(default=None, description="Dotted path to the offending input.")
    message: str


class ApiError(BaseModel):
    """The structured failure placed in `error`."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str = Field(description="Human-readable summary. Safe to show a user.")
    details: list[ErrorDetail] = Field(default_factory=list)
    # True when the same request could succeed later - a rate limit or a
    # timeout, not a malformed sequence.
    retryable: bool = False


class Meta(BaseModel):
    """Context attached to every response, success or failure."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    # Mirrors the X-Request-ID header that asgi-correlation-id sets, so a
    # caller can quote it in a bug report and it can be grepped in the logs.
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None


class Envelope(BaseModel, Generic[T]):
    """`{data, meta, error}` - the shape every v1 resource endpoint returns."""

    model_config = ConfigDict(frozen=True)

    data: T | None = None
    meta: Meta = Field(default_factory=Meta)
    error: ApiError | None = None

    @classmethod
    def ok(
        cls,
        data: T,
        *,
        request_id: str | None = None,
        duration_ms: float | None = None,
    ) -> Envelope[T]:
        return cls(
            data=data,
            meta=Meta(request_id=request_id, duration_ms=duration_ms),
            error=None,
        )

    @classmethod
    def fail(
        cls,
        code: ErrorCode,
        message: str,
        *,
        details: list[ErrorDetail] | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> Envelope[T]:
        return cls(
            data=None,
            meta=Meta(request_id=request_id),
            error=ApiError(
                code=code,
                message=message,
                details=details or [],
                retryable=retryable,
            ),
        )


def error_payload(
    code: ErrorCode,
    message: str,
    *,
    retryable: bool = False,
    request_id: str | None = None,
) -> dict[str, Any]:
    """A serialised failure envelope, for exception handlers.

    Handlers return `JSONResponse`, which needs a plain dict rather than a
    model, and cannot know the concrete `T` at that point.
    """
    return Envelope[Any].fail(
        code, message, retryable=retryable, request_id=request_id
    ).model_dump(mode="json")

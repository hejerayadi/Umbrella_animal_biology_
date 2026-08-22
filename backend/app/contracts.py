from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from fastapi import Request
from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseMeta(BaseModel):
    api_version: str = "v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = None
    duration_ms: int | None = None
    pagination: dict[str, Any] | None = None


class ProblemDetail(BaseModel):
    code: str
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


class ApiResponse(BaseModel, Generic[T]):
    data: T | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    error: ProblemDetail | None = None


class ApiProblem(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str,
        *,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.errors = errors or []


def response_meta(request: Request, pagination: dict[str, Any] | None = None) -> ResponseMeta:
    started = getattr(request.state, "started_at", None)
    duration_ms = None
    if started is not None:
        import time

        duration_ms = max(0, round((time.perf_counter() - started) * 1000))
    return ResponseMeta(
        request_id=getattr(request.state, "request_id", None),
        duration_ms=duration_ms,
        pagination=pagination,
    )


def success(request: Request, data: Any, *, pagination: dict[str, Any] | None = None) -> dict[str, Any]:
    return ApiResponse[Any](
        data=data, meta=response_meta(request, pagination=pagination)
    ).model_dump(mode="json")


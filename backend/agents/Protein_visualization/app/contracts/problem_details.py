from typing import Any

from pydantic import BaseModel, Field


class ProblemDetails(BaseModel):
    """RFC 9457-compatible API error body."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None
    trace_id: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)

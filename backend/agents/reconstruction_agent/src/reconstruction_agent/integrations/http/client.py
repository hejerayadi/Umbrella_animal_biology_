"""One pooled, paced, retrying HTTP client per external service.

Every integration in this agent goes through a `ServiceClient`, so the things
that are easy to get subtly wrong - connection reuse, courtesy pacing, backoff,
turning a transport failure into a typed domain error - are written once and
behave identically for NCBI, EMBL-EBI and NVIDIA.

Pacing is per service instance, which means each service needs exactly one
shared client. Two clients for the same host each honour the limit separately
and together exceed it, which is how a politely configured agent still earns a
block.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx
from aiolimiter import AsyncLimiter

from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import (
    ExternalServiceError,
    RateLimitError,
    ServiceTimeoutError,
)
from reconstruction_agent.integrations.http.retry import RETRYABLE_STATUS, RetryPolicy, with_retry


class ServiceClient:
    """An HTTP client bound to one upstream service."""

    def __init__(
        self,
        *,
        service: str,
        base_url: str,
        timeout: float = 30.0,
        requests_per_second: float = 3.0,
        retry_policy: RetryPolicy | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.service = service
        self._retry = retry_policy or RetryPolicy()
        self._limiter = _limiter_for(requests_per_second)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers=headers or {},
            follow_redirects=True,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        deadline_remaining: Any | None = None,
    ) -> httpx.Response:
        """Send one request, paced and retried, returning a successful response."""

        async def attempt() -> httpx.Response:
            async with self._limiter:
                try:
                    response = await self._client.request(
                        method,
                        path,
                        params=params,
                        data=data,
                        json=json_body,
                        headers=headers,
                    )
                except httpx.TimeoutException as error:
                    raise ServiceTimeoutError(
                        self.service, f"request timed out: {error}"
                    ) from error
                except httpx.TransportError as error:
                    raise ExternalServiceError(
                        self.service, f"transport failure: {error}"
                    ) from error

            self._raise_for_status(response)
            return response

        return await with_retry(attempt, self._retry, deadline_remaining=deadline_remaining)

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate an unsuccessful response into a typed domain error.

        Done here rather than through `raise_for_status` so that a 429 becomes a
        `RateLimitError` carrying the server's own `Retry-After`, which the
        backoff then honours instead of guessing.
        """
        status = response.status_code
        if status < 400:
            return

        body = response.text[:500]

        if status == 429:
            raise RateLimitError(
                self.service,
                f"rate limited (HTTP 429): {body}",
                retry_after=_retry_after_seconds(response),
                details={"status": status},
            )

        if status in RETRYABLE_STATUS:
            raise ExternalServiceError(
                self.service,
                f"HTTP {status}: {body}",
                retryable=True,
                details={"status": status},
            )

        raise ExternalServiceError(
            self.service,
            f"HTTP {status}: {body}",
            code=ErrorCode.UPSTREAM_UNAVAILABLE,
            retryable=False,
            details={"status": status},
        )

    async def get_text(self, path: str, **kwargs: Any) -> str:
        response = await self.request("GET", path, **kwargs)
        return response.text

    async def get_json(self, path: str, **kwargs: Any) -> Any:
        response = await self.request("GET", path, **kwargs)
        try:
            return response.json()
        except ValueError as error:
            raise ExternalServiceError(
                self.service, f"response was not valid JSON: {error}"
            ) from error

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", path, **kwargs)


def _limiter_for(requests_per_second: float) -> AsyncLimiter:
    """A rate limiter for `requests_per_second`, including rates below one.

    aiolimiter refuses to grant more than its capacity, and capacity is the
    rate times the period. Expressed as "N per second" a rate below one gives a
    capacity below one, so *every* acquisition raises - which silently rules
    out exactly the pacing the politest services ask for. NCBI BLAST wants no
    more than one request every ten seconds; that is 0.1/s, and it has to be
    expressed as one per ten seconds instead.
    """
    rate = max(requests_per_second, 0.01)
    if rate >= 1.0:
        return AsyncLimiter(rate, 1.0)
    return AsyncLimiter(1.0, 1.0 / rate)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The `Retry-After` header as seconds, when it is present and numeric."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # The header also permits an HTTP date. Rare here, and the backoff
        # curve is a fine fallback, so it is not worth parsing.
        return None

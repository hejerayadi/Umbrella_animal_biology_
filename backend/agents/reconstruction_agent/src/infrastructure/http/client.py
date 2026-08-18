"""The shared HTTP client every external service call goes through.

Combines the three concerns that all of them need - pacing, retrying, and
turning transport failures into this agent's own exception types - so that
`infrastructure/ncbi` and `infrastructure/embl_ebi` only have to describe
their endpoints.
"""
from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from configuration.logging import get_logger
from domain.exceptions import ExternalServiceError, RateLimitError
from infrastructure.http.rate_limiter import AsyncRateLimiter
from infrastructure.http.retry import RetryPolicy, with_retry

_log = get_logger(__name__)


class ServiceClient:
    """An async HTTP client scoped to one external service.

    Owns its `httpx.AsyncClient`, so it must be closed - use it as an async
    context manager, or call `aclose()`. The agent builds one per service at
    startup and reuses it, which is what keeps connections pooled and the rate
    limiter authoritative.
    """

    def __init__(
        self,
        service: str,
        base_url: str,
        *,
        timeout: float = 30.0,
        requests_per_second: float | None = None,
        retry_policy: RetryPolicy | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.service = service
        self.base_url = base_url.rstrip("/")
        self._retry = retry_policy or RetryPolicy()
        self._limiter = (
            AsyncRateLimiter(rate=requests_per_second) if requests_per_second else None
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers or {},
            follow_redirects=True,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """One paced, retried request. Raises `ExternalServiceError` on failure.

        `data` is form-encoded (EMBL-EBI's job APIs want that); `json` is a
        JSON body (Azure and NVIDIA want that). Pass one or the other.
        """

        async def _send() -> httpx.Response:
            if self._limiter is not None:
                await self._limiter.acquire()

            response = await self._client.request(
                method, path, params=params, data=data, json=json, headers=headers
            )

            # Translated before `raise_for_status` so the retry layer sees a
            # RateLimitError carrying the server's own backoff hint.
            if response.status_code == 429:
                raise RateLimitError(self.service, _retry_after(response))

            response.raise_for_status()
            return response

        try:
            return await with_retry(_send, policy=self._retry, service=self.service)
        except RateLimitError:
            raise
        except httpx.HTTPStatusError as error:
            raise ExternalServiceError(
                self.service,
                f"HTTP {error.response.status_code} for {method} {path}",
                retryable=False,
            ) from error
        except httpx.TransportError as error:
            raise ExternalServiceError(
                self.service, f"transport failure for {method} {path}: {error}", retryable=True
            ) from error

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def get_text(self, path: str, **kwargs: Any) -> str:
        response = await self.get(path, **kwargs)
        return response.text

    async def get_json(self, path: str, **kwargs: Any) -> Any:
        response = await self.get(path, **kwargs)
        try:
            return response.json()
        except ValueError as error:
            raise ExternalServiceError(
                self.service, f"expected JSON from {path}, got {response.text[:200]!r}"
            ) from error

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ServiceClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _retry_after(response: httpx.Response) -> float | None:
    """The server's Retry-After header in seconds, when it sent a usable one."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # The HTTP-date form is legal but rare here; our backoff curve covers it.
        return None

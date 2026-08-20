import logging
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.agents.Protein_visualization.app.domain.exceptions import UpstreamServiceError
from backend.agents.Protein_visualization.app.observability.logging import log_event, log_stage

logger = logging.getLogger("app.tools.http")


class RetryableHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _safe_path(value: str) -> str:
    """Keep operational routing information without logging query parameters."""
    parsed = urlsplit(value)
    return parsed.path or "/"


def _retry_status_code(exc: BaseException | None) -> int | None:
    return exc.status_code if isinstance(exc, RetryableHttpError) else None


def _safe_error_detail(exc: Exception) -> str:
    """Describe a provider failure without echoing URLs, queries, or response bodies."""
    if isinstance(exc, RetryableHttpError):
        return f"HTTP {exc.status_code}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "request timed out"
    if isinstance(exc, httpx.ConnectError):
        return "connection failed"
    if isinstance(exc, ValueError):
        return "invalid JSON response"
    return type(exc).__name__


class JsonHttpClient:
    def __init__(
        self,
        service: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.service = service
        self.max_retries = max_retries
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        method_name = method.upper()
        route = _safe_path(path)
        max_attempts = self.max_retries + 1

        def before_sleep(retry_state: RetryCallState) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            wait_seconds = retry_state.next_action.sleep if retry_state.next_action else 0.0
            log_event(
                logger,
                "protein.tool.request.retried",
                logging.WARNING,
                status="retrying",
                provider=self.service,
                method=method_name,
                route=route,
                attempt=retry_state.attempt_number,
                max_attempts=max_attempts,
                backoff_ms=round(wait_seconds * 1000),
                elapsed_ms=round((retry_state.seconds_since_start or 0.0) * 1000),
                status_code=_retry_status_code(exc),
                error_code=type(exc).__name__ if exc else None,
            )

        with log_stage(
            logger,
            "protein.tool.request",
            capability="external_http",
            provider=self.service,
            method=method_name,
            route=route,
            max_attempts=max_attempts,
        ) as outcome:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(multiplier=0.2, max=2),
                    retry=retry_if_exception_type(
                        (RetryableHttpError, httpx.TimeoutException, httpx.ConnectError)
                    ),
                    before_sleep=before_sleep,
                    reraise=True,
                ):
                    with attempt:
                        response = await self.client.request(method_name, path, **kwargs)
                        if response.status_code in {429, 500, 502, 503, 504}:
                            raise RetryableHttpError(response.status_code)
                        response.raise_for_status()
                        outcome["attempts"] = attempt.retry_state.attempt_number
                        outcome["status_code"] = response.status_code
                        if content_length := response.headers.get("content-length"):
                            outcome["response_bytes"] = int(content_length)
                        if response.status_code == 204:
                            return {}
                        return response.json()
            except (httpx.HTTPError, ValueError, RetryableHttpError) as exc:
                outcome["upstream_error_code"] = type(exc).__name__
                outcome["status_code"] = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else _retry_status_code(exc)
                )
                raise UpstreamServiceError(self.service, _safe_error_detail(exc)) from None
            raise UpstreamServiceError(self.service, "request produced no response")

    async def get_text(self, url: str) -> str:
        with log_stage(
            logger,
            "protein.tool.request",
            capability="external_http",
            provider=self.service,
            method="GET",
            route=_safe_path(url),
            max_attempts=1,
        ) as outcome:
            try:
                response = await self.client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                outcome["upstream_error_code"] = type(exc).__name__
                outcome["status_code"] = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                raise UpstreamServiceError(self.service, _safe_error_detail(exc)) from None
            outcome["attempts"] = 1
            outcome["status_code"] = response.status_code
            outcome["response_bytes"] = len(response.content)
            return response.text

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.domain.exceptions import UpstreamServiceError


class RetryableHttpError(Exception):
    pass


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
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential(multiplier=0.2, max=2),
                retry=retry_if_exception_type(
                    (RetryableHttpError, httpx.TimeoutException, httpx.ConnectError)
                ),
                reraise=True,
            ):
                with attempt:
                    response = await self.client.request(method, path, **kwargs)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise RetryableHttpError(f"HTTP {response.status_code}")
                    response.raise_for_status()
                    if response.status_code == 204:
                        return {}
                    return response.json()
        except (httpx.HTTPError, ValueError, RetryableHttpError) as exc:
            raise UpstreamServiceError(self.service, str(exc)) from exc
        raise UpstreamServiceError(self.service, "request produced no response")

    async def get_text(self, url: str) -> str:
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise UpstreamServiceError(self.service, str(exc)) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

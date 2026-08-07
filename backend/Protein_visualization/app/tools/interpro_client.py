from typing import Any

import httpx

from app.configuration.settings import Settings
from app.tools.http import JsonHttpClient


class InterProClient(JsonHttpClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            "InterPro",
            settings.interpro_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
            client,
        )

    async def annotations(self, accession: str) -> list[dict[str, Any]]:
        payload = await self.request_json("GET", f"/entry/all/protein/uniprot/{accession}")
        return list(payload.get("results", []))

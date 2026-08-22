from typing import Any

import httpx

from backend.agents.Protein_visualization.app.configuration.settings import Settings
from backend.agents.Protein_visualization.app.tools.http import JsonHttpClient


class AlphaFoldClient(JsonHttpClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            "AlphaFold DB",
            settings.alphafold_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
            client,
        )

    async def predictions(self, accession: str) -> list[dict[str, Any]]:
        payload = await self.request_json("GET", f"/prediction/{accession}")
        return payload if isinstance(payload, list) else [payload]

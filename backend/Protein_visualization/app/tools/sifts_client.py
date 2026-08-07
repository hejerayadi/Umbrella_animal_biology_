from typing import Any

import httpx

from app.configuration.settings import Settings
from app.tools.http import JsonHttpClient


class SiftsClient(JsonHttpClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            "PDBe SIFTS",
            settings.sifts_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
            client,
        )

    async def mappings(self, pdb_id: str) -> dict[str, Any]:
        return await self.request_json("GET", f"/mappings/uniprot/{pdb_id.lower()}")

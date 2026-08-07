from typing import Any
from urllib.parse import quote

import httpx

from app.configuration.settings import Settings
from app.tools.http import JsonHttpClient


class UniProtClient(JsonHttpClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            "UniProt",
            settings.uniprot_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
            client,
        )

    async def get_entry(self, accession: str) -> dict[str, Any]:
        return await self.request_json("GET", f"/uniprotkb/{quote(accession)}.json")

    async def search(self, query: str, organism: str | None = None) -> list[dict[str, Any]]:
        terms = [f"({query})"]
        if organism:
            terms.append(f'(organism_name:"{organism}")')
        payload = await self.request_json(
            "GET",
            "/uniprotkb/search",
            params={"query": " AND ".join(terms), "format": "json", "size": 5},
        )
        return list(payload.get("results", []))

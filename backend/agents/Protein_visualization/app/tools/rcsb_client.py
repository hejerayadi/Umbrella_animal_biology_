import httpx

from backend.agents.Protein_visualization.app.configuration.settings import Settings
from backend.agents.Protein_visualization.app.tools.http import JsonHttpClient


class RCSBClient(JsonHttpClient):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            "RCSB PDB",
            settings.rcsb_search_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
            client,
        )
        self.files_base_url = settings.rcsb_files_base_url.rstrip("/")
        self.data = JsonHttpClient(
            "RCSB PDB Data",
            settings.rcsb_data_base_url,
            settings.http_timeout_seconds,
            settings.http_max_retries,
        )

    async def find_by_uniprot(self, accession: str, limit: int = 10) -> list[str]:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": (
                                "rcsb_polymer_entity_container_identifiers."
                                "reference_sequence_identifiers.database_accession"
                            ),
                            "operator": "exact_match",
                            "value": accession,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": (
                                "rcsb_polymer_entity_container_identifiers."
                                "reference_sequence_identifiers.database_name"
                            ),
                            "operator": "exact_match",
                            "value": "UniProt",
                        },
                    },
                ],
            },
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": limit},
                "results_verbosity": "compact",
            },
        }
        payload = await self.request_json("POST", "/query", json=query)
        return [
            item if isinstance(item, str) else item["identifier"]
            for item in payload.get("result_set", [])
            if isinstance(item, str) or item.get("identifier")
        ]

    async def entry(self, pdb_id: str) -> dict:
        return await self.data.request_json("GET", f"/rest/v1/core/entry/{pdb_id}")

    async def polymer_entity(self, pdb_id: str, entity_id: str) -> dict:
        return await self.data.request_json("GET", f"/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}")

    def structure_urls(self, pdb_id: str) -> dict[str, str]:
        normalized = pdb_id.upper()
        return {
            "data": f"{self.files_base_url}/download/{normalized}.cif",
            "viewer": f"https://www.rcsb.org/3d-view/{normalized}",
        }

    async def close(self) -> None:
        await super().close()
        await self.data.close()

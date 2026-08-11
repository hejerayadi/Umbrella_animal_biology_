from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NCBIClientError(RuntimeError):
    def __init__(self, accession: str, status_code: int):
        super().__init__(f"NCBI request failed for accession {accession}: HTTP {status_code}")
        self.accession = accession
        self.status_code = status_code


class NCBIClient:
    def __init__(self, base_url: str = "https://api.ncbi.nlm.nih.gov"):
        self.base_url = base_url

    def _request_json(self, url: str) -> dict[str, Any]:
        import urllib.request

        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_assembly_metadata(self, accession: str) -> dict[str, Any]:
        url = f"{self.base_url}/datasets/v2alpha/genome/accession/{accession}"
        try:
            payload = self._request_json(url)
        except Exception as exc:
            raise NCBIClientError(accession, 500) from exc
        assemblies = payload.get("assemblies", [])
        if not assemblies:
            raise NCBIClientError(accession, 404)
        assembly = assemblies[0]
        if assembly.get("assembly_level") not in {"Chromosome", "Complete Genome"}:
            raise NCBIClientError(accession, 400)
        return assembly

    def download_fasta(self, accession: str, dest_path: Path) -> Path:
        """Download a FASTA file from NCBI E-utilities efetch and write it to *dest_path*.

        Raises :class:`NCBIClientError` on network failures, invalid accessions,
        or non-2xx HTTP responses.
        """
        import httpx

        # TODO: rate limiting NCBI E-utilities (3 req/s sans clé, 10 req/s avec clé)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "nuccore",
            "id": accession,
            "rettype": "fasta",
            "retmode": "text",
        }

        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise NCBIClientError(accession, 408) from exc
        except httpx.HTTPError as exc:
            raise NCBIClientError(accession, 503) from exc

        if response.status_code != 200:
            raise NCBIClientError(accession, response.status_code)

        body = response.text.strip()

        # NCBI returns 200 with an error message for invalid accessions
        if not body or not body.startswith(">"):
            raise NCBIClientError(accession, 404)

        dest_path.write_text(body, encoding="utf-8")
        return dest_path


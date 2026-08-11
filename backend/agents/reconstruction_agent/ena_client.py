from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ENAClientError(RuntimeError):
    def __init__(self, project_accession: str, status_code: int):
        super().__init__(f"ENA request failed for project {project_accession}: HTTP {status_code}")
        self.project_accession = project_accession
        self.status_code = status_code


class ENAClient:
    def __init__(self, base_url: str = "https://www.ebi.ac.uk/ena"):
        self.base_url = base_url

    def fetch_project_runs(self, project_accession: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/portal/api/filereports?accession={project_accession}"
        try:
            import urllib.request

            with urllib.request.urlopen(url) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ENAClientError(project_accession, 503) from exc
        if isinstance(payload, list):
            return payload
        return []

    def download_fasta(self, run_accession: str, dest_path: Path) -> Path:
        """Download FASTA for one ENA run and write it to *dest_path*.

        Raises :class:`ENAClientError` on network failures, invalid accessions,
        or non-2xx HTTP responses.
        """
        import httpx

        # TODO: rate limiting ENA (50 req/s documented by EBI, HTTP 429 on excess)
        url = f"https://www.ebi.ac.uk/ena/browser/api/fasta/{run_accession}"

        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = client.get(url)
        except httpx.TimeoutException as exc:
            raise ENAClientError(run_accession, 408) from exc
        except httpx.HTTPError as exc:
            raise ENAClientError(run_accession, 503) from exc

        if response.status_code != 200:
            raise ENAClientError(run_accession, response.status_code)

        body = response.text.strip()

        # ENA returns 200 with empty body or non-FASTA content for invalid accessions
        if not body or not body.startswith(">"):
            raise ENAClientError(run_accession, 404)

        dest_path.write_text(body, encoding="utf-8")
        return dest_path


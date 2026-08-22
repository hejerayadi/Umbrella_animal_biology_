from __future__ import annotations

import pytest
import requests

from backend.agents.image_generation_agent.tools.uniprot import call_uniprot


class FakeResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        if "/uniprotkb/search" in url:
            return self.responses["search"]
        if "/uniprotkb/" in url:
            return self.responses.get("entry", self.responses["search"])
        raise AssertionError(f"Unexpected URL: {url}")


UNIPROT_ENTRY = {
    "primaryAccession": "P01308",
    "entryType": "UniProtKB reviewed (Swiss-Prot)",
    "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Insulin"}},
    },
    "organism": {"scientificName": "Homo sapiens"},
    "sequence": {"value": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT", "length": 110},
}


def test_call_uniprot_valid_gene_and_species():
    session = FakeSession(
        {
            "search": FakeResponse(200, {"results": [UNIPROT_ENTRY]}),
        }
    )

    result = call_uniprot("INS", "Homo sapiens", session=session)

    assert result.success is True
    assert result.accession == "P01308"
    assert result.protein_name == "Insulin"
    assert result.organism == "Homo sapiens"
    assert result.sequence_length == 110
    assert result.sequence.startswith("MALW")


def test_call_uniprot_unknown_gene():
    session = FakeSession({"search": FakeResponse(200, {"results": []})})

    result = call_uniprot("NOTAREALGENE", "Homo sapiens", session=session)

    assert result.success is False
    assert "No UniProt entry found" in (result.error or "")


def test_call_uniprot_api_failure():
    session = FakeSession({"search": FakeResponse(503, {"results": []})})

    result = call_uniprot("INS", "Homo sapiens", session=session)

    assert result.success is False
    assert "UniProt request failed" in (result.error or "")


def test_call_uniprot_requires_gene():
    result = call_uniprot("  ", "Homo sapiens")

    assert result.success is False
    assert result.error == "Gene symbol is required."


def test_call_uniprot_fetches_full_entry_when_search_payload_is_incomplete():
    partial = {"primaryAccession": "P01308", "entryType": "UniProtKB reviewed (Swiss-Prot)"}
    session = FakeSession(
        {
            "search": FakeResponse(200, {"results": [partial]}),
            "entry": FakeResponse(200, UNIPROT_ENTRY),
        }
    )

    result = call_uniprot("INS", "Homo sapiens", session=session)

    assert result.success is True
    assert result.accession == "P01308"
    assert any("/uniprotkb/P01308.json" in call[0] for call in session.calls)

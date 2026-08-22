from __future__ import annotations

import requests

from backend.agents.image_generation_agent.tools.pdb import call_pdb


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
    def __init__(self, search_payload, entity_payload, entry_payload):
        self.search_payload = search_payload
        self.entity_payload = entity_payload
        self.entry_payload = entry_payload

    def post(self, url, json=None, timeout=None):
        return FakeResponse(200, self.search_payload)

    def get(self, url, timeout=None):
        if "/polymer_entity/" in url:
            return FakeResponse(200, self.entity_payload)
        if "/entry/" in url:
            return FakeResponse(200, self.entry_payload)
        raise AssertionError(f"Unexpected URL: {url}")


ENTITY = {
    "rcsb_polymer_entity_container_identifiers": {
        "reference_sequence_identifiers": [
            {
                "database_name": "UniProt",
                "database_accession": "P01308",
                "reference_sequence_coverage": 0.92,
            }
        ],
        "auth_asym_ids": ["A"],
    },
    "entity_poly": {"rcsb_sample_sequence_length": 101},
    "rcsb_entity_source_organism": [{"ncbi_scientific_name": "Homo sapiens"}],
}

ENTRY = {
    "struct": {"title": "Insulin structure"},
    "rcsb_entry_info": {"resolution_combined": [1.5]},
    "exptl": [{"method": "X-RAY DIFFRACTION"}],
}


def test_call_pdb_returns_ranked_structure():
    session = FakeSession(
        search_payload={"result_set": ["4ins_1", "1ai0_1"]},
        entity_payload=ENTITY,
        entry_payload=ENTRY,
    )

    result = call_pdb("P01308", 110, session=session)

    assert result.found is True
    assert result.pdb_id == "4INS"
    assert result.experimental_method == "X-RAY DIFFRACTION"
    assert result.resolution == 1.5
    assert result.sequence_coverage == 0.92
    assert result.title == "Insulin structure"
    assert result.organism == "Homo sapiens"
    assert result.chain == "A"
    assert result.selection_reason


def test_call_pdb_no_suitable_structure_when_search_empty():
    session = FakeSession(
        search_payload={"result_set": []},
        entity_payload=ENTITY,
        entry_payload=ENTRY,
    )

    result = call_pdb("P01308", 110, session=session)

    assert result.found is False
    assert result.message == "No suitable structure found for the UniProt accession."


def test_call_pdb_no_suitable_structure_when_metadata_missing():
    session = FakeSession(
        search_payload={"result_set": ["4ins_1"]},
        entity_payload={
            "rcsb_polymer_entity_container_identifiers": {
                "reference_sequence_identifiers": [
                    {"database_name": "UniProt", "database_accession": "OTHER"}
                ]
            }
        },
        entry_payload=ENTRY,
    )

    result = call_pdb("P01308", 110, session=session)

    assert result.found is False
    assert "No suitable structure found" in (result.message or "")


def test_call_pdb_api_failure():
    class FailingSession:
        def post(self, url, json=None, timeout=None):
            raise requests.Timeout("timed out")

    result = call_pdb("P01308", 110, session=FailingSession())

    assert result.found is False
    assert "RCSB PDB search failed" in (result.message or "")

from __future__ import annotations

import json

import requests

from backend.agents.image_generation_agent.scientific_critic import ScientificCritic
from backend.agents.image_generation_agent.tool_schemas import (
    CriticResult,
    PDBStructureResult,
    UniProtResult,
    WebSearchHit,
    WebSearchResult,
)


def test_deterministic_critic_abstain_without_uniprot():
    critic = ScientificCritic(base_url="", api_key=None)

    result = critic.review(
        gene="INS",
        species="Homo sapiens",
        instruction="Draw insulin",
        uniprot=UniProtResult(success=False, error="No UniProt entry found."),
        pdb=None,
        web_search=None,
    )

    assert result.verdict == "ABSTAIN"
    assert "No UniProt entry found" in result.reasons[0]


def test_deterministic_critic_accept_with_uniprot_and_pdb():
    critic = ScientificCritic(base_url="", api_key=None)

    result = critic.review(
        gene="INS",
        species="Homo sapiens",
        instruction="Draw insulin",
        uniprot=UniProtResult(
            success=True,
            accession="P01308",
            protein_name="Insulin",
            organism="Homo sapiens",
            sequence_length=110,
        ),
        pdb=PDBStructureResult(
            found=True,
            pdb_id="4INS",
            experimental_method="X-RAY DIFFRACTION",
            resolution=1.5,
            sequence_coverage=1.0,
        ),
        web_search=None,
    )

    assert result.verdict == "ACCEPT"
    assert any("P01308" in reason for reason in result.reasons)


def test_deterministic_critic_revise_on_organism_mismatch():
    critic = ScientificCritic(base_url="", api_key=None)

    result = critic.review(
        gene="INS",
        species="Mus musculus",
        instruction="Draw insulin",
        uniprot=UniProtResult(
            success=True,
            accession="P01308",
            protein_name="Insulin",
            organism="Homo sapiens",
        ),
        pdb=None,
        web_search=None,
    )

    assert result.verdict == "REVISE"
    assert any("differs" in reason for reason in result.reasons)


def test_llm_audit_can_only_make_verdict_stricter():
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "output_text": json.dumps(
                    {"verdict": "ACCEPT", "reasons": ["Model tried to relax the verdict."]}
                )
            }

        @staticmethod
        def raise_for_status():
            return None

    class FakeSession:
        def post(self, url, headers=None, params=None, json=None, timeout=None):
            return FakeResponse()

    critic = ScientificCritic(
        base_url="https://example.cognitiveservices.azure.com",
        api_key="test-key",
        session=FakeSession(),
    )

    deterministic = CriticResult(
        verdict="REVISE",
        reasons=("Partial structure coverage.",),
        source="deterministic",
    )
    result = critic._llm_audit(
        deterministic=deterministic,
        gene="INS",
        species="Homo sapiens",
        instruction="Draw insulin",
        uniprot=UniProtResult(success=True, accession="P01308", organism="Homo sapiens"),
        pdb=PDBStructureResult(found=True, pdb_id="4INS", sequence_coverage=0.5),
        web_search=WebSearchResult(success=True, results=(WebSearchHit("A", "https://a", "snippet"),)),
    )

    review = critic.review(
        gene="INS",
        species="Homo sapiens",
        instruction="Draw insulin",
        uniprot=UniProtResult(success=True, accession="P01308", organism="Homo sapiens"),
        pdb=PDBStructureResult(found=True, pdb_id="4INS", sequence_coverage=0.5),
        web_search=None,
    )

    assert review.verdict == "REVISE"
    assert result.verdict == "ACCEPT"

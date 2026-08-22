from __future__ import annotations

from backend.agents.image_generation_agent.evidence_workflow import gather_protein_evidence
from backend.agents.image_generation_agent.orchestrator_logic import run_orchestrator_logic
from backend.agents.image_generation_agent.schema import AgentRequest, AgentStatus
from backend.agents.image_generation_agent.tool_schemas import (
    CriticResult,
    EvidenceBundle,
    PDBStructureResult,
    UniProtResult,
    WebSearchHit,
    WebSearchResult,
)


def test_gather_protein_evidence_runs_full_pipeline_with_mocks():
    calls = {"web_search": 0}

    def fake_uniprot(gene, species, session=None, timeout=30):
        return UniProtResult(
            success=True,
            accession="P01308",
            protein_name="Insulin",
            organism="Homo sapiens",
            sequence_length=110,
        )

    def fake_pdb(accession, sequence_length=None, session=None, timeout=30, max_candidates=10):
        assert accession == "P01308"
        return PDBStructureResult(found=False, message="No suitable structure found for the UniProt accession.")

    def fake_web_search(query, session=None, timeout=30, api_key=None):
        calls["web_search"] += 1
        return WebSearchResult(
            success=True,
            results=(WebSearchHit("Insulin review", "https://example.com", "Compact hormone."),),
        )

    def fake_critic(bundle, *, gene, species, instruction, critic=None):
        return CriticResult(verdict="REVISE", reasons=("No PDB structure found.",), source="deterministic")

    bundle = gather_protein_evidence(
        gene="INS",
        species="Homo sapiens",
        instruction="Draw insulin",
        uniprot_fn=fake_uniprot,
        pdb_fn=fake_pdb,
        web_search_fn=fake_web_search,
        critic_fn=fake_critic,
    )

    assert bundle.skipped is False
    assert bundle.uniprot and bundle.uniprot.success
    assert bundle.pdb and not bundle.pdb.found
    assert bundle.web_search and bundle.web_search.success
    assert bundle.critic and bundle.critic.verdict == "REVISE"
    assert calls["web_search"] == 1


def test_run_orchestrator_logic_fails_on_critic_abstain(monkeypatch):
    def fake_gather(**kwargs):
        return EvidenceBundle(
            skipped=False,
            uniprot=UniProtResult(success=False, error="No UniProt entry found."),
            critic=CriticResult(verdict="ABSTAIN", reasons=("No UniProt entry found.",)),
        )

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.gather_protein_evidence",
        fake_gather,
    )

    result = run_orchestrator_logic(
        AgentRequest(instruction="Draw insulin", context={"gene": "INS", "species": "Homo sapiens"}),
    )

    assert result.status == AgentStatus.FAILED
    assert "No UniProt entry found" in str(result.output)


def test_run_orchestrator_logic_includes_evidence_in_prompt(monkeypatch):
    captured: dict[str, str] = {}

    class FakeFluxClient:
        def generate_image(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "https://example.com/generated.jpg"

    def fake_gather(**kwargs):
        return EvidenceBundle(
            skipped=False,
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
                chain="A",
            ),
            critic=CriticResult(verdict="ACCEPT", reasons=("Evidence is coherent.",)),
        )

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.gather_protein_evidence",
        fake_gather,
    )

    result = run_orchestrator_logic(
        AgentRequest(
            instruction="Draw human insulin",
            context={"gene": "INS", "species": "Homo sapiens"},
        ),
        flux_client=FakeFluxClient(),
    )

    assert result.status == AgentStatus.COMPLETED
    assert "P01308" in captured["prompt"]
    assert "4INS" in captured["prompt"]
    assert "Scientific Critic" in captured["prompt"]
    assert result.output["confidence_score"] > 0.35

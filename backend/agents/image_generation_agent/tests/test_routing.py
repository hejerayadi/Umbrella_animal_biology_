from __future__ import annotations

from backend.agents.Protein_visualization.logic import ProteinVisualizationLogic
from backend.agents.Protein_visualization.schema import AgentRequest, AgentStatus


def test_missing_protein_and_species_fails():
    result = ProteinVisualizationLogic().run(
        AgentRequest(instruction="Visualize the protein", context={})
    )
    assert result.status == AgentStatus.FAILED
    assert "Missing protein and species" in str(result.output)


def test_missing_protein_fails():
    result = ProteinVisualizationLogic().run(
        AgentRequest(
            instruction="Visualize",
            context={"species": "Homo sapiens"},
        )
    )
    assert result.status == AgentStatus.FAILED
    assert "Missing protein information" in str(result.output)


def test_missing_species_fails():
    result = ProteinVisualizationLogic().run(
        AgentRequest(
            instruction="Visualize insulin",
            context={"gene": "INS"},
        )
    )
    assert result.status == AgentStatus.FAILED
    assert "Missing species information" in str(result.output)


def test_missing_traits_needs_trait_discovery_agent():
    result = ProteinVisualizationLogic().run(
        AgentRequest(
            instruction="Create a 2D visualization of human insulin",
            context={"species": "Homo sapiens", "gene": "INS"},
        )
    )
    assert result.status == AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Trait Discovery Agent"
    assert result.prompt_to_target_agent is not None
    assert "morphological" in result.prompt_to_target_agent.lower()


def test_traits_in_context_completes_with_mock_flux(monkeypatch):
    captured: dict[str, str] = {}

    class FakeFluxClient:
        def generate_image(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "https://example.com/generated.jpg"

    monkeypatch.setattr(
        "backend.agents.Protein_visualization.orchestrator_logic.FluxClient",
        lambda: FakeFluxClient(),
    )

    result = ProteinVisualizationLogic().run(
        AgentRequest(
            instruction="Visualize insulin structural traits",
            context={
                "species": "Homo sapiens",
                "gene": "INS",
                "traits": [
                    "Compact globular hormone structure",
                    {"name": "Disulfide bonds", "description": "Stabilize the folded form"},
                ],
            },
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.output["image"] == "https://example.com/generated.jpg"
    assert len(result.output["traits_used"]) == 2
    assert 0.0 < result.output["confidence_score"] <= 1.0
    assert "scientific illustration style" in captured["prompt"]
    assert "Compact globular hormone structure" in captured["prompt"]

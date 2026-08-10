from __future__ import annotations

import pytest

from backend.agents.image_generation_agent.logic import ImageGenerationLogic
from backend.agents.image_generation_agent.schema import AgentRequest, AgentStatus


@pytest.fixture
def fake_flux(monkeypatch):
    """Capture the prompt instead of spending FLUX.2-pro credits."""
    captured: dict[str, str] = {}

    class FakeFluxClient:
        def generate_image(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "https://example.com/generated.jpg"

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.FluxClient",
        lambda: FakeFluxClient(),
    )
    return captured


def test_no_subject_at_all_fails():
    result = ImageGenerationLogic().run(
        AgentRequest(instruction="Draw something", context={})
    )
    assert result.status == AgentStatus.FAILED
    assert "Missing subject to illustrate" in str(result.output)


def test_species_alone_is_enough_to_draw(fake_flux):
    """No protein and no traits - a species on its own gets drawn.

    This is the "Draw an Arctic fox" path. It must not pause for the Trait
    agent: that turned a drawing request into a genomics pipeline that dead-ends
    in a five-gene stub. See the note at the top of orchestrator_logic.
    """
    result = ImageGenerationLogic().run(
        AgentRequest(
            instruction="Draw an Arctic fox",
            context={"species": "Vulpes lagopus"},
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.output["image"] == "https://example.com/generated.jpg"
    assert result.output["traits_used"] == []
    assert "Vulpes lagopus" in fake_flux["prompt"]


def test_protein_alone_is_enough_to_draw(fake_flux):
    result = ImageGenerationLogic().run(
        AgentRequest(instruction="Visualize insulin", context={"gene": "INS"})
    )

    assert result.status == AgentStatus.COMPLETED
    assert "INS" in fake_flux["prompt"]


def test_extractor_gene_name_counts_as_a_subject(fake_flux):
    """The orchestrator's extractor writes "gene_name", not "gene"."""
    result = ImageGenerationLogic().run(
        AgentRequest(
            instruction="Draw the protein FGF5 makes",
            context={"gene_name": "FGF5"},
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert "FGF5" in fake_flux["prompt"]


def test_this_agent_never_asks_for_another_agent(fake_flux):
    """Nothing this agent returns should re-open the trait escalation path."""
    for context in (
        {"species": "Vulpes lagopus"},
        {"gene": "INS"},
        {"species": "Homo sapiens", "gene": "INS"},
    ):
        result = ImageGenerationLogic().run(
            AgentRequest(instruction="Draw it", context=context)
        )
        assert result.status is not AgentStatus.NEEDS_AGENT
        assert result.target_agent is None


def test_traits_still_enrich_the_prompt_when_present(fake_flux):
    """Traits are optional, not ignored."""
    result = ImageGenerationLogic().run(
        AgentRequest(
            instruction="Draw an Arctic fox",
            context={
                "species": "Vulpes lagopus",
                "traits": ["Dense white winter coat", "Short rounded ears"],
            },
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.output["traits_used"] == [
        "Dense white winter coat",
        "Short rounded ears",
    ]
    assert "Dense white winter coat" in fake_flux["prompt"]
    # Trait data should raise confidence above the no-traits baseline.
    assert result.output["confidence_score"] > 0.35


def test_traits_in_context_completes_with_mock_flux(monkeypatch):
    captured: dict[str, str] = {}

    class FakeFluxClient:
        def generate_image(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "https://example.com/generated.jpg"

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.FluxClient",
        lambda: FakeFluxClient(),
    )

    result = ImageGenerationLogic().run(
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


def test_species_and_traits_complete_without_any_protein(monkeypatch):
    """The plain "draw this animal" path: no gene, no protein, still renders."""
    captured: dict[str, str] = {}

    class FakeFluxClient:
        def generate_image(self, prompt: str) -> str:
            captured["prompt"] = prompt
            return "https://example.com/mammoth.jpg"

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.FluxClient",
        lambda: FakeFluxClient(),
    )

    result = ImageGenerationLogic().run(
        AgentRequest(
            instruction="Draw a woolly mammoth",
            context={
                "species": "Mammuthus primigenius",
                "traits": ["Long curved tusks", "Dense woolly coat"],
            },
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.output["image"] == "https://example.com/mammoth.jpg"
    assert "Mammuthus primigenius" in captured["prompt"]
    assert "Long curved tusks" in captured["prompt"]
    assert "Protein" not in captured["prompt"]

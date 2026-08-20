from __future__ import annotations

from backend.agents.image_generation_agent.orchestrator_logic import organize_result
from backend.agents.image_generation_agent.prompt_builder import build_visualization_prompt
from backend.agents.image_generation_agent.schema import AgentRequest


def test_organize_result_creates_flexible_sections():
    request = AgentRequest(
        instruction="Show mammoth traits",
        context={
            "species": "Mammuthus primigenius",
            "protein_name": "Keratin",
            "traits": {
                "morphological": ["Long curved tusks", "Dense woolly coat"],
                "physiological": "Cold-adapted metabolism",
            },
        },
    )

    organized = organize_result(request)
    sections = organized["visualization_input"]["sections"]
    section_types = {section["type"] for section in sections}

    assert "user_intent" in section_types
    assert "protein" in section_types
    assert "species" in section_types
    assert "morphological" in section_types
    assert "physiological" in section_types


def test_build_prompt_includes_requirements_and_sections():
    organized = {
        "visualization_input": {
            "sections": [
                {"type": "structural", "content": "Alpha-helical domains"},
                {"type": "functional", "content": "Glucose regulation"},
            ]
        }
    }

    prompt = build_visualization_prompt(
        organized,
        instruction="Visualize insulin",
        context={},
    )

    assert "Create a professional 2D scientific visualization" in prompt
    assert "Alpha-helical domains" in prompt
    assert "Glucose regulation" in prompt
    assert "no invented biological claims" in prompt
    assert "scientific illustration style" in prompt

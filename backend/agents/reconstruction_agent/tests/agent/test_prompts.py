"""The model-guided nodes load Markdown resources, not Python prompt modules."""

from __future__ import annotations

import pytest

from reconstruction_agent.agent.prompts import load_prompt


@pytest.mark.parametrize("name", ["planner", "critic"])
def test_prompt_is_loaded_from_a_markdown_template(name: str) -> None:
    prompt = load_prompt(name)

    assert prompt.system
    assert prompt.user_template


def test_planner_template_renders_its_runtime_context() -> None:
    user = load_prompt("planner").render_user(
        gap_id="gap_1",
        gap_length=45,
        target="Ursus maritimus",
        known="record loaded",
    )

    assert "Gap gap_1, 45 bases" in user
    assert "Ursus maritimus" in user


def test_unknown_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown prompt"):
        load_prompt("not-a-prompt")

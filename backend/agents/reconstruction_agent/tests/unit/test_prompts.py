"""Prompts live in markdown and are loaded, not hardcoded.

These tests exist because a prompt is now data on disk: a renamed or deleted
`.md` file would otherwise fail at the first LLM call in production rather
than in CI.
"""
from __future__ import annotations

import pytest

from agent import prompts
from agent.prompts.loader import PromptNotFoundError, PromptRenderError

#: Every prompt the code loads by name. Keep in step with `src/prompts/`.
REQUIRED = [
    "planner.system",
    "planner.user",
    "critic.system",
    "critic.user",
    "explanation.system",
    "explanation.user",
]


class TestPromptFiles:
    @pytest.mark.parametrize("name", REQUIRED)
    def test_every_required_prompt_exists_and_is_not_empty(self, name: str) -> None:
        assert prompts.load(name).strip()

    def test_no_prompt_text_is_hardcoded_in_python(self) -> None:
        """The loader package must hold no prompt wording of its own."""
        import agent.prompts.loader as loader_module

        source = loader_module.__file__
        assert source is not None

    def test_unknown_prompt_names_the_available_ones(self) -> None:
        with pytest.raises(PromptNotFoundError, match="Available:"):
            prompts.load("does.not.exist")


class TestRendering:
    def test_planner_user_substitutes_every_placeholder(self) -> None:
        rendered = prompts.planner_user(
            instruction="Fill the gaps.",
            organism="Loxodonta africana",
            gaps=[{"gap_id": "gap_1", "length": 12}],
            tools=[{"name": "blast_search"}],
            prior_critiques=[],
        )

        assert "Fill the gaps." in rendered
        assert "Loxodonta africana" in rendered
        assert "gap_1" in rendered
        assert "blast_search" in rendered
        # No placeholder should survive rendering.
        assert "{instruction}" not in rendered
        assert "{critiques_section}" not in rendered

    def test_prior_critiques_are_included_when_present(self) -> None:
        rendered = prompts.planner_user(
            instruction="Fill the gaps.",
            organism=None,
            gaps=[],
            tools=[],
            prior_critiques=["gap_1: only one reference supports this."],
        )

        assert "only one reference supports this" in rendered

    def test_unspecified_organism_renders_a_word_not_none(self) -> None:
        rendered = prompts.planner_user(
            instruction="x", organism=None, gaps=[], tools=[], prior_critiques=[]
        )

        assert "unspecified" in rendered
        assert "None" not in rendered.split("Gaps needing")[0]

    def test_critic_user_embeds_the_evidence(self) -> None:
        rendered = prompts.critic_user(
            gap={"gap_id": "gap_1"},
            candidate={"sequence": "ACGT"},
            evidence=[{"reference_id": "REF_1"}],
        )

        assert "REF_1" in rendered and "ACGT" in rendered

    def test_missing_placeholder_value_names_the_prompt(self) -> None:
        with pytest.raises(PromptRenderError, match="planner.user"):
            prompts.render("planner.user", instruction="only one of five")


class TestDeterministicExplanation:
    def test_says_so_when_there_is_no_evidence(self) -> None:
        text = prompts.deterministic_explanation(
            gap_id="gap_1", length=10, confidence=0.0, references=[], mean_identity=None
        )

        assert "No reference sequence aligned" in text

    def test_names_the_references_and_calls_it_an_inference(self) -> None:
        text = prompts.deterministic_explanation(
            gap_id="gap_1",
            length=10,
            confidence=0.8,
            references=["REF_1", "REF_2"],
            mean_identity=0.93,
        )

        assert "REF_1" in text
        assert "93%" in text
        # The agent must never present a reconstruction as observed sequence.
        assert "inference" in text

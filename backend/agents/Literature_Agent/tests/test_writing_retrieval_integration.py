"""
Tests d'integration -- couvre la version tool-calling reelle de
WritingSupportAgent (le LLM decide lui-meme quels outils appeler) et la
protection des noms scientifiques dans la correction de style.

Usage (depuis Literature_Agent/) :
    pytest tests/test_writing_retrieval_integration.py -v
(aucun appel reseau : Azure, Qdrant et LanguageTool sont tous mockes)
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from agents.Literature_Agent.schema import AgentRequest, AgentStatus
from agents.Literature_Agent.subagents.writing.scientific_writing import (
    WritingSupportAgent,
    ScientificWritingOrchestrator,
)

MODULE = "agents.Literature_Agent.subagents.writing.scientific_writing"


def _fake_paper_hit(target_text: str):
    hit = MagicMock()
    hit.payload = {"target_text": target_text}
    return hit


def _fake_citation_hit(intent: str, context: str):
    hit = MagicMock()
    hit.payload = {"intent": intent, "context": context}
    return hit


def _tool_call_message(tool_name: str, arguments: dict):
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(arguments)

    message = MagicMock()
    message.content = None
    message.tool_calls = [tool_call]
    return message


def _final_message(content: str):
    message = MagicMock()
    message.content = content
    message.tool_calls = []
    return message


@pytest.fixture
def _skip_real_style_correction():
    """LanguageTool telecharge ~200 Mo au premier lancement et fait des
    appels reseau -- on le mocke dans la plupart des tests pour rester
    rapide et deterministe (sauf le test dedie a la protection des noms
    scientifiques, qui a besoin de tester la vraie logique de filtrage)."""
    with patch(f"{MODULE}._apply_style_correction", side_effect=lambda text: (text, True)):
        yield


# ---------------------------------------------------------------------------
# 1. Le modele appelle un outil, le resultat est reinjecte, puis le draft final
# ---------------------------------------------------------------------------

@patch(f"{MODULE}.search_kb_papers")
@patch(f"{MODULE}.call_llm_with_tools")
def test_agent_calls_a_tool_and_uses_its_result(
    mock_call_llm_with_tools, mock_search_kb_papers, _skip_real_style_correction
):
    mock_search_kb_papers.return_value = [_fake_paper_hit("Retrieved abstract example about migration.")]

    mock_call_llm_with_tools.side_effect = [
        _tool_call_message("get_abstract_examples", {"query": "tiger migration"}),
        _final_message("Final generated abstract."),
    ]

    request = AgentRequest(instruction="Write an abstract about tiger migration", context={})
    result = WritingSupportAgent().run(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output["draft"] == "Final generated abstract."
    assert result.output["tools_used"] == ["get_abstract_examples"]

    second_call_messages = mock_call_llm_with_tools.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "Retrieved abstract example about migration." in tool_messages[0]["content"]


# ---------------------------------------------------------------------------
# 2. Le contenu source fourni par l'utilisateur atteint bien le prompt
# ---------------------------------------------------------------------------

@patch(f"{MODULE}.call_llm_with_tools")
def test_source_content_reaches_the_prompt(mock_call_llm_with_tools, _skip_real_style_correction):
    mock_call_llm_with_tools.return_value = _final_message("Abstract condensed from the source content.")

    request = AgentRequest(
        instruction="Write an abstract from this content",
        context={"source_content": "This paper studies migratory patterns of Siberian tigers over a decade."},
    )
    result = WritingSupportAgent().run(request)

    assert result.status == AgentStatus.COMPLETED
    sent_messages = mock_call_llm_with_tools.call_args.kwargs["messages"]
    user_content = sent_messages[1]["content"]
    assert "Source content to work from:" in user_content
    assert "migratory patterns of Siberian tigers" in user_content


# ---------------------------------------------------------------------------
# 3. Degradation gracieuse si le LLM est indisponible
# ---------------------------------------------------------------------------

@patch(f"{MODULE}.call_llm_with_tools")
def test_llm_failure_returns_failed_honestly(mock_call_llm_with_tools):
    mock_call_llm_with_tools.side_effect = RuntimeError("Azure OpenAI is not configured")

    request = AgentRequest(instruction="Write an abstract", context={})
    result = WritingSupportAgent().run(request)

    assert result.status == AgentStatus.FAILED
    assert result.output["draft"] is None


# ---------------------------------------------------------------------------
# 4. Les exemples retrouves par les outils ne deviennent jamais des references citables
# ---------------------------------------------------------------------------

@patch(f"{MODULE}.search_citations")
@patch(f"{MODULE}.call_llm_with_tools")
def test_tool_results_never_become_citable_references(
    mock_call_llm_with_tools, mock_search_citations, _skip_real_style_correction
):
    mock_search_citations.return_value = [
        _fake_citation_hit("Background", "A Qdrant citation example, not a real reference.")
    ]
    mock_call_llm_with_tools.side_effect = [
        _tool_call_message("get_citation_examples", {"query": "tiger conservation"}),
        _final_message("Final draft."),
    ]

    request = AgentRequest(instruction="Write an abstract", context={})
    result = WritingSupportAgent().run(request)

    assert result.output["references_used"] == []
    assert "Qdrant citation example" not in str(result.output["references_used"])


# ---------------------------------------------------------------------------
# 5. Flux bout-en-bout via l'orchestrateur
# ---------------------------------------------------------------------------

@patch(f"{MODULE}.call_llm_with_tools")
@patch(f"{MODULE}.call_llm")
def test_orchestrator_writing_route_end_to_end(
    mock_call_llm, mock_call_llm_with_tools, _skip_real_style_correction
):
    mock_call_llm.return_value = "writing_support"
    mock_call_llm_with_tools.return_value = _final_message("Final generated abstract.")

    request = AgentRequest(
        instruction="Write an abstract about tiger population decline",
        context={"discovery_output": {"papers": ["Smith et al. 2020"], "is_placeholder": False}},
    )
    result = ScientificWritingOrchestrator().run(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output["draft"] == "Final generated abstract."
    assert result.output["references_used"] == ["Smith et al. 2020"]


# ---------------------------------------------------------------------------
# 6. Les noms scientifiques (nomenclature binomiale) survivent a la correction
# ---------------------------------------------------------------------------

def _fake_lt_match(offset: int, error_length: int):
    match = MagicMock()
    match.offset = offset
    match.error_length = error_length
    return match


def test_scientific_names_are_protected_from_style_correction():
    """Test dedie a la vraie logique de _apply_style_correction (pas mocke
    par _skip_real_style_correction) : verifie qu'un match qui tomberait sur
    un nom scientifique binomial (ex. Panthera tigris) est filtre avant
    d'etre applique."""
    from agents.Literature_Agent.subagents.writing.scientific_writing import (
        _apply_style_correction,
        _language_tool,
    )

    text = "This study focuses on Panthera tigris altaica in the wild."
    # 23..31 lands inside "Panthera tigris altaica" -> must be filtered out.
    bogus_match_on_scientific_name = _fake_lt_match(offset=23, error_length=8)
    # 0..18 is "This study focuses". The naive binomial pattern matched that
    # too, so ordinary prose at the start of a sentence was protected and
    # never corrected. This match must survive.
    real_match_on_plain_prose = _fake_lt_match(offset=0, error_length=18)

    fake_tool = MagicMock()
    fake_tool.check.return_value = [
        bogus_match_on_scientific_name,
        real_match_on_plain_prose,
    ]

    # `_language_tool` is an lru_cache singleton (one JVM start instead of one
    # per draft), so the MagicMock below would otherwise stay cached for every
    # later caller in the process.
    _language_tool.cache_clear()
    try:
        with patch("language_tool_python.LanguageTool", return_value=fake_tool), \
             patch("language_tool_python.utils.correct") as mock_correct:
            mock_correct.side_effect = lambda t, matches: t
            corrected, applied = _apply_style_correction(text)
    finally:
        _language_tool.cache_clear()

    assert applied is True
    passed_matches = mock_correct.call_args.args[1]
    assert bogus_match_on_scientific_name not in passed_matches
    assert real_match_on_plain_prose in passed_matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
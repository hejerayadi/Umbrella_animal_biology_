"""
Unit tests for workflows.llm.tool_loop.invoke_tool_loop_with_fallback.

Regression coverage for the FGF5 gene-mapper bug: when the model is forced
onto its final (tool_choice="none") turn before it's actually resolved
everything it wanted, it can write the tool call it wanted to make as
plain-text JSON (e.g. {"name": "resolve_go_term_name", "parameters": {...}})
instead of the required final-answer schema. That text used to be parsed as
if it *were* the final answer, silently failing with a "Model did not return
a go_id" error and forcing the deterministic fallback even though the model
never actually got a chance to answer.
"""
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from workflows.llm import tool_loop


@tool
async def fake_tool_a(x: str) -> str:
    """A fake tool for loop tests."""
    return f"resolved:{x}"


@tool
async def fake_tool_b(x: str) -> str:
    """Another fake tool for loop tests."""
    return f"other:{x}"


class _ScriptedLLM:
    """Stands in for a bind_tools-bound ChatNVIDIA client, replaying a fixed
    sequence of AIMessage responses regardless of which bind_tools() variant
    (with-tools vs tool_choice='none') is asked for."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, convo):
        msg = self.responses[self.calls]
        self.calls += 1
        return msg


def _patch_llm(monkeypatch, llm):
    monkeypatch.setattr(tool_loop, "get_llm", lambda temperature=0.1, model=None: llm)
    monkeypatch.setattr(tool_loop, "_candidate_models", lambda model: [None])


@pytest.mark.asyncio
async def test_disguised_tool_call_on_final_turn_is_honored_and_recovered(monkeypatch):
    """The exact FGF5 shape: forced-final turn produces a tool-call-shaped
    JSON blob as text. The loop should execute that tool call for real and
    grant one more turn, rather than treating the blob as a malformed final
    answer."""
    responses = [
        AIMessage(
            content='{"name": "fake_tool_a", "parameters": {"x": "GO:0022008"}}',
            tool_calls=[],
        ),
        AIMessage(
            content='{"go_id": "GO:0008543", "go_name": "FGFR signaling", "reasoning": "ok"}',
            tool_calls=[],
        ),
    ]
    _patch_llm(monkeypatch, _ScriptedLLM(responses))

    parsed, tool_call_log = await tool_loop.invoke_tool_loop_with_fallback(
        "system", "human", [fake_tool_a, fake_tool_b], max_turns=1,
    )

    assert parsed == {
        "go_id": "GO:0008543",
        "go_name": "FGFR signaling",
        "reasoning": "ok",
    }
    assert tool_call_log == [
        {"name": "fake_tool_a", "args": {"x": "GO:0022008"}, "result": "resolved:GO:0022008"}
    ]


@pytest.mark.asyncio
async def test_disguised_tool_call_grace_only_granted_once(monkeypatch):
    """If the model does this twice in a row, the loop must fail cleanly
    instead of looping forever or silently accepting the second blob as an
    answer."""
    responses = [
        AIMessage(
            content='{"name": "fake_tool_a", "parameters": {"x": "GO:0022008"}}',
            tool_calls=[],
        ),
        AIMessage(
            content='{"name": "fake_tool_b", "parameters": {"x": "GO:0000001"}}',
            tool_calls=[],
        ),
    ]
    _patch_llm(monkeypatch, _ScriptedLLM(responses))

    with pytest.raises(RuntimeError, match="non-JSON final answer"):
        await tool_loop.invoke_tool_loop_with_fallback(
            "system", "human", [fake_tool_a, fake_tool_b], max_turns=1,
        )


@pytest.mark.asyncio
async def test_legitimate_final_answer_is_not_mistaken_for_a_tool_call(monkeypatch):
    """A real final answer should never be misdetected as a disguised tool
    call just because it happens to contain a 'name' key."""
    responses = [
        AIMessage(
            content='{"go_id": "GO:0042633", "go_name": "hair cycle", "reasoning": "direct match"}',
            tool_calls=[],
        ),
    ]
    _patch_llm(monkeypatch, _ScriptedLLM(responses))

    parsed, tool_call_log = await tool_loop.invoke_tool_loop_with_fallback(
        "system", "human", [fake_tool_a, fake_tool_b], max_turns=1,
    )

    assert parsed["go_id"] == "GO:0042633"
    assert tool_call_log == []


def test_as_disguised_tool_call_helper_shapes():
    tools_by_name = {"fake_tool_a": fake_tool_a}

    # Genuine disguised call, "parameters" key.
    assert tool_loop._as_disguised_tool_call(
        {"name": "fake_tool_a", "parameters": {"x": "1"}}, tools_by_name
    ) == ("fake_tool_a", {"x": "1"})

    # Genuine disguised call, "args" key.
    assert tool_loop._as_disguised_tool_call(
        {"name": "fake_tool_a", "args": {"x": "1"}}, tools_by_name
    ) == ("fake_tool_a", {"x": "1"})

    # Tool name not among the bound tools -> not a disguised call.
    assert tool_loop._as_disguised_tool_call(
        {"name": "not_a_real_tool", "parameters": {"x": "1"}}, tools_by_name
    ) is None

    # A legitimate answer that happens to have a "name" key but no args shape.
    assert tool_loop._as_disguised_tool_call(
        {"name": "fake_tool_a", "go_id": "GO:1"}, tools_by_name
    ) is None

    # Not a dict at all.
    assert tool_loop._as_disguised_tool_call(None, tools_by_name) is None

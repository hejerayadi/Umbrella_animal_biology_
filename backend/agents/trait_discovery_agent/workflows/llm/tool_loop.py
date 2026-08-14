
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .client import _candidate_models, _parse_json_object, _retry_on_capacity, _is_missing_model_error, get_llm

logger = logging.getLogger(__name__)

# Hard ceiling on tool-call/tool-result round-trips inside a single bind_tools
# decision, so a model that keeps calling tools instead of answering can't spin
# forever. One real decision (§0.1 of the guide) should resolve in 1-2 turns.
MAX_TOOL_TURNS = 6


async def invoke_tool_loop_with_fallback(
    system_prompt: str,
    human_prompt: str,
    tools: list,
    *,
    temperature: float = 0.1,
    model: str | None = None,
    max_turns: int = MAX_TOOL_TURNS,
) -> tuple[dict, list[dict]]:
    """Run a bind_tools ReAct-style loop against NIM, with the same 404-driven
    model fallback as invoke_with_fallback.

    The model is given typed, direct access to `tools` (per guide §2/§5) and may
    call them zero or more times before returning a final JSON object as plain
    text. Returns (parsed_final_json, tool_call_log) — the tool_call_log lets
    the caller validate the final answer mechanically against what the tools
    actually returned (grounding rule, §0.1), rather than trusting the model's
    say-so.

    Raises RuntimeError if the model never produces a final JSON answer within
    max_turns, or if every candidate model 404s.
    """
    tools_by_name = {t.name: t for t in tools}

    last_error: Exception | None = None
    for candidate_model in _candidate_models(model):
        llm = get_llm(temperature=temperature, model=candidate_model)
        llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
        llm_final_turn = llm.bind_tools(
            tools, tool_choice="none", parallel_tool_calls=False
        )

        convo: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        tool_call_log: list[dict] = []

        try:
            for turn in range(max_turns):
                is_last_turn = turn == max_turns - 1
                active_llm = llm_final_turn if is_last_turn else llm_with_tools
                ai_msg = await _retry_on_capacity(lambda: active_llm.ainvoke(convo))
                convo.append(ai_msg)

                tool_calls = getattr(ai_msg, "tool_calls", None)
                if not tool_calls:
                    parsed = _parse_json_object(ai_msg.content)
                    if parsed is None:
                        raise RuntimeError(
                            f"Model returned a non-JSON final answer: {ai_msg.content!r}"
                        )
                    return parsed, tool_call_log

                if len(tool_calls) > 1:
                    # This NIM model only accepts a single tool call per turn.
                    # parallel_tool_calls=False should prevent this, but if the
                    # endpoint still returns several, keep only the first so the
                    # next request (which replays this AI message) doesn't 400.
                    tool_calls = tool_calls[:1]
                    ai_msg.tool_calls = tool_calls
                    if hasattr(ai_msg, "additional_kwargs"):
                        tc_raw = ai_msg.additional_kwargs.get("tool_calls")
                        if isinstance(tc_raw, list) and len(tc_raw) > 1:
                            ai_msg.additional_kwargs["tool_calls"] = tc_raw[:1]

                for call in tool_calls:
                    tool = tools_by_name[call["name"]]
                    result = await tool.ainvoke(call["args"])
                    tool_call_log.append(
                        {"name": call["name"], "args": call["args"], "result": result}
                    )
                    convo.append(
                        ToolMessage(
                            content=json.dumps(result, default=str),
                            tool_call_id=call["id"],
                        )
                    )

            raise RuntimeError(
                f"Tool-calling loop exceeded {max_turns} turns without a final answer"
            )
        except Exception as exc:
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")

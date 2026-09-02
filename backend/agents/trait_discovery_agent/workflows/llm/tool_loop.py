import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .client import (
    NonJSONFinalAnswerError,
    TruncatedCompletionError,
    _candidate_models,
    _parse_json_object,
    _retry_on_capacity,
    _is_advance_worthy_error,
    _is_truncated_completion,
    _reasoning_off_preamble,
    get_llm,
)

logger = logging.getLogger(__name__)

# Hard ceiling on tool-call/tool-result round-trips inside a single bind_tools
# decision, so a model that keeps calling tools instead of answering can't spin
# forever. One real decision (§0.1 of the guide) should resolve in 1-2 turns,
# but hosted NIM models occasionally need an extra round-trip or two under
# real latency/load (see the ~60s silent-poll note below) even with the
# repeated-identical-call guard in the loop -- observed on BRCA1 despite that
# guard, i.e. not every turn-exhaustion case is a repeated call. 8 gives that
# headroom without meaningfully raising the cost of a model that's genuinely
# stuck (the deterministic fallback in each subagent still catches that case
# either way).
MAX_TOOL_TURNS = 8


def _coerce_stringified_json_args(args: dict) -> dict:
    """Some NIM-hosted models (observed with meta/llama-3.1-70b-instruct) emit
    list/dict-typed tool arguments as a JSON-encoded string instead of a
    native JSON array/object — e.g. go_ids='["GO:0008284", "GO:0008593"]'
    instead of go_ids=["GO:0008284", "GO:0008593"]. The tool's Pydantic
    schema then rejects it as `str` where `list[str]` was expected, and the
    whole tool call fails before the tool body ever runs (§9 catches this as
    a generic exception and falls back — silently discarding a real,
    otherwise-successful model decision every time this happens).

    Only strings that actually parse as a JSON array or object are touched;
    plain string arguments (a go_id, a gene_symbol, etc.) pass through
    unchanged. This runs once, at the loop level, so every tool bound via
    invoke_tool_loop_with_fallback benefits — not just the one tool where
    this was first observed.

    Raises TruncatedCompletionError if a value LOOKS like the start of a
    JSON array/object (`[` or `{`) but fails to parse — that combination
    essentially only happens when max_tokens cut generation off mid-value
    (observed verbatim: go_ids='["GO:0051726", "GO:0008630", "GO:0'). This
    used to fall through the `continue` below and hand the raw truncated
    string to the tool, which raised an opaque, non-advance-worthy Pydantic
    ValidationError several layers away from the actual cause, discarding
    the LLM's pick and falling back to the deterministic heuristic every
    time a batched call (e.g. resolve_go_term_names over a dozen+ GO ids)
    ran long. Raising here instead lets the caller treat it the same as any
    other truncated completion — see _is_truncated_completion.
    """
    coerced = dict(args)
    for key, value in args.items():
        if isinstance(value, str) and value[:1] in "[{":
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TruncatedCompletionError(
                    f"Tool argument {key!r} looks like truncated JSON "
                    f"(max_tokens cut the completion off mid-value): {value!r}"
                ) from exc
            if isinstance(parsed_value, (list, dict)):
                coerced[key] = parsed_value
    return coerced


def _as_disguised_tool_call(parsed: dict | None, tools_by_name: dict) -> tuple[str, dict] | None:
    """Detect a model that, when forced to answer with tool_choice="none", writes
    the tool call it wanted to make as plain-text JSON instead of the required
    final-answer schema — e.g. {"name": "resolve_go_term_name",
    "parameters": {"go_id": "GO:..."}} instead of {"go_id": ..., "go_name": ...,
    "reasoning": ...}.

    Returns (tool_name, tool_args) if `parsed` looks like this shape and names a
    tool the loop actually has bound, else None (including when `parsed` is a
    legitimate final answer that merely happens to share a key name).
    """
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or name not in tools_by_name:
        return None
    args = parsed.get("parameters")
    if args is None:
        args = parsed.get("args")
    if not isinstance(args, dict):
        return None
    return name, args


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
            SystemMessage(content=_reasoning_off_preamble() + system_prompt),
            HumanMessage(content=human_prompt),
        ]
        tool_call_log: list[dict] = []
        # A model forced onto its final (tool_choice="none") turn can still try
        # to "call" a tool by writing the call out as plain-text JSON instead of
        # the real answer schema (see _as_disguised_tool_call). We honor that
        # once — running the tool it wanted and pushing the turn ceiling out by
        # one so it gets a genuine forced-final turn afterward — rather than
        # burning the whole decision on a formatting slip.
        grace_used = False
        turns_limit = max_turns

        try:
            turn = 0
            while turn < turns_limit:
                is_last_turn = turn == turns_limit - 1
                active_llm = llm_final_turn if is_last_turn else llm_with_tools
                turn_start = time.monotonic()
                logger.info(
                    "Turn %d/%d: invoking %s (tool_choice=%s) — note: NIM can silently "
                    "poll for up to ~60s on a 202 before this returns anything.",
                    turn + 1, turns_limit, candidate_model,
                    "none" if is_last_turn else "auto",
                )
                ai_msg = await _retry_on_capacity(lambda: active_llm.ainvoke(convo))
                logger.info(
                    "Turn %d/%d: got a response after %.1fs.",
                    turn + 1, turns_limit, time.monotonic() - turn_start,
                )
                if _is_truncated_completion(ai_msg):
                    # Caught here, before parsing .content or executing a
                    # tool call with possibly-incomplete arguments, since not
                    # every truncation produces an obviously-malformed value
                    # (see TruncatedCompletionError) -- finish_reason=="length"
                    # is the API telling us directly, so there's no reason to
                    # wait for a downstream parse/Pydantic failure to infer it.
                    raise TruncatedCompletionError(
                        f"{candidate_model}: completion truncated by max_tokens "
                        f"(finish_reason=length) on turn {turn + 1}/{turns_limit}"
                    )
                convo.append(ai_msg)

                tool_calls = getattr(ai_msg, "tool_calls", None)
                if not tool_calls:
                    parsed = _parse_json_object(ai_msg.content)
                    disguised = _as_disguised_tool_call(parsed, tools_by_name)

                    if disguised is not None and not grace_used:
                        # Forced not to call a tool, the model wrote the call out
                        # as text instead of answering. Run it for real, tell the
                        # model, and give it exactly one more turn to answer.
                        grace_used = True
                        turns_limit += 1
                        name, args = disguised
                        logger.warning(
                            "Model wrote a disguised tool call instead of a final "
                            "answer on its forced turn (%s(%s)); executing it and "
                            "granting one extra turn to answer.",
                            name, args,
                        )
                        tool = tools_by_name[name]
                        args = _coerce_stringified_json_args(args)
                        result = await tool.ainvoke(args)
                        tool_call_log.append({"name": name, "args": args, "result": result})
                        convo.append(HumanMessage(
                            content=(
                                f"You wrote a call to {name}({args}) as plain text "
                                f"instead of answering. Here is what it would have "
                                f"returned: {json.dumps(result, default=str)}. You "
                                "must now reply with ONLY the final JSON object — no "
                                "more tool calls."
                            )
                        ))
                        turn += 1
                        continue

                    if parsed is None or disguised is not None:
                        if ai_msg.content is None:
                            reasoning = (ai_msg.additional_kwargs or {}).get("reasoning_content")
                            logger.warning(
                                "%s: content is None (likely still mid hidden-analysis "
                                "channel at max_tokens); reasoning_content length=%s",
                                candidate_model, len(reasoning) if reasoning else 0,
                            )
                        raise NonJSONFinalAnswerError(
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
                    call_args = _coerce_stringified_json_args(call["args"])
                    if call_args != call["args"]:
                        logger.info(
                            "Coerced stringified-JSON args for %s: %r -> %r",
                            call["name"], call["args"], call_args,
                        )
                    # A model that calls the same (name, args) pair it already
                    # called earlier in this loop learns nothing new from
                    # re-running it -- these tools are deterministic reads
                    # (re-fetch the same candidate list), so a repeat is a
                    # model that's stuck, not one gathering information. Left
                    # unchecked this is exactly what burns through
                    # MAX_TOOL_TURNS without ever reaching a final answer
                    # (observed on MC1R/HRAS). Reuse the earlier result and
                    # tell it plainly instead of re-invoking and hoping the
                    # next turn is different.
                    repeat = next(
                        (
                            logged for logged in tool_call_log
                            if logged["name"] == call["name"] and logged["args"] == call_args
                        ),
                        None,
                    )
                    if repeat is not None:
                        logger.warning(
                            "Model repeated an identical tool call %s(%s); reusing "
                            "the earlier result instead of re-invoking, and telling "
                            "it plainly to stop calling and answer.",
                            call["name"], call_args,
                        )
                        result = repeat["result"]
                        convo.append(
                            ToolMessage(
                                content=(
                                    json.dumps(result, default=str)
                                    + " (note: identical to your earlier call -- this "
                                    "won't return anything new. You already have "
                                    "everything you need; reply with ONLY the final "
                                    "JSON object now.)"
                                ),
                                tool_call_id=call["id"],
                            )
                        )
                        continue
                    result = await tool.ainvoke(call_args)
                    tool_call_log.append(
                        {"name": call["name"], "args": call_args, "result": result}
                    )
                    convo.append(
                        ToolMessage(
                            content=json.dumps(result, default=str),
                            tool_call_id=call["id"],
                        )
                    )
                turn += 1

            raise RuntimeError(
                f"Tool-calling loop exceeded {turns_limit} turns without a final answer "
                f"(calls made: {[(c['name'], c['args']) for c in tool_call_log]!r})"
            )
        except Exception as exc:
            last_error = exc
            if not _is_advance_worthy_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")
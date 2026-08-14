import asyncio
import json
import logging
import os
import random
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")
FALLBACK_MODELS = (DEFAULT_MODEL,)

# Hard ceiling on tool-call/tool-result round-trips inside a single bind_tools
# decision, so a model that keeps calling tools instead of answering can't spin
# forever. One real decision (§0.1 of the guide) should resolve in 1-2 turns.
MAX_TOOL_TURNS = 6


def _get_api_key() -> str | None:
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.1, model: str | None = None) -> ChatNVIDIA:
    """Create a ChatNVIDIA client with the configured API key explicitly passed in."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_NIM_API_KEY is not set. Copy .env.example to .env and add your NIM key."
        )
    return ChatNVIDIA(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=512,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )


def _is_missing_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message and ("not found" in message or "function" in message)


def _is_capacity_error(exc: Exception) -> bool:
    """503s from a saturated local NIM worker pool ('request limit reached',
    ResourceExhausted) are transient — worth a short retry, not an immediate
    fallback or model swap."""
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


MAX_CAPACITY_RETRIES = 3
CAPACITY_RETRY_BASE_SECONDS = 1.5


async def _retry_on_capacity(coro_fn):
    """Call coro_fn() (a zero-arg async callable), retrying with jittered
    exponential backoff on transient worker-pool exhaustion (503)."""
    last_error: Exception | None = None
    for attempt in range(MAX_CAPACITY_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            if not _is_capacity_error(exc) or attempt == MAX_CAPACITY_RETRIES:
                raise
            last_error = exc
            delay = CAPACITY_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "NIM worker pool exhausted (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_CAPACITY_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_error  # pragma: no cover - unreachable


async def invoke_with_fallback(
    prompt: ChatPromptTemplate,
    payload: dict,
    *,
    temperature: float = 0.1,
    model: str | None = None,
):
    """Invoke a prompt against NIM, retrying with a known-good fallback model on 404s.

    The account behind this workspace can expose different model/function IDs over time.
    When one alias disappears, the graph should still be able to complete with an
    alternate model rather than failing hard at the first escalation node.
    """
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        llm = get_llm(temperature=temperature, model=candidate_model)
        chain = prompt | llm
        try:
            return await _retry_on_capacity(lambda: chain.ainvoke(payload))
        except Exception as exc:  # pragma: no cover - exercised in live NIM runs
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")


async def invoke_json_with_fallback(
    system_prompt: str,
    human_prompt: str,
    *,
    temperature: float = 0.1,
    model: str | None = None,
) -> dict:
    """Invoke NIM for a plain JSON answer with NO tools bound.

    Use this when the caller has already fetched and resolved everything the
    model needs (e.g. all candidate names) and the model's only job is to
    reason over that fixed context and answer. Skipping bind_tools entirely
    here removes a whole class of failure modes seen with the tool-loop path:
    a model that still calls a "double-check" tool despite already having
    full information, re-discovers unnamed data, burns its turn budget, and
    ends up emitting a fake tool-call-shaped string as its "final answer"
    once forced to stop calling tools.

    Raises RuntimeError if the model's response isn't valid JSON, or if every
    candidate model 404s.
    """
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        llm = get_llm(temperature=temperature, model=candidate_model)
        convo = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        try:
            ai_msg = await _retry_on_capacity(lambda: llm.ainvoke(convo))
            parsed = _parse_json_object(ai_msg.content)
            if parsed is None:
                raise RuntimeError(
                    f"Model returned a non-JSON final answer: {ai_msg.content!r}"
                )
            return parsed
        except Exception as exc:
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")


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
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
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


def _parse_json_object(content: str) -> dict | None:
    """Best-effort extraction of a single JSON object from a model's final text,
    tolerating markdown code fences."""
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
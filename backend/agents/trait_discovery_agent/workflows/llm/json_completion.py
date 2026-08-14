import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .client import _candidate_models, _parse_json_object, _retry_on_capacity, _is_missing_model_error, get_llm

logger = logging.getLogger(__name__)


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
    last_error: Exception | None = None
    for candidate_model in _candidate_models(model):
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
    model needs and the model's only job is to reason over that fixed context
    and answer. For decisions where the model should fetch/resolve data
    itself, use invoke_tool_loop_with_fallback (tool_loop.py) instead.

    Raises RuntimeError if the model's response isn't valid JSON, or if every
    candidate model 404s.
    """
    last_error: Exception | None = None
    for candidate_model in _candidate_models(model):
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

"""The LLM interface the planner and critic talk to.

Deliberately narrow: the agent needs "given these messages, return text" and
nothing more. Keeping the surface this small is what lets the null
implementation below be a real, honest substitute rather than a stub that
breaks the moment it is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in an LLM exchange."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """What the reasoning layer requires of a model backend."""

    @property
    def available(self) -> bool:
        """Whether calls to this client will actually reach a model."""
        ...

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        """Return the model's reply to `messages`."""
        ...


class NullLLMClient:
    """Stands in when no provider is configured.

    `available` is False, which every caller checks before planning: the graph
    then runs its deterministic fallback plan instead of an LLM-chosen one.
    Calling `complete` anyway is a programming error, so it raises loudly
    rather than returning an empty string that would look like a model answer.
    """

    @property
    def available(self) -> bool:
        return False

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        raise RuntimeError(
            "No LLM provider configured (LLM_PROVIDER=none). Check `available` before "
            "calling `complete`, or configure a provider."
        )


class OpenAILLMClient:
    """OpenAI-compatible chat completions, via langchain-openai.

    The import is deferred to construction so that `langchain-openai` stays an
    optional dependency: an install that never sets LLM_PROVIDER=openai does
    not need the package present.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "LLM_PROVIDER=openai requires the 'llm' extra: uv sync --extra llm"
            ) from error

        self._model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    @property
    def available(self) -> bool:
        return True

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        payload = [(message.role, message.content) for message in messages]
        response = await self._model.ainvoke(payload)
        content = response.content
        # langchain returns either a string or a list of content blocks
        # depending on the model; normalise to text.
        if isinstance(content, str):
            return content
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )

"""The LLM interface the planner and critic talk to.

Deliberately narrow: the agent needs "given these messages, return text" and
nothing more. Keeping the surface this small is what lets the null
implementation below be a real, honest substitute rather than a stub that
breaks the moment it is used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from configuration.logging import get_logger
from domain.exceptions import ExternalServiceError
from infrastructure.http.client import ServiceClient
from infrastructure.http.retry import RetryPolicy

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in an LLM exchange."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    """A model reply and what it cost.

    Tokens are metered by the budget policy. Clients that cannot report usage
    return zeros, which the policy treats as free rather than as an error -
    an unmetered provider should not stall the loop.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


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

    async def complete_with_usage(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """The reply plus its token cost, for budget metering."""
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

    async def complete_with_usage(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        raise RuntimeError(
            "No LLM provider configured (LLM_PROVIDER=none). Check `available` before "
            "calling the model, or configure a provider."
        )


class AzureOpenAIClient:
    """Azure OpenAI chat completions, over plain HTTP.

    Deliberately not the openai SDK: this agent needs one endpoint, and going
    direct means the same `ServiceClient` handles retries, timeouts and error
    translation as for every other external call - rather than a second
    retry stack with its own behaviour.

    Azure's URL layout differs from OpenAI's: the deployment name is in the
    path and the API version is a required query parameter. A mismatch there
    surfaces as a 404 on the deployment, which reads like a missing model.
    """

    def __init__(
        self,
        *,
        base_url: str,
        deployment: str,
        api_key: str,
        api_version: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self._deployment = deployment
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = ServiceClient(
            service="azure_openai",
            base_url=base_url,
            timeout=timeout,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay=2.0),
            headers={"api-key": api_key, "Content-Type": "application/json"},
        )

    @property
    def available(self) -> bool:
        return True

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        return (await self.complete_with_usage(messages, max_tokens=max_tokens)).text

    async def complete_with_usage(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """A completion plus what it cost.

        The token count is what the budget policy meters against, and it is
        only available here - the service reports it, we cannot infer it.
        """
        path = f"/openai/deployments/{self._deployment}/chat/completions"
        payload: dict[str, Any] = {
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "max_completion_tokens": max_tokens or self._max_tokens,
        }
        # Reasoning-tuned deployments reject an explicit temperature; only send
        # one when it differs from the default the service already applies.
        if self._temperature:
            payload["temperature"] = self._temperature

        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        try:
            response = await self._client.request(
                "POST",
                path,
                params={"api-version": self._api_version},
                json=payload,
            )
        except ExternalServiceError:
            # Structured output is not supported by every deployment or API
            # version, and the rejection looks like an ordinary 400. Retry once
            # without it rather than losing the call - the callers all parse
            # tolerantly anyway.
            if json_schema is None:
                raise
            _log.warning("azure_json_schema_unsupported", detail="retrying without schema")
            payload.pop("response_format", None)
            response = await self._client.request(
                "POST",
                path,
                params={"api-version": self._api_version},
                json=payload,
            )

        try:
            body = response.json()
            text = str(body["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ExternalServiceError(
                "azure_openai", f"unexpected completion payload: {response.text[:300]!r}"
            ) from error

        usage = body.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


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

    async def complete_with_usage(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Reply with zero-cost usage.

        langchain does not surface token counts consistently across providers,
        and reporting a wrong number would be worse than reporting none: the
        budget would throttle on fiction. Zero means "unmetered" to the policy.
        """
        return Completion(text=await self.complete(messages, max_tokens=max_tokens))

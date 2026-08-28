"""Azure OpenAI, reached through langchain-openai.

Matches `backend/orchestrator/llm.py` and the Protein agent: the same wire
format, the same `with_structured_output(..., include_raw=True)` call. The
`include_raw` flag is the part that matters here - without it a reply the model
could not shape into the schema raises, and a planning call that fails takes
down a run that has real evidence in hand. With it, the parse failure arrives
as data and this client answers None, which routes the caller onto its
deterministic fallback.

Construction is lazy and cached. Building the client eagerly would make every
process that merely imports the agent pay for a network-capable object it may
never use, and would turn a missing key into an import error rather than a
configuration state the health endpoint can report.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.config.settings import AzureOpenAISettings, LlmSettings
from reconstruction_agent.integrations.llm.base import StructuredT
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)


class AzureOpenAIClient:
    """Structured completions from an Azure OpenAI deployment."""

    def __init__(self, azure: AzureOpenAISettings, llm: LlmSettings) -> None:
        self._azure = azure
        self._llm = llm
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self._azure.configured)

    def _chat_model(self) -> Any:
        """The underlying chat model, built once per process."""
        if self._model is not None:
            return self._model

        from langchain_openai import AzureChatOpenAI

        assert self._azure.api_key is not None  # guarded by `available`
        self._model = AzureChatOpenAI(
            azure_endpoint=self._azure.base_url,
            azure_deployment=self._azure.deployment,
            api_key=self._azure.api_key.get_secret_value(),
            api_version=self._azure.api_version,
            temperature=self._llm.temperature,
            max_tokens=self._llm.max_tokens,
        )
        return self._model

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[StructuredT],
    ) -> StructuredT | None:
        """One structured completion, or None if the model gave nothing usable."""
        if not self.available:
            return None

        try:
            model = self._chat_model().with_structured_output(schema, include_raw=True)
            reply = await model.ainvoke(
                [("system", system), ("human", user)],
            )
        except Exception as error:  # noqa: BLE001 - transport, auth, quota, parse
            # Deliberately broad. Every failure mode here has the same correct
            # response - fall back to the deterministic strategy - and letting
            # any of them escape would discard evidence already gathered.
            _log.warning("llm_call_failed", schema=schema.__name__, error=str(error))
            return None

        parsed = reply.get("parsed") if isinstance(reply, dict) else None
        if parsed is None:
            _log.warning(
                "llm_reply_unparseable",
                schema=schema.__name__,
                error=str(reply.get("parsing_error")) if isinstance(reply, dict) else None,
            )
            return None
        return parsed if isinstance(parsed, schema) else None

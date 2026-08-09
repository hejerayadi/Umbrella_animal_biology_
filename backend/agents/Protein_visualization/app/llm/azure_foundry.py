"""Azure AI Foundry / Azure OpenAI client.

LangChain is used for one thing only: turning an evidence pack into a validated
structured output. No chain, agent, memory or tool-calling loop lives here — the
workflow is LangGraph's job and the biological facts are the tools' job.
"""

import json
import logging
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from backend.agents.Protein_visualization.app.configuration.settings import Settings
from backend.agents.Protein_visualization.app.llm.prompts import (
    PROTEIN_EXPLANATION_SYSTEM_PROMPT,
    SCIENTIFIC_CRITIC_SYSTEM_PROMPT,
)
from backend.agents.Protein_visualization.app.llm.schemas import CriticOutput, ExplanationOutput

logger = logging.getLogger("app.llm")


class AzureFoundryClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.llm_provider != "disabled"
            and self.settings.azure_openai_base_url
            and self.settings.azure_openai_deployment
            and self.settings.azure_openai_api_key
        )

    @cached_property
    def _model(self) -> Any:
        from langchain_openai import AzureChatOpenAI

        assert self.settings.azure_openai_base_url
        assert self.settings.azure_openai_deployment
        assert self.settings.azure_openai_api_key
        parameters: dict[str, Any] = {
            "azure_endpoint": self.settings.azure_openai_base_url.rstrip("/"),
            "azure_deployment": self.settings.azure_openai_deployment,
            "api_key": self.settings.azure_openai_api_key,
            "api_version": self.settings.azure_ai_api_version,
            "timeout": self.settings.http_timeout_seconds,
            "max_retries": self.settings.http_max_retries,
        }
        # Some deployments reject an explicit temperature; leave it unset then.
        if self.settings.azure_temperature is not None:
            parameters["temperature"] = self.settings.azure_temperature
        return AzureChatOpenAI(**parameters)

    async def explain(self, context: dict[str, object]) -> ExplanationOutput:
        return await self._invoke(PROTEIN_EXPLANATION_SYSTEM_PROMPT, context, ExplanationOutput)

    async def critique(self, context: dict[str, object]) -> CriticOutput:
        return await self._invoke(SCIENTIFIC_CRITIC_SYSTEM_PROMPT, context, CriticOutput)

    async def _invoke[T: BaseModel](
        self, system_prompt: str, context: dict[str, object], schema: type[T]
    ) -> T:
        if not self.enabled:
            raise RuntimeError("Azure OpenAI is not configured")
        structured = self._model.with_structured_output(schema)
        result = await structured.ainvoke(
            [
                ("system", system_prompt),
                ("human", json.dumps(context, default=str)),
            ]
        )
        # Some deployments return the parsed model, others the raw dict.
        return result if isinstance(result, schema) else schema.model_validate(result)

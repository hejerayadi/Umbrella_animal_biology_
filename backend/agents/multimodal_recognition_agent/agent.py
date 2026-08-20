"""The Recognition Agent's entry class.

`api.py` imports this and calls `run(request) -> AgentResult`. That signature is
the whole interface the rest of the platform depends on; everything below it is
this agent's own business.

Providers are built once, here, rather than per request - the same reason
`api.py` builds the agent at import time. Today they are mocks and cost nothing;
the real BioCLIP-2 provider will load a model and the real Qdrant retriever will
open a connection, and neither belongs in the request path.

Retrieval mode is chosen at construction:

- `mock` (default) uses the local fixture retriever. Development and tests only.
  Its provenance says `mock_local_development`, so nothing produced this way can
  be mistaken for the Sprint 2 Qdrant deliverable.
- `real` uses the Qdrant adapter, which refuses to construct until Chahd's
  collection contract is configured.
"""
from __future__ import annotations

import logging
import os

from .adapters.bioclip import MockBioCLIP2Provider
from .adapters.qdrant_mock import MockQdrantRetriever
from .adapters.qdrant_real import RealQdrantRetriever
from .adapters.reasoning_llm import ReasoningLLM, build_reasoning_llm
from .adapters.retrieval import RetrievalProvider
from .adapters.taxonomy import MockTaxonomyProvider
from .config import RecognitionConfig
from .domain.errors import RecognitionError
from .schema import AgentRequest, AgentResult, AgentStatus
from .text_analysis import RuleBasedTextAnalyzer
from .workflows import RecognitionWorkflow

_logger = logging.getLogger(__name__)


def _build_retriever(config: RecognitionConfig) -> RetrievalProvider:
    if config.retrieval_mode == "real":
        # Reads the key straight from the environment and hands it to the client.
        # It is never stored on config, never logged and never returned.
        return RealQdrantRetriever(config.qdrant, api_key=os.getenv("QDRANT_API_KEY"))

    _logger.info(
        "[Recognition] retrieval mode 'mock': using local development fixtures. "
        "This is NOT the Sprint 2 Qdrant deliverable."
    )
    return MockQdrantRetriever(config.mock_embedding_dimension)


class RecognitionAgent:
    """The Multimodal Species Recognition Agent."""

    def __init__(
        self,
        config: RecognitionConfig | None = None,
        *,
        retriever: RetrievalProvider | None = None,
        taxonomy_provider: MockTaxonomyProvider | None = None,
        reasoning_llm: ReasoningLLM | None = None,
    ) -> None:
        self.config = config or RecognitionConfig.from_env()

        embedding_provider = MockBioCLIP2Provider(
            dimension=self.config.mock_embedding_dimension,
            version=self.config.mock_provider_version,
            image_seed_overrides=self.config.image_seed_overrides,
        )
        taxonomy = taxonomy_provider or MockTaxonomyProvider()

        # The analyser knows which names exist so it can tell "a species I know
        # that the image did not retrieve" (a conflict) from "a word I do not
        # recognise" (no signal). It still cannot add a candidate.
        analyzer = RuleBasedTextAnalyzer(known_names=taxonomy.known_names())

        # Disabled unless RECOGNITION_REASONING_LLM_ENABLED is explicitly set.
        # `build_reasoning_llm` opens no connection - the client, if any, is
        # constructed lazily on the first permitted call.
        llm = reasoning_llm if reasoning_llm is not None else build_reasoning_llm(
            enabled=self.config.reasoning_llm_enabled,
            timeout_seconds=self.config.reasoning_llm_timeout_seconds,
        )

        self._workflow = RecognitionWorkflow(
            config=self.config,
            embedding_provider=embedding_provider,
            retriever=retriever if retriever is not None else _build_retriever(self.config),
            taxonomy_provider=taxonomy,
            text_analyzer=analyzer,
            reasoning_llm=llm,
        )

    def run(self, request: AgentRequest) -> AgentResult:
        """One request in, one `AgentResult` out. Always."""
        try:
            return self._workflow.run(request.instruction, request.context)
        except RecognitionError as exc:
            # The workflow already converts its own errors; this catches one
            # raised while building a provider (e.g. an unfrozen Qdrant contract).
            return AgentResult(status=AgentStatus.FAILED, output=exc.as_output())

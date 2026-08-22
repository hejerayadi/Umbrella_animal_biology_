"""The Recognition Agent's entry class.

`api.py` imports this and calls `run(request) -> AgentResult`. That signature is
the whole interface the rest of the platform depends on; everything below it is
this agent's own business.

Providers are built once, here, rather than per request - the same reason
`api.py` builds the agent at import time. Today the classifier is a mock and
costs nothing; the real BioCLIP-2 provider will load a model, and that does not
belong in the request path.

There is no retriever to build and no retrieval mode to choose. The agent has
one source of species candidates - `BioCLIP2Classifier.classify` - and swapping
the Sprint 2 mock for real BioCLIP-2 label classification later means passing a
different object here and changing nothing else.
"""
from __future__ import annotations

import logging

from .adapters.bioclip import BioCLIP2Classifier, build_classifier
from .adapters.reasoning_llm import ReasoningLLM, build_recognition_llm
from .adapters.taxonomy import MockTaxonomyProvider, build_taxonomy_provider
from .config import RecognitionConfig
from .domain.errors import RecognitionError
from .schema import AgentRequest, AgentResult, AgentStatus
from .text_analysis import RuleBasedTextAnalyzer
from .workflows import RecognitionWorkflow

_logger = logging.getLogger(__name__)


class RecognitionAgent:
    """The Multimodal Species Recognition Agent."""

    def __init__(
        self,
        config: RecognitionConfig | None = None,
        *,
        classifier: BioCLIP2Classifier | None = None,
        taxonomy_provider: MockTaxonomyProvider | None = None,
        reasoning_llm: ReasoningLLM | None = None,
    ) -> None:
        self.config = config or RecognitionConfig.from_env()

        # An injected provider wins outright - that is how the offline suite
        # supplies stubs with no environment, no network and no credentials.
        # Nothing else may choose a provider: when none is injected the factory
        # decides, and the factory refuses any mode it cannot honestly build.
        if classifier is None:
            classifier = build_classifier(self.config)
            _logger.info(
                "[Recognition] classifier mode=%s provider=%s. Real BioCLIP-2 "
                "inference is NOT executed in mock mode.",
                self.config.bioclip_provider_mode,
                getattr(classifier, "provider_name", "unknown"),
            )

        if taxonomy_provider is None:
            taxonomy = build_taxonomy_provider(self.config)
            _logger.info(
                "[Recognition] taxonomy mode=%s. No GBIF or NCBI call is made in "
                "mock mode.",
                self.config.taxonomy_provider_mode,
            )
        else:
            taxonomy = taxonomy_provider

        # The analyser knows which names exist so it can tell "a species I know
        # that the classifier did not return" (a conflict) from "a word I do not
        # recognise" (no signal). It still cannot add a candidate.
        analyzer = RuleBasedTextAnalyzer(known_names=taxonomy.known_names())

        # Disabled unless RECOGNITION_LLM_PROVIDER_MODE says otherwise. No
        # connection is opened here - the Azure client, if any, is built lazily
        # on the first permitted call.
        llm = reasoning_llm if reasoning_llm is not None else build_recognition_llm(
            self.config.reasoning_llm_provider_mode,
            timeout_seconds=self.config.reasoning_llm_timeout_seconds,
        )

        self._workflow = RecognitionWorkflow(
            config=self.config,
            classifier=classifier,
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
            # raised outside a node.
            return AgentResult(status=AgentStatus.FAILED, output=exc.as_output())

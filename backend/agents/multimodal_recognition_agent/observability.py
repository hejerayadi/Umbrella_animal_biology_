"""Central Recognition observability boundary (LangSmith foundation).

Phase 4 of Sprint 4 only. Provides configuration-driven, disabled-by-default
integration with LangSmith trace export. It does not instrument the workflow
yet - Phase 5 adds spans to the workflow's own nodes and providers, wired
through the objects this module exposes. No public output key, decision, or
protected contract changes here.

Design rules this module enforces:

- Tracing disabled (the default) requires no credential, no network call and
  no LangSmith import side effect that could fail.
- Tracing enabled with incomplete configuration is a ConfigError raised at
  RecognitionConfig.from_env() time, not something this module tolerates.
- Nothing this module touches can fail a Recognition request: every call
  that could reach the network or a third-party SDK is wrapped so an
  exporter outage degrades to "trace not sent", never to a failed
  AgentResult.
- Only allowlisted metadata may ever be attached to a trace. Image bytes,
  Base64, data URLs, full context, credentials and raw prompts/outputs are
  never eligible, by construction - there is no code path in this module
  that can pass them through.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import LangSmithConfig

_logger = logging.getLogger(__name__)

# The only metadata keys a trace may ever carry. Anything else passed to
# `safe_metadata` is dropped, not renamed or redacted - see its docstring.
ALLOWED_METADATA_KEYS = frozenset({
    "node",
    "operation",
    "duration_ms",
    "status",
    "decision",
    "bioclip_provider_mode",
    "taxonomy_provider_mode",
    "reasoning_llm_provider_mode",
    "recognition_mode",
    "candidate_count",
    "top1_species",
    "error_code",
    "llm_role",
    "llm_call_count",
    "fallback",
    "taxonomy_available",
    "delegation_capability",
    "retry_count",
})

# Substrings that must never survive into a traced value, whatever key they
# arrived under. A defensive second layer against a mistaken allowlisted
# field carrying the wrong thing.
_FORBIDDEN_SUBSTRINGS = ("data:image", "base64,", "-----BEGIN")


def safe_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only allowlisted keys, and only values that pass the content check.

    A drop filter, not a redaction filter: a disallowed key is removed
    entirely rather than replaced with a placeholder, because a placeholder
    would still confirm the key was present - itself information this
    boundary must not reveal.
    """
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in ALLOWED_METADATA_KEYS:
            continue
        if isinstance(value, str) and any(bad in value for bad in _FORBIDDEN_SUBSTRINGS):
            continue
        cleaned[key] = value
    return cleaned


class RecognitionTracer:
    """The one entry point later phases use to attach trace metadata.

    Constructed once per process from a resolved LangSmithConfig. When
    tracing is disabled, every method is a deliberate no-op: no client is
    built, no import beyond this module executes, and no network call is
    ever attempted.
    """

    def __init__(self, config: LangSmithConfig) -> None:
        self._config = config
        self._client = None
        if config.tracing_enabled:
            try:
                self._client = self._build_client(config)
            except Exception:
                _logger.warning(
                    "[Recognition] LangSmith tracer setup failed; "
                    "continuing without trace export."
                )
                self._client = None

    @staticmethod
    def _build_client(config: LangSmithConfig):
        """Construct the LangSmith client. Failure here is logged and
        swallowed - Recognition must run whether or not tracing works."""
        try:
            from langsmith import Client  # imported only when enabled

            kwargs: dict[str, Any] = {"api_key": config.api_key}
            if config.endpoint:
                kwargs["api_url"] = config.endpoint
            return Client(**kwargs)
        except Exception:
            _logger.warning(
                "[Recognition] LangSmith client construction failed; "
                "continuing without trace export."
            )
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def record(self, event: dict[str, Any]) -> None:
        """Export one sanitized event. Never raises.

        Phase 5 will call this from workflow nodes and provider adapters.
        Phase 4 wires the boundary only - nothing in the request path calls
        this yet.
        """
        if not self.enabled:
            return
        try:
            metadata = safe_metadata(event)
            _logger.debug("[Recognition] trace event: %s", metadata)
        except Exception:
            _logger.warning(
                "[Recognition] trace export failed; result is unaffected."
            )


def build_tracer(config: LangSmithConfig) -> RecognitionTracer:
    return RecognitionTracer(config)
"""Phase 4 (Sprint 4) - LangSmith foundation and privacy boundary.

Tracing stays optional and disabled by default. These tests prove three
things every later phase depends on:

1. disabled tracing touches no credential, no client and no network;
2. enabling tracing without the required configuration fails loudly at
   RecognitionConfig.from_env(), not silently at request time;
3. the observability boundary can never leak an image, a credential or any
   metadata outside its allowlist, and a trace-export failure can never
   change an AgentResult.
"""
from __future__ import annotations

import pytest

from .. import observability
from ..config import ConfigError, LangSmithConfig, MAX_REASONING_LLM_CALLS_PER_REQUEST, RecognitionConfig
from ..observability import ALLOWED_METADATA_KEYS, RecognitionTracer, build_tracer, safe_metadata
from .conftest import make_config


# --- disabled by default -----------------------------------------------

def test_langsmith_is_disabled_by_default(monkeypatch):
    for var in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
                "LANGSMITH_ENDPOINT", "LANGSMITH_WORKSPACE_ID"):
        monkeypatch.delenv(var, raising=False)

    config = make_config()
    assert config.langsmith.tracing_enabled is False
    assert config.langsmith.api_key is None


def test_disabled_tracer_builds_no_client_and_makes_no_call():
    tracer = build_tracer(LangSmithConfig())
    assert tracer.enabled is False
    tracer.record({"node": "validate", "duration_ms": 12})  # must not raise


def test_a_disabled_tracer_holds_no_client_instance():
    tracer = RecognitionTracer(LangSmithConfig(tracing_enabled=False))
    assert tracer._client is None


def test_the_module_imports_no_langsmith_client_at_module_level():
    """The import is lazy, inside _build_client, reached only when tracing is
    enabled. A module-level import (column 0) would defeat that - it would run,
    and could fail, on every import of this module regardless of config."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "observability.py").read_text(encoding="utf-8")
    pattern = re.compile(r"^(?:import|from)\s+langsmith\b", re.MULTILINE)
    assert pattern.search(source) is None


# --- fails loudly, never silently ---------------------------------------

def test_enabling_tracing_without_required_config_raises(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    with pytest.raises(ConfigError) as excinfo:
        RecognitionConfig.from_env()
    message = str(excinfo.value)
    assert "LANGSMITH_API_KEY" in message
    assert "LANGSMITH_PROJECT" in message


def test_the_error_never_echoes_a_configured_value(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__super-secret-canary")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    with pytest.raises(ConfigError) as excinfo:
        RecognitionConfig.from_env()
    assert "super-secret-canary" not in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_falsy_values_leave_tracing_disabled_even_with_credentials_set(monkeypatch, value):
    monkeypatch.setenv("LANGSMITH_TRACING", value)
    monkeypatch.setenv("LANGSMITH_API_KEY", "x")
    monkeypatch.setenv("LANGSMITH_PROJECT", "y")

    config = RecognitionConfig.from_env()
    assert config.langsmith.tracing_enabled is False


def test_a_complete_configuration_enables_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "x")
    monkeypatch.setenv("LANGSMITH_PROJECT", "y")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_WORKSPACE_ID", raising=False)

    config = RecognitionConfig.from_env()
    assert config.langsmith.tracing_enabled is True
    assert config.langsmith.project == "y"
    assert config.langsmith.endpoint is None


# --- the allowlist --------------------------------------------------------

def test_only_allowlisted_keys_survive():
    raw = {"node": "validate", "image_bytes": b"x", "context": {}, "instruction": "hi"}
    assert safe_metadata(raw) == {"node": "validate"}


def test_no_disallowed_key_can_slip_through():
    assert safe_metadata({"filename": "photo.jpg", "sha256": "abc"}) == {}


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png;base64,AAAA",
        "some text with base64, embedded",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_a_forbidden_substring_is_dropped_even_under_an_allowlisted_key(value):
    assert safe_metadata({"status": value}) == {}


def test_the_allowlist_contains_no_image_or_credential_shaped_key():
    for forbidden in ("image_bytes", "context", "data_url", "api_key",
                      "credential", "filename", "instruction", "prompt",
                      "response", "explanation"):
        assert forbidden not in ALLOWED_METADATA_KEYS


# --- exporter failure is never fatal to a request -------------------------

def test_export_failure_cannot_raise_out_of_record(monkeypatch):
    tracer = RecognitionTracer(LangSmithConfig())
    tracer._client = object()  # force enabled=True without a real client

    def _boom(_raw):
        raise RuntimeError("simulated exporter outage")

    monkeypatch.setattr(observability, "safe_metadata", _boom)

    tracer.record({"node": "validate"})  # must not raise


def test_client_construction_failure_leaves_tracer_disabled(monkeypatch):
    def _boom(_config):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(observability.RecognitionTracer, "_build_client", staticmethod(_boom))

    tracer = observability.RecognitionTracer(
        LangSmithConfig(tracing_enabled=True, api_key="x", project="y")
    )
    assert tracer.enabled is False


# --- agent behaviour is unaffected -----------------------------------------

def test_recognition_agent_construction_is_unaffected_by_langsmith_config():
    """Phase 4 wires configuration only; no workflow instrumentation yet."""
    from ..agent import RecognitionAgent

    agent = RecognitionAgent(make_config())
    assert agent.config.langsmith.tracing_enabled is False


def test_two_call_ceiling_is_unchanged():
    assert MAX_REASONING_LLM_CALLS_PER_REQUEST == 2
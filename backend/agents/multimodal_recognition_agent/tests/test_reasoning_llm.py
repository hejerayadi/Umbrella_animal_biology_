"""The optional reasoning-LLM boundary.

Every test here uses a local spy or fake. No credential is read, no client is
constructed, no network call is made.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from ..adapters.reasoning_llm import (
    BoundedAzureReasoningLLM,
    NullReasoningLLM,
    ReasoningBudget,
    ReasoningRequest,
    ReasoningResult,
    build_reasoning_llm,
    sanitize_result,
)


class SpyLLM:
    """Counts calls and records exactly what it was handed."""

    name = "spy"

    def __init__(self, result=None, raises=None, enabled=True):
        self.enabled = enabled
        self.calls = 0
        self.seen: list[ReasoningRequest] = []
        self._result = result
        self._raises = raises

    def analyze(self, request):
        self.calls += 1
        self.seen.append(request)
        if self._raises is not None:
            raise self._raises
        return self._result


# --- the disabled default --------------------------------------------------

def test_null_adapter_is_disabled_and_returns_nothing():
    llm = NullReasoningLLM()
    assert llm.enabled is False
    assert llm.name == "disabled"
    assert llm.analyze(ReasoningRequest(instruction="x", rule_intent="recognition")) is None


def test_builder_returns_the_disabled_adapter_by_default():
    assert isinstance(build_reasoning_llm(enabled=False, timeout_seconds=10.0), NullReasoningLLM)


def test_builder_constructs_no_client_when_enabled(monkeypatch):
    """Enabling must not open a connection - the client is lazy."""
    for name in ("AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)

    llm = build_reasoning_llm(enabled=True, timeout_seconds=10.0)

    assert isinstance(llm, BoundedAzureReasoningLLM)
    # No credentials -> not enabled -> deterministic path, and no client built.
    assert llm.enabled is False
    assert llm._client is None
    assert llm.analyze(ReasoningRequest(instruction="x", rule_intent="recognition")) is None


# --- the request payload: what the model can and cannot receive ------------

def test_request_has_only_the_four_sanitized_fields():
    """No field exists for image data, context, or a credential - so none can
    be passed, by construction rather than by discipline."""
    names = {f.name for f in dataclasses.fields(ReasoningRequest)}
    assert names == {"instruction", "rule_intent", "candidate_names", "candidate_species_ids"}

    for forbidden in ("image", "image_bytes", "data_url", "base64", "vector",
                      "embedding", "context", "api_key", "credential", "tools", "url"):
        assert forbidden not in names


def test_request_is_frozen_so_nothing_can_be_attached():
    request = ReasoningRequest(instruction="x", rule_intent="recognition")
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.instruction = "y"
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        request.image_bytes = b"leak"


# --- the response schema ---------------------------------------------------

def test_result_has_only_the_five_allowed_fields():
    names = {f.name for f in dataclasses.fields(ReasoningResult)}
    assert names == {"intent", "taxon_hint", "location_hint", "habitat_hint", "language"}
    # It cannot carry a species candidate, a score, or a taxonomy identifier.
    for forbidden in ("species", "candidates", "similarity_score", "gbif_id", "ncbi_taxid"):
        assert forbidden not in names


@pytest.mark.parametrize("raw", [None, "a string", 42, [], {"intent": "not_an_intent"}])
def test_malformed_output_is_rejected_entirely(raw):
    assert sanitize_result(raw) is None


def test_unknown_keys_are_dropped():
    result = sanitize_result({
        "intent": "scientific_follow_up",
        "taxon_hint": "lion",
        "species": "Panthera leo",      # not part of the schema
        "gbif_id": 5219404,             # not part of the schema
        "similarity_score": 0.99,       # not part of the schema
    })
    assert result == ReasoningResult(intent="scientific_follow_up", taxon_hint="lion")


def test_overlong_hints_are_dropped():
    assert sanitize_result({"taxon_hint": "x" * 500}).taxon_hint is None


def test_non_string_hints_are_dropped():
    assert sanitize_result({"taxon_hint": ["lion"], "location_hint": 5}).taxon_hint is None


def test_implausible_language_codes_are_dropped():
    assert sanitize_result({"language": "a whole sentence about language"}).language is None
    assert sanitize_result({"language": "fr"}).language == "fr"


# --- the request-scoped budget --------------------------------------------

def test_budget_allows_exactly_its_maximum():
    budget = ReasoningBudget(max_calls=1)
    assert budget.consume() is True
    assert budget.consume() is False
    assert budget.calls_made == 1


def test_budget_is_per_request_not_a_shared_lifetime_counter():
    """Two requests each get a fresh allowance; neither can borrow the other's."""
    first, second = ReasoningBudget(max_calls=1), ReasoningBudget(max_calls=1)
    assert first.consume() is True
    assert second.consume() is True
    assert first.calls_made == second.calls_made == 1


def test_zero_budget_permits_no_call():
    assert ReasoningBudget(max_calls=0).consume() is False


# --- failure modes ---------------------------------------------------------

def test_provider_exception_is_contained_by_the_adapter(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-used-no-call-is-made")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "not-used")

    llm = BoundedAzureReasoningLLM(timeout_seconds=0.01)
    assert llm.enabled is True

    # Replace the lazy client factory so nothing reaches the network.
    def _boom():
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(llm, "_ensure_client", _boom)

    assert llm.analyze(ReasoningRequest(instruction="x", rule_intent="recognition")) is None


def test_adapter_never_raises_out_of_analyze(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-used-no-call-is-made")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "not-used")
    llm = BoundedAzureReasoningLLM()
    monkeypatch.setattr(llm, "_ensure_client", lambda: (_ for _ in ()).throw(TimeoutError()))
    assert llm.analyze(ReasoningRequest(instruction="x", rule_intent="recognition")) is None


# --- no secrets anywhere in configuration ---------------------------------

def test_config_model_holds_no_credential_field():
    from ..config import RecognitionConfig

    names = {f.name for f in dataclasses.fields(RecognitionConfig)}
    for forbidden in ("api_key", "azure_api_key", "base_url", "deployment", "secret", "token"):
        assert forbidden not in names


def test_adapter_reads_no_credential_at_construction_time():
    """The constructor may check presence; it must not keep the value."""
    source = inspect.getsource(BoundedAzureReasoningLLM.__init__)
    assert "os.environ[" not in source  # only presence checks in __init__

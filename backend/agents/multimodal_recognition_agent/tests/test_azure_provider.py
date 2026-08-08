"""The real Azure GPT-5 mini provider - exercised entirely offline.

Every test here injects a fake client. No network call, no credential, no
token. What is being tested is the wiring: which deployment is addressed, what
is put in the payload, what happens when the reply is wrong, and above all that
one request can never cost more than two calls.
"""
from __future__ import annotations

import json

import pytest

from ..adapters.reasoning_llm import (
    ALLOWED_PLAN_STEPS,
    AzureConfigurationError,
    AzureGPT5MiniProvider,
    AzureSettings,
    ExplainRequest,
    FakeGPT5MiniProvider,
    NullRecognitionLLM,
    PlanRequest,
    build_recognition_llm,
    sanitize_plan,
)
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY, RecognitionConfig
from ..schema import AgentRequest, AgentStatus
from .conftest import StubClassifier, image_entry, make_config, png_bytes, prediction

DEPLOYMENT = "umbrella-gpt5-mini"

_AZURE_ENV = {
    "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
    "AZURE_OPENAI_API_KEY": "test-key-never-used-no-call-is-made",
    "AZURE_OPENAI_DEPLOYMENT": DEPLOYMENT,
}


# --- a fake Azure client ---------------------------------------------------

class _Response:
    def __init__(self, text):
        self.output_text = text


class FakeAzureClient:
    """Stands in for `openai.OpenAI`. Records every request, sends nothing."""

    def __init__(self, replies=None, raises=None):
        self._replies = list(replies or [])
        self._raises = raises
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if not self._replies:
            return _Response(None)
        return _Response(self._replies.pop(0))


def settings(**overrides) -> AzureSettings:
    base = dict(
        base_url=_AZURE_ENV["AZURE_OPENAI_BASE_URL"],
        api_key=_AZURE_ENV["AZURE_OPENAI_API_KEY"],
        deployment=DEPLOYMENT,
    )
    base.update(overrides)
    return AzureSettings(**base)


def valid_plan_json() -> str:
    return json.dumps({
        "steps": list(ALLOWED_PLAN_STEPS), "intent": "recognition", "top_k": 30,
        "taxon_hint": None, "location_hint": None, "habitat_hint": None,
        "language": "en", "requested_capability": None,
    })


def azure_env(monkeypatch, **overrides):
    values = {**_AZURE_ENV, **overrides}
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


# ===========================================================================
# Provider selection
# ===========================================================================

def test_the_code_default_is_disabled():
    """A fake brain must never switch itself on in a running service."""
    assert make_config().reasoning_llm_provider_mode == "disabled"
    assert isinstance(build_recognition_llm("disabled"), NullRecognitionLLM)


def test_fake_mode_selects_the_fake_provider():
    provider = build_recognition_llm("fake")
    assert isinstance(provider, FakeGPT5MiniProvider)
    assert provider.name == "fake-gpt-5-mini"


def test_azure_mode_selects_the_real_provider(monkeypatch):
    azure_env(monkeypatch)
    provider = build_recognition_llm("azure", client=FakeAzureClient())
    assert isinstance(provider, AzureGPT5MiniProvider)
    assert provider.name == "azure-gpt-5-mini"


def test_the_configured_deployment_is_used(monkeypatch):
    azure_env(monkeypatch)
    assert build_recognition_llm("azure", client=FakeAzureClient()).deployment == DEPLOYMENT


@pytest.mark.parametrize(
    "missing", ["AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
)
def test_incomplete_azure_configuration_fails_loudly(monkeypatch, missing):
    """Asking for the real model and silently getting a fake one would be worse
    than failing."""
    azure_env(monkeypatch, **{missing: None})

    with pytest.raises(AzureConfigurationError) as caught:
        build_recognition_llm("azure")

    assert missing in str(caught.value)


def test_the_error_names_variables_never_values(monkeypatch):
    azure_env(monkeypatch, AZURE_OPENAI_BASE_URL=None)
    with pytest.raises(AzureConfigurationError) as caught:
        build_recognition_llm("azure")
    message = str(caught.value)
    assert _AZURE_ENV["AZURE_OPENAI_API_KEY"] not in message
    assert "test-key" not in message


def test_an_unknown_mode_is_rejected():
    with pytest.raises(AzureConfigurationError):
        build_recognition_llm("gpt7-turbo")


def test_config_rejects_an_unknown_mode_from_the_environment(monkeypatch):
    from ..config import ConfigError

    monkeypatch.setenv("RECOGNITION_LLM_PROVIDER_MODE", "nonsense")
    with pytest.raises(ConfigError):
        RecognitionConfig.from_env()


def test_the_legacy_enable_flag_still_selects_azure(monkeypatch):
    monkeypatch.delenv("RECOGNITION_LLM_PROVIDER_MODE", raising=False)
    monkeypatch.setenv("RECOGNITION_REASONING_LLM_ENABLED", "true")
    assert RecognitionConfig.from_env().reasoning_llm_provider_mode == "azure"


def test_building_the_provider_opens_no_connection(monkeypatch):
    azure_env(monkeypatch)
    provider = build_recognition_llm("azure")
    assert provider._client is None  # lazy: nothing constructed, nothing sent


# ===========================================================================
# Call 1 - planning
# ===========================================================================

def test_the_planner_addresses_the_configured_deployment():
    client = FakeAzureClient(replies=[valid_plan_json()])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    provider.plan(PlanRequest(instruction="Identify this animal.", has_image=True,
                              image_media_type="image/png", rule_intent="recognition"))

    assert client.calls[0]["model"] == DEPLOYMENT
    assert client.calls[0]["store"] is False
    assert client.calls[0]["reasoning"] == {"effort": "low"}


def test_a_valid_plan_survives_sanitisation():
    client = FakeAzureClient(replies=[valid_plan_json()])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    raw = provider.plan(PlanRequest(instruction="Identify this animal.", has_image=True,
                                    image_media_type="image/png", rule_intent="recognition"))
    plan = sanitize_plan(raw, default_top_k=30, rule_intent="recognition")

    assert plan is not None and plan.source == "llm"


def test_a_fenced_json_reply_is_still_parsed():
    """Models wrap JSON in a ``` fence often enough that not handling it would
    send good plans to the fallback. This is parsing, not retrying."""
    client = FakeAzureClient(replies=["```json\n" + valid_plan_json() + "\n```"])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    raw = provider.plan(PlanRequest(instruction="x", has_image=True,
                                    image_media_type="image/png", rule_intent="recognition"))
    assert sanitize_plan(raw, default_top_k=30, rule_intent="recognition") is not None
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "reply",
    ["not json at all", "", None, "{broken", json.dumps({"steps": ["compare"]}),
     json.dumps({"steps": ["classify_image"], "intent": "hack"})],
)
def test_an_invalid_reply_yields_no_usable_plan_and_no_second_call(reply):
    client = FakeAzureClient(replies=[reply])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    raw = provider.plan(PlanRequest(instruction="x", has_image=True,
                                    image_media_type="image/png", rule_intent="recognition"))
    assert sanitize_plan(raw, default_top_k=30, rule_intent="recognition") is None
    assert len(client.calls) == 1, "a bad reply must not trigger a retry"


def test_a_provider_exception_is_contained_and_not_retried():
    client = FakeAzureClient(raises=TimeoutError("simulated"))
    provider = AzureGPT5MiniProvider(settings(), client=client)

    assert provider.plan(PlanRequest(instruction="x", has_image=True,
                                     image_media_type="image/png",
                                     rule_intent="recognition")) is None
    assert len(client.calls) == 1


# ===========================================================================
# Call 2 - explanation
# ===========================================================================

def _explain_request():
    return ExplainRequest(
        decision="identified", text_alignment="neutral", primary_species="Panthera leo",
        candidate_names=("Panthera leo",), top_score=0.96, margin=0.5,
        taxonomy_status="mock_verified", recognition_mode="mock_classification",
        classifier_version="sprint2-mock-bioclip2-classifier-v1",
        visual_evidence_sufficient=True,
    )


def test_the_explainer_addresses_the_configured_deployment():
    client = FakeAzureClient(replies=["Panthera leo is the highest-ranked label."])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    assert provider.explain(_explain_request())
    assert client.calls[0]["model"] == DEPLOYMENT


def test_the_explainer_failure_returns_none_without_retry():
    client = FakeAzureClient(raises=RuntimeError("simulated"))
    provider = AzureGPT5MiniProvider(settings(), client=client)

    assert provider.explain(_explain_request()) is None
    assert len(client.calls) == 1


# ===========================================================================
# What is - and is not - sent to Azure
# ===========================================================================

def test_no_image_key_or_vector_reaches_the_azure_payload():
    client = FakeAzureClient(replies=[valid_plan_json(), "Panthera leo is the match."])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    provider.plan(PlanRequest(instruction="Identify this animal.", has_image=True,
                              image_media_type="image/png", rule_intent="recognition"))
    provider.explain(_explain_request())

    payload = json.dumps(client.calls, default=str)
    for forbidden in ("data:image", "base64", "iVBOR", "test-key-never-used",
                      "AZURE_OPENAI_API_KEY", "query_vector", "image_bytes"):
        assert forbidden not in payload, f"{forbidden!r} reached the Azure payload"


def test_the_planner_is_told_an_image_exists_never_what_it_contains():
    client = FakeAzureClient(replies=[valid_plan_json()])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    provider.plan(PlanRequest(instruction="Identify this animal.", has_image=True,
                              image_media_type="image/png", rule_intent="recognition"))

    sent = client.calls[0]["input"]
    assert "image_present: True" in sent
    assert "image_media_type: image/png" in sent


def test_output_tokens_and_timeout_are_bounded():
    client = FakeAzureClient(replies=[valid_plan_json()])
    provider = AzureGPT5MiniProvider(
        settings(max_output_tokens=123, timeout_seconds=7.0), client=client
    )
    provider.plan(PlanRequest(instruction="x", has_image=True,
                              image_media_type="image/png", rule_intent="recognition"))
    assert client.calls[0]["max_output_tokens"] == 123


# ===========================================================================
# The two-call budget, end to end through the workflow
# ===========================================================================

def build_agent(predictions=None, *, llm=None):
    return RecognitionAgent(
        make_config(),
        classifier=StubClassifier(predictions or []),
        reasoning_llm=llm,
    )


def request_with(instruction="Identify this animal."):
    return AgentRequest(
        instruction=instruction,
        context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())},
    )


def test_a_valid_request_costs_exactly_two_azure_calls():
    client = FakeAzureClient(replies=[valid_plan_json(), "Panthera leo is the match."])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with())

    assert result.status is AgentStatus.COMPLETED
    assert provider.plan_calls == 1
    assert provider.explain_calls == 1
    assert len(client.calls) == 2, f"expected exactly 2 Azure calls, got {len(client.calls)}"


def test_an_invalid_request_costs_zero_azure_calls():
    client = FakeAzureClient(replies=[valid_plan_json()])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent([], llm=provider).run(
        AgentRequest(instruction="Identify this animal.", context={})
    )

    assert result.status is AgentStatus.FAILED
    assert len(client.calls) == 0


# --- a failed planner forfeits the explanation call ------------------------
#
# The contract:
#   invalid request before the planner  = 0 calls
#   planner failed                      = 1 call, never 2
#   planner ok + explainer ok           = 2 calls
#   planner ok + explainer failed       = 2 calls
#
# A provider that has just answered off-contract does not get asked again on
# the same request - that would be a retry wearing a different hat.

@pytest.mark.parametrize(
    "reply, label",
    [
        ("not json at all", "malformed planner response"),
        ("", "empty planner response"),
        (json.dumps({"steps": ["compare"]}), "forbidden action"),
        (json.dumps({"steps": ["classify_image"], "intent": "hack"}), "invalid schema"),
    ],
)
def test_a_failed_planner_costs_one_azure_call_only(reply, label):
    # A second reply is queued deliberately: if the explainer were called, the
    # count would be 2 and this test would catch it.
    client = FakeAzureClient(replies=[reply, "Panthera leo is the match."])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with())

    provenance = result.output["recognition_provenance"]
    assert len(client.calls) == 1, f"{label}: expected 1 Azure call, got {len(client.calls)}"
    assert provider.explain_calls == 0
    assert provenance["plan_source"] == "deterministic"
    assert provenance["explanation_source"] == "deterministic"
    # And the workflow still completed safely on the deterministic path.
    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "identified"


@pytest.mark.parametrize(
    "failure, label",
    [
        (TimeoutError("simulated timeout"), "planner timeout"),
        (PermissionError("401 invalid api key"), "authentication error"),
        (ConnectionError("simulated outage"), "total Azure outage"),
    ],
)
def test_a_planner_exception_costs_one_azure_call_only(failure, label):
    client = FakeAzureClient(raises=failure)
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with())

    provenance = result.output["recognition_provenance"]
    assert len(client.calls) == 1, f"{label}: expected 1 Azure call, got {len(client.calls)}"
    assert provider.explain_calls == 0
    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "identified"
    assert provenance["plan_source"] == "deterministic"
    assert provenance["explanation_source"] == "deterministic"
    assert provenance["reasoning_llm_calls"] == 1


def test_a_successful_planner_with_a_failing_explainer_costs_two():
    """The other side of the rule: the planner earned the second call."""
    client = FakeAzureClient(replies=[valid_plan_json(), None])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with())

    assert len(client.calls) == 2
    assert result.output["recognition_provenance"]["plan_source"] == "llm"
    assert result.output["recognition_provenance"]["explanation_source"] == "deterministic"
    assert result.status is AgentStatus.COMPLETED


def test_the_whole_call_budget_matrix():
    """Every row of the contract, in one place."""
    plan_ok, explain_ok = valid_plan_json(), "Panthera leo is the match."
    predictions = [prediction("panthera_leo", 0.96, "Panthera leo")]

    # invalid request, before the planner
    client = FakeAzureClient(replies=[plan_ok, explain_ok])
    build_agent([], llm=AzureGPT5MiniProvider(settings(), client=client)).run(
        AgentRequest(instruction="Identify this animal.", context={})
    )
    assert len(client.calls) == 0

    # planner failed
    client = FakeAzureClient(replies=["not json", explain_ok])
    build_agent(predictions, llm=AzureGPT5MiniProvider(settings(), client=client)).run(
        request_with()
    )
    assert len(client.calls) == 1

    # planner ok + explainer ok
    client = FakeAzureClient(replies=[plan_ok, explain_ok])
    build_agent(predictions, llm=AzureGPT5MiniProvider(settings(), client=client)).run(
        request_with()
    )
    assert len(client.calls) == 2

    # planner ok + explainer failed
    client = FakeAzureClient(replies=[plan_ok, "This is Ursus maritimus."])
    build_agent(predictions, llm=AzureGPT5MiniProvider(settings(), client=client)).run(
        request_with()
    )
    assert len(client.calls) == 2


def test_azure_cannot_change_the_science():
    """The model plans and phrases. The candidates, scores and decision are the
    deterministic code's, and identical with or without Azure."""
    predictions = [
        prediction("panthera_leo", 0.96, "Panthera leo"),
        prediction("panthera_tigris", 0.40, "Panthera tigris"),
    ]
    client = FakeAzureClient(replies=[valid_plan_json(), "Panthera leo is the match."])
    with_azure = build_agent(
        predictions, llm=AzureGPT5MiniProvider(settings(), client=client)
    ).run(request_with()).output
    without = build_agent(predictions, llm=NullRecognitionLLM()).run(request_with()).output

    assert with_azure["species"] == without["species"]
    assert with_azure["recognition"]["decision"] == without["recognition"]["decision"]
    assert (
        with_azure["recognition_candidates"][0]["classification_score"]
        == without["recognition_candidates"][0]["classification_score"]
    )


def test_an_ungrounded_azure_explanation_is_replaced_not_published():
    client = FakeAzureClient(
        replies=[valid_plan_json(), "This is Ursus maritimus, a 400 kg polar bear."]
    )
    provider = AzureGPT5MiniProvider(settings(), client=client)

    result = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with())

    explanation = result.output["recognition"]["explanation"]
    assert "Ursus maritimus" not in explanation
    assert result.output["recognition_provenance"]["explanation_source"] == "deterministic"
    assert len(client.calls) == 2  # no corrective third call


def test_provenance_names_the_azure_provider():
    client = FakeAzureClient(replies=[valid_plan_json(), "Panthera leo is the match."])
    provider = AzureGPT5MiniProvider(settings(), client=client)

    provenance = build_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
    ).run(request_with()).output["recognition_provenance"]

    assert provenance["reasoning_llm_provider"] == "azure-gpt-5-mini"
    assert provenance["reasoning_llm_enabled"] is True
    assert provenance["reasoning_llm_calls"] == 2


def test_no_secret_appears_in_any_output_or_log(caplog):
    import logging

    client = FakeAzureClient(raises=RuntimeError("boom https://secret.invalid key=abc"))
    provider = AzureGPT5MiniProvider(settings(), client=client)

    with caplog.at_level(logging.DEBUG, logger="backend.agents.multimodal_recognition_agent"):
        result = build_agent(
            [prediction("panthera_leo", 0.96, "Panthera leo")], llm=provider
        ).run(request_with())

    haystack = json.dumps(result.output, default=str) + "".join(
        r.getMessage() for r in caplog.records
    )
    for secret in ("test-key-never-used", "secret.invalid", "key=abc",
                   _AZURE_ENV["AZURE_OPENAI_BASE_URL"]):
        assert secret not in haystack, secret

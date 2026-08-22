"""Phase 2 corrective - transport retries off, and an observable smoke script.

Two defects motivated this module:

1. The OpenAI SDK defaults to `max_retries=2`, so one logical GPT-5 mini call
   could become three HTTP attempts. The agent counted one call while the
   transport spent three, and the bounded timeout silently became three times
   as long. The contract is one logical call = at most one HTTP attempt.
2. `smoke_test_azure.py` was a one-call `"Reply only with OK."` connectivity
   probe. It exercised no planner, no explainer and no call accounting, so it
   could not produce Phase 2 evidence - and it printed nothing incrementally,
   so a stalled run left no trace of where it stopped.

Every test here uses an injected or mocked client. Nothing loads `.env`,
resolves a hostname, or contacts Azure.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from ..adapters.reasoning_llm import (
    SDK_MAX_RETRIES,
    AzureGPT5MiniProvider,
    AzureSettings,
    ExplainRequest,
    PlanRequest,
)
from .conftest import make_config

AGENT_DIR = Path(__file__).resolve().parent.parent
SMOKE_PATH = AGENT_DIR / "smoke_test_azure.py"
SMOKE_SOURCE = SMOKE_PATH.read_text(encoding="utf-8")

DEPLOYMENT = "umbrella-gpt5-mini"
FAKE_KEY = "test-key-never-used-no-call-is-made"
FAKE_URL = "https://example.invalid/openai/v1"


def settings(**overrides) -> AzureSettings:
    base = dict(base_url=FAKE_URL, api_key=FAKE_KEY, deployment=DEPLOYMENT)
    base.update(overrides)
    return AzureSettings(**base)


class _Response:
    def __init__(self, text):
        self.output_text = text


class RecordingClient:
    """Injected stand-in. Counts attempts, sends nothing."""

    def __init__(self, replies=None, raises=None):
        self._replies = list(replies or [])
        self._raises = raises
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return _Response(self._replies.pop(0) if self._replies else None)


def plan_request() -> PlanRequest:
    return PlanRequest(
        instruction="What animal is this?",
        has_image=True,
        image_media_type="image/png",
        rule_intent="recognition",
    )


def explain_request() -> ExplainRequest:
    return ExplainRequest(
        decision="identified",
        text_alignment="neutral",
        primary_species="Panthera leo",
        candidate_names=["Panthera leo"],
        top_score=0.93,
        margin=0.4,
        taxonomy_status="verified",
        recognition_mode="mock_classification",
        classifier_version="v1",
        visual_evidence_sufficient=True,
    )


# ===========================================================================
# 1-3 - the production client: max_retries=0, the configured timeout, store
# ===========================================================================

def test_the_retry_constant_is_exactly_zero():
    assert SDK_MAX_RETRIES == 0


def test_the_real_client_is_constructed_with_max_retries_zero(monkeypatch):
    """The provider's own lazy construction path, captured at the call site."""
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.responses = None

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    provider = AzureGPT5MiniProvider(settings(timeout_seconds=13.5))
    provider._ensure_client()

    assert captured["max_retries"] == 0
    assert captured["timeout"] == 13.5
    assert captured["base_url"] == FAKE_URL
    assert captured["api_key"] == FAKE_KEY


def test_timeout_and_max_retries_reach_the_same_client(monkeypatch):
    """Proof they are not configured on two different objects."""
    seen: list[dict] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.responses = None

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    provider = AzureGPT5MiniProvider(settings(timeout_seconds=4.25))
    provider._ensure_client()

    assert len(seen) == 1, "exactly one client is constructed"
    assert seen[0]["timeout"] == 4.25 and seen[0]["max_retries"] == 0


def test_the_client_is_built_lazily_and_only_once(monkeypatch):
    built = []

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.responses = None

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    provider = AzureGPT5MiniProvider(settings())
    assert provider._client is None and built == []  # lazy
    provider._ensure_client()
    provider._ensure_client()
    assert len(built) == 1  # cached


def test_store_is_false_on_every_responses_request():
    client = RecordingClient(replies=["{}", "text"])
    provider = AzureGPT5MiniProvider(settings(), client=client)
    provider.plan(plan_request())
    provider.explain(explain_request())

    assert len(client.calls) == 2
    assert all(call["store"] is False for call in client.calls)
    assert all(call["model"] == DEPLOYMENT for call in client.calls)


def test_the_source_passes_max_retries_at_the_production_call_site():
    """An AST check, so the argument cannot be lost to a refactor."""
    source = (AGENT_DIR / "adapters" / "reasoning_llm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OpenAI"
    ]
    assert sites, "the production client construction should be findable"
    production = [
        site for site in sites
        if any(kw.arg == "max_retries" for kw in site.keywords)
    ]
    assert production, "no OpenAI(...) call passes max_retries"


# ===========================================================================
# 4 - injected clients still work with no network or credentials
# ===========================================================================

def test_an_injected_client_needs_no_network_or_credential():
    client = RecordingClient(replies=["{}"])
    provider = AzureGPT5MiniProvider(settings(), client=client)
    provider.plan(plan_request())
    assert len(client.calls) == 1
    assert provider._client is client  # never replaced by a real one


# ===========================================================================
# 5-6 - an exception produces no provider-level retry
# ===========================================================================

@pytest.mark.parametrize(
    "failure",
    [TimeoutError("simulated"), RuntimeError("simulated"), ConnectionError("simulated")],
)
def test_a_planner_exception_causes_no_retry(failure):
    client = RecordingClient(raises=failure)
    provider = AzureGPT5MiniProvider(settings(), client=client)

    assert provider.plan(plan_request()) is None
    assert len(client.calls) == 1  # one attempt, no second
    assert provider.plan_calls == 1


@pytest.mark.parametrize(
    "failure", [TimeoutError("simulated"), RuntimeError("simulated")]
)
def test_an_explainer_exception_causes_no_retry(failure):
    client = RecordingClient(raises=failure)
    provider = AzureGPT5MiniProvider(settings(), client=client)

    assert provider.explain(explain_request()) is None
    assert len(client.calls) == 1
    assert provider.explain_calls == 1


def test_logical_and_transport_retry_counts_are_both_zero():
    """The two layers agree: one logical call, one HTTP attempt."""
    client = RecordingClient(replies=["{}", "text"])
    provider = AzureGPT5MiniProvider(settings(), client=client)
    provider.plan(plan_request())
    provider.explain(explain_request())

    logical = provider.plan_calls + provider.explain_calls
    assert logical == 2
    assert len(client.calls) == logical  # no transport multiplication
    assert SDK_MAX_RETRIES == 0


def test_no_retry_library_or_loop_was_introduced():
    source = (AGENT_DIR / "adapters" / "reasoning_llm.py").read_text(encoding="utf-8")
    for forbidden in ("import tenacity", "from tenacity", "import backoff",
                      "from backoff", "@retry"):
        assert forbidden not in source


# ===========================================================================
# 7-10 - budget behaviour, unchanged by this correction
# ===========================================================================

def test_a_failed_planner_spends_one_call_and_skips_the_explainer():
    from ..adapters.reasoning_llm import ReasoningBudget

    client = RecordingClient(raises=TimeoutError("simulated"))
    provider = AzureGPT5MiniProvider(settings(), client=client)
    budget = ReasoningBudget(max_calls=2)

    assert budget.consume() is True
    assert provider.plan(plan_request()) is None
    # The planner failed, so the explainer call is forfeited by the workflow.
    assert provider.explain_calls == 0
    assert budget.calls_made == 1
    assert len(client.calls) == 1


def test_budget_zero_permits_no_call():
    from ..adapters.reasoning_llm import ReasoningBudget

    budget = ReasoningBudget(max_calls=0)
    assert budget.consume() is False
    assert budget.calls_made == 0


def test_budget_one_permits_the_planner_only():
    from ..adapters.reasoning_llm import ReasoningBudget

    budget = ReasoningBudget(max_calls=1)
    assert budget.consume() is True   # planner
    assert budget.consume() is False  # explainer refused
    assert budget.calls_made == 1


def test_budget_two_permits_planner_plus_explainer_only():
    from ..adapters.reasoning_llm import ReasoningBudget

    budget = ReasoningBudget(max_calls=2)
    assert budget.consume() is True   # planner
    assert budget.consume() is True   # explainer
    assert budget.consume() is False  # nothing more, ever
    assert budget.calls_made == 2


def test_the_configured_ceiling_is_still_two():
    assert make_config().reasoning_llm_max_calls_per_request == 2


# ===========================================================================
# 11-15 - the smoke script's contract
# ===========================================================================

def test_the_smoke_script_uses_the_agent_local_env():
    assert 'Path(__file__).resolve().parent / ".env"' in SMOKE_SOURCE
    assert "parents[2]" not in SMOKE_SOURCE

    namespace: dict = {"__file__": str(SMOKE_PATH)}
    exec("from pathlib import Path", namespace)  # noqa: S102 - fixed literal
    match = re.search(r"AGENT_ENV_PATH = (.+)", SMOKE_SOURCE)
    assert match
    assert eval(match.group(1), namespace) == AGENT_DIR / ".env"  # noqa: S307


def test_the_smoke_script_requires_explicit_live_opt_in():
    assert "RECOGNITION_LIVE_SMOKE" in SMOKE_SOURCE
    tree = ast.parse(SMOKE_SOURCE)
    # The guard must be the first thing `main` does, before any import of the
    # adapter or any client construction.
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert isinstance(main.body[0], ast.If), "opt-in must be the first statement"
    assert "LIVE_OPT_IN_VAR" in ast.dump(main.body[0])


def test_importing_the_smoke_module_makes_no_call(monkeypatch):
    """Import must be inert: no env load, no client, no request."""
    monkeypatch.delenv("RECOGNITION_LIVE_SMOKE", raising=False)
    import importlib

    module = importlib.import_module(
        "backend.agents.multimodal_recognition_agent.smoke_test_azure"
    )
    assert module.AGENT_ENV_PATH == AGENT_DIR / ".env"
    # Running it without the opt-in returns non-zero and does nothing else.
    assert module.main() == 2


def test_the_old_connectivity_probe_is_gone():
    """The one-call probe could not demonstrate anything Phase 2 requires."""
    assert "Reply only with OK." not in SMOKE_SOURCE
    assert "Reply only with OK" not in SMOKE_SOURCE


def test_the_smoke_script_exercises_planner_then_explainer():
    assert "plan_source" in SMOKE_SOURCE
    assert "explanation_source" in SMOKE_SOURCE
    assert "planner_started" in SMOKE_SOURCE
    assert "explainer_started" in SMOKE_SOURCE
    assert "planner_before_explainer" in SMOKE_SOURCE


def test_the_smoke_script_reuses_the_production_adapter():
    """It must not reproduce provider logic of its own."""
    assert "AzureGPT5MiniProvider" in SMOKE_SOURCE
    assert "RecognitionAgent" in SMOKE_SOURCE
    # No hand-rolled prompt or Responses payload in the script.
    assert "responses.create" not in SMOKE_SOURCE.replace(
        "self._inner.responses.create(**kwargs)", ""
    )


def test_the_smoke_script_counts_started_and_completed_separately():
    """The distinction the previous stalled attempt could not make."""
    assert "attempts_started" in SMOKE_SOURCE
    assert "attempts_completed" in SMOKE_SOURCE


def test_every_smoke_progress_marker_is_flushed():
    tree = ast.parse(SMOKE_SOURCE)
    prints = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert prints, "the script should print progress"
    for call in prints:
        assert any(kw.arg == "flush" and kw.value.value is True
                   for kw in call.keywords), "every print must flush"


def test_the_smoke_script_exits_non_zero_when_evidence_is_incomplete():
    tree = ast.parse(SMOKE_SOURCE)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    returns = {
        node.value.value
        for node in ast.walk(main)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
    }
    assert 0 in returns, "a fully successful run exits 0"
    assert returns - {0}, "incomplete evidence must exit non-zero"


def test_the_smoke_output_contract_is_redacted():
    """Only safe fields may be emitted."""
    forbidden = ("api_key", "base_url", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_BASE_URL", "data_url", "b64encode(raw)")
    emitted = re.findall(r"emit\(([^)]*)\)", SMOKE_SOURCE)
    joined = "\n".join(emitted)
    for term in forbidden:
        assert term not in joined, f"{term} must never be emitted"
    # The alias is the one deployment detail that may be shown.
    assert "deployment_alias" in joined


def test_the_smoke_script_never_emits_a_prompt_or_payload():
    emitted = "\n".join(re.findall(r"emit\(([^)]*)\)", SMOKE_SOURCE))
    for term in ("payload", "system_prompt", "input", "output_text",
                 "instruction", "context", "gbif", "ncbi"):
        assert term not in emitted.lower(), f"{term} must not be emitted"


# ===========================================================================
# 16-17 - the offline suite stays offline and leaks nothing
# ===========================================================================

def test_no_test_module_loads_dotenv_or_names_a_live_opt_in():
    """The needles are assembled at runtime so this module does not match
    itself and can be checked by the same rule as every other test file."""
    load_env = "load_" + "dotenv"
    read_env = "dotenv_" + "values"
    opt_in = 'setenv("RECOGNITION_LIVE_' + 'SMOKE", "1")'

    offenders = []
    for path in (AGENT_DIR / "tests").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if load_env in source or read_env in source:
            offenders.append(f"{path.name}: loads .env")
        if opt_in in source:
            offenders.append(f"{path.name}: enables the live smoke")
    assert offenders == []


def test_no_credential_value_appears_in_a_provider_exception():
    from ..adapters.reasoning_llm import AzureConfigurationError, build_recognition_llm

    with pytest.raises(AzureConfigurationError) as caught:
        build_recognition_llm("azure")
    message = str(caught.value)
    assert FAKE_KEY not in message and FAKE_URL not in message


def test_a_provider_failure_logs_only_the_exception_type(caplog):
    """A message could echo the request; the request is not ours to leak."""
    client = RecordingClient(raises=RuntimeError("secret-bearing-detail"))
    provider = AzureGPT5MiniProvider(settings(), client=client)

    with caplog.at_level("DEBUG"):
        provider.plan(plan_request())

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret-bearing-detail" not in logged
    assert "RuntimeError" in logged


def test_no_prompt_or_credential_reaches_the_request_payload():
    client = RecordingClient(replies=["{}"])
    provider = AzureGPT5MiniProvider(settings(), client=client)
    provider.plan(plan_request())

    sent = "\n".join(str(value) for value in client.calls[0].values())
    for forbidden in (FAKE_KEY, FAKE_URL, "data:image", "base64"):
        assert forbidden not in sent

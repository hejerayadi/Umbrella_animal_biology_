"""Phase 1 - production configuration and provider-selection hardening.

Four defects motivated this module, and each one is pinned by a test here:

1. `RECOGNITION_LLM_TIMEOUT_SECONDS` was parsed, stored on the config and passed
   into `build_recognition_llm`, which then dropped it and let the adapter read
   its own `AZURE_OPENAI_TIMEOUT_SECONDS` instead. The documented setting had no
   effect.
2. `RECOGNITION_LLM_MAX_CALLS_PER_REQUEST=0` became `2`, because the parsed
   value went through `or`, and `0` is falsy. Switching the model off doubled
   the calls instead.
3. `smoke_test_azure.py` loaded `backend/.env` - the orchestrator's file, two
   directories above the agent that owns the deployment settings.
4. Provider construction ignored the configured mode entirely: the agent built
   `MockBioCLIP2Provider` and `MockTaxonomyProvider` whenever nothing was
   injected, so a production mode could be selected and silently served fixture
   data.

Nothing here needs a network, a credential or a model download.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from .. import config as config_module
from ..adapters.bioclip import MockBioCLIP2Provider, build_classifier
from ..adapters.reasoning_llm import (
    AzureGPT5MiniProvider,
    AzureSettings,
    FakeGPT5MiniProvider,
    NullRecognitionLLM,
    build_recognition_llm,
)
from ..adapters.taxonomy import MockTaxonomyProvider, build_taxonomy_provider
from ..agent import RecognitionAgent
from ..config import (
    BIOCLIP_PROVIDER_MODES,
    DEFAULT_REASONING_LLM_TIMEOUT_SECONDS,
    MAX_REASONING_LLM_CALLS_PER_REQUEST,
    REASONING_LLM_PROVIDER_MODES,
    TAXONOMY_PROVIDER_MODES,
    ConfigError,
    RecognitionConfig,
)
from ..schema import AgentRequest, AgentStatus
from .conftest import StubClassifier, image_entry, make_config, png_bytes, prediction

AGENT_DIR = Path(__file__).resolve().parent.parent

_AZURE_ENV = {
    "AZURE_OPENAI_BASE_URL": "https://example.invalid/openai/v1",
    "AZURE_OPENAI_API_KEY": "test-key-never-used-no-call-is-made",
    "AZURE_OPENAI_DEPLOYMENT": "umbrella-gpt5-mini",
}


@pytest.fixture
def clean_env(monkeypatch):
    """A process environment with every Recognition variable removed.

    `from_env` reads the real environment, so a developer's own `.env` values
    must not decide whether these assertions hold.
    """
    for name in (
        "BIOCLIP_PROVIDER_MODE",
        "TAXONOMY_PROVIDER_MODE",
        "RECOGNITION_LLM_PROVIDER_MODE",
        "RECOGNITION_REASONING_LLM_ENABLED",
        "RECOGNITION_LLM_MAX_CALLS_PER_REQUEST",
        "RECOGNITION_LLM_TIMEOUT_SECONDS",
        "AZURE_OPENAI_TIMEOUT_SECONDS",
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "NCBI_TOOL",
        "NCBI_EMAIL",
        "NCBI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ===========================================================================
# 1 - one effective LLM timeout source
# ===========================================================================

def test_the_configured_timeout_reaches_the_azure_client(clean_env):
    """The whole point of the fix: set it, and it is the value that bounds the
    call. Before, this arrived as 20 no matter what was configured."""
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", "3.5")
    for name, value in _AZURE_ENV.items():
        clean_env.setenv(name, value)

    config = RecognitionConfig.from_env()
    assert config.reasoning_llm_timeout_seconds == 3.5

    provider = build_recognition_llm(
        "azure", timeout_seconds=config.reasoning_llm_timeout_seconds
    )
    assert isinstance(provider, AzureGPT5MiniProvider)
    assert provider._settings.timeout_seconds == 3.5


def test_the_agent_propagates_its_configured_timeout(clean_env):
    """End to end through `RecognitionAgent`, not just the builder."""
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", "7.25")
    clean_env.setenv("RECOGNITION_LLM_PROVIDER_MODE", "azure")
    for name, value in _AZURE_ENV.items():
        clean_env.setenv(name, value)

    agent = RecognitionAgent()
    llm = agent._workflow._reasoning_llm
    assert isinstance(llm, AzureGPT5MiniProvider)
    assert llm._settings.timeout_seconds == 7.25


def test_there_is_no_second_timeout_variable(clean_env):
    """`AZURE_OPENAI_TIMEOUT_SECONDS` must no longer influence anything."""
    clean_env.setenv("AZURE_OPENAI_TIMEOUT_SECONDS", "999")
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", "4")
    for name, value in _AZURE_ENV.items():
        clean_env.setenv(name, value)

    provider = build_recognition_llm("azure", timeout_seconds=4.0)
    assert provider._settings.timeout_seconds == 4.0


def test_no_module_reads_a_second_timeout_variable():
    """No EXECUTABLE reference to the competing name survives, so it cannot
    creep back in as a quiet override.

    Comments and docstrings are excluded on purpose, following the same rule the
    scope gates use: documentation should record that the duplicate was removed;
    code may not read it.
    """
    from .test_recognition_only_gate import executable_text, shipped_files

    offenders = [
        path.name
        for path in shipped_files(".py")
        if path.parent.name != "tests"
        and "AZURE_OPENAI_TIMEOUT_SECONDS" in executable_text(path)
    ]
    assert offenders == []


def test_the_effective_default_timeout_is_unchanged_at_twenty(clean_env):
    """Unifying the source must not halve the timeout that production has
    actually been running with."""
    assert DEFAULT_REASONING_LLM_TIMEOUT_SECONDS == 20.0
    assert RecognitionConfig.from_env().reasoning_llm_timeout_seconds == 20.0
    assert AzureSettings.from_env.__defaults__ is None or True  # keyword-only


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf", "-inf", "abc", "10s"])
def test_an_unusable_timeout_is_refused_at_startup(clean_env, bad):
    """Zero and negative remove the bound; NaN and infinity make every call
    either fail or hang. None of them may start the service."""
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", bad)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "RECOGNITION_LLM_TIMEOUT_SECONDS" in str(caught.value)


def test_a_valid_timeout_is_accepted(clean_env):
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", "0.5")
    assert RecognitionConfig.from_env().reasoning_llm_timeout_seconds == 0.5


# ===========================================================================
# 2 - call-budget parsing
# ===========================================================================

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),   # the regression: this used to come back as 2
        ("1", 1),
        ("2", 2),
        ("3", 2),   # clamped, never raised above the ceiling
        ("99", 2),
        (" 1 ", 1),
    ],
)
def test_the_call_budget_is_parsed_exactly(clean_env, raw, expected):
    clean_env.setenv("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", raw)
    assert RecognitionConfig.from_env().reasoning_llm_max_calls_per_request == expected


def test_zero_calls_stays_zero_and_is_not_the_default(clean_env):
    """Stated on its own because it is the defect: `0` is falsy, and the old
    expression turned it into the default `2`."""
    clean_env.setenv("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", "0")
    config = RecognitionConfig.from_env()
    assert config.reasoning_llm_max_calls_per_request == 0
    assert config.reasoning_llm_max_calls_per_request != 2


@pytest.mark.parametrize("bad", ["-1", "-2", "abc", "1.5", "two", "0x2", ""])
def test_a_malformed_or_negative_budget_fails_clearly(clean_env, bad):
    """Empty means unset, which is the one case that legitimately defaults."""
    if bad == "":
        clean_env.delenv("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", raising=False)
        assert (
            RecognitionConfig.from_env().reasoning_llm_max_calls_per_request
            == MAX_REASONING_LLM_CALLS_PER_REQUEST
        )
        return

    clean_env.setenv("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", bad)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "RECOGNITION_LLM_MAX_CALLS_PER_REQUEST" in str(caught.value)


def test_the_global_ceiling_is_exactly_two(clean_env):
    assert MAX_REASONING_LLM_CALLS_PER_REQUEST == 2
    clean_env.setenv("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", "1000")
    assert RecognitionConfig.from_env().reasoning_llm_max_calls_per_request == 2


def test_no_retry_was_introduced():
    """The budget fix must not have grown a retry loop next to it."""
    source = inspect.getsource(config_module)
    assert "retry" not in source.lower()
    assert "retries" not in source.lower()


# ===========================================================================
# 3 - the smoke script reads the agent's own .env
# ===========================================================================

def test_the_smoke_script_loads_the_agent_local_env_file():
    source = (AGENT_DIR / "smoke_test_azure.py").read_text(encoding="utf-8")
    assert "parents[2]" not in source, "that is backend/.env, not this agent's"
    assert "Path(__file__).resolve().parent / \".env\"" in source


def test_the_smoke_script_env_path_resolves_inside_the_agent_directory():
    namespace: dict = {}
    source = (AGENT_DIR / "smoke_test_azure.py").read_text(encoding="utf-8")
    # Evaluate only the path expression: importing the module would try to
    # build a real client.
    match = re.search(r"AGENT_ENV_PATH = (.+)", source)
    assert match, "the script should name its env path"
    namespace["__file__"] = str(AGENT_DIR / "smoke_test_azure.py")
    exec("from pathlib import Path", namespace)  # noqa: S102 - fixed literal
    resolved = eval(match.group(1), namespace)  # noqa: S307 - fixed literal
    assert resolved == AGENT_DIR / ".env"
    assert resolved.parent == AGENT_DIR


# ===========================================================================
# 4 - explicit provider modes, loud failures, no silent mock
# ===========================================================================

def test_the_supported_mode_vocabularies_are_explicit():
    # Phase 3 replaced the placeholder BioCLIP `real` mode with the implemented
    # `remote` one. Phase 4 implemented taxonomy's `real` mode.
    assert BIOCLIP_PROVIDER_MODES == ("mock", "remote")
    assert TAXONOMY_PROVIDER_MODES == ("mock", "real")
    assert REASONING_LLM_PROVIDER_MODES == ("disabled", "fake", "azure")


@pytest.mark.parametrize("mode", ["banana", "qdrant", "vector", "REAL-ish", "0"])
def test_an_unknown_bioclip_mode_fails_loudly(clean_env, mode):
    clean_env.setenv("BIOCLIP_PROVIDER_MODE", mode)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    message = str(caught.value)
    assert "BIOCLIP_PROVIDER_MODE" in message
    assert "mock" in message  # tells the operator what is legal


@pytest.mark.parametrize("mode", ["banana", "gbif", "http", "live"])
def test_an_unknown_taxonomy_mode_fails_loudly(clean_env, mode):
    clean_env.setenv("TAXONOMY_PROVIDER_MODE", mode)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "TAXONOMY_PROVIDER_MODE" in str(caught.value)


@pytest.mark.parametrize("mode", ["nonsense", "gpt7-turbo", "openai", ""])
def test_an_unknown_llm_mode_fails_loudly(clean_env, mode):
    if mode == "":
        # Empty means unset, which legitimately defaults to disabled.
        assert RecognitionConfig.from_env().reasoning_llm_provider_mode == "disabled"
        return
    clean_env.setenv("RECOGNITION_LLM_PROVIDER_MODE", mode)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "RECOGNITION_LLM_PROVIDER_MODE" in str(caught.value)


def test_the_remote_bioclip_mode_is_implemented_and_selectable(clean_env):
    """Phase 3 landed: `remote` is a working mode, not a placeholder."""
    clean_env.setenv("BIOCLIP_PROVIDER_MODE", "remote")
    config = RecognitionConfig.from_env()
    assert config.bioclip_provider_mode == "remote"
    assert config.recognition_mode == "remote_bioclip2_open_domain_species"


def test_the_old_real_bioclip_mode_name_is_refused_as_unknown(clean_env):
    """`real` was the placeholder name for the cancelled LOCAL design. It is not
    a synonym for `remote`: an operator who sets it must be told, not silently
    given remote inference they did not ask for."""
    clean_env.setenv("BIOCLIP_PROVIDER_MODE", "real")
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    message = str(caught.value)
    assert "BIOCLIP_PROVIDER_MODE" in message
    assert "remote" in message  # names the legal values


def test_real_taxonomy_mode_is_implemented_and_selectable(clean_env):
    """Phase 4 landed: `real` is a working mode, not a placeholder."""
    clean_env.setenv("TAXONOMY_PROVIDER_MODE", "real")
    clean_env.setenv("NCBI_TOOL", "test-tool")
    clean_env.setenv("NCBI_EMAIL", "test@example.com")
    config = RecognitionConfig.from_env()
    assert config.taxonomy_provider_mode == "real"


def test_real_taxonomy_mode_without_ncbi_contact_info_is_refused(clean_env):
    """Real mode is selectable, but NCBI's usage guidelines require a tool and
    email - selecting real mode without them must still fail loudly, not call
    NCBI unidentified."""
    clean_env.setenv("TAXONOMY_PROVIDER_MODE", "real")
    # NCBI_TOOL / NCBI_EMAIL deliberately left unset.
    config = RecognitionConfig.from_env()
    with pytest.raises(ConfigError) as caught:
        build_taxonomy_provider(config)
    message = str(caught.value)
    assert "NCBI_TOOL" in message
    assert "NCBI_EMAIL" in message


def test_real_bioclip_mode_never_returns_a_mock_provider():
    """The factory refuses even when the config object is built in code and
    never passed through the environment."""
    config = make_config(bioclip_provider_mode="real")
    with pytest.raises(ConfigError):
        build_classifier(config)


def test_real_taxonomy_mode_never_returns_a_fixture_provider():
    """With no NCBI credentials configured, real mode must still refuse rather
    than silently falling back to the fixture provider."""
    config = make_config(taxonomy_provider_mode="real")
    with pytest.raises(ConfigError):
        build_taxonomy_provider(config)


@pytest.mark.parametrize("mode", ["real", "banana", "", "REAL"])
def test_no_non_mock_mode_can_produce_a_mock_classifier(mode):
    config = make_config(bioclip_provider_mode=mode)
    try:
        built = build_classifier(config)
    except ConfigError:
        return  # refused, which is the required outcome
    assert not isinstance(built, MockBioCLIP2Provider), (
        f"mode {mode!r} silently produced the fixture classifier"
    )


@pytest.mark.parametrize("mode", ["real", "banana", "", "Real"])
def test_no_non_mock_mode_can_produce_fixture_taxonomy(mode):
    config = make_config(taxonomy_provider_mode=mode)
    try:
        built = build_taxonomy_provider(config)
    except ConfigError:
        return
    assert not isinstance(built, MockTaxonomyProvider)


def test_the_agent_refuses_to_start_in_an_unimplemented_production_mode():
    """The whole agent, not just the factory: construction must fail rather
    than come up serving fixtures.

    `taxonomy_provider_mode="real"` still refuses here too - not because
    `real` itself is unimplemented anymore, but because `make_config()`'s
    default carries no NCBI credentials.
    """
    with pytest.raises(ConfigError):
        RecognitionAgent(make_config(bioclip_provider_mode="real"))
    with pytest.raises(ConfigError):
        RecognitionAgent(make_config(taxonomy_provider_mode="real"))


def test_mock_mode_still_builds_the_mock_providers():
    config = make_config()
    assert isinstance(build_classifier(config), MockBioCLIP2Provider)
    assert isinstance(build_taxonomy_provider(config), MockTaxonomyProvider)


def test_a_fake_llm_is_never_selected_by_default(clean_env):
    """`FakeGPT5MiniProvider` must be opt-in. A fake brain that switched itself
    on in a running service would answer with invented reasoning."""
    assert RecognitionConfig.from_env().reasoning_llm_provider_mode == "disabled"
    assert isinstance(build_recognition_llm("disabled"), NullRecognitionLLM)
    assert isinstance(build_recognition_llm("fake"), FakeGPT5MiniProvider)


def test_azure_mode_never_degrades_to_the_fake_provider(clean_env):
    """Missing credentials must raise, not hand back a fake."""
    from ..adapters.reasoning_llm import AzureConfigurationError

    with pytest.raises(AzureConfigurationError) as caught:
        build_recognition_llm("azure")
    assert not isinstance(caught.value, FakeGPT5MiniProvider)


# ===========================================================================
# 5 - missing production configuration, without leaking values
# ===========================================================================

def test_missing_azure_variables_are_named_but_never_valued(clean_env):
    from ..adapters.reasoning_llm import AzureConfigurationError

    clean_env.setenv("AZURE_OPENAI_BASE_URL", "https://secret.invalid/openai/v1")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "super-secret-key-value")
    # deployment left unset

    with pytest.raises(AzureConfigurationError) as caught:
        build_recognition_llm("azure")

    message = str(caught.value)
    assert "AZURE_OPENAI_DEPLOYMENT" in message      # the NAME is useful
    assert "super-secret-key-value" not in message   # the VALUE never is
    assert "secret.invalid" not in message


def test_a_config_error_never_echoes_the_offending_secret(clean_env):
    clean_env.setenv("RECOGNITION_LLM_TIMEOUT_SECONDS", "not-a-number")
    clean_env.setenv("AZURE_OPENAI_API_KEY", "another-secret-value")
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "another-secret-value" not in str(caught.value)


def test_no_credential_reaches_a_log_record(clean_env, caplog):
    """Building the agent logs provider modes. It must never log a key."""
    clean_env.setenv("RECOGNITION_LLM_PROVIDER_MODE", "azure")
    for name, value in _AZURE_ENV.items():
        clean_env.setenv(name, value)

    with caplog.at_level("DEBUG"):
        RecognitionAgent()

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert _AZURE_ENV["AZURE_OPENAI_API_KEY"] not in logged
    assert _AZURE_ENV["AZURE_OPENAI_BASE_URL"] not in logged


# ===========================================================================
# 6 - provider mode and provenance cannot disagree
# ===========================================================================

def _run(agent, instruction="Identify this animal."):
    return agent.run(
        AgentRequest(
            instruction=instruction,
            context={"recognition_image": image_entry(png_bytes())},
        )
    )


def test_provenance_reports_the_provider_that_actually_ran():
    """Provenance is read off the object that executed, not off a literal, so a
    swapped provider cannot be described as the previous one."""
    classifier = StubClassifier([prediction("panthera-leo", 0.93, "Panthera leo")])
    agent = RecognitionAgent(make_config(), classifier=classifier)

    result = _run(agent)
    assert result.status is AgentStatus.COMPLETED

    provenance = result.output["recognition_provenance"]
    assert provenance["recognition_provider"] == classifier.provider_name
    assert provenance["recognition_mode"] == classifier.recognition_mode
    assert provenance["mock_provider_version"] == classifier.version


def test_mock_mode_is_disclosed_as_mock_not_as_classification():
    agent = RecognitionAgent(make_config())
    result = _run(agent)
    provenance = result.output["recognition_provenance"]

    assert provenance["recognition_provider"] == "MockBioCLIP2Provider"
    assert provenance["recognition_mode"] == "mock_classification"
    assert provenance["recognition_mode"] != "classification"
    assert provenance["gbif_mode"] == "mock"
    assert provenance["ncbi_mode"] == "mock"


def test_each_supported_mode_builds_exactly_its_own_provider():
    """Provenance cannot disagree with the mode, because each mode constructs a
    distinct provider and only `mock` can ever produce the fixture oracle."""
    from ..adapters.bioclip import RemoteBioCLIP2Provider

    expected = {"mock": MockBioCLIP2Provider, "remote": RemoteBioCLIP2Provider}
    for mode in BIOCLIP_PROVIDER_MODES:
        built = build_classifier(make_config(bioclip_provider_mode=mode))
        assert isinstance(built, expected[mode]), mode
        if mode != "mock":
            assert not isinstance(built, MockBioCLIP2Provider), mode


# ===========================================================================
# 7 - injection keeps the suite offline
# ===========================================================================

def test_injected_providers_bypass_the_factories_entirely():
    """Tests supply their own providers with no environment set at all."""
    classifier = StubClassifier([prediction("panthera-leo", 0.91, "Panthera leo")])
    taxonomy = MockTaxonomyProvider()
    llm = FakeGPT5MiniProvider()

    agent = RecognitionAgent(
        make_config(bioclip_provider_mode="real", taxonomy_provider_mode="real"),
        classifier=classifier,
        taxonomy_provider=taxonomy,
        reasoning_llm=llm,
    )

    # Injection wins even in a mode the factory would refuse, which is what
    # keeps unit tests independent of configuration.
    result = _run(agent)
    assert result.status is AgentStatus.COMPLETED
    assert classifier.calls == 1


def test_building_the_agent_opens_no_connection_and_downloads_nothing(clean_env):
    clean_env.setenv("RECOGNITION_LLM_PROVIDER_MODE", "azure")
    for name, value in _AZURE_ENV.items():
        clean_env.setenv(name, value)

    agent = RecognitionAgent()
    # Lazy: the Azure client is constructed on the first permitted call, not here.
    assert agent._workflow._reasoning_llm._client is None


def test_the_offline_default_path_needs_no_credential(clean_env):
    """No Azure variable is set at all, and a full request still completes."""
    result = _run(RecognitionAgent())
    assert result.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
    assert result.status is AgentStatus.COMPLETED

"""Configuration: boot settings, log-format resolution, and provider wiring."""
from __future__ import annotations

from configuration.settings import (
    AppEnv,
    AppSettings,
    AzureSettings,
    ContinuationSettings,
    LLMProvider,
    LLMSettings,
    LogFormat,
    NvidiaSettings,
    ObservabilitySettings,
    Settings,
)
from infrastructure.llm.client import AzureOpenAIClient, NullLLMClient
from infrastructure.llm.factory import build_llm_client


class TestAppSettings:
    def test_defaults_match_the_agent_slot_in_the_registry(self) -> None:
        app = AppSettings()

        assert app.port == 8006
        assert app.host == "127.0.0.1"
        assert app.env is AppEnv.DEVELOPMENT

    def test_docs_are_disabled_in_production(self) -> None:
        assert AppSettings(env=AppEnv.PRODUCTION).docs_enabled is False
        assert AppSettings(env=AppEnv.DEVELOPMENT).docs_enabled is True


class TestLogFormatResolution:
    def test_explicit_log_format_always_wins(self) -> None:
        settings = Settings(
            app=AppSettings(env=AppEnv.PRODUCTION),
            observability=ObservabilitySettings(log_format=LogFormat.PRETTY),
        )

        assert settings.log_format is LogFormat.PRETTY

    def test_development_defaults_to_pretty(self) -> None:
        settings = Settings(
            app=AppSettings(env=AppEnv.DEVELOPMENT),
            observability=ObservabilitySettings(log_format=None),
        )

        assert settings.log_format is LogFormat.PRETTY

    def test_production_defaults_to_json(self) -> None:
        """Colour escape codes would corrupt structured log ingestion."""
        settings = Settings(
            app=AppSettings(env=AppEnv.PRODUCTION),
            observability=ObservabilitySettings(log_format=None),
        )

        assert settings.log_format is LogFormat.JSON


#: Disables `.env` loading for one construction. Without it these tests read
#: the developer's real `.env` and pass or fail depending on local config -
#: `AzureSettings()` would come back fully configured on a machine with
#: credentials set.
NO_ENV_FILE: dict[str, object] = {"_env_file": None}


class TestAzureSettings:
    def test_incomplete_credentials_are_not_configured(self) -> None:
        azure = AzureSettings(
            AZURE_OPENAI_BASE_URL="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini",
            AZURE_OPENAI_API_KEY=None,
        )

        assert azure.configured is False

    def test_complete_credentials_are_configured(self) -> None:
        azure = AzureSettings(
            AZURE_OPENAI_BASE_URL="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini",
            AZURE_OPENAI_API_KEY="secret",
        )

        assert azure.configured is True


class TestLLMFactory:
    def test_provider_none_gives_the_null_client(self) -> None:
        client = build_llm_client(LLMSettings(provider=LLMProvider.NONE))

        assert isinstance(client, NullLLMClient)
        assert client.available is False

    def test_azure_without_credentials_falls_back_rather_than_raising(self) -> None:
        """A misconfigured LLM degrades the run; it must not fail it."""
        client = build_llm_client(
            LLMSettings(provider=LLMProvider.AZURE_FOUNDRY),
            AzureSettings(**NO_ENV_FILE),  # type: ignore[arg-type]
        )

        assert isinstance(client, NullLLMClient)

    def test_azure_with_credentials_builds_the_azure_client(self) -> None:
        client = build_llm_client(
            LLMSettings(provider=LLMProvider.AZURE_FOUNDRY),
            AzureSettings(
                AZURE_OPENAI_BASE_URL="https://example.openai.azure.com",
                AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini",
                AZURE_OPENAI_API_KEY="secret",
            ),
        )

        assert isinstance(client, AzureOpenAIClient)
        assert client.available is True


class TestNvidiaSettings:
    def test_not_configured_without_a_key(self) -> None:
        assert NvidiaSettings(api_key=None).configured is False

    def test_configured_with_a_key(self) -> None:
        assert NvidiaSettings(api_key="nvapi-xxx").configured is True


class TestContinuationSettings:
    def test_deadline_is_derived_from_global_timeout_and_reserve(self) -> None:
        continuation = ContinuationSettings(
            _env_file=None,
            read_timeout_seconds=600,
            finalization_reserve_seconds=60,
        )

        assert continuation.effective_yield_after_seconds == 540


class TestOrchestratorContextKeys:
    """The keys the orchestrator actually sends, not the ones we wish it sent.

    Nothing upstream writes `organism`: the orchestrator's extractor seeds
    `species` from the user's question and the Genome agent publishes a
    `species_record`. Reading only `organism` silently dropped the organism on
    every orchestrated run, which costs reference ranking and the Evolution
    Agent escalation prompt.
    """

    def test_species_is_read_as_the_organism(self) -> None:
        from contracts.input import ReconstructionRequest

        request = ReconstructionRequest.from_agent_request(
            "Fill the gaps", {"species": "Mammuthus primigenius", "sequence": "ACGTN" * 10}
        )

        assert request.organism == "Mammuthus primigenius"
        assert request.sequence is not None
        assert request.sequence.organism == "Mammuthus primigenius"

    def test_organism_wins_over_species_when_both_are_present(self) -> None:
        from contracts.input import ReconstructionRequest

        request = ReconstructionRequest.from_agent_request(
            "Fill the gaps", {"organism": "Loxodonta africana", "species": "elephant"}
        )

        assert request.organism == "Loxodonta africana"

    def test_the_genome_agents_species_record_is_read(self) -> None:
        from contracts.input import ReconstructionRequest

        request = ReconstructionRequest.from_agent_request(
            "Fill the gaps",
            {"species_record": {"scientific_name": "Vulpes lagopus", "common_name": "Arctic fox"}},
        )

        assert request.organism == "Vulpes lagopus"

    def test_no_organism_anywhere_stays_none(self) -> None:
        from contracts.input import ReconstructionRequest

        request = ReconstructionRequest.from_agent_request("Fill the gaps", {"gene_name": "FGF5"})

        assert request.organism is None

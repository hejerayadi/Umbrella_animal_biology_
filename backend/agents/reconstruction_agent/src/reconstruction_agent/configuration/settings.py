"""Every knob this agent reads from the environment, in one typed object.

Nothing else in the package calls `os.getenv`. Modules take a `Settings` (or
one of its nested sections) as an argument, which is what makes them testable
without touching the process environment.

`get_settings()` is cached: the agent's `.env` is read once per process, at
first use, not per request.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The agent's own .env, four levels up from this file:
# configuration/ -> reconstruction_agent/ -> src/ -> <agent root>
_AGENT_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _AGENT_ROOT / ".env"


class LLMProvider(str, Enum):
    """Which LLM backend the planner and critic talk to.

    NONE is a first-class choice, not an error: the agent falls back to a
    deterministic plan so it stays runnable in CI and offline development.
    """

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    NONE = "none"


def _config(env_prefix: str = "") -> SettingsConfigDict:
    """Settings config for one section.

    Built by this helper rather than by spreading a base `model_config`:
    `SettingsConfigDict` already carries a default `env_prefix`, so spreading
    it alongside an explicit one is a duplicate-keyword error.
    """
    return SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix=env_prefix,
    )


class _Base(BaseSettings):
    model_config = _config()


class NCBISettings(_Base):
    """Access to NCBI E-utilities (esearch/efetch)."""

    model_config = _config("NCBI_")

    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key: str | None = None
    tool_name: str = "umbrella-reconstruction-agent"
    contact_email: str | None = None

    @property
    def requests_per_second(self) -> int:
        """NCBI's documented anonymous limit is 3/s, raised to 10/s with a key.

        The rate limiter reads this rather than a hardcoded constant so that
        supplying a key actually buys the extra throughput.
        """
        return 10 if self.api_key else 3


class EMBLEBISettings(_Base):
    """Access to EMBL-EBI's job-based REST tools (BLAST, MAFFT)."""

    model_config = _config("EMBL_EBI_")

    base_url: str = "https://www.ebi.ac.uk/Tools/services/rest"
    # EMBL-EBI rejects submissions without a contact address, so this is
    # required in practice - but only at call time, not at import time.
    contact_email: str | None = None
    poll_interval_seconds: float = 5.0
    poll_timeout_seconds: float = 600.0


class LLMSettings(_Base):
    """Which model the planner/critic use, and how to reach it."""

    model_config = _config("LLM_")

    provider: LLMProvider = LLMProvider.NONE
    model: str = "gpt-5-mini"
    api_key: str | None = None
    base_url: str | None = None
    # Reconstruction planning must be reproducible; default to greedy decoding.
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.provider is not LLMProvider.NONE


class HTTPSettings(_Base):
    model_config = _config("HTTP_")

    timeout_seconds: float = 30.0
    max_retries: int = 3


class ObservabilitySettings(_Base):
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="umbrella-reconstruction-agent", alias="LANGSMITH_PROJECT"
    )


class Settings(_Base):
    """Root settings object. Build it with `get_settings()`."""

    agent_name: str = Field(default="reconstruction_agent", alias="AGENT_NAME")
    agent_mode: str = Field(default="local", alias="AGENT_MODE")

    max_iterations: int = Field(default=6, alias="RECONSTRUCTION_MAX_ITERATIONS", ge=1, le=50)
    min_confidence: float = Field(
        default=0.55, alias="RECONSTRUCTION_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    max_gap_length: int = Field(default=5000, alias="RECONSTRUCTION_MAX_GAP_LENGTH", ge=1)

    ncbi: NCBISettings = Field(default_factory=NCBISettings)
    embl_ebi: EMBLEBISettings = Field(default_factory=EMBLEBISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton.

    Cached so that reading `.env` and validating stays a startup cost. Tests
    that need different values should build `Settings(...)` directly, or call
    `get_settings.cache_clear()`.
    """
    return Settings()

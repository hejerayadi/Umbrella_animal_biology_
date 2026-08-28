"""Typed runtime settings - the only place in the agent that reads the environment.

Every other module receives a `Settings` (or one of its sub-models) rather than
calling `os.getenv`, so the full configuration surface is visible here and
testable by constructing the object directly.

Settings are grouped to match the `.env` file's sections, each group binding to
its own environment prefix. The `.env` path is absolute and anchored to the
agent directory: `backend/run_agents.py` launches this service with
`cwd = repository root`, so a relative `env_file=".env"` would silently resolve
against the wrong directory and load nothing.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `backend/agents/reconstruction_agent/` - this file is at
#: `src/reconstruction_agent/config/settings.py`, hence three levels up.
SERVICE_ROOT = Path(__file__).resolve().parents[3]

_ENV_FILE = SERVICE_ROOT / ".env"


def _config(prefix: str = "") -> SettingsConfigDict:
    """Shared loader configuration, differing only by environment prefix.

    `extra="ignore"` is required rather than cosmetic: every group sees the
    whole environment, so a group without it would reject the other groups'
    variables.
    """
    return SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=prefix,
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )


class Environment(str, Enum):
    """Deployment environment. Decides log format and whether /docs is exposed."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LlmProvider(str, Enum):
    """Which LLM backs planning and criticism.

    `NONE` is a first-class choice, not a degraded mode: the graph must reach a
    scientifically valid answer on the deterministic fallback plan alone, and
    the offline test suite runs this way.
    """

    AZURE_OPENAI = "azure_openai"
    NONE = "none"


class AppSettings(BaseSettings):
    """Application boot."""

    model_config = _config("APP_")

    env: Environment = Environment.DEVELOPMENT
    name: str = "Umbrella Reconstruction Agent"
    host: str = "127.0.0.1"
    #: 8006 is this agent's slot in backend/registry.py.
    port: int = 8006
    reload: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_enabled(self) -> bool:
        """Interactive docs are a development affordance, not a public surface."""
        return self.env is not Environment.PRODUCTION


class ObservabilitySettings(BaseSettings):
    """Logging, correlation and tracing."""

    model_config = _config()

    log_level: str = "INFO"
    #: None follows APP_ENV: pretty in development, json everywhere else.
    log_format: Literal["pretty", "json"] | None = None

    #: Unique per HTTP attempt.
    request_id_header: str = "X-Request-Id"
    #: Stable for a whole orchestrator run, across retries.
    trace_id_header: str = "X-Trace-Id"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "umbrella-reconstruction-agent"


class HttpSettings(BaseSettings):
    """Defaults for every outbound HTTP client."""

    model_config = _config("HTTP_")

    timeout_seconds: float = 30.0
    max_retries: int = 3


class NcbiSettings(BaseSettings):
    """NCBI E-utilities: sequences, assembly metadata and taxonomy.

    Also the authority for taxonomy in this agent - lineages, ranks, and the
    vernacular names that EMBL-EBI's database labels are resolved against.
    """

    model_config = _config("NCBI_")

    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key: SecretStr | None = None
    tool_name: str = "umbrella-reconstruction-agent"
    contact_email: str | None = None
    #: Lifetime of the taxonomy cache. Taxonomy does not change during a run,
    #: and a lineage lookup repeated per hit would dominate the rate limit.
    cache_ttl_seconds: float = 3600.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def requests_per_second(self) -> float:
        """NCBI allows 3/s anonymously and 10/s with a key.

        Paced well under each cap rather than just under it. Sitting at 9/s
        against a published 10/s ceiling was measured earning a block partway
        through a run - and NCBI enforces by going silent, not by answering
        429, so every later call in that run timed out instead of failing
        usefully. The headroom costs a few seconds and buys the run finishing.
        """
        return 6.0 if self.api_key else 2.0


class EmblEbiSettings(BaseSettings):
    """EMBL-EBI MAFFT, the one job service still used.

    Homology search moved to the NCBI BLAST URL API. Alignment stays here
    because NCBI publishes no alignment service and no MAFFT binary is
    installed anywhere in this repository.
    """

    model_config = _config("EMBL_EBI_")

    base_url: str = "https://www.ebi.ac.uk/Tools/services/rest"
    #: REQUIRED. EMBL-EBI rejects anonymous job submissions outright, so
    #: without it every gap that reaches alignment comes back unresolved.
    contact_email: str | None = None
    poll_interval_seconds: float = 5.0
    #: Below the run deadline on purpose - a poll that outlives the run is waste.
    poll_timeout_seconds: float = 280.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configured(self) -> bool:
        return bool(self.contact_email)


class NvidiaSettings(BaseSettings):
    """NVIDIA NIM access for Evo 2."""

    model_config = _config("NVIDIA_")

    base_url: str = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b"
    model: str = "arc/evo2-40b"
    api_key: SecretStr | None = None
    timeout_seconds: float = 120.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configured(self) -> bool:
        """Without a key the Evo 2 tool is not registered at all."""
        return bool(self.api_key)


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI deployment backing the planner and critic."""

    model_config = _config("AZURE_OPENAI_")

    base_url: str | None = None
    deployment: str | None = None
    api_key: SecretStr | None = None
    #: Azure pins the wire format by date; a mismatch surfaces as a 404 on the
    #: deployment rather than a version error.
    api_version: str = "2024-10-21"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.deployment and self.api_key)


class LlmSettings(BaseSettings):
    """How the agent talks to a language model, if at all."""

    model_config = _config("LLM_")

    provider: LlmProvider = LlmProvider.AZURE_OPENAI
    temperature: float = 0.0
    max_tokens: int = 2048


class NcbiBlastSettings(BaseSettings):
    """The official NCBI BLAST URL API.

    A different service from E-utilities, with its own etiquette and no API
    key: NCBI issues keys for E-utilities only, so nothing here is
    authenticated. What NCBI asks for instead is identification and restraint -
    a `tool` and `email` on every call, no more than one request every ten
    seconds, and no more than one poll a minute for any single search.
    """

    model_config = _config("NCBI_BLAST_")

    #: The host only. The endpoint is a path, because httpx treats a base URL
    #: as a directory and appends a slash - and `/Blast.cgi/` is served as the
    #: interactive BLAST form, not as the API, so every submission comes back
    #: as a megabyte of HTML with no request id in it.
    base_url: str = "https://blast.ncbi.nlm.nih.gov"
    endpoint_path: str = "/Blast.cgi"
    #: `core_nt` is the curated core of the nucleotide collection; `nt` is the
    #: whole of it and is markedly slower.
    database: str = "core_nt"
    #: The URL API accepts only the base program names. `megablast` is not one
    #: of them - it is `blastn` with a flag, and passing it as a program earns
    #: "Message ID#7 Error: Unrecognized program name".
    program: str = "blastn"
    #: Off, despite the query being two flanks of a close relative - which is
    #: exactly what megablast is for.
    #:
    #: Measured: on a 1 kb polar bear flank query, megablast returned twenty
    #: hits at identity 1.000 and *none* spanning the gap. It is a local
    #: aligner and splits at the 45-base indel, giving two HSPs of 0.50
    #: coverage each, neither crossing the junction. Gapped blastn bridges the
    #: indel in one HSP, which is the alignment a reconstruction needs.
    megablast: bool = False
    #: Legacy BLAST XML. Not in the current documented list (XML2, XML2_S,
    #: JSON2, JSON2_S, SAM), but still served and the format the parser reads.
    #: Recorded here so the day it stops being served is a config change.
    format_type: str = "XML"
    tool_name: str = "umbrella-reconstruction-agent"
    #: NCBI asks to be able to contact whoever is submitting.
    contact_email: str | None = None
    #: NCBI asks for no more than one request every 10 seconds.
    requests_per_second: float = 0.1
    #: NCBI asks that a single search not be polled more than once a minute.
    #: Honoured literally: the cost is that a fast search is noticed late.
    poll_interval_seconds: float = 60.0
    poll_timeout_seconds: float = 600.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def configured(self) -> bool:
        return bool(self.contact_email)


class HomologySettings(BaseSettings):
    """How homologues are searched for."""

    model_config = _config("HOMOLOGY_")

    #: Taxonomic ranks tried as search scopes, narrowest first. Each is looked
    #: up in the target lineage; ranks the target has no taxon at are skipped.
    #: Nothing here names a clade - only how specific to be.
    ncbi_scope_ranks: tuple[str, ...] = ("family", "order", "class")
    #: Exclude the record being repaired from its own search. A record cannot
    #: be evidence about its own unresolved region, and leaving it in makes a
    #: ground-truth measurement a lookup of the answer.
    exclude_target_accession: bool = True

    #: How long the narrowest scope may still be waited for once a wider one
    #: has already returned usable evidence. The narrowest scope gives the
    #: closest relatives, so it is worth waiting for - but not unboundedly:
    #: BLAST queue times vary by two orders of magnitude between identical
    #: submissions, and letting one slow queue consume the homology budget
    #: leaves nothing to fetch and align the hits that did arrive.
    front_runner_grace_seconds: float = Field(default=25.0, ge=0.0)


class ReconstructionSettings(BaseSettings):
    """Agent behaviour, budgets and the run deadline."""

    model_config = _config("RECONSTRUCTION_")

    max_iterations: int = Field(default=6, ge=1)
    #: A candidate below this is reported UNRESOLVED rather than returned.
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    max_gap_length: int = Field(default=5000, ge=1)
    #: Draft assemblies carry hundreds of N-runs; a run commits to what it can
    #: finish inside the deadline and reports the rest.
    max_gaps_per_run: int = Field(default=8, ge=1)

    #: The whole run is one HTTP call. The orchestrator allows 600 s, so
    #: finishing well inside that is what keeps findings from being discarded.
    deadline_seconds: float = Field(default=300.0, gt=0.0)
    finalization_reserve_seconds: float = Field(default=30.0, ge=0.0)
    homology_budget_seconds: float = Field(default=200.0, gt=0.0)
    alignment_budget_seconds: float = Field(default=40.0, gt=0.0)
    arbitration_budget_seconds: float = Field(default=30.0, ge=0.0)

    max_tool_calls: int = Field(default=24, ge=1)
    max_blast_calls: int = Field(default=8, ge=1)
    max_mafft_calls: int = Field(default=8, ge=1)
    max_evo2_calls: int = Field(default=4, ge=0)
    max_llm_calls: int = Field(default=12, ge=0)


class Settings(BaseSettings):
    """The composed configuration for one process."""

    model_config = _config()

    app: AppSettings = Field(default_factory=AppSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    ncbi: NcbiSettings = Field(default_factory=NcbiSettings)
    embl_ebi: EmblEbiSettings = Field(default_factory=EmblEbiSettings)
    ncbi_blast: NcbiBlastSettings = Field(default_factory=NcbiBlastSettings)
    homology: HomologySettings = Field(default_factory=HomologySettings)
    nvidia: NvidiaSettings = Field(default_factory=NvidiaSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    reconstruction: ReconstructionSettings = Field(default_factory=ReconstructionSettings)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_log_format(self) -> Literal["pretty", "json"]:
        """Explicit LOG_FORMAT wins; otherwise the environment decides."""
        if self.observability.log_format is not None:
            return self.observability.log_format
        return "pretty" if self.app.env is Environment.DEVELOPMENT else "json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_enabled(self) -> bool:
        """True when an LLM will actually be called.

        A provider selected but not configured is treated as absent rather than
        as an error: the deterministic plan still produces a valid answer, and
        failing the whole service over a missing key would be worse.
        """
        if self.llm.provider is LlmProvider.NONE:
            return False
        return self.azure_openai.configured


@lru_cache
def get_settings() -> Settings:
    """The process-wide settings instance.

    Cached because reading and validating the environment on every request is
    pointless work; tests construct `Settings(...)` directly instead of calling
    this, so a developer's real `.env` cannot change an assertion.
    """
    return Settings()

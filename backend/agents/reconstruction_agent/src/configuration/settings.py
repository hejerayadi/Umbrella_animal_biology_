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

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The agent's own .env, three levels up from this file:
# configuration/ -> src/ -> <agent root>
_AGENT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _AGENT_ROOT / ".env"

# The backend's .env, two levels up: agents/ -> backend/. The database URL is
# declared there once, for the whole of Umbrella, and this agent reads it
# rather than keeping a second copy that could drift.
#
# Reading a config file is not importing a package: the agent still has its own
# `.venv` and never imports `backend`, so the isolation that lets it pin its
# own dependencies is untouched.
_BACKEND_ENV_FILE = _AGENT_ROOT.parents[1] / ".env"

#: Where prompt markdown lives. Prompts are data, not code - see
#: `agent/prompts/loader.py`.
PROMPTS_DIR = _AGENT_ROOT / "src" / "prompts"


class AppEnv(str, Enum):
    """Which deployment this process is.

    Drives defaults that should differ between a laptop and a server -
    log format, and whether the interactive API docs are exposed.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogFormat(str, Enum):
    """How log records are rendered.

    `PRETTY` is rich-rendered colour for a terminal; `JSON` is one object per
    line for a log aggregator. Never pretty in production - the colour escape
    codes corrupt structured ingestion.
    """

    PRETTY = "pretty"
    JSON = "json"


class LLMProvider(str, Enum):
    """Which LLM backend the planner and critic talk to.

    NONE is a first-class choice, not an error: the agent falls back to a
    deterministic plan so it stays runnable in CI and offline development.
    """

    AZURE_FOUNDRY = "azure_foundry"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
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
        # Without this, a field declared with an `alias` can ONLY be set by its
        # alias, and `Settings(yield_after_seconds=...)` is silently discarded
        # by `extra="ignore"` - leaving the default in place with no error.
        # Tests and programmatic construction both need the field name to work.
        populate_by_name=True,
    )


class _Base(BaseSettings):
    model_config = _config()


class AppSettings(_Base):
    """How the process boots: environment, and where uvicorn binds."""

    model_config = _config("APP_")

    env: AppEnv = AppEnv.DEVELOPMENT
    name: str = "Umbrella Reconstruction Agent"
    host: str = "127.0.0.1"
    # 8006 is this agent's slot in backend/registry.py. Changing it here alone
    # would leave the orchestrator calling the old port.
    port: int = 8006
    reload: bool = False

    #: Browser origins allowed to call this agent directly. Empty in
    #: production, where the only client is the orchestrator and a browser has
    #: no business reaching the agent at all. `*` in development so the local
    #: test page - opened from disk, which sends `Origin: null` - can drive it.
    cors_origins: str = Field(default="", alias="APP_CORS_ORIGINS")

    @property
    def is_production(self) -> bool:
        return self.env is AppEnv.PRODUCTION

    @property
    def allowed_origins(self) -> list[str]:
        """Origins to send CORS headers for.

        Explicit entries always win. Outside production an unset value means
        `*`, so the local test page works with no configuration; in production
        an unset value means no CORS at all.
        """
        configured = [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]
        if configured or self.is_production:
            return configured
        return ["*"]

    @property
    def docs_enabled(self) -> bool:
        """Interactive docs are a development convenience, not a public surface."""
        return self.env is not AppEnv.PRODUCTION


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


class AzureSettings(_Base):
    """Azure OpenAI / AI Foundry access for the planner and critic."""

    model_config = _config("AZURE_")

    openai_base_url: str | None = Field(default=None, alias="AZURE_OPENAI_BASE_URL")
    openai_deployment: str | None = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    # Azure pins the wire format by date; the SDK default drifts, and a mismatch
    # surfaces as a confusing 404 on the deployment rather than a version error.
    openai_api_version: str = Field(
        default="2024-10-21", alias="AZURE_OPENAI_API_VERSION"
    )
    foundry_project_endpoint: str | None = Field(
        default=None, alias="AZURE_FOUNDRY_PROJECT_ENDPOINT"
    )

    @property
    def configured(self) -> bool:
        return bool(self.openai_base_url and self.openai_deployment and self.openai_api_key)


class NvidiaSettings(_Base):
    """NVIDIA NIM access for the Evo 2 genomic foundation model.

    Evo 2 scores how plausible a nucleotide sequence is, which is a different
    kind of evidence from homology: it can flag a consensus that aligns well
    but reads as biologically implausible.
    """

    model_config = _config("NVIDIA_")

    api_key: str | None = None
    base_url: str = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b"
    model: str = "arc/evo2-40b"
    timeout_seconds: float = 120.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class LLMSettings(_Base):
    """Which model the planner/critic use, and how to reach it."""

    model_config = _config("LLM_")

    provider: LLMProvider = LLMProvider.NONE
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    # Reconstruction planning must be reproducible; default to greedy decoding.
    temperature: float = 0.0
    max_tokens: int = 2048

    @property
    def enabled(self) -> bool:
        return self.provider is not LLMProvider.NONE


class HTTPSettings(_Base):
    model_config = _config("HTTP_")

    timeout_seconds: float = 30.0
    max_retries: int = 3


class ObservabilitySettings(_Base):
    """Logging and tracing.

    `log_format` has no default of its own here: `Settings` resolves it from
    `APP_ENV` when it is not set explicitly, so production cannot accidentally
    emit colour codes into a log pipeline.
    """

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: LogFormat | None = Field(default=None, alias="LOG_FORMAT")
    correlation_id_header: str = Field(default="X-Request-ID", alias="REQUEST_ID_HEADER")
    #: The orchestrator's run-wide id, stable across CONTINUE retries. Distinct
    #: from the per-attempt request id above, and the key the checkpoint uses.
    correlation_id_header_trace: str = Field(default="X-Trace-Id", alias="TRACE_ID_HEADER")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="umbrella-reconstruction-agent", alias="LANGSMITH_PROJECT"
    )


class BudgetSettings(_Base):
    """Hard caps on what one reconstruction may consume.

    These exist because the agent runs inside someone else's request. The
    orchestrator allows 120 s per HTTP call and only three CONTINUE retries
    (`backend/orchestrator/langgraph/nodes/worker_node.py`), so an unbounded
    loop does not merely cost money - it gets the whole run force-failed and
    every finding discarded.
    """

    model_config = _config("RECONSTRUCTION_")

    # Total tool invocations across every slice of one run.
    max_tool_calls: int = Field(default=12, ge=1)
    # Per-tool caps for the expensive ones. BLAST and MAFFT are submit-and-poll
    # jobs measured in tens of seconds; a handful of each fills the wall clock.
    max_blast_calls: int = Field(default=4, ge=0)
    max_mafft_calls: int = Field(default=6, ge=0)
    # Prompt + completion tokens across the whole run.
    max_llm_tokens: int = Field(default=60_000, ge=0)

    def per_tool(self) -> dict[str, int]:
        return {"blast_search": self.max_blast_calls, "mafft_align": self.max_mafft_calls}


class ContinuationSettings(_Base):
    """How the agent yields back to the orchestrator mid-run.

    The orchestrator gives each agent 120 s per call and retries a CONTINUE
    three times with 1/2/4 s backoff - so four HTTP slices in total, and the
    fifth is converted to FAILED. Both numbers are mirrored here because the
    agent has to stay inside them without importing `backend`.
    """

    #: Yield before the orchestrator's 120 s read timeout. This doubles as the
    #: deadline for each individual tool call - see `BudgetPolicy.
    #: remaining_seconds` - because the yield check only runs between graph
    #: nodes and so cannot interrupt a submit-and-poll tool on its own. The
    #: remaining margin covers the critic's LLM call, which runs after the
    #: tools, and serialising the result.
    yield_after_seconds: float = Field(default=75.0, alias="AGENT_YIELD_AFTER_SECONDS", gt=0)
    #: Slices the orchestrator will grant: the first call plus three retries.
    #: On the last one the agent must return COMPLETED with whatever it has,
    #: because another CONTINUE would be turned into FAILED.
    max_slices: int = Field(default=4, alias="AGENT_MAX_SLICES", ge=1)
    #: Header carrying the orchestrator's run-wide correlation id. It is the
    #: only identifier stable across retries, so it keys the checkpoint.
    trace_id_header: str = Field(default="X-Trace-Id", alias="TRACE_ID_HEADER")


class DatabaseSettings(_Base):
    """Where LangGraph checkpoints and the run audit table live.

    The same database and schema as the rest of Umbrella - one database, no
    dedicated namespace. The agent's tables are told apart by name
    (`reconstruction_runs`, LangGraph's `checkpoint*`), not by schema.

    **The URL is declared once, in `backend/.env` as `DATABASE_URL`.** This
    section reads that file directly so there is no second copy to drift out of
    step when the password or port changes. `RECONSTRUCTION_DATABASE_URL` is
    still honoured and wins when set, which is how a container or CI job points
    the agent somewhere else without editing a file.

    One thing must still stay separate: the Alembic version table. The backend
    runs its own migration history in the default `alembic_version`, and two
    histories sharing that row would each treat the other's revision as
    unknown and try to "repair" it. See `migrations/env.py`.

    No URL at all means in-memory checkpointing and no audit rows - the
    supported mode for tests and for running the agent without infrastructure.
    """

    # Both files, agent last so it can override; a missing file is ignored.
    # No `env_prefix`: the aliases below are explicit, and a prefix would stop
    # `DATABASE_URL` from being seen at all.
    model_config = SettingsConfigDict(
        env_file=(_BACKEND_ENV_FILE, _ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RECONSTRUCTION_DATABASE_URL", "DATABASE_URL"),
    )
    #: Bounded hard, and low. psycopg's default connect timeout is ~130 s -
    #: longer than the 120 s the orchestrator allows for the whole request, so
    #: an unreachable database would hang past the deadline instead of falling
    #: back to in-memory checkpointing.
    connect_timeout_seconds: int = Field(
        default=5, alias="RECONSTRUCTION_DB_CONNECT_TIMEOUT", ge=1
    )
    @property
    def configured(self) -> bool:
        return bool(self.database_url)


class Settings(_Base):
    """Root settings object. Build it with `get_settings()`."""

    max_iterations: int = Field(default=6, alias="RECONSTRUCTION_MAX_ITERATIONS", ge=1, le=50)
    # Measured, not guessed: `scripts/tune_settings.py` sweeps 360 synthetic
    # gaps with known answers, and this is the lowest threshold that accepts no
    # incorrect reconstruction. Precision is weighted above recall because a
    # wrong base propagates silently into every downstream analysis, while an
    # unresolved gap stays visibly unresolved.
    #
    # 0.15, not the 0.65 this used to be. That figure was measured against the
    # OLD confidence score, which separated correct from incorrect at 0.575 -
    # barely better than chance - so the threshold had to be set high to
    # compensate. The rebuilt score separates them at 0.987, and
    # `scripts/tuning_baseline.json` records precision 1.000 from 0.15 upward.
    # Leaving the old default here meant any deployment that did not set the
    # variable explicitly ran at recall 0.289 and refused most gaps it could
    # have filled correctly - while `.env.example` and `ConfidencePolicy` had
    # both already moved to 0.15.
    min_confidence: float = Field(
        default=0.15, alias="RECONSTRUCTION_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    max_gap_length: int = Field(default=5000, alias="RECONSTRUCTION_MAX_GAP_LENGTH", ge=1)

    app: AppSettings = Field(default_factory=AppSettings)
    ncbi: NCBISettings = Field(default_factory=NCBISettings)
    embl_ebi: EMBLEBISettings = Field(default_factory=EMBLEBISettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    nvidia: NvidiaSettings = Field(default_factory=NvidiaSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    budgets: BudgetSettings = Field(default_factory=BudgetSettings)
    continuation: ContinuationSettings = Field(default_factory=ContinuationSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def log_format(self) -> LogFormat:
        """The effective log format.

        An explicit `LOG_FORMAT` always wins. Otherwise it follows the
        environment: pretty for a developer's terminal, JSON anywhere the
        output is collected by something other than a human.
        """
        if self.observability.log_format is not None:
            return self.observability.log_format
        return LogFormat.PRETTY if self.app.env is AppEnv.DEVELOPMENT else LogFormat.JSON

    @property
    def agent_name(self) -> str:
        return "reconstruction_agent"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton.

    Cached so that reading `.env` and validating stays a startup cost. Tests
    that need different values should build `Settings(...)` directly, or call
    `get_settings.cache_clear()`.
    """
    return Settings()

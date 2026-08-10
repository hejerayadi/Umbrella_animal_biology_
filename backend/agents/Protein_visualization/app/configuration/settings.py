from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVICE_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Umbrella Protein Agent"
    app_version: str = "1.1.0"
    app_env: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_reload: bool = False
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_format: str = "json"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    uniprot_base_url: str = "https://rest.uniprot.org"
    rcsb_search_base_url: str = "https://search.rcsb.org/rcsbsearch/v2"
    rcsb_data_base_url: str = "https://data.rcsb.org"
    rcsb_files_base_url: str = "https://files.rcsb.org"
    alphafold_base_url: str = "https://alphafold.ebi.ac.uk/api"
    interpro_base_url: str = "https://www.ebi.ac.uk/interpro/api"
    sifts_base_url: str = "https://www.ebi.ac.uk/pdbe/api"

    http_timeout_seconds: float = Field(default=15.0, gt=0)
    http_max_retries: int = Field(default=2, ge=0, le=10)
    max_structure_candidates: int = Field(default=10, ge=1, le=100)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    min_sequence_coverage: float = Field(default=0.3, ge=0.0, le=1.0)

    # PostgreSQL is out of scope for this sprint; the local SQLite file keeps the
    # repository layer runnable without provisioning a database.
    persistence_enabled: bool = False
    database_url: str = "sqlite:///./protein_agent.db"

    # Qdrant runs as a managed cluster; URL and API key come from the environment.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "protein_knowledge"
    embedding_provider: str = "bge-m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = Field(default=1024, gt=0)
    internal_ingestion_api_key: str | None = None

    llm_provider: str = "disabled"
    azure_openai_base_url: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_key: str | None = None
    azure_foundry_project_endpoint: str | None = None
    azure_ai_endpoint: str | None = None
    azure_ai_api_key: str | None = None
    azure_ai_deployment: str | None = None
    azure_ai_api_version: str = "2024-10-21"
    # Left unset for deployments that reject an explicit temperature.
    azure_temperature: float | None = 0.0

    # Optional: USD per 1,000 tokens for the deployed model, so a response can
    # report an estimated cost alongside the token counts Azure itself returns.
    # Azure OpenAI pricing depends on region, contract and model version and is
    # not discoverable from the API, so this is never guessed - unset means the
    # response reports tokens and latency only, no cost.
    azure_input_price_per_1k_usd: float | None = Field(default=None, ge=0)
    azure_output_price_per_1k_usd: float | None = Field(default=None, ge=0)

    @property
    def local_base_url(self) -> str:
        return f"http://{self.app_host}:{self.app_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()

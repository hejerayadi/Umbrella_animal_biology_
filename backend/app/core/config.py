from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_name: str = "Umbrella API"
    api_prefix: str = "/api/v1"
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )

    database_url: str = (
        "postgresql+psycopg://umbrella:umbrella_dev@localhost:5432/umbrella"
    )
    redis_url: str = "redis://:umbrella_dev@localhost:6379/0"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        if not isinstance(value, str) or not value.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            return value

        url = make_url(value)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        if url.host and url.host.endswith(".supabase.com") and "sslmode" not in url.query:
            url = url.update_query_dict({"sslmode": "require"})

        return url.render_as_string(hide_password=False)

    session_cookie_name: str = "umbrella_session"
    session_idle_ttl_seconds: int = 86_400
    session_absolute_ttl_seconds: int = 604_800
    restricted_session_ttl_seconds: int = 1_800
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    email_verification_ttl_minutes: int = 30
    password_reset_ttl_minutes: int = 30
    invitation_ttl_hours: int = 72
    audit_retention_days: int = 365

    # A deterministic development-only Fernet key. Production must override it.
    auth_encryption_key: str = "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="

    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "no-reply@umbrella.local"
    smtp_from_name: str = "Umbrella"
    smtp_starttls: bool = False

    login_rate_limit: int = 5
    login_rate_window_seconds: int = 900
    public_rate_limit: int = 5
    public_rate_window_seconds: int = 3600

    request_id_header: str = Field(default="X-Request-Id")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

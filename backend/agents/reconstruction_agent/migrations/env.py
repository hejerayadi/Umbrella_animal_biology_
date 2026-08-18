"""Alembic environment for the Reconstruction Agent.

The database URL comes from the agent's own settings rather than `alembic.ini`,
so the migration and the running agent can never point at different databases,
and no credential is committed.

This owns only `reconstruction_runs`. LangGraph creates and manages its own
checkpoint tables through `AsyncPostgresSaver.setup()`, called at startup -
version-controlling someone else's schema would break the moment they change
it, and their setup is already idempotent.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# `prepend_sys_path = src` in alembic.ini covers the normal invocation; this
# makes the module importable when alembic is driven programmatically too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from configuration.settings import get_settings  # noqa: E402
from infrastructure.persistence.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_settings = get_settings()
_SCHEMA = _settings.database.schema_name


def _database_url() -> str:
    """The agent's configured URL, as a synchronous driver.

    Alembic runs its own migrations synchronously; the async driver the agent
    uses at runtime would fail here with "greenlet_spawn has not been called".
    """
    url = _settings.database.database_url
    if not url:
        raise RuntimeError(
            "RECONSTRUCTION_DATABASE_URL is not set. Point it at the Postgres from "
            "docker/compose.yml before running migrations."
        )
    return url.replace("+asyncpg", "").replace("postgresql+psycopg", "postgresql+psycopg")


def _include_object(obj: object, name: str | None, type_: str, *_: object) -> bool:
    """Keep autogenerate away from LangGraph's tables.

    They live in the same schema but are not ours; without this, `--autogenerate`
    would propose dropping them on the first run.
    """
    langgraph_tables = {
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
    }
    return not (type_ == "table" and name in langgraph_tables)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=_SCHEMA,
        include_schemas=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the configured database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        # The schema has to exist before the version table can be created in it.
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_SCHEMA}"'))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=_SCHEMA,
            include_schemas=True,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

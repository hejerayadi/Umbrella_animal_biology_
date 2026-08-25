"""Alembic environment for the Reconstruction Agent.

The database URL comes from the agent's own settings rather than `alembic.ini`,
so the migration and the running agent can never point at different databases,
and no credential is committed. Those settings read `backend/.env`, where
`DATABASE_URL` is declared once for the whole of Umbrella.

## Sharing one database with the backend

The agent writes into the same database and schema as `backend/` - one
database, no dedicated namespace. Two things make that safe:

1. **A separate version table.** `backend/migrations` runs its own history in
   the default `alembic_version`. If this one used it too, each would find the
   other's revision id unrecognised and try to "repair" it - in practice
   stamping over it, then failing every subsequent upgrade. `ALEMBIC_VERSION`
   below keeps the two histories apart while the tables sit side by side.

2. **Distinct table names.** `reconstruction_runs` and LangGraph's
   `checkpoint*` tables cannot collide with `users`, `invitations` and the
   rest. `include_object` below also stops autogenerate proposing to drop
   anything this history does not own - as an allow-list of what *is* ours,
   not a deny-list of what is not, so it cannot drift as the backend grows.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# `prepend_sys_path = src` in alembic.ini covers the normal invocation; this
# makes the module importable when alembic is driven programmatically too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from configuration.settings import get_settings  # noqa: E402
from infrastructure.persistence.models import Base  # noqa: E402
from infrastructure.persistence.repository import psycopg_url  # noqa: E402

#: This agent's own migration bookkeeping, kept out of the backend's
#: `alembic_version`. Renaming it would orphan every applied migration.
ALEMBIC_VERSION = "alembic_version_reconstruction"

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
_settings = get_settings()


def _database_url() -> str:
    """The agent's configured URL, as a synchronous driver.

    Alembic runs its migrations synchronously; the async driver the agent uses
    at runtime would fail here with "greenlet_spawn has not been called".
    """
    url = _settings.database.database_url
    if not url:
        raise RuntimeError(
            "No database URL. It is normally read from backend/.env (DATABASE_URL); "
            "set RECONSTRUCTION_DATABASE_URL to override that."
        )
    # Deliberately NOT `%`-escaped. `backend/migrations/env.py` escapes because
    # it goes through `config.set_main_option`, which interpolates; this sets
    # the section dict directly, which does not. Escaping here turned the
    # percent-encoded username `%40` into `%%40` and authentication failed.
    #
    # Normalised onto psycopg 3: Supabase hands out a bare `postgresql://`,
    # which SQLAlchemy would route to psycopg2 - not installed here.
    return psycopg_url(url)


def _include_object(obj: object, name: str | None, type_: str, *_: object) -> bool:
    """Keep autogenerate to the tables this history owns.

    An allow-list drawn from our own metadata, not a list of the backend's
    tables: the agent shares a schema with `users`, `invitations`, LangGraph's
    `checkpoint*` and whatever the backend adds next, and a deny-list would
    silently go stale the day someone adds a table. Anything this history did
    not declare is simply not ours to alter.
    """
    if type_ == "table":
        return name in target_metadata.tables
    # Columns, indexes and constraints belong to a table already filtered above.
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=ALEMBIC_VERSION,
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
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=ALEMBIC_VERSION,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Checkpoint selection and the run audit table.

No database here: these cover the decisions made *around* persistence, which is
where the mistakes that matter live - silently losing durability, or letting an
audit write fail a reconstruction that otherwise succeeded.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest

from configuration.settings import DatabaseSettings
from infrastructure.persistence.checkpoints import _libpq_url, build_checkpointer
from infrastructure.persistence.models import ReconstructionRun
from infrastructure.persistence.repository import RunRepository, psycopg_url


class TestDatabaseSettings:
    def test_no_url_means_not_configured(self) -> None:
        assert DatabaseSettings(_env_file=None).configured is False  # type: ignore[call-arg]

    def test_a_url_means_configured(self) -> None:
        settings = DatabaseSettings(
            _env_file=None,  # type: ignore[call-arg]
            database_url="postgresql://user:pw@localhost/db",
        )

        assert settings.configured is True

    def test_the_url_is_read_from_the_backend_env_file(self) -> None:
        """One URL for all of Umbrella, declared in backend/.env as DATABASE_URL.

        Skipped when that file is absent - a fresh checkout has no .env, and
        this is about resolution order, not about the developer's machine.
        """
        backend_env = Path(__file__).resolve().parents[4] / ".env"
        if not backend_env.is_file() or "DATABASE_URL=" not in backend_env.read_text(
            encoding="utf-8"
        ):
            pytest.skip("backend/.env has no DATABASE_URL on this machine")

        assert DatabaseSettings().configured is True

    def test_an_explicit_override_wins_over_the_shared_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """How a container or CI job points the agent at another database."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://shared:pw@h/umbrella")
        monkeypatch.setenv("RECONSTRUCTION_DATABASE_URL", "postgresql://own:pw@h/agent")

        assert DatabaseSettings().database_url == "postgresql://own:pw@h/agent"

    def test_the_shared_url_is_used_when_there_is_no_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RECONSTRUCTION_DATABASE_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://shared:pw@h/umbrella")

        assert DatabaseSettings().database_url == "postgresql://shared:pw@h/umbrella"

    def test_the_connect_timeout_is_short_enough_to_degrade_in_time(self) -> None:
        """psycopg waits ~130 s by default, outliving the orchestrator's 120 s."""
        assert DatabaseSettings(_env_file=None).connect_timeout_seconds <= 30  # type: ignore[call-arg]


class TestUrlNormalisation:
    """One configured URL, two consumers that disagree about its shape.

    Both bugs below reached a live database before being caught, and both fail
    silently: the agent keeps running, just without durable checkpoints.
    """

    def test_a_bare_postgres_url_gets_the_psycopg3_driver(self) -> None:
        """Supabase hands out a bare `postgresql://`, and SQLAlchemy would route
        that to psycopg2 - not installed - failing with a bare
        `ModuleNotFoundError` that says nothing about the URL."""
        assert psycopg_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"

    def test_an_explicit_driver_is_left_alone(self) -> None:
        assert psycopg_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"

    def test_a_supabase_pooler_url_normalises(self) -> None:
        url = psycopg_url(
            "postgresql://postgres.abc:pw@aws-0-eu-north-1.pooler.supabase.com:5432/postgres"
        )

        assert url.startswith("postgresql+psycopg://")
        assert "pooler.supabase.com:5432/postgres" in url

    def test_libpq_strips_the_sqlalchemy_dialect_suffix(self) -> None:
        """libpq rejects `+psycopg` with a misleading `missing "=" after ...`."""
        url = _libpq_url("postgresql+psycopg://u:p@h:5432/db")

        assert url.startswith("postgresql://u:p@h:5432/db")
        assert "+psycopg" not in url

    def test_libpq_bounds_the_connect_wait(self) -> None:
        """The default outlives the orchestrator's whole request budget."""
        assert "connect_timeout=" in _libpq_url("postgresql://u:p@h/db")

    def test_an_explicit_connect_timeout_is_respected(self) -> None:
        url = _libpq_url("postgresql://u:p@h/db?connect_timeout=30")

        assert url.count("connect_timeout=") == 1
        assert "connect_timeout=30" in url

    def test_no_search_path_is_forced(self) -> None:
        """One database, one schema: the agent shares the backend's."""
        assert "search_path" not in unquote(_libpq_url("postgresql://u:p@h/db"))

    def test_a_percent_encoded_username_survives(self) -> None:
        """The dev database's user is an email address, so `@` is encoded."""
        url = _libpq_url("postgresql+psycopg://a%40b.com:pw@h/db")

        assert "a%40b.com" in url


class TestCheckpointerSelection:
    async def test_no_database_falls_back_to_memory(self) -> None:
        handle = await build_checkpointer(DatabaseSettings(_env_file=None))  # type: ignore[call-arg]

        assert handle.saver is not None
        # The flag /health reports, so a deployment that silently lost
        # durability is visible rather than merely slower.
        assert handle.durable is False

    async def test_an_unreachable_database_degrades_rather_than_refusing_to_start(
        self,
    ) -> None:
        """An agent that will not boot because a checkpoint store is down is
        worse than one that boots and cannot resume."""
        settings = DatabaseSettings(
            _env_file=None,  # type: ignore[call-arg]
            database_url="postgresql://nobody:nope@127.0.0.1:1/nonexistent",
        )

        handle = await build_checkpointer(settings)

        assert handle.durable is False
        assert handle.saver is not None

    async def test_closing_a_memory_handle_is_safe(self) -> None:
        handle = await build_checkpointer(DatabaseSettings(_env_file=None))  # type: ignore[call-arg]

        await handle.aclose()  # must not raise


class TestRunRepository:
    def test_disabled_without_a_database(self) -> None:
        assert RunRepository(DatabaseSettings(_env_file=None)).enabled is False  # type: ignore[call-arg]

    async def test_recording_without_a_database_is_a_no_op(self) -> None:
        repository = RunRepository(DatabaseSettings(_env_file=None))  # type: ignore[call-arg]

        await repository.record(
            trace_id="t",
            run_id="r",
            sequence_id="s",
            organism=None,
            status="completed",
            stop_reason=None,
            slice_count=1,
            iterations=1,
            gaps_total=1,
            gaps_resolved=0,
            tool_calls_used=0,
            llm_tokens_used=0,
            summary=None,
        )  # must not raise

    async def test_a_write_failure_never_fails_the_run(self) -> None:
        """Losing an audit row must not cost a reconstruction that succeeded."""
        settings = DatabaseSettings(
            _env_file=None,  # type: ignore[call-arg]
            database_url="postgresql://nobody:nope@127.0.0.1:1/nonexistent",
        )
        repository = RunRepository(settings)

        await repository.record(
            trace_id="t",
            run_id="r",
            sequence_id="s",
            organism=None,
            status="completed",
            stop_reason=None,
            slice_count=1,
            iterations=1,
            gaps_total=1,
            gaps_resolved=0,
            tool_calls_used=0,
            llm_tokens_used=0,
            summary=None,
        )  # swallowed, not raised

        await repository.aclose()


class TestRunModel:
    def test_keyed_by_trace_id_not_run_id(self) -> None:
        """`run_id` changes per HTTP call; only the trace id gathers the slices
        of one logical run."""
        primary = [column.name for column in ReconstructionRun.__table__.primary_key]

        assert primary == ["trace_id"]

    def test_shares_the_default_schema(self) -> None:
        """One database, no dedicated namespace; the table name distinguishes it."""
        assert ReconstructionRun.__table__.schema is None
        assert ReconstructionRun.__tablename__ == "reconstruction_runs"

    @pytest.mark.parametrize(
        "column",
        ["slice_count", "gaps_resolved", "tool_calls_used", "llm_tokens_used", "stop_reason"],
    )
    def test_records_what_the_run_cost(self, column: str) -> None:
        assert column in ReconstructionRun.__table__.columns

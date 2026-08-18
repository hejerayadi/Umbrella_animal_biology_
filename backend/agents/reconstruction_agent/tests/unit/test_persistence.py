"""Checkpoint selection and the run audit table.

No database here: these cover the decisions made *around* persistence, which is
where the mistakes that matter live - silently losing durability, or letting an
audit write fail a reconstruction that otherwise succeeded.
"""
from __future__ import annotations

from urllib.parse import unquote

import pytest

from configuration.settings import DatabaseSettings
from infrastructure.persistence.checkpoints import _libpq_url, build_checkpointer
from infrastructure.persistence.models import ReconstructionRun
from infrastructure.persistence.repository import RunRepository, _async_url


class TestDatabaseSettings:
    def test_no_url_means_not_configured(self) -> None:
        assert DatabaseSettings(_env_file=None).configured is False  # type: ignore[call-arg]

    def test_a_url_means_configured(self) -> None:
        settings = DatabaseSettings(
            _env_file=None,  # type: ignore[call-arg]
            database_url="postgresql://user:pw@localhost/db",
        )

        assert settings.configured is True

    def test_the_schema_defaults_to_the_agent_namespace(self) -> None:
        """Kept out of the backend's namespace so the two migrate separately."""
        assert DatabaseSettings(_env_file=None).schema_name == "reconstruction"  # type: ignore[call-arg]


class TestUrlNormalisation:
    """One configured URL, two consumers that disagree about its shape.

    Both bugs below reached a live database before being caught, and both fail
    silently: the agent keeps running, just without durable checkpoints.
    """

    def test_a_bare_postgres_url_gets_the_async_driver(self) -> None:
        """The sync driver would block the event loop the agent runs on."""
        assert _async_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"

    def test_an_explicit_driver_is_left_alone(self) -> None:
        assert _async_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"

    def test_libpq_strips_the_sqlalchemy_dialect_suffix(self) -> None:
        """libpq rejects `+psycopg` with a misleading `missing "=" after ...`."""
        url = _libpq_url("postgresql+psycopg://u:p@h:5432/db", "reconstruction")

        assert url.startswith("postgresql://u:p@h:5432/db")
        assert "+psycopg" not in url

    def test_libpq_pins_the_search_path_to_the_agent_schema(self) -> None:
        """Without this the saver creates its tables in `public`, mixed in with
        the backend's."""
        url = _libpq_url("postgresql://u:p@h/db", "reconstruction")

        assert "options=" in url
        assert "search_path" in unquote(url)
        assert "reconstruction" in unquote(url)

    def test_an_explicit_options_parameter_is_respected(self) -> None:
        """An operator who set options deliberately should keep them."""
        url = _libpq_url("postgresql://u:p@h/db?options=-cstatement_timeout%3D5000", "recon")

        assert url.count("options=") == 1
        assert "statement_timeout" in unquote(url)

    def test_a_percent_encoded_username_survives(self) -> None:
        """The dev database's user is an email address, so `@` is encoded."""
        url = _libpq_url("postgresql+psycopg://a%40b.com:pw@h/db", "recon")

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

    def test_lives_in_the_agent_schema(self) -> None:
        assert ReconstructionRun.__table__.schema == "reconstruction"

    @pytest.mark.parametrize(
        "column",
        ["slice_count", "gaps_resolved", "tool_calls_used", "llm_tokens_used", "stop_reason"],
    )
    def test_records_what_the_run_cost(self, column: str) -> None:
        assert column in ReconstructionRun.__table__.columns

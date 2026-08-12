"""Tracing configuration.

The assertions that matter here ask LangSmith itself whether tracing is on,
rather than only checking which strings were written to ``os.environ``. Those
two can disagree - LangSmith reads several names in a fixed precedence and
memoises the answer - and it is the disagreement that would silently cost a
deployment its traces.
"""

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from langsmith import utils as ls_utils

from backend.agents.Protein_visualization.app.configuration.settings import Settings
from backend.agents.Protein_visualization.app.observability import tracing
from backend.agents.Protein_visualization.app.observability.tracing import (
    configure_tracing,
    get_tracing_status,
    trace_config,
)

_MANAGED_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Give each test a pristine environment, and hand the process back unchanged.

    ``configure_tracing`` writes to the real process environment on purpose -
    that is the only channel LangChain reads - so the variables are cleared
    going in as well as restored coming out. Without the clear, a
    ``create_app()`` in any earlier test file leaves ``LANGSMITH_TRACING=false``
    behind, and these tests would be asserting against that instead of against
    the settings they pass.
    """
    saved = {key: value for key, value in os.environ.items() if key.startswith(_MANAGED_PREFIXES)}
    saved_status = tracing.get_tracing_status()
    _clear_managed_environment()
    try:
        yield
    finally:
        _clear_managed_environment()
        os.environ.update(saved)
        tracing._status = saved_status
        tracing._reset_langsmith_env_cache()


def _clear_managed_environment() -> None:
    for key in [key for key in os.environ if key.startswith(_MANAGED_PREFIXES)]:
        del os.environ[key]
    tracing._reset_langsmith_env_cache()


def _settings(**overrides: object) -> Settings:
    """Settings built from the arguments alone, ignoring any .env on disk."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_tracing_is_off_when_not_requested() -> None:
    status = configure_tracing(_settings())

    assert status.enabled is False
    assert status.reason == "LANGSMITH_TRACING is not enabled"
    assert ls_utils.tracing_is_enabled() is False


def test_tracing_requested_without_api_key_stays_off() -> None:
    status = configure_tracing(_settings(langsmith_tracing=True))

    assert status.enabled is False
    assert "LANGSMITH_API_KEY" in (status.reason or "")
    assert ls_utils.tracing_is_enabled() is False


def test_tracing_enabled_publishes_the_langsmith_environment() -> None:
    status = configure_tracing(
        _settings(
            langsmith_tracing=True,
            langsmith_api_key="lsv2_test_key",
            langsmith_project="protein-agent-tests",
            # A trailing slash here produces //runs/batch against the API.
            langsmith_endpoint="https://api.smith.langchain.com/",
        )
    )

    assert status.enabled is True
    assert status.project == "protein-agent-tests"
    assert status.endpoint == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test_key"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert ls_utils.tracing_is_enabled() is True
    assert ls_utils.get_tracer_project() == "protein-agent-tests"


def test_disabling_overrides_tracing_inherited_from_the_shell() -> None:
    """A LANGCHAIN_TRACING_V2 in the environment must not outrank the settings."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    tracing._reset_langsmith_env_cache()
    assert ls_utils.tracing_is_enabled() is True

    status = configure_tracing(_settings(langsmith_tracing=False))

    assert status.enabled is False
    assert ls_utils.tracing_is_enabled() is False


def test_sampling_rate_is_published_when_set() -> None:
    configure_tracing(
        _settings(
            langsmith_tracing=True,
            langsmith_api_key="lsv2_test_key",
            langsmith_tracing_sampling_rate=0.25,
        )
    )

    assert os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] == "0.25"


def test_unset_sampling_rate_clears_a_stale_one() -> None:
    """An earlier value in the environment must not survive a config that drops it."""
    os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] = "0.25"

    configure_tracing(
        _settings(
            langsmith_tracing=True,
            langsmith_api_key="lsv2_test_key",
            langsmith_tracing_sampling_rate=None,
        )
    )

    assert "LANGSMITH_TRACING_SAMPLING_RATE" not in os.environ


def test_status_survives_between_calls() -> None:
    configure_tracing(_settings(langsmith_tracing=True, langsmith_api_key="lsv2_test_key"))

    assert get_tracing_status().enabled is True


def test_trace_config_carries_identity_and_drops_empty_metadata() -> None:
    analysis_id = uuid4()

    config = trace_config(
        run_name="protein_workflow",
        run_id=analysis_id,
        thread_id=str(analysis_id),
        tags=["protein-visualization-agent", ""],
        gene="TP53",
        taxon_id=9606,
        mutation=None,
    )

    # The root run id is the analysis id, so a log line locates its own trace.
    assert config["run_id"] == analysis_id
    assert config["configurable"] == {"thread_id": str(analysis_id)}
    assert config["tags"] == ["protein-visualization-agent"]
    assert config["metadata"] == {"gene": "TP53", "taxon_id": 9606}

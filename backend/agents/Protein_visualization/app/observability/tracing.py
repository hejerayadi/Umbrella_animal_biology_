"""LangSmith tracing for the protein workflow.

LangChain and LangGraph do not read this service's settings: they read the
process environment. Everything else here comes from ``.env`` through
pydantic-settings, which loads that file *without* exporting it. Closing that
gap is the whole job of :func:`configure_tracing` - validate the LangSmith
settings once at startup, then publish them under the names the LangChain
tracer looks for.

Tracing is enabled only when it is both asked for and usable. ``LANGSMITH_TRACING=true``
without an API key leaves every LLM call and every graph node paying for a
tracer whose exports can only be rejected, so that combination is reported as a
warning and tracing stays off rather than half-on.

Nothing here is required for the workflow to run. When LangSmith is unreachable
or misconfigured the agent keeps answering; the structured logs in
``app.observability.logging`` remain the authoritative record either way.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from backend.agents.Protein_visualization.app.configuration.settings import Settings
from backend.agents.Protein_visualization.app.observability.logging import log_event

if TYPE_CHECKING:  # imported for the annotations only; langchain stays lazy here
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger("app.tracing")

# LangSmith resolves each setting by trying the LANGSMITH_ prefix before the
# legacy LANGCHAIN_ one, and ``*_TRACING_V2`` before ``*_TRACING``. Writing the
# highest-precedence name of each pair is what makes this function decisive:
# a LANGCHAIN_TRACING_V2=true inherited from the shell can no longer quietly
# re-enable tracing that these settings turned off.
ENV_TRACING_V2 = "LANGSMITH_TRACING_V2"
ENV_TRACING = "LANGSMITH_TRACING"
ENV_API_KEY = "LANGSMITH_API_KEY"
ENV_ENDPOINT = "LANGSMITH_ENDPOINT"
ENV_PROJECT = "LANGSMITH_PROJECT"
ENV_SAMPLING_RATE = "LANGSMITH_TRACING_SAMPLING_RATE"


@dataclass(frozen=True)
class TracingStatus:
    """What startup decided about tracing, for logs and the readiness probe."""

    enabled: bool
    project: str | None = None
    endpoint: str | None = None
    sampling_rate: float | None = None
    # Why tracing is off. Always set when ``enabled`` is False, so an operator
    # who expected traces can tell "never asked for" from "asked for, unusable".
    reason: str | None = None


_status = TracingStatus(enabled=False, reason="configure_tracing has not run")


def get_tracing_status() -> TracingStatus:
    """The decision the last :func:`configure_tracing` call reached."""
    return _status


def configure_tracing(settings: Settings) -> TracingStatus:
    """Publish the LangSmith environment and report whether tracing is on."""
    global _status

    if not settings.langsmith_tracing:
        _status = _disable("LANGSMITH_TRACING is not enabled")
    elif not settings.langsmith_api_key:
        _status = _disable(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing",
            level=logging.WARNING,
        )
    else:
        _status = _enable(settings)
    return _status


def _enable(settings: Settings) -> TracingStatus:
    project = settings.langsmith_project.strip() or Settings.model_fields["langsmith_project"].default
    endpoint = settings.langsmith_endpoint.strip().rstrip("/")

    assert settings.langsmith_api_key  # guaranteed by configure_tracing
    os.environ[ENV_TRACING_V2] = "true"
    os.environ[ENV_TRACING] = "true"
    os.environ[ENV_API_KEY] = settings.langsmith_api_key
    os.environ[ENV_ENDPOINT] = endpoint
    os.environ[ENV_PROJECT] = project
    if settings.langsmith_tracing_sampling_rate is None:
        os.environ.pop(ENV_SAMPLING_RATE, None)
    else:
        os.environ[ENV_SAMPLING_RATE] = str(settings.langsmith_tracing_sampling_rate)
    _reset_langsmith_env_cache()

    log_event(
        logger,
        "protein.tracing.enabled",
        capability="tracing",
        provider="langsmith",
        project=project,
        endpoint=endpoint,
        sampling_rate=settings.langsmith_tracing_sampling_rate,
    )
    return TracingStatus(
        enabled=True,
        project=project,
        endpoint=endpoint,
        sampling_rate=settings.langsmith_tracing_sampling_rate,
    )


def _disable(reason: str, level: int = logging.INFO) -> TracingStatus:
    os.environ[ENV_TRACING_V2] = "false"
    os.environ[ENV_TRACING] = "false"
    _reset_langsmith_env_cache()
    log_event(
        logger,
        "protein.tracing.disabled",
        level,
        capability="tracing",
        provider="langsmith",
        detail=reason,
    )
    return TracingStatus(enabled=False, reason=reason)


def _reset_langsmith_env_cache() -> None:
    """Drop LangSmith's memoised reads of the environment.

    ``langsmith.utils.get_env_var`` and ``get_tracer_project`` are both
    ``lru_cache``d, so any consultation that happened before this function ran -
    an import-time check, an earlier call with different settings, the previous
    test in the file - would otherwise pin the stale answer for the life of the
    process and silently ignore everything written above.
    """
    try:
        from langsmith import utils as ls_utils
    except ImportError:  # pragma: no cover - langsmith ships with langchain-core
        return
    for name in ("get_env_var", "get_tracer_project"):
        cache_clear = getattr(getattr(ls_utils, name, None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def trace_config(
    *,
    run_name: str,
    run_id: UUID,
    thread_id: str,
    tags: Sequence[str] = (),
    **metadata: Any,
) -> "RunnableConfig":
    """Build the LangGraph invocation config that makes a run findable later.

    ``run_id`` is deliberately this service's own ``analysis_id``. LangSmith
    accepts a caller-supplied UUID for the root run, so the id already printed
    on every log line of an analysis *is* the id of its trace - no searching by
    timestamp to line the two up. LangChain strips ``run_id`` from the config it
    hands to child runs, so only the root takes it.

    ``metadata`` is what the LangSmith filter bar queries, so the biological
    identity of the run belongs there: gene, accession, taxon. ``None`` values
    are dropped rather than sent as nulls, which would otherwise litter the
    filters with keys that match nothing.

    Building this costs nothing when tracing is off - ``thread_id`` is required
    by the graph's checkpointer regardless, and LangChain ignores the rest.
    """
    return cast(
        "RunnableConfig",
        {
            "run_id": run_id,
            "run_name": run_name,
            "configurable": {"thread_id": thread_id},
            "tags": [tag for tag in tags if tag],
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        },
    )


def child_run_config(run_name: str, tags: Sequence[str] = (), **metadata: Any) -> "RunnableConfig":
    """Name and annotate a nested LangChain call without detaching it from its trace.

    A config passed explicitly to ``ainvoke`` replaces the inherited ``tags`` and
    ``metadata`` wholesale rather than merging into them, and an inherited
    ``callbacks`` is what keeps the call underneath the graph node that made it.
    So the ambient config is read first and extended, never rebuilt: the run
    gains a readable name and its own fields while staying in the same tree.
    """
    from langchain_core.runnables import ensure_config

    config = ensure_config()
    config["run_name"] = run_name
    config["tags"] = [*config.get("tags", []), *(tag for tag in tags if tag)]
    config["metadata"] = {
        **config.get("metadata", {}),
        **{key: value for key, value in metadata.items() if value is not None},
    }
    return config

"""Process-wide singletons, injected into routes.

Built once, not per request: the tool registry owns HTTP clients with
connection pools and rate limiters, and the checkpointer owns a database pool.
Rebuilding either per request would defeat both and quietly exceed NCBI's
request budget.

The checkpointer and the audit repository need an event loop to open, so they
are created in the app's lifespan (`api/app.py`) and handed here, rather than
being built lazily on first use like the rest.
"""
from __future__ import annotations

from functools import lru_cache

from application.reconstruction_service import ReconstructionService
from configuration.logging import get_logger
from configuration.settings import Settings, get_settings
from infrastructure.persistence.checkpoints import CheckpointerHandle, build_checkpointer
from infrastructure.persistence.repository import RunRepository
from observability.events import EventEmitter
from tools.registry import ToolRegistry, build_default_registry

_log = get_logger(__name__)

#: Set by `startup()`; cleared by `shutdown()`.
_checkpointer: CheckpointerHandle | None = None
_repository: RunRepository | None = None
_service: ReconstructionService | None = None


@lru_cache(maxsize=1)
def get_event_emitter() -> EventEmitter:
    return EventEmitter()


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return build_default_registry(get_settings())


async def startup(settings: Settings | None = None) -> None:
    """Open the resources that need an event loop."""
    global _checkpointer, _repository, _service

    settings = settings or get_settings()
    _checkpointer = await build_checkpointer(settings.database)
    _repository = RunRepository(settings.database)
    _service = ReconstructionService(
        settings,
        get_tool_registry(),
        events=get_event_emitter(),
        checkpointer=_checkpointer.saver,
        runs=_repository,
    )

    _log.info(
        "dependencies_ready",
        durable_checkpoints=_checkpointer.durable,
        audit_enabled=_repository.enabled,
    )


async def shutdown() -> None:
    """Close them again, in reverse order."""
    global _checkpointer, _repository, _service

    if _repository is not None:
        await _repository.aclose()
    if _checkpointer is not None:
        await _checkpointer.aclose()
    _checkpointer = _repository = _service = None


def get_service() -> ReconstructionService:
    """The agent's use case, wired to real services.

    Falls back to an un-checkpointed service when `startup()` has not run -
    which is the case for a `TestClient` built without the lifespan. The agent
    still works; it simply cannot resume across calls.
    """
    if _service is not None:
        return _service
    return ReconstructionService(
        get_settings(), get_tool_registry(), events=get_event_emitter()
    )


def get_checkpointer() -> CheckpointerHandle | None:
    return _checkpointer


def get_app_settings() -> Settings:
    """Settings, as a FastAPI dependency."""
    return get_settings()


#: Alias used by routes that need settings alongside the service, so the
#: dependency reads as what it is at the call site.
get_settings_dependency = get_app_settings


def reset() -> None:
    """Drop the cached singletons.

    For tests that need a differently-configured agent in the same process.
    """
    global _service
    get_event_emitter.cache_clear()
    get_tool_registry.cache_clear()
    _service = None

"""Process-wide singletons, injected into routes.

Built once at first use, not per request: the tool registry owns HTTP clients
with connection pools and rate limiters, and rebuilding it per request would
defeat both and quietly exceed NCBI's request budget.
"""
from __future__ import annotations

from functools import lru_cache

from ..application.reconstruction_service import ReconstructionService
from ..configuration.settings import Settings, get_settings
from ..observability.events import EventEmitter
from ..tools.registry import ToolRegistry, build_default_registry


@lru_cache(maxsize=1)
def get_event_emitter() -> EventEmitter:
    return EventEmitter()


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return build_default_registry(get_settings())


@lru_cache(maxsize=1)
def get_service() -> ReconstructionService:
    """The agent's use case, wired to real services."""
    return ReconstructionService(
        get_settings(),
        get_tool_registry(),
        events=get_event_emitter(),
    )


def get_app_settings() -> Settings:
    """Settings, as a FastAPI dependency."""
    return get_settings()


def reset() -> None:
    """Drop the cached singletons.

    For tests that need a differently-configured agent in the same process.
    """
    get_event_emitter.cache_clear()
    get_tool_registry.cache_clear()
    get_service.cache_clear()

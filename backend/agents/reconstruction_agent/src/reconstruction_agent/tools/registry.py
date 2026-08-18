"""The set of tools available to the planner, keyed by name.

The graph never imports a tool directly - it asks the registry. That is what
lets the planner choose from a described catalogue, and lets tests swap the
whole toolset for fakes without touching the graph.
"""
from __future__ import annotations

from typing import Any

from ..configuration.logging import get_logger
from ..configuration.settings import Settings
from ..domain.exceptions import ToolExecutionError
from .contracts import Tool

_log = get_logger(__name__)


class ToolRegistry:
    """A name -> tool mapping with a planner-facing catalogue."""

    def __init__(self, tools: list[Tool[Any, Any]] | None = None) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolExecutionError(name, "no tool with that name is registered") from error

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalogue(self) -> list[dict[str, Any]]:
        """Tool descriptions for the planner prompt."""
        return [tool.describe() for tool in self._tools.values()]

    async def run(self, name: str, payload: Any) -> Any:
        """Execute a registered tool by name.

        Unexpected exceptions are wrapped as `ToolExecutionError` so a bug in
        one tool surfaces as a named tool failure rather than an anonymous
        traceback from inside the graph.
        """
        tool = self.get(name)
        try:
            return await tool.run(payload)
        except Exception as error:  # noqa: BLE001 - re-raised as a named failure
            raise ToolExecutionError(name, str(error)) from error


def build_default_registry(settings: Settings) -> ToolRegistry:
    """The production toolset, wired to real external services.

    Constructed lazily per agent process. Clients are long-lived on purpose:
    they own the connection pools and rate limiters, and rebuilding them per
    request would defeat both.
    """
    from ..infrastructure.embl_ebi.blast_client import BlastClient
    from ..infrastructure.embl_ebi.mafft_client import MafftClient
    from ..infrastructure.ncbi.client import NCBIClient
    from .blast.tool import BlastSearchTool
    from .evo.tool import EvolutionaryContextTool
    from .mafft.tool import MafftAlignmentTool
    from .ncbi.tool import NCBISearchTool

    timeout = settings.http.timeout_seconds

    registry = ToolRegistry(
        [
            NCBISearchTool(NCBIClient(settings.ncbi, timeout=timeout)),
            BlastSearchTool(BlastClient(settings.embl_ebi, timeout=timeout)),
            MafftAlignmentTool(MafftClient(settings.embl_ebi, timeout=timeout)),
            EvolutionaryContextTool(),
        ]
    )
    _log.info("Registered %d tools: %s", len(registry.names), ", ".join(registry.names))
    return registry

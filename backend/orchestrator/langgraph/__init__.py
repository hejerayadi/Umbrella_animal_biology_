"""LangGraph execution engine for the Global Scientific Orchestrator.

Split into three layers so each file has one reason to change:

- `nodes/`          - what each step of the graph does
- `graph.py`        - how those steps are wired together
- `orchestrator.py` - the public object the rest of the app calls

Import from this package (`backend.orchestrator.langgraph`) rather than from
the modules inside it, so the internal layout stays free to move.
"""
from __future__ import annotations

from .graph import build_orchestrator_graph
from .orchestrator import GlobalOrchestrator

__all__ = ["GlobalOrchestrator", "build_orchestrator_graph"]

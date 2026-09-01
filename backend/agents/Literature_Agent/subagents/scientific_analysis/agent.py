"""
Callable entry point for the Scientific Analysis subagent.

Mirrors the adapter pattern used by subagents/retrieval_knowledge/agent.py:
the engine (qdrant_kb/, graph.py, tools.py) is never imported at module scope,
so a missing Qdrant/Azure config degrades this into a call-time failure the
Knowledge Discovery sub-orchestrator can handle, rather than an ImportError
that would take the whole Literature Agent service down at start.
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Whether a real analysis can be attempted. Never raises."""
    try:
        from .config import is_qdrant_configured, is_azure_configured
        return is_qdrant_configured() and is_azure_configured()
    except Exception:  # noqa: BLE001
        return False


def run_scientific_analysis(query: str, context_papers: list | None = None, context_records: list | None = None) -> dict:
    """
    Run the Scientific Analysis agent (QA / Contradiction / Gap Detection)
    for the given query.

    Args:
        query: The scientific question or claim to analyze.
        context_papers: Optional list of paper citation strings from the search agent.
        context_records: Optional list of record dicts with title/summary from the search agent.

    Raises nothing: a failure comes back as {"status": "error", ...} so the
    caller (subagents/discovery/knowledge_discovery.py) can degrade gracefully
    instead of the whole discovery pipeline failing.
    """
    if not is_configured():
        _logger.info("scientific analysis skipped: Qdrant or Azure not configured")
        return {
            "status": "not_configured",
            "answer": "",
            "tool_calls_made": [],
        }

    try:
        from .graph import run_scientific_analysis_agent
    except Exception as exc:  # noqa: BLE001
        _logger.warning("scientific analysis engine unavailable (%s): %s", type(exc).__name__, exc)
        return {"status": "error", "answer": "", "tool_calls_made": [], "error": str(exc)}

    return run_scientific_analysis_agent(query, context_papers=context_papers, context_records=context_records)

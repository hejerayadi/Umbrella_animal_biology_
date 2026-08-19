"""HTTP boundary for the Reconstruction Agent - launcher shim.

The agent itself lives in `src/`, whose top-level packages (`api`, `domain`,
`agent`, ...) are installed into this agent's own `.venv`. This file exists
because `backend/run_agents.py` launches every agent as
`backend.agents.<folder>.api:app`, and that convention is shared with eight
other agents - keeping the shim is cheaper than special-casing the launcher.

Everything real is in `api.app`; nothing but the re-export belongs here.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.reconstruction_agent.api:app --port 8006

or, for the whole system:

    python -m backend.run_agents --setup   # once, to build the venvs
    python -m backend.run_agents
"""
from __future__ import annotations

# Absolute import of the installed top-level `api` package, not this module.
# Python 3 resolves it against sys.path, and this file's own dotted name is
# `backend.agents.reconstruction_agent.api`, so the two never collide.
from api.app import app, create_app

__all__ = ["app", "create_app"]

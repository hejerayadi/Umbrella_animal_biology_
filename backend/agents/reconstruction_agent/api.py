"""HTTP boundary for the Reconstruction Agent - launcher shim.

The agent itself lives in `src/reconstruction_agent/`, installed into this
agent's own `.venv` as the top-level `reconstruction_agent` package. This file
exists because `backend/run_agents.py` launches every agent as
`backend.agents.<folder>.api:app`, a convention shared with eight other agents -
keeping the shim is cheaper than special-casing the launcher.

Everything real is in `reconstruction_agent.main`; nothing but the re-export
belongs here.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.reconstruction_agent.api:app --port 8006

or, for the whole system:

    python -m backend.run_agents --setup   # once, to build the venvs
    python -m backend.run_agents
"""

from __future__ import annotations

# Absolute import of the installed top-level `reconstruction_agent` package,
# not this directory. Python 3 resolves it against sys.path, and this module's
# own dotted name is `backend.agents.reconstruction_agent.api`, so the two
# never collide.
from reconstruction_agent.main import app, create_app

__all__ = ["app", "create_app"]

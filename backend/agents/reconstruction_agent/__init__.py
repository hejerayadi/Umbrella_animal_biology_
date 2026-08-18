"""Reconstruction Agent - repository-level package marker.

The agent's implementation lives in `src/reconstruction_agent/` and is
installed into this agent's own `.venv` as the top-level `reconstruction_agent`
package. This module stays deliberately empty of imports: `backend/registry.py`
reads only `card.json` and imports no code from `backend/agents/`, and eagerly
importing the agent here would make the whole backend depend on this agent's
dependencies being installed.

See `api.py` for the launcher entry point.
"""

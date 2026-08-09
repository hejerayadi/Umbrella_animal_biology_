"""Protein Visualization Agent.

The agent is the LangGraph workflow under `app/`. `api.py` publishes it: the
inter-agent `POST /execute` for the Grand Orchestrator, and the versioned
scientific API for the frontend viewer.

Nothing is exported here on purpose. `backend/registry.py` reads this agent's
`card.json` and reaches it over HTTP, and importing the workflow eagerly would
pull httpx, LangGraph and the Qdrant client into any process that merely walks
the `backend.agents` package.
"""

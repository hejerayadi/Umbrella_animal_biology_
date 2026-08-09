"""HTTP boundary for the Protein Visualization Agent.

This is the service `backend/run_agents.py` starts on port 8008, and it serves
the real agent - there is no mock implementation behind it:

* `POST /execute` is the inter-agent contract the Grand Orchestrator calls. It
  runs the actual LangGraph workflow over UniProt, RCSB PDB, AlphaFold, InterPro
  and SIFTS, with Azure for the explanation and critic passes and Qdrant for
  retrieval. The adapter lives in `app/api/execute.py`.
* `/api/v1/...` is the scientific API the frontend viewer calls, mounted from
  `app/main.py` - the full analysis with the Mol* scene, every annotation and
  every evidence record.
* `/docs` documents both.

One process, one set of clients, one workflow. Splitting them would mean two
different answers to the same question depending on who asked.

Run it (from the repository root):

    uv run --project backend/agents/Protein_visualization \
        python -m uvicorn backend.agents.Protein_visualization.api:app --port 8008
"""

from __future__ import annotations

from fastapi import FastAPI

from .app.api.execute import router as execute_router
from .app.main import create_app

app: FastAPI = create_app()
app.include_router(execute_router)

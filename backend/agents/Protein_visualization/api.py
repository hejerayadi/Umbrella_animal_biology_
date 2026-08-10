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
* `/console` is the test console: a single Bootstrap page for driving either
  endpoint by hand and inspecting the run. Served from here rather than opened
  as a file so it is same-origin with the API it calls.
* `/docs` documents the endpoints.

One process, one set of clients, one workflow. Splitting them would mean two
different answers to the same question depending on who asked.

Run it (from the repository root):

    uv run --project backend/agents/Protein_visualization \
        python -m uvicorn backend.agents.Protein_visualization.api:app --port 8008
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .app.api.execute import router as execute_router
from .app.configuration.settings import get_settings
from .app.main import create_app

CONSOLE_DIR = Path(__file__).parent / "console"

app: FastAPI = create_app()
app.include_router(execute_router)

# The console drives real analyses against real providers, and shows the raw
# request and response of every call. That is a developer tool, so it is not
# published from a production deployment.
if get_settings().app_env.lower() != "production" and CONSOLE_DIR.is_dir():
    app.mount("/console", StaticFiles(directory=CONSOLE_DIR, html=True), name="console")

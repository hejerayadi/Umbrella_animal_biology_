"""Run the Protein Agent backend using settings loaded from .env.

Serves `api:app`, the same object `backend/run_agents.py` starts - so a locally
served agent and an orchestrator-started one expose the same routes: the
inter-agent `POST /execute` and the versioned scientific API alike. Serving
`app.main:app` here instead would give two entry points that answer
differently.

The import path is absolute rather than module-relative: `app.main` only
resolves when the process starts inside this agent's folder, while every module
under `app/` imports itself as `backend.agents.Protein_visualization.app.*`.
Both facts have to agree, so the server is addressed from the repository root.
"""

import uvicorn

from backend.agents.Protein_visualization.app.configuration.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "backend.agents.Protein_visualization.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()

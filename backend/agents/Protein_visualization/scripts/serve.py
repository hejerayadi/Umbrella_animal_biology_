"""Run the Protein Agent backend using settings loaded from .env.

Serves `api:app`, the same object `backend/run_agents.py` starts - so a locally
served agent and an orchestrator-started one expose the same routes, including
the inter-agent `POST /execute`. Serving `app.main:app` here instead would give
two entry points that answer differently.
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

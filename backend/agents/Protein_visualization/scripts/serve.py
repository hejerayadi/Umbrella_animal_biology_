"""Run the Protein Agent backend using settings loaded from .env."""

import uvicorn

from backend.agents.Protein_visualization.app.configuration.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        # Import path, not a module-relative name: `app.main` only resolves
        # when the process starts inside this agent's folder, and every module
        # under `app/` now imports itself as
        # `backend.agents.Protein_visualization.app.*`. Both facts have to
        # agree, so the server is addressed from the repository root too.
        "backend.agents.Protein_visualization.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()

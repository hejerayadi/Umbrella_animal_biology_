"""Run the Protein Agent backend using settings loaded from .env."""

import uvicorn

from backend.agents.Protein_visualization.app.configuration.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )


if __name__ == "__main__":
    main()

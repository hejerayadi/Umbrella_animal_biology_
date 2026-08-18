"""Run the agent standalone: `python -m api`.

Binds to APP_HOST/APP_PORT from the environment, so the same command works on
a laptop and in a container without arguments. The orchestrator path goes
through `backend/run_agents.py` instead - see the repository README.
"""
from __future__ import annotations

import uvicorn

from configuration.logging import configure_logging, get_logger
from configuration.runtime import use_selector_event_loop
from configuration.settings import get_settings

_log = get_logger(__name__)


def main() -> None:
    use_selector_event_loop()
    settings = get_settings()
    configure_logging(settings.observability, log_format=settings.log_format)

    _log.info(
        "starting_uvicorn",
        host=settings.app.host,
        port=settings.app.port,
        environment=settings.app.env.value,
    )

    uvicorn.run(
        "api.app:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload,
        # Our structlog handler owns formatting; uvicorn's own config would
        # install a second set of handlers and print every line twice.
        log_config=None,
    )


if __name__ == "__main__":
    main()

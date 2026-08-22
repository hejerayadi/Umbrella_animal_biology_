"""HTTP boundary for the Biodiversity Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

Which implementation answers is chosen by BIODIVERSITY_AGENT_IMPL:

    BIODIVERSITY_AGENT_IMPL=mock          (default) the stub in `mock.py`
    BIODIVERSITY_AGENT_IMPL=orchestrator  the real LangGraph orchestrator,
                                          reached through `orchestrator_adapter.py`

The default stays `mock` so this endpoint cannot start failing because an LLM
key is missing on someone's machine. The orchestrator path needs this agent's
own venv and a configured LLM backend for intent classification.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.biodiversity_agent.api:app --port 8003
"""
from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .mock import BiodiversityMock
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Biodiversity Agent")

# ---------------------------------------------------------------- maps
#
# Every worker renders a self-contained folium HTML file and reports it as a
# ``file://`` URI. That is the right answer for the Streamlit dashboards, which
# read the file off disk - but a browser will not load a ``file://`` URL from a
# page served over http, so a web frontend got a link it could never render.
#
# So the files are served here instead, and ``map_url`` is rewritten on the way
# out to an address the frontend can actually put in an iframe. The workers are
# left alone: they keep writing paths, and this boundary publishes them.

_MAPS_DIR = (Path(__file__).resolve().parent / "outputs" / "maps")

# A container does not know its own public address, so it has to be told.
_PUBLIC_URL = os.getenv("BIODIVERSITY_PUBLIC_URL", "http://localhost:8003").rstrip("/")


def _local_map_name(value: object) -> str | None:
    """The filename if ``value`` points at one of our rendered maps, else None.

    Accepts both a ``file://`` URI and a bare path, because the workers are not
    perfectly consistent and this is the layer that makes them look that way.
    The file must actually exist in the maps directory - that is what keeps
    this from rewriting arbitrary strings that happen to end in ``.html``.
    """
    if not isinstance(value, str) or not value.lower().endswith(".html"):
        return None
    if value.startswith(("http://", "https://")):
        return None  # already published

    raw = value
    if value.startswith("file:"):
        parsed = urlparse(value)
        raw = unquote(parsed.path or "")
        # ``file:///C:/x`` parses to ``/C:/x`` on Windows.
        if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]

    name = Path(raw).name
    if not name:
        return None
    return name if (_MAPS_DIR / name).is_file() else None


def _publish_map_urls(value):
    """Rewrite every rendered-map reference in a response to an HTTP URL.

    Walks the whole payload rather than touching the known keys: ``map_url``
    appears at the top level, inside ``output``, and again inside
    ``biodiversity_report.findings``, and the responder LLM reads whichever it
    finds - which is how a raw ``file://`` path ended up quoted in chat.
    """
    if isinstance(value, str):
        name = _local_map_name(value)
        return f"{_PUBLIC_URL}/maps/{name}" if name else value
    if isinstance(value, dict):
        return {key: _publish_map_urls(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_publish_map_urls(item) for item in value]
    # Worker payloads are dataclasses (``SpeciesDistributionOutput``,
    # ``BiodiversityHotspotOutput``), not dicts, and FastAPI serializes them
    # after this runs - so descend into their attributes too, or a nested
    # ``map_url`` survives as a ``file://`` path in the JSON.
    if hasattr(value, "__dict__") and not isinstance(value, type):
        for key, item in list(vars(value).items()):
            published = _publish_map_urls(item)
            if published is not item:
                try:
                    setattr(value, key, published)
                except Exception:  # noqa: BLE001 - frozen/slotted payloads
                    pass
        return value
    return value


@app.get("/maps/{name}")
def get_map(name: str):
    """One rendered map, as HTML. Only files this agent wrote are reachable."""

    target = (_MAPS_DIR / Path(name).name).resolve()
    # Path traversal guard: the resolved file must sit directly in the maps dir.
    if target.parent != _MAPS_DIR.resolve() or not target.is_file():
        return JSONResponse(status_code=404, content={"detail": f"No map named {name!r}"})
    return FileResponse(target, media_type="text/html")


def _build_agent():
    """Pick the implementation this process serves.

    A failure to build the orchestrator falls back to the mock rather than
    killing the service: an agent that answers something is more useful to the
    Global Orchestrator than a port that never opens.
    """
    impl = os.getenv("BIODIVERSITY_AGENT_IMPL", "mock").strip().lower()
    if impl != "orchestrator":
        # print, not log: uvicorn owns the logging config and swallows module
        # INFO records, but stdout is streamed by run_agents with the agent's
        # name in front. Which implementation is serving is the single most
        # confusing thing to get wrong, so it must be impossible to miss.
        print(
            f"[Biodiversity] serving MOCK (BIODIVERSITY_AGENT_IMPL={impl!r}). "
            f"Set BIODIVERSITY_AGENT_IMPL=orchestrator for the real orchestrator.",
            flush=True,
        )
        return BiodiversityMock()

    try:
        from .orchestrator_adapter import OrchestratorBiodiversityAgent

        agent = OrchestratorBiodiversityAgent()
        print("[Biodiversity] serving the LangGraph ORCHESTRATOR", flush=True)
        return agent
    except Exception as exc:  # noqa: BLE001 - startup must not depend on it
        # Loud, because this looks exactly like the flag not working.
        print(
            f"[Biodiversity] BIODIVERSITY_AGENT_IMPL=orchestrator but it could NOT be "
            f"built ({type(exc).__name__}: {exc}); falling back to MOCK.",
            flush=True,
        )
        _logger.warning("orchestrator build failed", exc_info=True)
        return BiodiversityMock()


# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = _build_agent()


@app.get("/health")
def health() -> dict:
    """Which implementation this process is actually serving.

    Cheap way to answer "did the environment variable take?" without sending a
    real request and inferring the answer from the shape of the output.
    """
    return {
        "agent": "Biodiversity",
        "implementation": type(_agent).__name__,
        "is_orchestrator": type(_agent).__name__ == "OrchestratorBiodiversityAgent",
    }


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        # The mock answers synchronously, the orchestrator is async. Awaiting
        # only when there is something to await lets one code path serve both
        # without duplicating the error handling below.
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result

        # Turn the workers' on-disk paths into URLs the caller can fetch.
        if result.map_url:
            name = _local_map_name(result.map_url)
            if name:
                result.map_url = f"{_PUBLIC_URL}/maps/{name}"
        result.output = _publish_map_urls(result.output)
        return result
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        return AgentResult(status=AgentStatus.FAILED, output=f"Biodiversity Agent error: {exc}")

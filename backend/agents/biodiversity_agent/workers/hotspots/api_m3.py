"""HTTP boundary that serves the real M3 to the project's own backend.

The agent already has an HTTP boundary, ``api.py``, but it builds
``BiodiversityMock``. That file is not ours to edit, so this is a second app with
the *same contract* - ``POST /execute`` in, one ``AgentResult`` out - wired to the
real Biodiversity sub-orchestrator with M3 injected and the other three features
left as their mocks.

Nothing in the project changes. The orchestrator finds an agent through
``backend/registry.py``, which already reads an environment variable per agent, so
pointing it here is configuration, not code:

    # 1. serve this app instead of the mock, on the Biodiversity agent's own port
    python -m uvicorn backend.agents.biodiversity_agent.workers.hotspots.api_m3:app --port 8003

    # 2. or run it beside the mock on a free port and redirect the orchestrator
    python -m uvicorn backend.agents.biodiversity_agent.workers.hotspots.api_m3:app --port 8013
    $env:BIODIVERSITY_AGENT_URL = "http://localhost:8013"   # PowerShell

Either way the chain is the project's real one:

    frontend  ->  POST /api/chat        (backend/api.py, port 8000)
              ->  GlobalOrchestrator    (planner, resolver, scheduler)
              ->  POST /execute         (this app)
              ->  BiodiversityOrchestrator -> HotspotsWorker
              ->  AgentResult           -> Responder -> answer

**The map.** ``render_folium`` writes an HTML file inside this agent's folder,
which is a path, not a URL - useless to a browser on another origin. This app
serves those files at ``GET /map/{name}`` and rewrites ``map_url`` to an absolute
URL, so any client can show the map in an iframe. The URL host is taken from
``M3_PUBLIC_URL`` when set, since a container does not know its own address.
"""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ...orchestrator import BiodiversityOrchestrator
from ...schema import AgentRequest, AgentResult, AgentStatus, BiodiversityFeature
from ..common.config import OUTPUT_DIR, REGIONS, regions_in_text
from ..habitat.mock import HabitatMock
from ..migration.mock import MigrationMock
from ..species_distribution.mock import SpeciesDistributionMock
from .status import M3Outcome
from .worker import HotspotsWorker

app = FastAPI(title="Biodiversity Agent (M3 live)")

# The map files are served to a browser that may be on another origin (the Vite
# dev server), so the iframe needs CORS. /execute is called server-to-server and
# would not need it; allowing both costs nothing here.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Built once at startup, not per request: the sub-orchestrator constructs a
# LangGraph graph, and doing that per call would put it in the request path.
_agent = BiodiversityOrchestrator(workers={
    BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsWorker(),
    BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: SpeciesDistributionMock(),
    BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
    BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
})

_PUBLIC_URL = os.getenv("M3_PUBLIC_URL", "http://localhost:8003").rstrip("/")


def _plain(value: Any) -> Any:
    """Dataclasses to JSON-safe structures, so ``output`` survives the wire."""

    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "value") and type(value).__name__.endswith(("Index", "Algorithm",
                                                                 "Status", "Outcome")):
        return value.value
    return value


def _publish_map(result: AgentResult) -> AgentResult:
    """Turn the local map path into a URL a browser can actually open."""

    if result.map_url and not result.map_url.startswith("http"):
        result.map_url = f"{_PUBLIC_URL}/map/{Path(result.map_url).name}"
    payload = result.output
    if hasattr(payload, "render_spec") and payload.render_spec.html_path:
        payload.render_spec.html_path = result.map_url or payload.render_spec.html_path
    return result


def _child_payload(result: AgentResult):
    """The worker's own payload, whichever way the aggregator wrapped it.

    The sub-orchestrator passes a single successful payload straight through, but
    nests a failed one under its feature key - so a caller that only reads
    ``result.output`` sees a different shape depending on the outcome.
    """

    payload = result.output
    if isinstance(payload, dict) and len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, dict) and {"status_detail", "known_regions"} & set(inner):
            return inner
    return payload


# What a hotspots question sounds like. These are the routing_triggers from
# card.json, and they exist because the Global Orchestrator's worker node posts
# only {instruction, context} - it never sets a feature, and the sub-orchestrator
# refuses to guess one. The legacy mock defaults to species distribution and then
# reports "Species not specified.", so a hotspots question cannot reach M3 through
# the project's own chain without this.
_HOTSPOT_TRIGGERS = (
    "hotspot", "richest", "most species", "species richness", "richness",
    "diversity highest", "biodiversity heatmap", "heatmap", "biodiversity of",
    "biodiversity in", "biodiversity between", "compare biodiversity",
)


def _feature_for(request: AgentRequest) -> str:
    """The feature this request is asking for, when the caller did not say.

    Region-scoped questions belong to M3; a question naming a species does not.
    """

    if request.feature:
        return request.feature
    text = (request.instruction or "").lower()
    if any(trigger in text for trigger in _HOTSPOT_TRIGGERS):
        return BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value
    if regions_in_text(text) and not (request.species_name
                                      or request.context.get("species")):
        # Names a study area and no species: nothing else it could be.
        return BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value
    return BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value


@app.post("/execute", response_model=None)
async def execute(request: AgentRequest) -> dict:
    """The agent's single endpoint. Always answers with an AgentResult shape.

    Deliberately not an HTTPException on failure: the orchestrator's worker node
    expects one schema back every time and already handles a FAILED status, so a
    500 with FastAPI's ``{"detail": ...}`` body would break that contract.
    """

    try:
        request.feature = _feature_for(request)
        result = _publish_map(await _agent.run(request))
        body = _plain(result)
        # The Responder writes the user-facing answer from what it is given, so
        # the two things a reader wants - the sentence and the map - are lifted to
        # the top level rather than left buried in the payload.
        payload = _child_payload(result)
        if hasattr(payload, "summary"):
            body["summary"] = payload.summary
            body["map_url"] = result.map_url
        elif isinstance(payload, dict) and payload.get("status_detail") ==                 M3Outcome.NEEDS_CLARIFICATION.value:
            # A question, not a failure: pass the options on so the caller can
            # ask the user instead of reporting an error.
            body["question"] = payload.get("message")
            body["options"] = payload.get("options") or payload.get("known_regions")
        return body
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        return _plain(AgentResult(
            status=AgentStatus.FAILED,
            output=f"Biodiversity Agent (M3) error: {exc}",
            source_agents=["Biodiversity Hotspots Agent"]))


@app.get("/map/{name}")
def get_map(name: str) -> FileResponse:
    """One rendered map. Only files this module wrote are reachable."""

    target = (OUTPUT_DIR / Path(name).name).resolve()
    if target.parent != OUTPUT_DIR.resolve() or not target.is_file():
        return FileResponse(status_code=404, path=OUTPUT_DIR / "missing.html")
    return FileResponse(target, media_type="text/html")


@app.get("/health")
def health() -> dict:
    """What is live here, so a deployment can tell the mock from the real thing."""

    return {
        "status": "ok",
        "hotspots": "live - GBIF, effort-corrected richness, DBSCAN",
        "other_features": "mock (M1, M2, M4 are not this module's scope)",
        "regions": sorted(REGIONS),
        "public_url": _PUBLIC_URL,
    }


@app.get("/")
def index() -> dict:
    """What this service is, so an accidental visit to / explains itself.

    There is no UI here: this app is the agent's HTTP boundary, called by the
    orchestrator. The dashboard is the interface for a human.
    """

    return {
        "service": "Biodiversity Agent - M3 Biodiversity Hotspots (live)",
        "called_by": "the Global Orchestrator, via POST /execute",
        "endpoints": {
            "POST /execute": "one AgentRequest in, one AgentResult out",
            "GET /map/{name}": "a rendered hotspot map, as HTML",
            "GET /health": "what is live here and which regions are known",
            "GET /docs": "the generated OpenAPI page",
        },
        "example": {
            "curl": ("curl -X POST http://localhost:8003/execute "
                     "-H 'Content-Type: application/json' "
                     "-d '{\"instruction\": \"hotspots in Madagascar\", \"context\": {}}'"),
        },
        "regions": sorted(REGIONS),
    }

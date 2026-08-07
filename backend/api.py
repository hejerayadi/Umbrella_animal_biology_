"""HTTP API exposing the Global Scientific Orchestrator to the frontend.

A thin FastAPI wrapper around `GlobalOrchestrator`. The frontend sends the
user's chat message here; this file hands it to the orchestrator (planner ->
worker -> capability resolver -> ... -> done) and sends back the result as
plain JSON.

Run it from the repository root with:

    uvicorn backend.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .orchestrator.langgraph import GlobalOrchestrator

# Makes the Planner/Resolver/worker log lines (see orchestrator/*.py) show
# up in the terminal running `uvicorn backend.api:app`, so you can watch the
# orchestrator's decisions live as requests come in.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
# Quiet the underlying HTTP client's own request/response logging so the
# console only shows the orchestrator's own reasoning, not raw network noise.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Umbrella Orchestrator API")

# Local development only: the frontend dev server's port can vary (it's
# auto-detected by its own tooling), so we allow any origin here rather than
# hardcoding one. Restrict this to a specific origin before deploying
# anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once, when the server starts - not on every request. Building it
# does real work (setting up the LangGraph graph and its LLM chains), so
# re-building it per-request would be slow and wasteful.
_orchestrator = GlobalOrchestrator()
_logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """What the frontend sends when the user submits a chat message."""

    query: str
    context: dict[str, Any] = {}


class ChatResponse(BaseModel):
    """What the frontend receives back once the orchestrator finishes.

    `answer` is the finished, human-readable reply written by the Responder.
    `execution_history` is the step-by-step log the orchestrator builds
    internally (e.g. "Planner -> Genome", "Genome -> completed") - the
    frontend turns this into the "Agent Thinking" timeline. `context` is the
    raw structured data the agents produced, kept for debugging and for
    future richer rendering in the UI.
    """

    answer: str
    execution_history: list[str]
    context: dict[str, Any]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one user query through the full orchestrator loop and return the result."""

    _logger.info("=== New request: %r ===", request.query)
    state = _orchestrator.run(request.query, initial_context=request.context)
    _logger.info(
        "=== Done: %d steps, final context keys=%s ===",
        len(state.execution_history),
        list(state.context),
    )
    return ChatResponse(
        # `final_answer` is set by whichever answer-writing node the workflow
        # ended at. The fallback only triggers if the graph somehow finished
        # without reaching one of them.
        answer=state.final_answer or "The orchestrator did not produce an answer for this request.",
        execution_history=state.execution_history,
        context=state.context,
    )

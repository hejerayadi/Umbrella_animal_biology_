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

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .image_store import IMAGE_STORE, MAX_IMAGE_BYTES, ImageRejected
from .orchestrator.langgraph import GlobalOrchestrator
from .orchestrator.langgraph.nodes.worker_node import IMAGE_ID_CONTEXT_KEY

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

    `image_url` is set when an agent generated an illustration. It is a URL
    into this same API, never the image itself - see `_publish_generated_image`.
    """

    answer: str
    execution_history: list[str]
    context: dict[str, Any]
    image_url: str | None = None


# The context key the Image Generation Agent publishes its result under, matching
# the `output` block of its card.json.
_GENERATED_IMAGE_KEY = "image"


def _publish_generated_image(context: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Move a generated image out of the context and behind a URL.

    FLUX.2-pro replies with a base64 data URI - ~440 KB of characters for one
    1024x1024 image. That is the same problem uploads have (see
    `image_store.py`), arriving from the other direction: this context is
    serialised into the chat response, and the frontend persists messages to
    localStorage, whose quota one image would exhaust.

    So the bytes are parked in the same store the upload path uses and the
    context carries the URL instead. An agent that already returned a plain http
    URL is passed through untouched - there is nothing to store.

    Returns the context to send back, plus the URL for `ChatResponse.image_url`.
    """
    image = context.get(_GENERATED_IMAGE_KEY)
    if not isinstance(image, str) or not image.strip():
        return context, None

    if image.startswith("http://") or image.startswith("https://"):
        return context, image

    if not image.startswith("data:"):
        # Neither a URL nor a data URI - nothing displayable. Leave it alone so
        # it still shows up in the debugging context rather than vanishing.
        return context, None

    try:
        stored = IMAGE_STORE.add_data_url(image, filename="generated.jpg")
    except ImageRejected as exc:
        # A generated image we cannot store is not worth failing the whole
        # answer for: the text is still correct without the picture.
        _logger.warning("Generated image could not be stored: %s", exc)
        return {**context, _GENERATED_IMAGE_KEY: f"<unstorable image: {exc}>"}, None

    url = f"/api/upload/{stored.image_id}"
    _logger.info(
        "=== Generated image stored: %d KB, id=%s ===", len(stored.data) // 1024, stored.image_id
    )
    return {**context, _GENERATED_IMAGE_KEY: url}, url


class UploadResponse(BaseModel):
    """What the frontend gets back after attaching an image.

    `image_id` is what it must put in the next chat request's context, under
    the key given by `context_key`. The bytes stay on the server.
    """

    image_id: str
    context_key: str
    media_type: str
    filename: str
    size_bytes: int


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Accept one image for the next chat message.

    Returns an id rather than echoing the image back: the frontend already has
    the file it just picked, and the orchestrator only needs a handle. See
    `image_store.py` for why the bytes must not travel in the chat context.
    """

    data = await file.read()
    try:
        stored = IMAGE_STORE.add(data, file.filename)
    except ImageRejected as exc:
        # A 400 with a readable reason - the frontend shows this to the user,
        # who can act on "too big" or "wrong format" but not on a stack trace.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _logger.info(
        "=== Image uploaded: %s %s (%d KB), id=%s ===",
        stored.filename, stored.media_type, len(stored.data) // 1024, stored.image_id,
    )
    return UploadResponse(
        image_id=stored.image_id,
        context_key=IMAGE_ID_CONTEXT_KEY,
        media_type=stored.media_type,
        filename=stored.filename,
        size_bytes=len(stored.data),
    )


@app.get("/api/upload/limits")
def upload_limits() -> dict[str, Any]:
    """What the frontend should enforce before uploading, so it can fail early.

    Declared BEFORE the `/{image_id}` route below: FastAPI matches in
    declaration order, so the other way round this path would be read as an
    image whose id is the literal string "limits".
    """
    return {
        "max_bytes": MAX_IMAGE_BYTES,
        "accepted_media_types": ["image/jpeg", "image/png", "image/webp"],
        "context_key": IMAGE_ID_CONTEXT_KEY,
    }


@app.get("/api/upload/{image_id}")
def uploaded_image(image_id: str) -> Response:
    """Serve a stored image back, so the chat can display what was sent.

    The frontend keeps only this URL on the message, not the image data. A
    base64 copy in the message would be persisted to localStorage, whose quota
    is around 5 MB - one photo would fill it and break the whole conversation
    history. A blob: URL would not survive a page reload either.

    404 once the store evicts it (see `image_store.py`). The chat treats that
    as "no image to show" rather than an error: the conversation text is still
    correct, and an old thumbnail is not worth persisting bytes for.
    """
    stored = IMAGE_STORE.get(image_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="That image is no longer available.")

    return Response(
        content=stored.data,
        media_type=stored.media_type,
        # Immutable: the id is derived per upload, so this URL's bytes can
        # never change. Lets the browser skip re-fetching on every render.
        headers={"Cache-Control": "private, max-age=3600, immutable"},
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one user query through the full orchestrator loop and return the result."""

    # An attached image arrives as an id in the context, not as bytes. The
    # planner is told one exists because it reads only the text, and the same
    # sentence routes differently with a photo attached.
    has_image = bool(request.context.get(IMAGE_ID_CONTEXT_KEY))

    _logger.info("=== New request: %r (image=%s) ===", request.query, has_image)
    state = _orchestrator.run(
        request.query, initial_context=request.context, has_image=has_image
    )
    _logger.info(
        "=== Done: %d steps, final context keys=%s ===",
        len(state.execution_history),
        list(state.context),
    )
    context, image_url = _publish_generated_image(state.context)
    return ChatResponse(
        # `final_answer` is set by whichever answer-writing node the workflow
        # ended at. The fallback only triggers if the graph somehow finished
        # without reaching one of them.
        answer=state.final_answer or "The orchestrator did not produce an answer for this request.",
        execution_history=state.execution_history,
        context=context,
        image_url=image_url,
    )

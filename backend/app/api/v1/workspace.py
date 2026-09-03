from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.image_store import IMAGE_STORE, MAX_IMAGE_BYTES, ImageRejected
from backend.orchestrator import events
from backend.orchestrator.langgraph import GlobalOrchestrator
from backend.orchestrator.langgraph.nodes.worker_node import IMAGE_ID_CONTEXT_KEY

from ...contracts import ApiProblem, success
from ...core.dependencies import Identity, require_user, require_user_csrf

router = APIRouter(tags=["workspace"])
logger = logging.getLogger("umbrella.workspace")

@lru_cache(maxsize=1)
def get_orchestrator() -> GlobalOrchestrator:
    # Auth and administration can run without LLM credentials. Build the
    # expensive scientific graph only when the first chat request needs it.
    return GlobalOrchestrator()


class ChatRequest(BaseModel):
    query: str
    context: dict[str, Any] = {}


def _publish_generated_image(
    context: dict[str, Any], owner_id: str
) -> tuple[dict[str, Any], str | None]:
    image = context.get("image")
    if not isinstance(image, str) or not image.strip():
        return context, None
    if image.startswith(("http://", "https://")):
        return context, image
    if not image.startswith("data:"):
        return context, None
    try:
        stored = IMAGE_STORE.add_data_url(
            image, filename="generated.jpg", owner_id=owner_id
        )
    except ImageRejected as exc:
        logger.warning("Generated image could not be stored: %s", exc)
        return {**context, "image": f"<unstorable image: {exc}>"}, None
    url = f"/api/v1/uploads/{stored.image_id}"
    return {**context, "image": url}, url


@router.post("/uploads")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    identity: Identity = Depends(require_user_csrf),
) -> dict:
    data = await file.read()
    try:
        stored = IMAGE_STORE.add(
            data, file.filename, owner_id=str(identity.user.id)
        )
    except ImageRejected as exc:
        raise ApiProblem(400, "UPLOAD_REJECTED", "Image rejected", str(exc)) from exc
    return success(request, {
        "image_id": stored.image_id,
        "context_key": IMAGE_ID_CONTEXT_KEY,
        "media_type": stored.media_type,
        "filename": stored.filename,
        "size_bytes": len(stored.data),
    })


@router.get("/uploads/limits")
async def upload_limits(
    request: Request, _: Identity = Depends(require_user)
) -> dict:
    return success(request, {
        "max_bytes": MAX_IMAGE_BYTES,
        "accepted_media_types": ["image/jpeg", "image/png", "image/webp"],
        "context_key": IMAGE_ID_CONTEXT_KEY,
    })


@router.get("/uploads/{image_id}")
async def uploaded_image(
    image_id: str,
    identity: Identity = Depends(require_user),
) -> Response:
    stored = IMAGE_STORE.get_for_owner(image_id, str(identity.user.id))
    if stored is None:
        raise ApiProblem(404, "IMAGE_NOT_FOUND", "Image not found", "That image is no longer available.")
    return Response(
        content=stored.data,
        media_type=stored.media_type,
        headers={"Cache-Control": "private, max-age=3600, immutable"},
    )


def _attached_image(payload: ChatRequest, identity: Identity) -> bool:
    """Whether this message carries a usable image, rejecting one that doesn't."""

    if not payload.context.get(IMAGE_ID_CONTEXT_KEY):
        return False
    image_id = str(payload.context[IMAGE_ID_CONTEXT_KEY])
    if IMAGE_STORE.get_for_owner(image_id, str(identity.user.id)) is None:
        raise ApiProblem(404, "IMAGE_NOT_FOUND", "Image not found", "The attached image is unavailable.")
    return True


def _chat_result(state: Any, owner_id: str) -> dict[str, Any]:
    """The finished chat payload, shared by the blocking and streaming routes."""

    context, image_url = _publish_generated_image(state.context, owner_id)
    return {
        "answer": state.final_answer
        or "The orchestrator did not produce an answer for this request.",
        "execution_history": state.execution_history,
        "context": context,
        "image_url": image_url,
    }


@router.post("/chat")
def chat(
    payload: ChatRequest,
    request: Request,
    identity: Identity = Depends(require_user_csrf),
) -> dict:
    has_image = _attached_image(payload, identity)
    state = get_orchestrator().run(
        payload.query, initial_context=payload.context, has_image=has_image
    )
    return success(request, _chat_result(state, str(identity.user.id)))


# How long the stream may go without sending anything before a comment line is
# sent to hold the connection open. A single worker agent can legitimately take
# ten minutes (see `_READ_TIMEOUT_SECONDS` in worker_node.py), which is longer
# than the idle timeout of most proxies and of some browsers.
_HEARTBEAT_SECONDS = 15.0

# Ends the stream. A queue carrying real events needs one value that cannot be
# one, and `None` is it.
_END = None

# Strong references to the graph runs currently in flight. See `stream()`.
_RUNNING: set[asyncio.Task[None]] = set()


def _sse(event: dict[str, Any]) -> str:
    """One Server-Sent Events frame. Newlines inside the JSON would split it."""

    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    identity: Identity = Depends(require_user_csrf),
) -> StreamingResponse:
    """The same run as `/chat`, narrated over Server-Sent Events as it happens.

    `/chat` answers once, at the end. A research question routed through three
    agents takes minutes to get there, and everything the orchestrator decided
    on the way - which agent it picked and why, who it had to bring in, what
    failed - arrived as history the user no longer needed. Here each of those
    decisions is sent the moment it is made, and the final answer streams in
    token by token as the responder writes it.

    Event shapes, all JSON in the `data:` field:

      {"type": "step",    "id", "agent", "text", "state": running|done|failed}
      {"type": "thought", "id", "text"}    - the model's own words on a step
      {"type": "answer",  "delta"}         - one chunk of the final answer
      {"type": "done",    "payload"}       - identical to `/chat`'s response
      {"type": "error",   "message"}

    `done` carries the whole result anyway, so a client that cannot stream can
    still ignore everything before it and behave exactly like a `/chat` caller.
    Steps are keyed by `id` and sent twice (running, then done): render them as
    one line that updates, not two.

    This route is deliberately outside the `success()` envelope the rest of the
    API uses - an envelope describes one complete response, and this is a
    sequence of them.
    """

    has_image = _attached_image(payload, identity)
    owner_id = str(identity.user.id)
    query = payload.query
    context = dict(payload.context)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def publish(event: dict[str, Any]) -> None:
        # Called from the graph's worker thread, so it may not touch the queue
        # directly - `call_soon_threadsafe` is the hand-off back to the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def run_graph() -> None:
        # The emitter is bound inside the thread that runs the graph. Nodes
        # read it from a ContextVar, and `to_thread` gives this call its own
        # copy of the context, so concurrent chats cannot cross streams.
        try:
            with events.emitting_to(publish):
                state = get_orchestrator().run(
                    query, initial_context=context, has_image=has_image
                )
            publish({"type": "done", "payload": _chat_result(state, owner_id)})
        except Exception as exc:  # noqa: BLE001 - the client gets told either way
            logger.exception("Streaming chat run failed")
            publish({"type": "error", "message": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _END)

    async def stream() -> AsyncIterator[str]:
        # Held in `_RUNNING` for as long as it runs. The event loop keeps only
        # a weak reference to a task, so a run whose reader disconnects - the
        # browser navigated away, the tab closed - could otherwise be collected
        # mid-flight. The graph cannot be cancelled anyway (it is blocking on
        # HTTP calls to nine services), so it is left to finish into a queue
        # nobody reads rather than abandoned part-way through.
        task = asyncio.create_task(asyncio.to_thread(run_graph))
        _RUNNING.add(task)
        task.add_done_callback(_RUNNING.discard)

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), _HEARTBEAT_SECONDS)
            except TimeoutError:
                # A comment line: keeps proxies and browsers from closing a
                # connection that is only quiet because an agent is slow.
                yield ": keep-alive\n\n"
                continue
            if event is _END:
                return
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx not to buffer, which would hold every event back
            # until the response ended and undo the whole point of this route.
            "X-Accel-Buffering": "no",
        },
    )

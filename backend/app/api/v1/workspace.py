from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from pydantic import BaseModel

from backend.image_store import IMAGE_STORE, MAX_IMAGE_BYTES, ImageRejected
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


@router.post("/chat")
def chat(
    payload: ChatRequest,
    request: Request,
    identity: Identity = Depends(require_user_csrf),
) -> dict:
    has_image = bool(payload.context.get(IMAGE_ID_CONTEXT_KEY))
    if has_image:
        image_id = str(payload.context[IMAGE_ID_CONTEXT_KEY])
        if IMAGE_STORE.get_for_owner(image_id, str(identity.user.id)) is None:
            raise ApiProblem(404, "IMAGE_NOT_FOUND", "Image not found", "The attached image is unavailable.")
    state = get_orchestrator().run(
        payload.query, initial_context=payload.context, has_image=has_image
    )
    context, image_url = _publish_generated_image(
        state.context, str(identity.user.id)
    )
    return success(request, {
        "answer": state.final_answer
        or "The orchestrator did not produce an answer for this request.",
        "execution_history": state.execution_history,
        "context": context,
        "image_url": image_url,
    })

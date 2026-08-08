"""Server-side holding area for uploaded images.

The Multimodal Recognition Agent needs the image bytes, but the shared
`context` is the wrong way to carry them. That dict is broadcast: the worker
node POSTs it in full to every agent it calls, the Responder renders every key
into an LLM prompt, and the chat endpoint returns it to the browser. A 2 MB
photo becomes ~2.7 MB of base64 that would be re-sent on every agent call and
turned into hundreds of thousands of prompt tokens.

So the image is uploaded once, kept here, and only a short id travels in the
context. `worker_node` swaps that id back for the real data URL for the one
agent that needs it, and strips it for everyone else.

Deliberately in-memory and bounded. Nothing here survives a restart, which is
correct for a chat attachment: the alternative is files on disk that nothing
ever deletes. `_MAX_IMAGES` evicts oldest-first so a long session cannot grow
without limit.

Validation is intentionally duplicated with the Recognition agent's own
`validation.py` rather than imported. `backend/` imports no agent code - that
isolation is what lets a broken agent stop being the orchestrator's problem -
and an upload boundary should reject a malformed file whether or not the agent
that consumes it happens to be installed. The agent re-validates anyway; two
independent checks at a trust boundary is the point, not an accident.
"""
from __future__ import annotations

import base64
import binascii
import secrets
from collections import OrderedDict
from dataclasses import dataclass

# Kept in step with the Recognition agent's ALLOWED_MEDIA_TYPES. A type here
# that the agent rejects would only move the error later, not prevent it.
ALLOWED_MEDIA_TYPES: dict[str, tuple[bytes, ...]] = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),  # "WEBP" at offset 8 is checked separately
}

# 8 MB of decoded image. A phone photo exceeds this easily, which is why the
# frontend is expected to downscale before uploading rather than after.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# How many uploads to keep. Chat attachments are used within one request or
# two; this is a safety limit, not a cache size worth tuning.
_MAX_IMAGES = 32


class ImageRejected(ValueError):
    """The upload is not something we are willing to store or forward."""


@dataclass(frozen=True)
class StoredImage:
    """One accepted upload, ready to be handed to the Recognition agent."""

    image_id: str
    media_type: str
    filename: str
    data: bytes

    def as_data_url(self) -> str:
        """The `data:image/png;base64,...` form the Recognition agent accepts."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


def _sniff_media_type(data: bytes) -> str | None:
    """Identify the image from its leading bytes, ignoring any claimed type.

    A client-supplied content type is a claim, not evidence. The magic bytes
    are what the decoder will actually act on.
    """
    for media_type, signatures in ALLOWED_MEDIA_TYPES.items():
        for signature in signatures:
            if data.startswith(signature):
                if media_type == "image/webp":
                    # RIFF alone is a container; only WEBP is an image.
                    if len(data) >= 12 and data[8:12] == b"WEBP":
                        return media_type
                    continue
                return media_type
    return None


def _safe_filename(filename: str | None) -> str:
    """A display name only - never used to open, write or join a path."""
    if not filename:
        return "upload"
    # Strip anything path-like: this value is echoed back to the client and
    # forwarded to an agent, and neither should ever see a traversal attempt.
    cleaned = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return cleaned[:120] or "upload"


class ImageStore:
    """Bounded, in-memory store of uploaded images, keyed by opaque id."""

    def __init__(self, max_images: int = _MAX_IMAGES) -> None:
        self._images: OrderedDict[str, StoredImage] = OrderedDict()
        self._max_images = max_images

    def add(self, data: bytes, filename: str | None = None) -> StoredImage:
        """Validate and store raw image bytes. Raises `ImageRejected`."""

        if not data:
            raise ImageRejected("The uploaded file is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageRejected(
                f"The image is {len(data) // 1024} KB; the limit is "
                f"{MAX_IMAGE_BYTES // 1024} KB. Please resize it before uploading."
            )

        media_type = _sniff_media_type(data)
        if media_type is None:
            raise ImageRejected(
                "That file is not a JPEG, PNG or WebP image. Only those three are accepted."
            )

        stored = StoredImage(
            # Opaque and unguessable: the id is handed to the browser and comes
            # back on the next request, so it should not be enumerable.
            image_id=secrets.token_urlsafe(16),
            media_type=media_type,
            filename=_safe_filename(filename),
            data=data,
        )
        self._images[stored.image_id] = stored

        while len(self._images) > self._max_images:
            self._images.popitem(last=False)  # oldest first

        return stored

    def add_data_url(self, data_url: str, filename: str | None = None) -> StoredImage:
        """Store an image supplied as a `data:` URL rather than a file upload."""

        prefix, _, payload = data_url.partition(",")
        if not payload or not prefix.startswith("data:") or "base64" not in prefix:
            raise ImageRejected("Expected a base64 data URL of the form 'data:image/png;base64,...'.")
        try:
            return self.add(base64.b64decode(payload, validate=True), filename)
        except (binascii.Error, ValueError) as exc:
            if isinstance(exc, ImageRejected):
                raise
            raise ImageRejected("The data URL's base64 payload could not be decoded.") from exc

    def get(self, image_id: str) -> StoredImage | None:
        """Return a stored image, or None if the id is unknown or evicted."""
        return self._images.get(image_id)

    def __len__(self) -> int:
        return len(self._images)


# One store per process, matching how the orchestrator itself is built once.
IMAGE_STORE = ImageStore()

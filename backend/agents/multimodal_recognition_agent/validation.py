"""Paired image-and-text validation, and safe image decoding.

Every Recognition request carries exactly two things: a non-empty instruction,
and exactly one image under `context["recognition_image"]`. Neither alone is a
valid request, and both are enforced here rather than left to the caller.

The decode path is ordered so that each step is cheap relative to the one after
it. The size of the *encoded* string is checked before any decoding, the decoded
size before any parsing, the header dimensions before any pixel loading. A
hostile 200 MB payload is rejected having cost one integer comparison; a
decompression bomb is rejected from its header, before it can allocate anything.

Nothing in this module ever puts a value from the request into an exception, a
log line or a return value. Failures carry an `ErrorCode` and nothing else.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import re
import warnings
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ALLOWED_MEDIA_TYPES, RECOGNITION_IMAGE_CONTEXT_KEY, ValidationConfig
from .domain.errors import ErrorCode, RecognitionError
from .domain.models import NormalizedRecognitionInput

# "data:image/png;base64," - the media type is captured, everything else is fixed.
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)

# Anything that looks like it wants us to go and fetch something. We never do.
_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "file://", "//")

# Leading bytes that identify the real format, independent of what was declared.
_MAGIC = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),  # plus "WEBP" at offset 8, checked below
}

_CONTROL_CHARS = frozenset(chr(code) for code in range(32)) | {chr(127)}


def _safe_filename(raw: Any) -> str:
    """Accept an already-safe basename, or refuse. Never derive one.

    Deriving a basename from an unsafe path would silently accept input the
    contract rejects, and would mean the value we store is not the value that
    was sent. One deterministic rule: trim, then reject anything unsafe.
    """
    if not isinstance(raw, str):
        raise RecognitionError(ErrorCode.INVALID_FILENAME)

    name = raw.strip()
    if not name:
        raise RecognitionError(ErrorCode.INVALID_FILENAME)

    if "/" in name or "\\" in name:
        raise RecognitionError(ErrorCode.INVALID_FILENAME)
    if ".." in name:
        raise RecognitionError(ErrorCode.INVALID_FILENAME)
    # "C:" and friends - a drive-relative path with no separator.
    if len(name) >= 2 and name[1] == ":":
        raise RecognitionError(ErrorCode.INVALID_FILENAME)
    if any(char in _CONTROL_CHARS for char in name):
        raise RecognitionError(ErrorCode.INVALID_FILENAME)
    if name in (".", ".."):
        raise RecognitionError(ErrorCode.INVALID_FILENAME)

    return name


def _extract_image_entry(context: dict[str, Any]) -> tuple[str, str]:
    """Pull `data_url` and `filename` out of the one approved context key.

    Only `recognition_image` is read. Other context keys are never scanned for
    images: a legitimate context can hold another agent's `generated_image`, and
    that is not this agent's input.
    """
    if RECOGNITION_IMAGE_CONTEXT_KEY not in context:
        raise RecognitionError(ErrorCode.MISSING_IMAGE)

    entry = context[RECOGNITION_IMAGE_CONTEXT_KEY]

    # A list/tuple is the shape someone reaches for when they want to send two
    # images. The contract has no such representation, so it is malformed input
    # rather than "too many images".
    if not isinstance(entry, dict):
        raise RecognitionError(ErrorCode.MALFORMED_IMAGE_OBJECT)
    if set(entry) - {"data_url", "filename"}:
        raise RecognitionError(ErrorCode.MALFORMED_IMAGE_OBJECT)

    data_url = entry.get("data_url")
    if not isinstance(data_url, str) or not data_url:
        raise RecognitionError(ErrorCode.MALFORMED_IMAGE_OBJECT)
    if "filename" not in entry:
        raise RecognitionError(ErrorCode.INVALID_FILENAME)

    return data_url, _safe_filename(entry.get("filename"))


def _parse_data_url(data_url: str, config: ValidationConfig) -> tuple[str, str]:
    """Return (declared media type, base64 payload) without decoding anything."""

    stripped = data_url.strip()
    if stripped.lower().startswith(_REMOTE_PREFIXES):
        raise RecognitionError(ErrorCode.UNSUPPORTED_IMAGE_SOURCE)

    match = _DATA_URL_RE.match(stripped)
    if match is None:
        # A bare path ("/tmp/x.jpg", "C:\\x.jpg") is a source we refuse to
        # follow, which is worth distinguishing from a mangled data URL.
        if stripped.startswith(("/", ".")) or (len(stripped) >= 2 and stripped[1] == ":"):
            raise RecognitionError(ErrorCode.UNSUPPORTED_IMAGE_SOURCE)
        raise RecognitionError(ErrorCode.INVALID_DATA_URL)

    media_type = match.group(1).lower()
    payload = match.group(2)

    if media_type not in ALLOWED_MEDIA_TYPES:
        raise RecognitionError(ErrorCode.UNSUPPORTED_MEDIA_TYPE)
    if not payload:
        raise RecognitionError(ErrorCode.INVALID_DATA_URL)

    # Cheapest possible size guard: base64 expands 3 bytes to 4 characters, so a
    # payload longer than this cannot decode to an accepted image. Checked
    # before decoding, and without trusting padding characters to shrink it.
    if len(payload) > config.max_encoded_len:
        raise RecognitionError(ErrorCode.IMAGE_TOO_LARGE)

    return media_type, payload


def _decode(payload: str, config: ValidationConfig) -> bytes:
    """Strict base64 decode, then the authoritative size check."""
    try:
        # validate=True rejects characters outside the base64 alphabet instead
        # of silently discarding them, so what we hash is what was sent.
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RecognitionError(ErrorCode.INVALID_BASE64) from exc

    if not image_bytes:
        raise RecognitionError(ErrorCode.INVALID_BASE64)
    if len(image_bytes) > config.max_image_bytes:
        raise RecognitionError(ErrorCode.IMAGE_TOO_LARGE)

    return image_bytes


def _detect_format(image_bytes: bytes) -> str | None:
    """The real media type, read from the leading bytes."""
    for media_type, signatures in _MAGIC.items():
        for signature in signatures:
            if image_bytes.startswith(signature):
                if media_type == "image/webp":
                    # RIFF is a container; WEBP identifies itself at offset 8.
                    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
                        return media_type
                    continue
                return media_type
    return None


def _inspect_pixels(image_bytes: bytes, config: ValidationConfig) -> tuple[int, int]:
    """Validate the image with Pillow and return its stored dimensions.

    The whole Pillow interaction sits inside one warnings context, starting at
    the very first `open()`: a decompression bomb announces itself in the header,
    so the warning can fire before any of the later calls are reached.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                width, height = probe.size

                if width <= 0 or height <= 0:
                    raise RecognitionError(ErrorCode.CORRUPT_IMAGE)
                # Before any pixel loading: refuse an area we will not process.
                if width * height > config.max_image_pixels:
                    raise RecognitionError(ErrorCode.IMAGE_PIXELS_EXCEEDED)
                if width < config.min_image_width or height < config.min_image_height:
                    raise RecognitionError(ErrorCode.IMAGE_TOO_SMALL)

                # verify() detects structural damage but leaves the handle unusable.
                probe.verify()

            # Reopen from a fresh buffer and fully load, which is what actually
            # catches truncation. LOAD_TRUNCATED_IMAGES is left at its default
            # False so a truncated file raises instead of quietly succeeding.
            with Image.open(io.BytesIO(image_bytes)) as full:
                full.load()
                # Temporary, in-memory only: proves the EXIF/RGB path this image
                # will take later is sound. It never redefines `image_bytes` and
                # never feeds the hash.
                ImageOps.exif_transpose(full).convert("RGB")

        except RecognitionError:
            raise
        except Image.DecompressionBombWarning as exc:
            raise RecognitionError(ErrorCode.IMAGE_PIXELS_EXCEEDED) from exc
        except Image.DecompressionBombError as exc:
            raise RecognitionError(ErrorCode.IMAGE_PIXELS_EXCEEDED) from exc
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise RecognitionError(ErrorCode.CORRUPT_IMAGE) from exc

    return width, height


def decode_image(
    data_url: str, config: ValidationConfig
) -> tuple[bytes, str, str, int, int]:
    """Validate one data URL. Returns (bytes, media_type, sha256, width, height)."""

    declared_type, payload = _parse_data_url(data_url, config)
    image_bytes = _decode(payload, config)

    actual_type = _detect_format(image_bytes)
    if actual_type is None:
        raise RecognitionError(ErrorCode.CORRUPT_IMAGE)
    if actual_type != declared_type:
        raise RecognitionError(ErrorCode.MEDIA_TYPE_MISMATCH)

    width, height = _inspect_pixels(image_bytes, config)

    # Over exactly the base64-decoded encoded file bytes. This definition is the
    # one that must match Chahd's ingestion side before any sha-keyed fixture
    # mapping can be trusted.
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    return image_bytes, declared_type, image_sha256, width, height


def validate_paired_request(
    instruction: Any, context: Any, config: ValidationConfig
) -> NormalizedRecognitionInput:
    """The single entry point. Either returns a normalized input, or raises."""

    # Both halves are required. One image AND one non-empty text instruction, in
    # the same request - an image with nothing asked of it is not a Recognition
    # request, and neither is a sentence with no photograph. A whitespace-only
    # string is empty; a non-text instruction is malformed. Both refuse the same
    # way, because both mean "no usable instruction was supplied".
    if not isinstance(instruction, str) or not instruction.strip():
        raise RecognitionError(ErrorCode.EMPTY_INSTRUCTION)

    if not isinstance(context, dict):
        raise RecognitionError(ErrorCode.MISSING_IMAGE)

    data_url, filename = _extract_image_entry(context)
    image_bytes, media_type, image_sha256, width, height = decode_image(data_url, config)

    # Deep copy first, then remove the image key from the copy. The caller's
    # dict is never touched: the orchestrator reuses that object for every
    # subsequent agent, and mutating it here would silently break them.
    retained = copy.deepcopy(context)
    retained.pop(RECOGNITION_IMAGE_CONTEXT_KEY, None)

    return NormalizedRecognitionInput(
        instruction=instruction.strip(),
        image_bytes=image_bytes,
        context=retained,
        media_type=media_type,
        image_sha256=image_sha256,
        width=width,
        height=height,
        byte_size=len(image_bytes),
        filename=filename,
    )

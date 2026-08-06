"""Controlled failures.

Every way this agent can refuse a request is one of the codes below. The
important property is the constructor signature: `RecognitionError` takes an
`ErrorCode` and nothing else. There is no parameter through which a data URL, a
filename, a base64 fragment or a decoded byte could be attached to an exception
and then leak out through a log line, a traceback or an `AgentResult`.

The message is derived from the code alone, so two requests failing the same way
produce byte-identical text no matter what they contained.
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    # --- paired-input contract ---
    EMPTY_INSTRUCTION = "EMPTY_INSTRUCTION"
    MISSING_IMAGE = "MISSING_IMAGE"
    MALFORMED_IMAGE_OBJECT = "MALFORMED_IMAGE_OBJECT"
    INVALID_FILENAME = "INVALID_FILENAME"

    # --- image transport ---
    UNSUPPORTED_IMAGE_SOURCE = "UNSUPPORTED_IMAGE_SOURCE"
    INVALID_DATA_URL = "INVALID_DATA_URL"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    INVALID_BASE64 = "INVALID_BASE64"

    # --- image content ---
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    MEDIA_TYPE_MISMATCH = "MEDIA_TYPE_MISMATCH"
    IMAGE_PIXELS_EXCEEDED = "IMAGE_PIXELS_EXCEEDED"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"

    # --- embedding ---
    EMBEDDING_DIMENSION_MISMATCH = "EMBEDDING_DIMENSION_MISMATCH"

    # --- retrieval ---
    QDRANT_CONTRACT_NOT_FROZEN = "QDRANT_CONTRACT_NOT_FROZEN"
    QDRANT_CLIENT_UNAVAILABLE = "QDRANT_CLIENT_UNAVAILABLE"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    RETRIEVAL_TIMEOUT = "RETRIEVAL_TIMEOUT"
    COLLECTION_CONTRACT_MISMATCH = "COLLECTION_CONTRACT_MISMATCH"


# Fixed, data-free explanations. Written for the orchestrator's responder and
# ultimately for a person, so each one says what to change - without ever
# echoing what was sent.
_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.EMPTY_INSTRUCTION:
        "The instruction must be text when one is supplied. It is optional - the "
        "image is the primary evidence - but it cannot be a number or a structure.",
    ErrorCode.MISSING_IMAGE:
        "No image was provided. Recognition requires one image and one text instruction "
        "in the same request.",
    ErrorCode.MALFORMED_IMAGE_OBJECT:
        "The image entry must be a single object with a 'data_url' string and a "
        "'filename' string.",
    ErrorCode.INVALID_FILENAME:
        "The image filename is missing or is not a safe file name.",
    ErrorCode.UNSUPPORTED_IMAGE_SOURCE:
        "Only inline base64 data URLs are accepted. Remote URLs and file paths are "
        "never fetched.",
    ErrorCode.INVALID_DATA_URL:
        "The image data URL is malformed. Expected 'data:image/<format>;base64,<data>'.",
    ErrorCode.UNSUPPORTED_MEDIA_TYPE:
        "Unsupported image type. Accepted types are JPEG, PNG and WEBP.",
    ErrorCode.INVALID_BASE64:
        "The image payload is not valid base64.",
    ErrorCode.IMAGE_TOO_LARGE:
        "The image exceeds the maximum accepted size.",
    ErrorCode.MEDIA_TYPE_MISMATCH:
        "The image content does not match its declared media type.",
    ErrorCode.IMAGE_PIXELS_EXCEEDED:
        "The image pixel area exceeds the maximum accepted area.",
    ErrorCode.IMAGE_TOO_SMALL:
        "The image is smaller than the minimum accepted dimensions.",
    ErrorCode.CORRUPT_IMAGE:
        "The image could not be decoded. It may be truncated or corrupt.",
    ErrorCode.EMBEDDING_DIMENSION_MISMATCH:
        "The generated query vector length does not match the configured collection "
        "dimension.",
    ErrorCode.QDRANT_CONTRACT_NOT_FROZEN:
        "Real Qdrant retrieval is not configured. The Sprint 2 collection contract has "
        "not been supplied yet.",
    ErrorCode.QDRANT_CLIENT_UNAVAILABLE:
        "The Qdrant client library is not installed in this agent's environment.",
    ErrorCode.RETRIEVAL_UNAVAILABLE:
        "The reference collection could not be reached.",
    ErrorCode.RETRIEVAL_TIMEOUT:
        "The reference collection did not respond in time.",
    ErrorCode.COLLECTION_CONTRACT_MISMATCH:
        "The reference collection does not match the configured contract; no query was "
        "sent.",
}


class RecognitionError(Exception):
    """A controlled refusal. Carries a code - never a value from the request."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        # `Exception.args` is what `repr()` prints, so the fixed message is the
        # only thing that can ever appear there.
        super().__init__(_MESSAGES[code])

    @property
    def message(self) -> str:
        return _MESSAGES[self.code]

    def as_output(self) -> dict[str, str]:
        """The `AgentResult.output` payload for a failed request."""
        return {"error_code": self.code.value, "error": self.message}

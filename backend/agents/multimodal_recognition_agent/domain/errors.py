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

    # --- species classification ---
    CLASSIFICATION_FIXTURE_INVALID = "CLASSIFICATION_FIXTURE_INVALID"
    CLASSIFICATION_CONTRACT_VIOLATION = "CLASSIFICATION_CONTRACT_VIOLATION"
    CLASSIFICATION_UNAVAILABLE = "CLASSIFICATION_UNAVAILABLE"

    # --- taxonomy validation (mock GBIF / mock NCBI) ---
    TAXONOMY_UNAVAILABLE = "TAXONOMY_UNAVAILABLE"


# Fixed, data-free explanations. Written for the orchestrator's responder and
# ultimately for a person, so each one says what to change - without ever
# echoing what was sent.
_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.EMPTY_INSTRUCTION:
        "A non-empty text instruction is required. Recognition needs one image and one "
        "text instruction in the same request.",
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
    ErrorCode.CLASSIFICATION_FIXTURE_INVALID:
        "The Sprint 2 mock classification fixture is malformed. No species was named, "
        "because a broken test oracle must never be repaired into an answer.",
    ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION:
        "The classification provider returned predictions that break the agreed "
        "contract - unranked, duplicated, or scored outside the accepted range.",
    ErrorCode.CLASSIFICATION_UNAVAILABLE:
        "The species classification provider could not be reached.",
    ErrorCode.TAXONOMY_UNAVAILABLE:
        "A taxonomy source could not be reached. Candidates are reported as unverified; "
        "no identifier is ever inferred to fill the gap.",
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

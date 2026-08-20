"""Paired input validation: the happy path and every way it can be refused."""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
from ..domain.errors import ErrorCode, RecognitionError
from ..validation import validate_paired_request
from .conftest import data_url, image_entry, jpeg_bytes, png_bytes, sha256_of, webp_bytes

INSTRUCTION = "Identify this animal and explain the result."


def _fails_with(code, instruction, context, config):
    with pytest.raises(RecognitionError) as caught:
        validate_paired_request(instruction, context, config)
    assert caught.value.code is code
    return caught.value


# --- happy path ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, media_type",
    [
        (png_bytes(), "image/png"),
        (jpeg_bytes(), "image/jpeg"),
        (webp_bytes(), "image/webp"),
    ],
)
def test_accepts_each_supported_format(raw, media_type, validation_config):
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(raw, media_type, "obs.bin")}
    result = validate_paired_request(INSTRUCTION, context, validation_config)

    assert result.media_type == media_type
    assert result.instruction == INSTRUCTION
    assert result.filename == "obs.bin"
    assert result.width == 128 and result.height == 128


def test_image_bytes_are_the_original_encoded_file_bytes(validation_config):
    raw = png_bytes()
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(raw)}
    result = validate_paired_request(INSTRUCTION, context, validation_config)

    # Not re-encoded, not re-compressed, not normalised.
    assert result.image_bytes == raw
    assert result.byte_size == len(raw)
    # And the hash is over exactly those bytes.
    assert result.image_sha256 == sha256_of(raw)


def test_instruction_is_stripped(validation_config, valid_context):
    result = validate_paired_request("  identify this  ", valid_context, validation_config)
    assert result.instruction == "identify this"


# --- context handling ------------------------------------------------------

def test_caller_context_is_never_mutated(validation_config, valid_png):
    original = {
        RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(valid_png),
        "species": "seeded by the extractor",
    }
    snapshot = dict(original)

    result = validate_paired_request(INSTRUCTION, original, validation_config)

    # The orchestrator reuses this dict for every later agent.
    assert original == snapshot
    assert RECOGNITION_IMAGE_CONTEXT_KEY in original
    # The retained copy has the image removed, and is a different object.
    assert RECOGNITION_IMAGE_CONTEXT_KEY not in result.context
    assert result.context is not original
    assert result.context["species"] == "seeded by the extractor"


def test_other_context_keys_are_not_scanned_for_images(validation_config, valid_png):
    """`generated_image` is another agent's output, not this agent's input."""
    context = {
        RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(valid_png),
        "generated_image": {"data_url": data_url(png_bytes(), "image/png")},
    }
    result = validate_paired_request(INSTRUCTION, context, validation_config)
    assert result.context["generated_image"]  # survives untouched, unread


# --- instruction -----------------------------------------------------------

@pytest.mark.parametrize("instruction", ["", "   ", "\n\t ", None])
def test_absent_instruction_is_accepted(instruction, validation_config, valid_context):
    """The validated decisions make the instruction OPTIONAL: the image is the
    primary evidence, and text is context that may simply not be there."""
    result = validate_paired_request(instruction, valid_context, validation_config)
    assert result.instruction == ""
    assert result.image_sha256


@pytest.mark.parametrize("instruction", [42, [], {}, 3.5, True])
def test_non_text_instruction_is_still_rejected(instruction, validation_config, valid_context):
    """Optional is not the same as "anything goes" - a number is malformed
    input, not an absent instruction."""
    _fails_with(ErrorCode.EMPTY_INSTRUCTION, instruction, valid_context, validation_config)


def test_the_image_remains_mandatory(validation_config):
    """Text became optional. The image did not."""
    _fails_with(ErrorCode.MISSING_IMAGE, "Identify this animal.", {}, validation_config)


# --- image entry shape -----------------------------------------------------

def test_missing_image_key(validation_config):
    _fails_with(ErrorCode.MISSING_IMAGE, INSTRUCTION, {}, validation_config)


def test_non_dict_context(validation_config):
    _fails_with(ErrorCode.MISSING_IMAGE, INSTRUCTION, "not a dict", validation_config)


@pytest.mark.parametrize(
    "entry",
    [
        "just-a-string",
        123,
        [],                                        # a list is not the frozen shape
        [{"data_url": "x", "filename": "a.png"}],  # nor is a list of objects
        (),
        {"filename": "a.png"},                     # no data_url
        {"data_url": 5, "filename": "a.png"},      # data_url not a string
        {"data_url": "", "filename": "a.png"},
        {"data_url": "x", "filename": "a.png", "extra": 1},  # unknown key
    ],
)
def test_malformed_image_object(entry, validation_config):
    _fails_with(
        ErrorCode.MALFORMED_IMAGE_OBJECT,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


# --- filename --------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        None, "", "   ", 42,
        "../../etc/passwd",
        "dir/observation.png",
        "dir\\observation.png",
        "C:observation.png",
        "obs\x00.png",
        "obs\nname.png",
        "..",
        ".",
    ],
)
def test_invalid_filename(filename, validation_config, valid_png):
    entry = {"data_url": data_url(valid_png), "filename": filename}
    _fails_with(
        ErrorCode.INVALID_FILENAME,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_safe_basename_is_retained_verbatim(validation_config, valid_png):
    entry = {"data_url": data_url(valid_png), "filename": "  observation-1.png  "}
    result = validate_paired_request(
        INSTRUCTION, {RECOGNITION_IMAGE_CONTEXT_KEY: entry}, validation_config
    )
    # Trimmed, never rewritten.
    assert result.filename == "observation-1.png"


# --- data URL --------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/lion.jpg",
        "http://example.org/lion.jpg",
        "file:///tmp/lion.jpg",
        "//example.org/lion.jpg",
        "/var/data/lion.jpg",
        "./lion.jpg",
        "C:\\images\\lion.jpg",
    ],
)
def test_remote_or_path_sources_are_never_fetched(url, validation_config):
    entry = {"data_url": url, "filename": "lion.jpg"}
    _fails_with(
        ErrorCode.UNSUPPORTED_IMAGE_SOURCE,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


@pytest.mark.parametrize(
    "url",
    ["data:image/png,notbase64", "data:image/png;base64,", "notadataurl", "data:;base64,AAAA"],
)
def test_invalid_data_url(url, validation_config):
    entry = {"data_url": url, "filename": "lion.png"}
    _fails_with(
        ErrorCode.INVALID_DATA_URL,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


@pytest.mark.parametrize("media_type", ["image/gif", "image/tiff", "image/svg+xml", "image/bmp"])
def test_unsupported_media_type(media_type, validation_config, valid_png):
    entry = {"data_url": data_url(valid_png, media_type), "filename": "x.gif"}
    _fails_with(
        ErrorCode.UNSUPPORTED_MEDIA_TYPE,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_invalid_base64(validation_config):
    entry = {"data_url": "data:image/png;base64,!!!!not base64!!!!", "filename": "x.png"}
    _fails_with(
        ErrorCode.INVALID_BASE64,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_strict_base64_rejects_stray_characters(validation_config, valid_png):
    payload = base64.b64encode(valid_png).decode("ascii")
    # A newline inside the payload would be silently dropped without validate=True.
    entry = {"data_url": f"data:image/png;base64,{payload[:20]}\n{payload[20:]}", "filename": "x.png"}
    _fails_with(
        ErrorCode.INVALID_BASE64,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


# --- size and content ------------------------------------------------------

def test_oversized_encoded_payload_is_rejected_before_decoding(validation_config):
    from ..config import ValidationConfig

    tiny = ValidationConfig(
        max_image_bytes=64, max_image_pixels=25_000_000, min_image_width=1, min_image_height=1
    )
    entry = {"data_url": data_url(png_bytes()), "filename": "x.png"}
    _fails_with(
        ErrorCode.IMAGE_TOO_LARGE, INSTRUCTION, {RECOGNITION_IMAGE_CONTEXT_KEY: entry}, tiny
    )


def test_oversized_decoded_bytes_are_rejected(validation_config):
    from ..config import ValidationConfig

    raw = png_bytes(256, 256)
    # Big enough to pass the encoded-length bound, small enough to fail the real one.
    limit = ValidationConfig(
        max_image_bytes=len(raw) - 1,
        max_image_pixels=25_000_000,
        min_image_width=1,
        min_image_height=1,
    )
    entry = {"data_url": data_url(raw), "filename": "x.png"}
    _fails_with(
        ErrorCode.IMAGE_TOO_LARGE, INSTRUCTION, {RECOGNITION_IMAGE_CONTEXT_KEY: entry}, limit
    )


def test_media_type_mismatch(validation_config):
    # Real PNG bytes, declared as JPEG.
    entry = {"data_url": data_url(png_bytes(), "image/jpeg"), "filename": "x.jpg"}
    _fails_with(
        ErrorCode.MEDIA_TYPE_MISMATCH,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_pixel_area_exceeded(validation_config):
    from ..config import ValidationConfig

    limit = ValidationConfig(
        max_image_bytes=10_485_760, max_image_pixels=100, min_image_width=1, min_image_height=1
    )
    entry = {"data_url": data_url(png_bytes(128, 128)), "filename": "x.png"}
    _fails_with(
        ErrorCode.IMAGE_PIXELS_EXCEEDED,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        limit,
    )


def test_image_too_small(validation_config):
    entry = {"data_url": data_url(png_bytes(16, 16)), "filename": "x.png"}
    _fails_with(
        ErrorCode.IMAGE_TOO_SMALL,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_corrupt_image(validation_config):
    # A valid PNG signature followed by rubbish: passes magic-byte detection,
    # fails when Pillow tries to make sense of it.
    broken = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    entry = {"data_url": data_url(broken), "filename": "x.png"}
    _fails_with(
        ErrorCode.CORRUPT_IMAGE,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_truncated_image_is_corrupt(validation_config):
    raw = jpeg_bytes(256, 256)
    entry = {"data_url": data_url(raw[: len(raw) // 2], "image/jpeg"), "filename": "x.jpg"}
    _fails_with(
        ErrorCode.CORRUPT_IMAGE,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_unrecognised_bytes_are_corrupt(validation_config):
    entry = {"data_url": data_url(b"this is not an image at all"), "filename": "x.png"}
    _fails_with(
        ErrorCode.CORRUPT_IMAGE,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )


def test_decompression_bomb_is_refused_from_the_header(validation_config, monkeypatch):
    """Pillow warns from the header; the guard must already be in scope."""
    huge = io.BytesIO()
    Image.new("RGB", (2000, 2000)).save(huge, format="PNG")

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    entry = {"data_url": data_url(huge.getvalue()), "filename": "x.png"}
    _fails_with(
        ErrorCode.IMAGE_PIXELS_EXCEEDED,
        INSTRUCTION,
        {RECOGNITION_IMAGE_CONTEXT_KEY: entry},
        validation_config,
    )

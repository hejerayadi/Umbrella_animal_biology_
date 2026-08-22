"""The HTTP boundary.

`POST /execute` must accept an `AgentRequest` and return an `AgentResult` for
every outcome, including failures - the orchestrator parses exactly one schema
and a 500 with FastAPI's `{"detail": ...}` body would break it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ..api import app
from .conftest import image_entry, png_bytes

client = TestClient(app)

INSTRUCTION = "Identify this animal and explain the result."


def post(instruction, context):
    return client.post("/execute", json={"instruction": instruction, "context": context})


def valid_context():
    return {"recognition_image": image_entry(png_bytes())}


def test_valid_request_returns_the_shared_schema():
    response = post(INSTRUCTION, valid_context())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "target_agent", "prompt_to_target_agent", "output"}
    assert body["status"] in ("completed", "needs_agent", "continue", "failed")


def test_valid_request_completes_with_the_recognition_contract():
    body = post(INSTRUCTION, valid_context()).json()

    assert body["status"] == "completed"
    assert set(body["output"]) == {
        "recognition", "species", "species_id", "gbif_id", "ncbi_taxid",
        "recognition_candidates", "recognition_provenance",
    }


def test_an_unmatched_image_honestly_reports_not_identified():
    """No fixture was ingested for this image, so nothing should be claimed."""
    body = post(INSTRUCTION, valid_context()).json()
    assert body["output"]["recognition"]["decision"] == "not_identified"
    assert body["output"]["species"] is None


@pytest.mark.parametrize(
    "instruction, context, code",
    [
        (INSTRUCTION, {}, "MISSING_IMAGE"),
        (INSTRUCTION, {"recognition_image": "not-an-object"}, "MALFORMED_IMAGE_OBJECT"),
        (INSTRUCTION, {"recognition_image": [{"data_url": "x", "filename": "y.png"}]},
         "MALFORMED_IMAGE_OBJECT"),
        (INSTRUCTION, {"recognition_image": {
            "data_url": "data:image/png;base64,!!!", "filename": "x.png"}}, "INVALID_BASE64"),
        (INSTRUCTION, {"recognition_image": {
            "data_url": "https://example.org/a.png", "filename": "a.png"}},
         "UNSUPPORTED_IMAGE_SOURCE"),
    ],
)
def test_invalid_input_returns_a_controlled_failure_never_a_500(instruction, context, code):
    response = post(instruction, context if context is not None else valid_context())

    assert response.status_code == 200, "the boundary must never leak a 500"
    body = response.json()
    assert body["status"] == "failed"
    assert body["output"]["error_code"] == code


@pytest.mark.parametrize("instruction", ["", "   "])
def test_an_empty_instruction_is_refused_over_http(instruction):
    """Both halves are required: one image AND one non-empty text instruction."""
    response = post(instruction, valid_context())

    assert response.status_code == 200, "the boundary must never leak a 500"
    body = response.json()
    assert body["status"] == "failed"
    assert body["output"]["error_code"] == "EMPTY_INSTRUCTION"


def test_a_non_text_instruction_is_rejected_at_the_schema_boundary():
    """Two layers, deliberately.

    The shared `AgentRequest` declares `instruction: str`, so FastAPI rejects a
    number with 422 before the agent ever sees it. The agent's own
    `EMPTY_INSTRUCTION` guard covers the same case for in-process callers - see
    `test_validation.py::test_non_text_instruction_is_still_rejected`.
    """
    response = client.post(
        "/execute", json={"instruction": 42, "context": valid_context()}
    )
    assert response.status_code == 422


def test_missing_context_is_a_validation_error_not_a_crash():
    """`context` has no default in the shared schema, so FastAPI rejects it."""
    response = client.post("/execute", json={"instruction": INSTRUCTION})
    assert response.status_code == 422


def test_no_image_data_is_echoed_back():
    context = valid_context()
    payload = context["recognition_image"]["data_url"].split(",", 1)[1]

    response = post(INSTRUCTION, context)

    assert payload[:48] not in response.text
    assert "data:image" not in response.text


def test_openapi_still_describes_the_shared_contract():
    schema = client.get("/openapi.json").json()
    assert "/execute" in schema["paths"]
    assert "post" in schema["paths"]["/execute"]

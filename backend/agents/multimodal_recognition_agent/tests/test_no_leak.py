"""The leak tests.

Three independently checked markers, because one is not enough:

  E1  a substring of the raw base64 payload         (the encoded surface)
  B1  plaintext embedded in the decoded PNG bytes   (the decoded surface)
  C1  a value in an unrelated nested context entry  (someone else's data)

E1 and B1 have to be separate: a plaintext string inside PNG bytes does not
appear unchanged in the base64 of those bytes, so a single marker would silently
test only half of what matters.

None of the three may appear in a repr, a string conversion, a serialization, a
log record, an exception, an output dict, or any prompt payload this agent
builds. The structural assertions matter just as much: `image_bytes` and
`context` must be *absent* from serialized output, not merely blanked.
"""
from __future__ import annotations

import base64
import json
import logging

import pytest

from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
from ..domain.errors import RecognitionError
from ..schema import AgentRequest
from ..validation import validate_paired_request
from .conftest import (
    StubRetriever,
    data_url,
    image_entry,
    make_config,
    png_bytes,
    reference,
)

E1_LABEL = "encoded-surface"
B1 = "BBBB1111MARKERdecodedimagebytesXX"
C1 = "CCCC1111MARKERunrelatedcontextvalXX"

AGENT_PACKAGE_LOGGER = "backend.agents.multimodal_recognition_agent"


def _payload_and_markers():
    """Build one request carrying all three markers, and return them."""
    raw = png_bytes(marker=B1)
    assert B1.encode() in raw, "marker must survive into the decoded image bytes"

    encoded = base64.b64encode(raw).decode("ascii")
    # E1 is taken from the payload that actually exists, since base64 output
    # cannot be chosen freely.
    e1 = encoded[16:64]
    assert len(e1) == 48

    context = {
        RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": f"data:image/png;base64,{encoded}",
            "filename": "observation.png",
        },
        "generated_image": {"data_url": f"data:image/png;base64,{C1}AAAA"},
        "papers": [{"title": f"A study mentioning {C1}"}],
    }
    return raw, e1, context


def _assert_absent(haystack: str, e1: str, *, where: str) -> None:
    for name, marker in (("E1", e1), ("B1", B1), ("C1", C1)):
        assert marker not in haystack, f"{name} leaked into {where}"


# --- the model itself ------------------------------------------------------

def test_markers_absent_from_repr_str_and_serialization(config):
    _, e1, context = _payload_and_markers()
    model = validate_paired_request("Identify this animal.", context, config.validation)

    _assert_absent(repr(model), e1, where="repr()")
    _assert_absent(str(model), e1, where="str()")
    _assert_absent(json.dumps(model.model_dump(), default=str), e1, where="model_dump()")
    _assert_absent(model.model_dump_json(), e1, where="model_dump_json()")


def test_excluded_fields_are_absent_not_redacted(config):
    _, _, context = _payload_and_markers()
    model = validate_paired_request("Identify this animal.", context, config.validation)

    dumped = model.model_dump()
    assert "image_bytes" not in dumped
    assert "context" not in dumped
    assert "image_bytes" not in model.model_dump_json()
    assert "context" not in model.model_dump_json()
    # ...while the safe metadata is still there.
    assert dumped["image_sha256"] and dumped["media_type"] and dumped["byte_size"]


def test_extra_fields_are_forbidden(config):
    """`extra="forbid"` is what stops an image field being bolted on later."""
    from ..domain.models import NormalizedRecognitionInput

    with pytest.raises(Exception):
        NormalizedRecognitionInput(
            instruction="x",
            image_bytes=b"x",
            context={},
            media_type="image/png",
            image_sha256="0" * 64,
            width=64,
            height=64,
            byte_size=1,
            filename="x.png",
            sneaky_extra_image="leak",
        )


# --- logging ---------------------------------------------------------------

def test_markers_absent_from_every_log_record(config, caplog):
    _, e1, context = _payload_and_markers()

    with caplog.at_level(logging.DEBUG, logger=AGENT_PACKAGE_LOGGER):
        agent = RecognitionAgent(
            make_config(),
            retriever=StubRetriever([reference("panthera_leo", 0.9)]),
        )
        agent.run(AgentRequest(instruction="Identify this animal.", context=context))

    assert caplog.records, "expected the workflow to log something"
    for record in caplog.records:
        _assert_absent(record.getMessage(), e1, where="log message")
        _assert_absent(str(record.args), e1, where="log args")


# --- exceptions ------------------------------------------------------------

@pytest.mark.parametrize(
    "context_builder",
    [
        # every failure mode reachable with a marker-bearing payload
        lambda ctx, e1: {**ctx, RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": ctx[RECOGNITION_IMAGE_CONTEXT_KEY]["data_url"], "filename": "../" + B1}},
        lambda ctx, e1: {**ctx, RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": "data:image/gif;base64," + e1, "filename": "x.gif"}},
        lambda ctx, e1: {**ctx, RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": "data:image/png;base64,!!!" + e1, "filename": "x.png"}},
        lambda ctx, e1: {**ctx, RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": "https://example.org/" + B1 + ".png", "filename": "x.png"}},
    ],
)
def test_markers_absent_from_exceptions(config, context_builder):
    _, e1, context = _payload_and_markers()
    bad_context = context_builder(context, e1)

    with pytest.raises(RecognitionError) as caught:
        validate_paired_request("Identify this animal.", bad_context, config.validation)

    exc = caught.value
    _assert_absent(str(exc), e1, where="str(exception)")
    _assert_absent(repr(exc), e1, where="repr(exception)")
    _assert_absent(str(exc.args), e1, where="exception.args")


def test_every_error_message_is_fixed_and_data_free():
    """The same refusal produces byte-identical text regardless of the input."""
    from ..domain.errors import ErrorCode

    for code in ErrorCode:
        first = RecognitionError(code)
        second = RecognitionError(code)
        assert str(first) == str(second)
        assert first.args == second.args


# --- outputs ---------------------------------------------------------------

def test_markers_absent_from_the_agent_result(config):
    _, e1, context = _payload_and_markers()

    agent = RecognitionAgent(
        make_config(), retriever=StubRetriever([reference("panthera_leo", 0.9)])
    )
    result = agent.run(AgentRequest(instruction="Identify this animal.", context=context))

    serialized = json.dumps(
        {
            "status": result.status.value,
            "target_agent": result.target_agent,
            "prompt_to_target_agent": result.prompt_to_target_agent,
            "output": result.output,
        },
        default=str,
    )
    _assert_absent(serialized, e1, where="AgentResult")


def test_failure_result_carries_no_request_data(config):
    _, e1, context = _payload_and_markers()
    context = {**context, RECOGNITION_IMAGE_CONTEXT_KEY: {
        "data_url": "data:image/png;base64,!!!" + e1, "filename": "x.png"}}

    agent = RecognitionAgent(make_config(), retriever=StubRetriever([]))
    result = agent.run(AgentRequest(instruction="Identify this animal.", context=context))

    _assert_absent(json.dumps(result.output, default=str), e1, where="failed output")


def test_delegation_prompt_carries_no_request_data(config):
    """`prompt_to_target_agent` is the one free-text field this agent authors,
    and it reaches the orchestrator's LLM. It must carry a species name only."""
    _, e1, context = _payload_and_markers()

    agent = RecognitionAgent(
        make_config(), retriever=StubRetriever([reference("panthera_leo", 0.95)])
    )
    result = agent.run(
        AgentRequest(
            instruction="What is the evolutionary history of this animal?", context=context
        )
    )

    assert result.status.value == "needs_agent"
    _assert_absent(result.prompt_to_target_agent, e1, where="prompt_to_target_agent")
    assert result.output is None

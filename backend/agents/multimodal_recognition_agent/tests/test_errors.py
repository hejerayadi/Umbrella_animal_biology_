"""The error taxonomy's shape.

The point of these tests is the constructor signature. `RecognitionError` must
have no parameter through which a request value could be attached, because that
is what guarantees an exception cannot become a leak.
"""
from __future__ import annotations

import inspect

import pytest

from ..domain.errors import ErrorCode, RecognitionError


def test_constructor_accepts_only_a_code():
    parameters = list(inspect.signature(RecognitionError.__init__).parameters)
    assert parameters == ["self", "code"]


def test_every_code_has_a_message():
    for code in ErrorCode:
        error = RecognitionError(code)
        assert error.message
        assert str(error) == error.message


def test_as_output_shape():
    error = RecognitionError(ErrorCode.MISSING_IMAGE)
    output = error.as_output()
    assert output == {"error_code": "MISSING_IMAGE", "error": error.message}
    # An output dict, so a failure is still machine-readable downstream.
    assert isinstance(output, dict)


def test_code_is_preserved():
    error = RecognitionError(ErrorCode.CORRUPT_IMAGE)
    assert error.code is ErrorCode.CORRUPT_IMAGE


def test_cannot_attach_arbitrary_data():
    with pytest.raises(TypeError):
        RecognitionError(ErrorCode.MISSING_IMAGE, "data:image/png;base64,SECRET")  # type: ignore[call-arg]

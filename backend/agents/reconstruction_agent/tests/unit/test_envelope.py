"""The `{data, meta, error}` invariant.

Exactly one of `data`/`error` is non-null. Callers branch on that instead of
inspecting HTTP codes, so it has to hold unconditionally.
"""
from __future__ import annotations

from api.v1.envelope import (
    API_VERSION,
    ApiError,
    Envelope,
    ErrorCode,
    ErrorDetail,
    error_payload,
)


class TestEnvelope:
    def test_ok_populates_data_and_leaves_error_null(self) -> None:
        envelope = Envelope[dict].ok({"value": 1})

        assert envelope.data == {"value": 1}
        assert envelope.error is None

    def test_fail_populates_error_and_leaves_data_null(self) -> None:
        envelope = Envelope[dict].fail(ErrorCode.INTERNAL_ERROR, "boom")

        assert envelope.data is None
        assert envelope.error is not None
        assert envelope.error.code is ErrorCode.INTERNAL_ERROR

    def test_meta_is_always_present(self) -> None:
        """Even on failure - the request id is what a bug report quotes."""
        for envelope in (
            Envelope[dict].ok({}),
            Envelope[dict].fail(ErrorCode.TIMEOUT, "slow"),
        ):
            assert envelope.meta.api_version == API_VERSION
            assert envelope.meta.timestamp is not None

    def test_request_id_and_duration_ride_along(self) -> None:
        envelope = Envelope[dict].ok({}, request_id="abc-123", duration_ms=12.5)

        assert envelope.meta.request_id == "abc-123"
        assert envelope.meta.duration_ms == 12.5

    def test_retryable_is_carried_on_the_error(self) -> None:
        envelope = Envelope[dict].fail(
            ErrorCode.RATE_LIMITED, "slow down", retryable=True
        )

        assert envelope.error is not None
        assert envelope.error.retryable is True

    def test_details_carry_field_level_problems(self) -> None:
        envelope = Envelope[dict].fail(
            ErrorCode.VALIDATION_ERROR,
            "bad body",
            details=[ErrorDetail(field="sequence", message="required")],
        )

        assert envelope.error is not None
        assert envelope.error.details[0].field == "sequence"

    def test_serialises_to_exactly_three_keys(self) -> None:
        payload = Envelope[dict].ok({"value": 1}).model_dump(mode="json")

        assert set(payload) == {"data", "meta", "error"}

    def test_error_payload_is_json_ready(self) -> None:
        """Exception handlers return JSONResponse, which needs a plain dict."""
        payload = error_payload(ErrorCode.INTERNAL_ERROR, "boom", request_id="r1")

        assert payload["data"] is None
        assert payload["error"]["code"] == "internal_error"
        assert payload["meta"]["request_id"] == "r1"
        assert isinstance(payload["meta"]["timestamp"], str)


class TestErrorCodes:
    def test_codes_are_stable_snake_case_strings(self) -> None:
        """Clients branch on these; renaming one breaks callers."""
        for code in ErrorCode:
            assert code.value.islower()
            assert " " not in code.value

    def test_api_error_defaults_to_not_retryable(self) -> None:
        assert ApiError(code=ErrorCode.INVALID_SEQUENCE, message="x").retryable is False

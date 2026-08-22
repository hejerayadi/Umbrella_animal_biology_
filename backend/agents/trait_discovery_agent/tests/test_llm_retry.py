"""
Unit tests for workflows/llm/client.py's transient-error retry logic.

Before this, _is_capacity_error only matched 503/ResourceExhausted — a 429
(request-rate quota, as opposed to worker-pool exhaustion) skipped retry
entirely and propagated straight to whichever subagent called it, which
immediately gave up and fell back to deterministic behavior. That's fine as
a last resort, but on a rate-limited (e.g. free-tier) key it meant the LLM
disambiguation path essentially never ran at all — every call hit the quota
and bailed on the first attempt.
"""
import time

import pytest

import workflows.llm.client as client_module


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # Keep these tests fast regardless of the real defaults.
    monkeypatch.setattr(client_module, "CAPACITY_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "RATE_LIMIT_RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "RATE_LIMIT_RETRY_MAX_SECONDS", 0.05)
    monkeypatch.setattr(client_module, "MAX_CAPACITY_RETRIES", 3)
    monkeypatch.setattr(client_module, "MAX_RATE_LIMIT_RETRIES", 5)
    monkeypatch.setattr(client_module, "MIN_CALL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(client_module, "_last_call_started_at", 0.0)


def _nim_error(status: int, title: str) -> Exception:
    """Matches the bare Exception format langchain_nvidia_ai_endpoints
    actually raises (_common.py:_try_raise): f"[{status}] {title}\n{body}"."""
    return Exception(f"[{status}] {title}\n{{'status': {status}, 'title': '{title}'}}")


def test_is_rate_limit_error_matches_429():
    assert client_module._is_rate_limit_error(_nim_error(429, "Too Many Requests"))
    assert not client_module._is_rate_limit_error(_nim_error(503, "Service Unavailable"))
    assert not client_module._is_rate_limit_error(_nim_error(401, "Unauthorized"))


def test_is_capacity_error_unaffected_by_429_addition():
    assert client_module._is_capacity_error(Exception("503 Service Unavailable: request limit reached"))
    assert not client_module._is_capacity_error(_nim_error(429, "Too Many Requests"))


@pytest.mark.asyncio
async def test_retries_and_recovers_from_429():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _nim_error(429, "Too Many Requests")
        return "ok"

    result = await client_module._retry_on_capacity(flaky)
    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_503_capacity_retry_still_works():
    """Regression check: extending retry to cover 429 must not change 503
    handling, which existed before this change."""
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise Exception("503 Service Unavailable: request limit reached")
        return "ok"

    result = await client_module._retry_on_capacity(flaky)
    assert result == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_non_retryable_error_propagates_immediately():
    calls = []

    async def always_401():
        calls.append(1)
        raise _nim_error(401, "Unauthorized")

    with pytest.raises(Exception, match="401"):
        await client_module._retry_on_capacity(always_401)
    assert len(calls) == 1  # no retry attempted at all


@pytest.mark.asyncio
async def test_exhausts_rate_limit_retry_budget_and_raises(monkeypatch):
    monkeypatch.setattr(client_module, "MAX_RATE_LIMIT_RETRIES", 2)
    calls = []

    async def always_429():
        calls.append(1)
        raise _nim_error(429, "Too Many Requests")

    with pytest.raises(Exception, match="429"):
        await client_module._retry_on_capacity(always_429)
    assert len(calls) == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_throttle_spaces_out_consecutive_calls(monkeypatch):
    monkeypatch.setattr(client_module, "MIN_CALL_INTERVAL_SECONDS", 0.1)
    timestamps = []

    async def instant_ok():
        timestamps.append(time.monotonic())
        return "ok"

    for _ in range(3):
        await client_module._retry_on_capacity(instant_ok)

    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    assert all(g >= 0.09 for g in gaps), gaps


@pytest.mark.asyncio
async def test_throttle_disabled_by_default_adds_no_delay():
    # MIN_CALL_INTERVAL_SECONDS=0 via the autouse fixture above.
    async def instant_ok():
        return "ok"

    start = time.monotonic()
    await client_module._retry_on_capacity(instant_ok)
    await client_module._retry_on_capacity(instant_ok)
    assert time.monotonic() - start < 0.05

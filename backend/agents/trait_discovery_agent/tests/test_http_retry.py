"""
Unit tests for kb.sources._http_retry.request_with_retry.

Regression coverage for a live finding: running scripts/pathways_scenarios.py
against real KEGG showed the LLM tool loop firing several back-to-back
fetch_pathway_name calls (one per unresolved candidate) fast enough to trip
KEGG's rate limit. Before this fix, only httpx.TimeoutException was retried —
a 429 propagated straight up, killed the LLM pick, and silently forced the
deterministic fallback even though the LLM was otherwise working fine.
"""
import httpx
import pytest

from kb.sources._http_retry import request_with_retry


class _FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class _FlakyClient:
    """Returns a scripted sequence of responses/exceptions per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def request(self, method, url, **kwargs):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.asyncio
async def test_retries_on_429_and_eventually_succeeds():
    client = _FlakyClient([
        _FakeResponse(429),
        _FakeResponse(429),
        _FakeResponse(200),
    ])
    resp = await request_with_retry(
        client, "GET", "https://rest.kegg.jp/get/hsa03320",
        attempts=3, backoff_base=0.01,
    )
    assert resp.status_code == 200
    assert client.calls == 3


@pytest.mark.asyncio
async def test_respects_retry_after_header():
    """When the server sends Retry-After, we should honor it rather than
    guessing at a backoff — verified via the sleep call, not wall time."""
    import kb.sources._http_retry as retry_module

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    client = _FlakyClient([
        _FakeResponse(429, headers={"retry-after": "2"}),
        _FakeResponse(200),
    ])

    orig_sleep = retry_module.asyncio.sleep
    retry_module.asyncio.sleep = fake_sleep
    try:
        resp = await request_with_retry(client, "GET", "https://rest.kegg.jp/get/hsa03320")
    finally:
        retry_module.asyncio.sleep = orig_sleep

    assert resp.status_code == 200
    assert slept == [2.0]


@pytest.mark.asyncio
async def test_exhausts_attempts_and_raises():
    client = _FlakyClient([
        _FakeResponse(429),
        _FakeResponse(429),
        _FakeResponse(429),
    ])
    with pytest.raises(httpx.HTTPStatusError):
        await request_with_retry(
            client, "GET", "https://rest.kegg.jp/get/hsa03320",
            attempts=3, backoff_base=0.01,
        )
    assert client.calls == 3


@pytest.mark.asyncio
async def test_non_retryable_status_raises_immediately():
    """A 404 (e.g. a bad pathway id) should not be retried — only
    timeouts/429/5xx are transient."""
    client = _FlakyClient([_FakeResponse(404)])
    with pytest.raises(httpx.HTTPStatusError):
        await request_with_retry(client, "GET", "https://rest.kegg.jp/get/bogus")
    assert client.calls == 1


@pytest.mark.asyncio
async def test_timeout_is_still_retried():
    client = _FlakyClient([
        httpx.TimeoutException("timed out"),
        _FakeResponse(200),
    ])
    resp = await request_with_retry(
        client, "GET", "https://rest.kegg.jp/get/hsa03320", backoff_base=0.01,
    )
    assert resp.status_code == 200
    assert client.calls == 2


@pytest.mark.asyncio
async def test_connect_error_is_retried():
    """Regression: a dropped/reset connection (httpx.ConnectError / other
    httpx.TransportError subclasses that aren't a TimeoutException) is just
    as transient as a timeout in Docker/WSL networking, but used to fall
    through the except clause uncaught entirely — no retry, no warning log,
    instant failure that silently killed the deterministic UniProt fallback
    even though a retry would have succeeded."""
    client = _FlakyClient([
        httpx.ConnectError("connection reset by peer"),
        _FakeResponse(200),
    ])
    resp = await request_with_retry(
        client, "GET", "https://rest.uniprot.org/uniprotkb/search", backoff_base=0.01,
    )
    assert resp.status_code == 200
    assert client.calls == 2


@pytest.mark.asyncio
async def test_connect_error_exhausts_attempts_and_raises():
    client = _FlakyClient([
        httpx.ConnectError("connection reset by peer"),
        httpx.ConnectError("connection reset by peer"),
        httpx.ConnectError("connection reset by peer"),
    ])
    with pytest.raises(httpx.ConnectError):
        await request_with_retry(
            client, "GET", "https://rest.uniprot.org/uniprotkb/search",
            attempts=3, backoff_base=0.01,
        )
    assert client.calls == 3

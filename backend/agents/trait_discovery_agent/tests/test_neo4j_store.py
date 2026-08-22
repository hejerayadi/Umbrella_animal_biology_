"""
Unit tests for kb/neo4j_store.py — hermetic, no live Neo4j required (a fake
driver/session stands in, mirroring how kb/qdrant_store.py's callers get
faked in tests/test_gene_mapper.py etc.). Live end-to-end coverage against
the docker-compose `neo4j` service lives in
tests/integration/test_neo4j_store.py.
"""
import pytest

import kb.neo4j_store as neo4j_store
from kb.neo4j_store import upsert_trait_gene_relationship


class _FakeSession:
    def __init__(self, log, fail: bool = False):
        self._log = log
        self._fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        if self._fail:
            raise RuntimeError("simulated bolt connection error")
        self._log.append(params)
        return None


class _FakeDriver:
    def __init__(self, log, fail: bool = False):
        self._log = log
        self._fail = fail

    def session(self):
        return _FakeSession(self._log, fail=self._fail)


@pytest.fixture(autouse=True)
def _reset_driver_singleton(monkeypatch):
    # get_driver() caches a module-level singleton — clear it before/after
    # every test so tests don't leak a fake driver into each other.
    monkeypatch.setattr(neo4j_store, "_driver", None)
    yield
    monkeypatch.setattr(neo4j_store, "_driver", None)


@pytest.mark.asyncio
async def test_upsert_writes_scalar_params_only(monkeypatch):
    log: list[dict] = []
    monkeypatch.setattr(neo4j_store, "_NEO4J_AVAILABLE", True)
    monkeypatch.setattr(neo4j_store, "get_driver", lambda: _FakeDriver(log))

    ok = await upsert_trait_gene_relationship("fur growth", "FGF5", "18239092")

    assert ok is True
    assert len(log) == 1
    # Only trait_name/gene_symbol/pmid ever reach the query — this is the
    # enforcement mechanism for "never cache evidence content" (§6): the
    # function has no parameter a title/short_summary could travel through.
    assert log[0] == {
        "trait_name": "fur growth",
        "gene_symbol": "FGF5",
        "pmid": "18239092",
    }


@pytest.mark.asyncio
async def test_pmid_is_stringified(monkeypatch):
    log: list[dict] = []
    monkeypatch.setattr(neo4j_store, "_NEO4J_AVAILABLE", True)
    monkeypatch.setattr(neo4j_store, "get_driver", lambda: _FakeDriver(log))

    await upsert_trait_gene_relationship("fur growth", "FGF5", 18239092)  # type: ignore[arg-type]

    assert log[0]["pmid"] == "18239092"
    assert isinstance(log[0]["pmid"], str)


@pytest.mark.asyncio
async def test_missing_field_refuses_without_touching_driver(monkeypatch):
    log: list[dict] = []
    driver_calls = []
    monkeypatch.setattr(neo4j_store, "_NEO4J_AVAILABLE", True)
    monkeypatch.setattr(
        neo4j_store, "get_driver", lambda: (driver_calls.append(1), _FakeDriver(log))[1]
    )

    ok = await upsert_trait_gene_relationship("", "FGF5", "18239092")

    assert ok is False
    assert driver_calls == []  # never even asked for a driver


@pytest.mark.asyncio
async def test_query_failure_fails_soft(monkeypatch):
    log: list[dict] = []
    monkeypatch.setattr(neo4j_store, "_NEO4J_AVAILABLE", True)
    monkeypatch.setattr(neo4j_store, "get_driver", lambda: _FakeDriver(log, fail=True))

    ok = await upsert_trait_gene_relationship("fur growth", "FGF5", "18239092")

    assert ok is False  # never raises — §9, evidence is still valid upstream


@pytest.mark.asyncio
async def test_driver_unavailable_fails_soft(monkeypatch):
    monkeypatch.setattr(neo4j_store, "_NEO4J_AVAILABLE", False)

    ok = await upsert_trait_gene_relationship("fur growth", "FGF5", "18239092")

    assert ok is False

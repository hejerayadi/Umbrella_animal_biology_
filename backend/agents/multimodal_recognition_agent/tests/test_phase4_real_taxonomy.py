"""Phase 4 - live GBIF and NCBI taxonomy, exercised entirely offline.

Every HTTP session here is injected. Nothing in this module opens a socket,
reads `.env`, or needs an account: the real providers take a `session=` argument
precisely so their parsing, degradation and annotation rules can be driven
against fabricated responses.

The scenarios mirror what the two services actually return. GBIF answers with a
`matchType` plus a `usageKey`; NCBI Entrez answers with `esearchresult.idlist`.
Both can also be slow, rate-limited or down, and each of those is a fact about
the request - never a crash, and never an invented identifier.
"""
from __future__ import annotations

import dataclasses
import json as _json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from ..adapters.reasoning_llm import FakeGPT5MiniProvider
from ..adapters.taxonomy import (
    NCBI_PUBLISHED_LIMIT_WITHOUT_KEY,
    NCBI_REQUESTS_PER_SECOND_WITH_KEY,
    NCBI_REQUESTS_PER_SECOND_WITHOUT_KEY,
    MockTaxonomyProvider,
    RealGBIFProvider,
    RealNCBIProvider,
    RealTaxonomyProvider,
    _RateLimiter,
    build_taxonomy_provider,
)
from ..agent import RecognitionAgent
from ..api import app
from ..config import ConfigError
from ..domain.models import SpeciesCandidate
from ..schema import AgentRequest, AgentStatus
from .conftest import StubClassifier, image_entry, make_config, png_bytes, prediction

NCBI_TOOL = "umbrella-recognition-agent"
NCBI_EMAIL = "contact@example.invalid"

GBIF_ENDPOINT = "https://api.gbif.org/v1/species/match"
NCBI_ENDPOINT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


# --- injected transport ----------------------------------------------------

class FakeResponse:
    def __init__(self, payload=None, status: int = 200, json_raises=None):
        self._payload = payload
        self._status = status
        self._json_raises = json_raises

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError("HTTP " + str(self._status))

    def json(self):
        if self._json_raises is not None:
            raise self._json_raises
        return self._payload


class FakeSession:
    """Records the request and returns a canned response. Sends nothing."""

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self._raises is not None:
            raise self._raises
        return self._response


def gbif(response=None, raises=None, **kwargs) -> RealGBIFProvider:
    return RealGBIFProvider(session=FakeSession(response, raises), **kwargs)


def ncbi(response=None, raises=None, **kwargs) -> RealNCBIProvider:
    return RealNCBIProvider(
        tool=NCBI_TOOL, email=NCBI_EMAIL,
        session=FakeSession(response, raises), **kwargs
    )


def gbif_ok(usage_key: int = 5219404, name: str = "Panthera leo") -> FakeResponse:
    return FakeResponse({
        "matchType": "EXACT", "usageKey": usage_key, "canonicalName": name,
        "rank": "SPECIES", "kingdom": "Animalia", "phylum": "Chordata",
        "class": "Mammalia", "order": "Carnivora", "family": "Felidae",
        "genus": "Panthera",
    })


def ncbi_ok(taxid: str = "9689") -> FakeResponse:
    return FakeResponse({"esearchresult": {"idlist": [taxid]}})


def gbif_none() -> FakeResponse:
    return FakeResponse({"matchType": "NONE"})


def ncbi_none() -> FakeResponse:
    return FakeResponse({"esearchresult": {"idlist": []}})


def candidates() -> list[SpeciesCandidate]:
    return [
        SpeciesCandidate(species_id="panthera_leo", scientific_name="Panthera leo",
                         common_name="lion", rank="species",
                         classification_score=0.91, taxonomy_status="unverified"),
        SpeciesCandidate(species_id="panthera_pardus", scientific_name="Panthera pardus",
                         common_name="leopard", rank="species",
                         classification_score=0.42, taxonomy_status="unverified"),
    ]


def identity_of(items) -> list[tuple]:
    """Everything taxonomy is forbidden to touch."""
    return [(c.species_id, c.scientific_name, c.common_name, c.rank,
             c.classification_score) for c in items]


def real_config(**overrides):
    base = make_config(taxonomy_provider_mode="real", **overrides)
    return dataclasses.replace(base, ncbi_tool=NCBI_TOOL, ncbi_email=NCBI_EMAIL)


# ===========================================================================
# GBIF response matrix
# ===========================================================================

def test_gbif_exact_match_yields_the_usage_key():
    result = gbif(gbif_ok()).lookup("panthera_leo", "Panthera leo")
    assert result.available and result.matched
    assert result.identifier == 5219404
    assert result.accepted_name == "Panthera leo"
    assert result.rank == "SPECIES"
    assert result.inconsistent is False


def test_gbif_uses_the_official_endpoint_and_a_bounded_timeout():
    provider = gbif(gbif_ok(), timeout_seconds=7.5)
    provider.lookup("panthera_leo", "Panthera leo")
    call = provider._session.calls[0]
    assert call["url"] == GBIF_ENDPOINT
    assert call["params"] == {"name": "Panthera leo"}
    assert call["timeout"] == 7.5


def test_gbif_no_match_returns_available_but_unmatched():
    result = gbif(gbif_none()).lookup("x", "Nonexistus fake")
    assert result.available and not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("match_type", ["FUZZY", "HIGHERRANK"])
def test_gbif_ambiguous_match_is_reported_and_never_borrowed(match_type):
    """A fuzzy hit is a real GBIF answer, but it is not this species. The key
    must not be taken from it - that is how one taxon acquires another's id."""
    result = gbif(FakeResponse({"matchType": match_type, "usageKey": 999})).lookup(
        "panthera_leo", "Panthera leo")
    assert result.available and not result.matched
    assert result.identifier is None
    assert result.inconsistent is True


def test_gbif_exact_match_without_a_usage_key_is_not_invented():
    result = gbif(FakeResponse({"matchType": "EXACT"})).lookup("x", "Panthera leo")
    assert result.available and not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("payload", [["list"], "text", 42, None])
def test_gbif_malformed_payload_is_contained(payload):
    result = gbif(FakeResponse(payload)).lookup("x", "Panthera leo")
    assert not result.matched
    assert result.identifier is None


def test_gbif_unparseable_json_is_contained():
    result = gbif(FakeResponse(json_raises=ValueError("not json"))).lookup(
        "x", "Panthera leo")
    assert not result.available and not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("status", [429, 503, 500, 404])
def test_gbif_http_error_degrades_without_raising(status):
    """429 rate limit and 5xx outage alike: unavailable, never an exception."""
    result = gbif(FakeResponse(None, status=status)).lookup("x", "Panthera leo")
    assert not result.available and not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("failure", [TimeoutError("timed out"),
                                     ConnectionError("reset"),
                                     OSError("network down")])
def test_gbif_transport_failure_degrades_without_raising(failure):
    result = gbif(raises=failure).lookup("x", "Panthera leo")
    assert not result.available and not result.matched
    assert result.identifier is None


# ===========================================================================
# NCBI response matrix
# ===========================================================================

def test_ncbi_exact_match_yields_the_taxid():
    result = ncbi(ncbi_ok()).lookup("panthera_leo", "Panthera leo")
    assert result.available and result.matched
    assert result.identifier == 9689
    assert result.rank == "species"


def test_ncbi_uses_the_official_endpoint_identifies_itself_and_bounds_the_call():
    provider = ncbi(ncbi_ok(), timeout_seconds=6.0)
    provider.lookup("panthera_leo", "Panthera leo")
    call = provider._session.calls[0]
    assert call["url"] == NCBI_ENDPOINT
    assert call["params"]["db"] == "taxonomy"
    assert call["params"]["term"] == '"Panthera leo"[Scientific Name]'
    # NCBI's usage guidelines require the caller to identify itself.
    assert call["params"]["tool"] == NCBI_TOOL
    assert call["params"]["email"] == NCBI_EMAIL
    assert call["timeout"] == 6.0


def test_ncbi_omits_the_api_key_when_none_is_configured():
    provider = ncbi(ncbi_ok())
    provider.lookup("panthera_leo", "Panthera leo")
    assert "api_key" not in provider._session.calls[0]["params"]


def test_ncbi_sends_the_api_key_only_when_configured():
    provider = RealNCBIProvider(tool=NCBI_TOOL, email=NCBI_EMAIL, api_key="k-123",
                                session=FakeSession(ncbi_ok()))
    provider.lookup("panthera_leo", "Panthera leo")
    assert provider._session.calls[0]["params"]["api_key"] == "k-123"


def test_ncbi_no_match_returns_available_but_unmatched():
    result = ncbi(ncbi_none()).lookup("x", "Nonexistus fake")
    assert result.available and not result.matched
    assert result.identifier is None


def test_ncbi_multiple_taxids_are_ambiguous_and_never_guessed():
    """Two taxids for one exact-quoted name is an ambiguous record. Picking the
    first would attach a real identifier to the wrong organism."""
    result = ncbi(FakeResponse({"esearchresult": {"idlist": ["9689", "12345"]}})).lookup(
        "panthera_leo", "Panthera leo")
    assert result.available and not result.matched
    assert result.identifier is None
    assert result.inconsistent is True


@pytest.mark.parametrize("payload", ["text", 42, None, {}, {"esearchresult": {}},
                                     {"esearchresult": {"idlist": "nope"}},
                                     {"esearchresult": {"idlist": ["abc"]}}])
def test_ncbi_malformed_payload_is_contained(payload):
    result = ncbi(FakeResponse(payload)).lookup("x", "Panthera leo")
    assert not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("status", [429, 503, 500])
def test_ncbi_http_error_degrades_without_raising(status):
    result = ncbi(FakeResponse(None, status=status)).lookup("x", "Panthera leo")
    assert not result.available and not result.matched
    assert result.identifier is None


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionError("reset")])
def test_ncbi_transport_failure_degrades_without_raising(failure):
    result = ncbi(raises=failure).lookup("x", "Panthera leo")
    assert not result.available and not result.matched
    assert result.identifier is None


# ===========================================================================
# Combined outcomes, and the annotation-only contract
# ===========================================================================

def test_both_services_succeed_gives_verified_and_both_identifiers():
    provider = RealTaxonomyProvider(gbif=gbif(gbif_ok()), ncbi=ncbi(ncbi_ok()))
    enriched, degraded = provider.enrich_all(candidates())

    assert degraded is False
    assert enriched[0].gbif_id == 5219404
    assert enriched[0].ncbi_taxid == 9689
    assert enriched[0].taxonomy_status == "verified"
    # `verified` and `mock_verified` are deliberately different words.
    assert enriched[0].taxonomy_status != "mock_verified"


def test_gbif_down_ncbi_up_is_partial_and_degraded():
    provider = RealTaxonomyProvider(
        gbif=gbif(FakeResponse(None, status=503)), ncbi=ncbi(ncbi_ok()))
    enriched, degraded = provider.enrich_all(candidates())

    assert degraded is True
    assert enriched[0].gbif_id is None
    assert enriched[0].ncbi_taxid == 9689
    assert enriched[0].taxonomy_status == "partial"


def test_ncbi_down_gbif_up_is_partial_and_degraded():
    provider = RealTaxonomyProvider(
        gbif=gbif(gbif_ok()), ncbi=ncbi(raises=TimeoutError("slow")))
    enriched, degraded = provider.enrich_all(candidates())

    assert degraded is True
    assert enriched[0].gbif_id == 5219404
    assert enriched[0].ncbi_taxid is None
    assert enriched[0].taxonomy_status == "partial"


def test_total_outage_leaves_every_identifier_null():
    provider = RealTaxonomyProvider(
        gbif=gbif(raises=ConnectionError("down")),
        ncbi=ncbi(raises=ConnectionError("down")))
    enriched, degraded = provider.enrich_all(candidates())

    assert degraded is True
    assert all(c.gbif_id is None and c.ncbi_taxid is None for c in enriched)
    assert all(c.taxonomy_status == "unverified" for c in enriched)


def test_no_match_from_either_service_is_not_a_degradation():
    """Both services answered. A name absent from a database is an answer, not
    an outage - the distinction is what the degraded flag exists to record."""
    provider = RealTaxonomyProvider(gbif=gbif(gbif_none()), ncbi=ncbi(ncbi_none()))
    enriched, degraded = provider.enrich_all(candidates())

    assert degraded is False
    assert enriched[0].gbif_id is None and enriched[0].ncbi_taxid is None
    assert enriched[0].taxonomy_status == "unverified"


@pytest.mark.parametrize("scenario", ["both_ok", "gbif_only", "ncbi_only", "neither",
                                      "ambiguous", "outage"])
def test_taxonomy_never_changes_candidates_order_identity_or_scores(scenario):
    """The whole safety property in one test: taxonomy annotates, full stop."""
    setups = {
        "both_ok": (gbif_ok(), ncbi_ok()),
        "gbif_only": (gbif_ok(), ncbi_none()),
        "ncbi_only": (gbif_none(), ncbi_ok()),
        "neither": (gbif_none(), ncbi_none()),
        "ambiguous": (FakeResponse({"matchType": "FUZZY", "usageKey": 1}),
                      FakeResponse({"esearchresult": {"idlist": ["1", "2"]}})),
        "outage": (FakeResponse(None, status=503), FakeResponse(None, status=503)),
    }
    g, n = setups[scenario]
    original = candidates()
    before = identity_of(original)

    provider = RealTaxonomyProvider(gbif=gbif(g), ncbi=ncbi(n))
    enriched, _ = provider.enrich_all(original)

    assert len(enriched) == len(original)           # never added or dropped
    assert identity_of(enriched) == before          # never reordered or rescored
    assert [c.species_id for c in enriched] == ["panthera_leo", "panthera_pardus"]


def test_an_identifier_is_never_borrowed_by_the_second_candidate():
    """Both candidates are looked up separately; a hit for one must not leak
    onto the other."""

    class PerNameSession(FakeSession):
        def get(self, url, params=None, timeout=None):
            self.calls.append({"url": url, "params": params, "timeout": timeout})
            name = (params or {}).get("name") or ""
            return gbif_ok() if "Panthera leo" in name else gbif_none()

    provider = RealTaxonomyProvider(
        gbif=RealGBIFProvider(session=PerNameSession()),
        ncbi=ncbi(ncbi_none()))
    enriched, _ = provider.enrich_all(candidates())

    assert enriched[0].gbif_id == 5219404
    assert enriched[1].gbif_id is None


def test_the_per_candidate_report_separates_the_two_sources():
    provider = RealTaxonomyProvider(
        gbif=gbif(gbif_ok()), ncbi=ncbi(FakeResponse(None, status=503)))
    _, report = provider.validate_candidate(candidates()[0])

    assert report["gbif"]["mode"] == "real" and report["gbif"]["available"] is True
    assert report["ncbi"]["mode"] == "real" and report["ncbi"]["available"] is False
    assert report["status"] == "partial"


# ===========================================================================
# Provider selection, and the facade the agent needs
# ===========================================================================

def test_real_mode_builds_the_live_provider_and_opens_no_session():
    built = build_taxonomy_provider(real_config())
    assert isinstance(built, RealTaxonomyProvider)
    assert not isinstance(built, MockTaxonomyProvider)
    assert built.gbif._session is None and built.ncbi._session is None


def test_real_mode_never_falls_back_to_the_fixture_provider():
    with pytest.raises(ConfigError):
        build_taxonomy_provider(make_config(taxonomy_provider_mode="real"))


def test_the_real_provider_exposes_the_same_facade_the_agent_requires():
    """`agent.py` calls `known_names()` while building its text analyser. A
    provider missing it cannot start the agent at all."""
    built = build_taxonomy_provider(real_config())
    for method in ("validate_candidate", "enrich", "enrich_all",
                   "known_names", "resolve_name"):
        assert callable(getattr(built, method)), method


def test_the_real_provider_reports_an_empty_local_catalogue():
    """Real mode owns no offline name list. Empty is the honest answer - the
    alternative is shipping fixture names into a live deployment."""
    built = build_taxonomy_provider(real_config())
    assert built.known_names() == {}
    assert built.resolve_name("Panthera leo") is None
    assert built.resolve_name("lion") is None
    assert built.resolve_name("") is None


def test_the_name_catalogue_opens_no_connection():
    provider = RealTaxonomyProvider(gbif=gbif(gbif_ok()), ncbi=ncbi(ncbi_ok()))
    provider.known_names()
    provider.resolve_name("Panthera leo")
    assert provider.gbif._session.calls == []
    assert provider.ncbi._session.calls == []


def test_the_agent_constructs_in_real_taxonomy_mode():
    """The defect this module exists to prevent: real mode used to raise
    AttributeError before the agent could be built."""
    agent = RecognitionAgent(real_config())
    assert isinstance(agent._workflow._taxonomy, RealTaxonomyProvider)


def test_agent_construction_in_real_mode_opens_no_connection():
    agent = RecognitionAgent(real_config())
    taxonomy = agent._workflow._taxonomy
    assert taxonomy.gbif._session is None and taxonomy.ncbi._session is None


# ===========================================================================
# Full agent, real taxonomy path, deterministic GPT, injected HTTP
# ===========================================================================

def run_agent(gbif_response, ncbi_response, *, raises=None):
    classifier = StubClassifier([
        prediction("panthera_leo", 0.91, "Panthera leo"),
        prediction("panthera_pardus", 0.42, "Panthera pardus"),
    ])
    taxonomy = RealTaxonomyProvider(
        gbif=gbif(gbif_response, raises=raises),
        ncbi=ncbi(ncbi_response, raises=raises),
    )
    agent = RecognitionAgent(real_config(), classifier=classifier,
                             taxonomy_provider=taxonomy,
                             reasoning_llm=FakeGPT5MiniProvider())
    return agent.run(AgentRequest(
        instruction="Identify this species.",
        context={"recognition_image": image_entry(png_bytes())},
    ))


def test_the_full_agent_completes_with_live_taxonomy_annotations():
    result = run_agent(gbif_ok(), ncbi_ok())

    assert result.status is AgentStatus.COMPLETED
    assert result.output["species"] == "Panthera leo"
    assert result.output["gbif_id"] == 5219404
    assert result.output["ncbi_taxid"] == 9689


def test_the_full_agent_returns_exactly_seven_top_level_keys():
    result = run_agent(gbif_ok(), ncbi_ok())
    assert sorted(result.output) == [
        "gbif_id", "ncbi_taxid", "recognition", "recognition_candidates",
        "recognition_provenance", "species", "species_id",
    ]
    assert len(result.output) == 7


def test_a_total_taxonomy_outage_still_completes_with_null_identifiers():
    result = run_agent(None, None, raises=ConnectionError("both down"))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["gbif_id"] is None
    assert result.output["ncbi_taxid"] is None
    assert result.output["recognition_provenance"]["taxonomy_degraded"] is True
    assert len(result.output) == 7


def test_taxonomy_outcome_never_changes_the_scientific_decision():
    """Classification and confidence run BEFORE taxonomy. Whatever the services
    say - or fail to say - the decision and the candidates are identical."""
    verified = run_agent(gbif_ok(), ncbi_ok())
    outage = run_agent(None, None, raises=ConnectionError("down"))

    def evidence(result):
        return [(c["species_id"], c["scientific_name"], c["classification_score"])
                for c in result.output["recognition_candidates"]]

    assert evidence(verified) == evidence(outage)
    assert verified.output["recognition"]["decision"] == outage.output["recognition"]["decision"]
    assert verified.output["species"] == outage.output["species"]
    assert verified.output["species_id"] == outage.output["species_id"]


def test_a_taxonomy_failure_is_never_an_http_500():
    """The orchestrator parses one schema. A 500 with FastAPI's error body
    would break it, so even a dead taxonomy service must return 200."""
    client = TestClient(app)
    response = client.post("/execute", json={
        "instruction": "Identify this species.",
        "context": {"recognition_image": image_entry(png_bytes())},
    })
    assert response.status_code == 200
    assert response.json()["status"] in ("completed", "needs_agent", "continue", "failed")


def test_no_raw_service_error_reaches_the_output():
    result = run_agent(None, None, raises=ConnectionError("gbif-internal-detail-xyz"))
    serialized = _json.dumps(result.output, default=str)
    assert "gbif-internal-detail-xyz" not in serialized
    assert "ConnectionError" not in serialized


# ===========================================================================
# NCBI rate limiting
#
# Enriching a Top-K list sends one NCBI request per candidate, back to back.
# Measured against the live service, 2 of 8 rapid requests came back
# rate-limited and a healthy species lost its taxid. The limiter spaces the
# requests instead; nothing here sleeps for real.
# ===========================================================================

class FakeClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self, start: float = 1000.0):
        self.now = float(start)
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        with self._lock:
            self.slept.append(duration)
            self.now += duration


def limiter(rate: float, clock: FakeClock) -> _RateLimiter:
    return _RateLimiter(rate, monotonic=clock.monotonic, sleep=clock.sleep)


def test_the_unkeyed_rate_sits_below_the_published_limit():
    """Pacing at exactly the documented cap left no margin - a live burst still
    saw 1 of 8 requests rejected at a measured 2.86/second. The send rate is
    therefore deliberately lower than what NCBI publishes."""
    assert NCBI_PUBLISHED_LIMIT_WITHOUT_KEY == 3.0
    assert NCBI_REQUESTS_PER_SECOND_WITHOUT_KEY == 2.0
    assert NCBI_REQUESTS_PER_SECOND_WITHOUT_KEY < NCBI_PUBLISHED_LIMIT_WITHOUT_KEY


def test_the_keyed_rate_is_unchanged_at_the_published_ten_per_second():
    """An API key gives the deployment its own budget, so no margin is needed."""
    assert NCBI_REQUESTS_PER_SECOND_WITH_KEY == 10.0


def test_an_unkeyed_provider_is_limited_to_two_per_second():
    provider = ncbi(ncbi_ok())
    assert provider.requests_per_second == NCBI_REQUESTS_PER_SECOND_WITHOUT_KEY == 2.0
    assert provider._rate_limiter.min_interval == pytest.approx(0.5)


def test_a_keyed_provider_is_limited_to_ten_per_second():
    provider = ncbi(ncbi_ok(), api_key="k-123")
    assert provider.requests_per_second == 10.0
    assert provider._rate_limiter.min_interval == pytest.approx(0.1)


@pytest.mark.parametrize(("rate", "calls"), [(2.0, 8), (10.0, 12)])
def test_requests_are_spaced_to_the_configured_rate(rate, calls):
    clock = FakeClock()
    lim = limiter(rate, clock)
    start = clock.now

    stamps = []
    for _ in range(calls):
        lim.acquire()
        stamps.append(clock.now)

    interval = 1.0 / rate
    # The first call is free; every later one is at least one interval on.
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(gap >= interval - 1e-9 for gap in gaps), gaps
    assert clock.now - start == pytest.approx(interval * (calls - 1))
    # Never faster than the published rate, across the whole window.
    elapsed = clock.now - start
    assert (calls - 1) / elapsed <= rate + 1e-9


def test_the_unkeyed_burst_that_broke_live_lookups_is_now_spaced():
    """Eight rapid requests - the exact shape that lost a taxid live."""
    clock = FakeClock()
    provider = RealNCBIProvider(
        tool=NCBI_TOOL, email=NCBI_EMAIL, session=FakeSession(ncbi_ok()),
        monotonic=clock.monotonic, sleep=clock.sleep)

    for _ in range(8):
        result = provider.lookup("panthera_leo", "Panthera leo")
        assert result.identifier == 9689  # none lost

    elapsed = clock.now - 1000.0
    assert elapsed == pytest.approx(0.5 * 7)
    # Inside our own send rate, and comfortably inside NCBI's published cap.
    assert 7 / elapsed <= NCBI_REQUESTS_PER_SECOND_WITHOUT_KEY + 1e-9
    assert 7 / elapsed < NCBI_PUBLISHED_LIMIT_WITHOUT_KEY


def test_the_first_request_is_not_delayed():
    clock = FakeClock()
    assert limiter(3.0, clock).acquire() == 0.0
    assert clock.slept == []


def test_the_limiter_uses_monotonic_time_not_the_wall_clock():
    """A wall-clock adjustment mid-request could push the deadline backwards
    and let a burst through, or forwards and stall one."""
    import inspect

    source = inspect.getsource(_RateLimiter)
    assert "time.time" not in source
    assert "datetime" not in source

    from ..adapters import taxonomy as module

    assert _RateLimiter(3.0)._monotonic is module.time.monotonic


def test_the_injected_clock_is_actually_the_one_consulted():
    clock = FakeClock()
    lim = limiter(3.0, clock)
    lim.acquire()
    clock.now += 10.0        # jump forward: the next call owes no wait
    assert lim.acquire() == 0.0
    assert clock.slept == []


def test_concurrent_callers_cannot_bypass_the_limiter():
    """Threads must queue behind the same deadline. If the lock were released
    before the wait, they would all wake together and burst."""
    clock = FakeClock()
    lim = limiter(3.0, clock)
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            lim.acquire()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    # Six concurrent callers, five of them spaced by one interval.
    assert len(clock.slept) == 5
    assert clock.now - 1000.0 == pytest.approx((1 / 3) * 5)


def test_the_offline_tests_never_really_sleep():
    """Wall-clock proof: 30 limited calls at 3/s would take ~10 real seconds."""
    clock = FakeClock()
    lim = limiter(3.0, clock)
    started = time.monotonic()
    for _ in range(30):
        lim.acquire()
    real_elapsed = time.monotonic() - started

    assert clock.now - 1000.0 == pytest.approx((1 / 3) * 29)  # simulated
    assert real_elapsed < 1.0                                  # actual


def test_rate_limiting_adds_no_retry():
    """Spacing replaces retrying - it must not quietly become a retry."""
    import inspect

    source = inspect.getsource(_RateLimiter).lower()
    for forbidden in ("retry", "attempt", "while true", "for _ in range"):
        assert forbidden not in source

    provider = ncbi(FakeResponse(None, status=429))
    provider.lookup("panthera_leo", "Panthera leo")
    assert len(provider._session.calls) == 1  # one attempt, never a second


def test_a_real_429_still_degrades_to_a_null_identifier():
    """The limiter reduces 429s; it cannot promise none. A rate-limited answer
    must still be a null identifier, not a crash and not a guess."""
    provider = ncbi(FakeResponse(None, status=429))
    result = provider.lookup("panthera_leo", "Panthera leo")

    assert not result.available and not result.matched
    assert result.identifier is None


def test_rate_limiting_preserves_candidates_order_scores_and_status():
    clock = FakeClock()
    original = candidates()
    before = identity_of(original)

    provider = RealTaxonomyProvider(
        gbif=gbif(gbif_ok()),
        ncbi=RealNCBIProvider(tool=NCBI_TOOL, email=NCBI_EMAIL,
                              session=FakeSession(ncbi_ok()),
                              monotonic=clock.monotonic, sleep=clock.sleep))
    enriched, degraded = provider.enrich_all(original)

    assert identity_of(enriched) == before
    assert [c.species_id for c in enriched] == ["panthera_leo", "panthera_pardus"]
    assert degraded is False
    assert all(c.taxonomy_status == "verified" for c in enriched)
    # Two candidates, so exactly one wait between the two NCBI requests.
    assert clock.slept == [pytest.approx(0.5)]


def test_the_seven_key_contract_survives_rate_limited_enrichment():
    clock = FakeClock()
    classifier = StubClassifier([
        prediction("panthera_leo", 0.91, "Panthera leo"),
        prediction("panthera_pardus", 0.42, "Panthera pardus"),
    ])
    taxonomy = RealTaxonomyProvider(
        gbif=gbif(gbif_ok()),
        ncbi=RealNCBIProvider(tool=NCBI_TOOL, email=NCBI_EMAIL,
                              session=FakeSession(ncbi_ok()),
                              monotonic=clock.monotonic, sleep=clock.sleep))
    agent = RecognitionAgent(real_config(), classifier=classifier,
                             taxonomy_provider=taxonomy,
                             reasoning_llm=FakeGPT5MiniProvider())
    result = agent.run(AgentRequest(
        instruction="Identify this species.",
        context={"recognition_image": image_entry(png_bytes())},
    ))

    assert result.status is AgentStatus.COMPLETED
    assert sorted(result.output) == [
        "gbif_id", "ncbi_taxid", "recognition", "recognition_candidates",
        "recognition_provenance", "species", "species_id",
    ]
    assert result.output["ncbi_taxid"] == 9689
    assert clock.slept  # the limiter really was engaged on this path


def test_the_limiter_refuses_a_nonsensical_rate():
    for bad in (0, -1, -0.5):
        with pytest.raises(ValueError):
            _RateLimiter(bad)


def test_gbif_is_not_rate_limited():
    """Only NCBI publishes a request-rate guideline. GBIF is left alone rather
    than slowed down for no stated reason."""
    provider = gbif(gbif_ok())
    assert not hasattr(provider, "_rate_limiter")

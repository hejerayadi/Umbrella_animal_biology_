"""Integration tests for the Reconstruction Agent end-to-end flow.

These tests are marked ``@pytest.mark.integration`` and are excluded from
the default ``pytest`` run.  Run them explicitly with::

    pytest tests/integration/ -m integration -v

Test 1 (this file) verifies that a POST to the Reconstruction Agent's
``/execute`` endpoint is accepted and returns HTTP 200.  Because no Global
Orchestrator is running in the CI environment, the test drives the FastAPI app
directly via Starlette's ``TestClient`` — this exercises the full request
path (HTTP boundary → LangGraph graph → nodes) without requiring any external
services.

Requirements: 11.1
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / external dependencies before the agent modules are imported so
# the tests can run without an Azure OpenAI key or a live Qdrant instance.
# Use ``setdefault`` so that an already-loaded stub (e.g. from conftest.py
# or another test module imported first) is not overwritten.
# ---------------------------------------------------------------------------

# 1. LLM dependency
_llm_mod = ModuleType("backend.orchestrator.llm")
_llm_mod.get_llm = lambda: MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("backend.orchestrator.llm", _llm_mod)

# ---------------------------------------------------------------------------
# Now it's safe to import the FastAPI app.
# ---------------------------------------------------------------------------
from starlette.testclient import TestClient  # noqa: E402

from backend.agents.reconstruction_agent.api import app  # noqa: E402
from backend.agents.reconstruction_agent.schema import GapPrediction  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router_mock(
    predicted_sequence: str,
    confidence: float,
    model_used: str = "Azure-GPT-5.1",
):
    """Patch DNAModelRouter.get() so predict() returns a fixed GapPrediction."""
    mock_router_instance = MagicMock()

    def _predict(flanks, gap, previous_attempts):
        return GapPrediction(
            gap=gap,
            predicted_sequence=predicted_sequence[: gap.length].ljust(gap.length, "N"),
            model_used=model_used,
            confidence=confidence,
        )

    mock_router_instance.predict.side_effect = _predict

    return patch(
        "backend.agents.reconstruction_agent.agent.DNAModelRouter.get",
        return_value=mock_router_instance,
    )


def _make_agent_request(genome: str = "ACGTACGT" + "N" * 15 + "ACGTACGT") -> dict:
    """Build a minimal valid AgentRequest payload as a JSON-serialisable dict."""
    return {
        "instruction": "Reconstruct the missing genome region.",
        "context": {
            "session_id": "test-session-integration-001",
            "genome": genome,
            "species": "test_species",
            "species_metadata": {
                "species_id": "test_species",
                "is_extinct": False,
            },
            "sequence_type": "nuclear",
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — POST /execute reaches Reconstruction Agent
# Validates: Requirements 11.1
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_post_execute_returns_http_200():
    """POST a valid AgentRequest to /execute and assert HTTP 200 is returned.

    The Reconstruction Agent's /execute endpoint must accept the request,
    run the internal LangGraph graph, and return a ReconstructionResult
    (any status) with HTTP 200.  This test confirms the routing layer is
    wired correctly and that the agent's HTTP boundary does not crash on a
    well-formed request.

    Validates: Requirements 11.1
    """
    # Mock the LLM so no real Azure call is made; return a valid JSON prediction.
    mock_llm_response = MagicMock()
    mock_llm_response.content = '{"predicted_sequence": "ACGTACGTACGTACG", "confidence": 0.9}'

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response

    with patch("backend.orchestrator.llm.get_llm", return_value=mock_llm):
        # Also patch the RetrievalPipeline so Qdrant is not required.
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/execute", json=_make_agent_request())

    assert response.status_code == 200, (
        f"Expected HTTP 200 from /execute, got {response.status_code}. "
        f"Response body: {response.text}"
    )


@pytest.mark.integration
def test_post_execute_response_body_is_valid_reconstruction_result():
    """The response body from /execute must be a valid ReconstructionResult JSON.

    Checks the key fields mandated by the agent contract so callers can rely
    on a stable, schema-conformant payload.

    Validates: Requirements 11.1
    """
    mock_llm_response = MagicMock()
    mock_llm_response.content = '{"predicted_sequence": "ACGTACGTACGTACG", "confidence": 0.9}'

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response

    with patch("backend.orchestrator.llm.get_llm", return_value=mock_llm):
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/execute", json=_make_agent_request())

    assert response.status_code == 200
    body = response.json()

    # The response must contain the required top-level fields from ReconstructionResult.
    assert "status" in body, "Response must include 'status' field"
    assert body["status"] in ("completed", "needs_agent", "continue", "failed"), (
        f"Unexpected status value: {body['status']}"
    )
    # A well-formed request with a genome present should not return 'failed'.
    assert body["status"] != "failed", (
        f"Agent returned FAILED unexpectedly: {body}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Extinct species emits NEEDS_AGENT
# Validates: Requirements 11.2, 4.1
# ---------------------------------------------------------------------------

def _make_extinct_agent_request() -> dict:
    """Build an AgentRequest for an extinct species with no evolution_analysis in context.

    The genome is crafted to contain exactly one in-scope N-run (20 bp) so that
    the graph reaches model_selector_predictor → validation_engine, where the
    NEEDS_AGENT check lives.  No ``evolution_analysis`` key is included.
    """
    return {
        "instruction": "Reconstruct the missing genome region for an extinct species.",
        "context": {
            "session_id": "test-session-integration-002",
            # In-scope gap: 20 N characters (10 ≤ 20 ≤ 500)
            "genome": "ACGTACGTACGT" + "N" * 20 + "ACGTACGTACGT",
            "species": "mammoth",
            "species_metadata": {
                "species_id": "mammuthus_primigenius",
                "is_extinct": True,
            },
            "sequence_type": "nuclear",
            # Deliberately omit "evolution_analysis" to trigger NEEDS_AGENT
        },
    }


@pytest.mark.integration
def test_extinct_species_emits_needs_agent():
    """POST an extinct-species AgentRequest with no evolution_analysis in context.

    The Reconstruction Agent must emit NEEDS_AGENT targeting the Evolution
    Agent so the Global Orchestrator can suspend the Reconstruction Agent and
    route to Evolution Agent for cross-species analysis.

    The test drives the FastAPI app directly via TestClient so no external
    services are needed.  The LLM is mocked to return a high-confidence
    prediction, which ensures the request reaches validation_engine (where
    the NEEDS_AGENT check lives) rather than failing at the LLM call.

    Validates: Requirements 11.2, 4.1
    """
    with _make_router_mock("ACGTACGTACGTACGTACGT", 0.95):
        # Patch RetrievalPipeline so no Qdrant instance is required.
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/execute", json=_make_extinct_agent_request())

    assert response.status_code == 200, (
        f"Expected HTTP 200 from /execute, got {response.status_code}. "
        f"Response body: {response.text}"
    )

    body = response.json()

    # --- Core assertions per task spec ---

    # 1. status must be NEEDS_AGENT
    assert body.get("status") == "needs_agent", (
        f"Expected status='needs_agent' for extinct species with no evolution_analysis, "
        f"got status='{body.get('status')}'. Full response: {body}"
    )

    # 2. target_agent must be "Evolution"
    assert body.get("target_agent") == "Evolution", (
        f"Expected target_agent='Evolution', got '{body.get('target_agent')}'. "
        f"Full response: {body}"
    )

    # 3. prompt_to_target_agent must be non-empty
    prompt = body.get("prompt_to_target_agent", "")
    assert isinstance(prompt, str) and prompt.strip(), (
        f"Expected a non-empty prompt_to_target_agent string, "
        f"got '{prompt}'. Full response: {body}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Evolution Agent resumption flow
# Validates: Requirements 11.3, 11.4, 4.3, 4.4
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_evolution_agent_resumption_flow():
    """Verify the full NEEDS_AGENT → Evolution Agent → resumption cycle.

    Step 1: POST an extinct-species request with no ``evolution_analysis``; assert
            the agent emits NEEDS_AGENT and capture the ``session_id``.
    Step 2: Simulate the mock Evolution Agent returning a non-empty
            ``evolution_analysis`` string.
    Step 3: Resume the Reconstruction Agent by POSTing the same request enriched
            with ``evolution_analysis`` in context, using the *same* ``session_id``
            (which becomes the LangGraph ``thread_id``).  Assert that the final
            result has ``status == COMPLETED`` and a non-empty
            ``reconstructed_sequence``.

    The test drives the FastAPI app directly via ``TestClient`` so no external
    services (Global Orchestrator, Evolution Agent port, Qdrant) are required.
    The MemorySaver checkpointer preserves the LangGraph state across the two
    calls when the same ``thread_id`` is reused.

    Validates: Requirements 11.3, 11.4, 4.3, 4.4
    """
    # Shared session_id — used as LangGraph thread_id in both invocations.
    SESSION_ID = "test-session-integration-003-resumption"

    # Genome with one in-scope N-run (20 bp, i.e. 10 ≤ 20 ≤ 500).
    GENOME = "ACGTACGTACGT" + "N" * 20 + "ACGTACGTACGT"

    # ---------------------------------------------------------------------------
    # Step 1 — POST extinct-species request; expect NEEDS_AGENT
    # ---------------------------------------------------------------------------
    first_request = {
        "instruction": "Reconstruct the missing genome region for an extinct species.",
        "context": {
            "session_id": SESSION_ID,
            "genome": GENOME,
            "species": "mammoth",
            "species_metadata": {
                "species_id": "mammuthus_primigenius",
                "is_extinct": True,
            },
            "sequence_type": "nuclear",
            # No "evolution_analysis" — triggers NEEDS_AGENT
        },
    }

    with _make_router_mock("ACGTACGTACGTACGTACGT", 0.95):
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            client = TestClient(app, raise_server_exceptions=False)
            step1_response = client.post("/execute", json=first_request)

    assert step1_response.status_code == 200, (
        f"Step 1: expected HTTP 200, got {step1_response.status_code}. "
        f"Body: {step1_response.text}"
    )
    step1_body = step1_response.json()
    assert step1_body.get("status") == "needs_agent", (
        f"Step 1: expected status='needs_agent', got '{step1_body.get('status')}'. "
        f"Full response: {step1_body}"
    )
    assert step1_body.get("target_agent") == "Evolution", (
        f"Step 1: expected target_agent='Evolution', "
        f"got '{step1_body.get('target_agent')}'"
    )

    # ---------------------------------------------------------------------------
    # Step 2 — Simulate mock Evolution Agent returning evolution_analysis
    # ---------------------------------------------------------------------------
    # Any non-empty string is a valid Evolution Agent response (Req 4.6).
    mock_evolution_analysis = (
        "Cross-species evolutionary analysis for mammuthus_primigenius: "
        "Genomic similarity to Loxodonta africana is 98.5% in the target region. "
        "Predicted sequence is consistent with elephant-lineage ancestral patterns."
    )
    assert mock_evolution_analysis.strip(), (
        "Step 2: mock evolution_analysis must be non-empty to satisfy Req 4.6"
    )

    # ---------------------------------------------------------------------------
    # Step 3 — Resume Reconstruction Agent with evolution_analysis injected
    # ---------------------------------------------------------------------------
    # Re-use the *same* SESSION_ID so LangGraph's MemorySaver checkpointer
    # restores the suspended graph state (validated_genome + gaps already set).
    resumption_request = {
        "instruction": "Reconstruct the missing genome region for an extinct species.",
        "context": {
            "session_id": SESSION_ID,   # same thread_id → MemorySaver resumption
            "genome": GENOME,
            "species": "mammoth",
            "species_metadata": {
                "species_id": "mammuthus_primigenius",
                "is_extinct": True,
            },
            "sequence_type": "nuclear",
            "evolution_analysis": mock_evolution_analysis,  # injected by Orchestrator
        },
    }

    with _make_router_mock("ACGTACGTACGTACGTACGT", 0.95):
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            # Re-use the same TestClient instance is not possible across separate
            # ``with`` blocks; the app-level MemorySaver lives in the ``graph``
            # module global, so a new TestClient pointing at the same ``app``
            # object will reuse the same in-memory checkpointer.
            client2 = TestClient(app, raise_server_exceptions=False)
            step3_response = client2.post("/execute", json=resumption_request)

    assert step3_response.status_code == 200, (
        f"Step 3: expected HTTP 200, got {step3_response.status_code}. "
        f"Body: {step3_response.text}"
    )
    step3_body = step3_response.json()

    # --- Core assertions (Req 11.4, 4.4) ---
    assert step3_body.get("status") == "completed", (
        f"Step 3: expected status='completed' after Evolution Agent resumption, "
        f"got '{step3_body.get('status')}'. Full response: {step3_body}"
    )

    reconstructed_sequence = step3_body.get("reconstructed_sequence", "")
    assert isinstance(reconstructed_sequence, str) and reconstructed_sequence.strip(), (
        f"Step 3: expected non-empty reconstructed_sequence after resumption, "
        f"got '{reconstructed_sequence}'. Full response: {step3_body}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Retrieval Pipeline returns ≥ 1000 bp flanking context
# Validates: Requirements 11.5, 9.3
# ---------------------------------------------------------------------------

class _FakeQdrantClient:
    """In-process stand-in for QdrantClient used in Test 4.

    Returns a fixed list of records ignoring server-side filters so the
    Python-level filtering logic inside ``get_flanking_context`` is exercised.
    Each record exposes a ``payload`` dict with ``window_sequence`` and
    ``is_extinct`` fields, matching the format expected by
    ``RetrievalPipeline.get_flanking_context``.
    """

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def search(
        self,
        *,
        collection_name: str,
        query_vector,
        limit: int,
        filter=None,  # noqa: A002
    ) -> list[dict]:
        """Return all stored records up to *limit* regardless of filter."""
        return self._records[:limit]


def _make_window_record(
    species_id: str,
    sequence: str,
    is_extinct: bool = False,
) -> dict:
    """Build a minimal Qdrant-style hit dict for a single 2000 bp window."""
    return {
        "payload": {
            "species_id": species_id,
            "is_extinct": is_extinct,
            "window_sequence": sequence,
            "accession": "GCF_000001905.1",
            "chromosome": "chr1",
            "gap_start": 1000,
            "gap_end": 1050,
            "gap_length": 50,
            "window_start": 0,
            "window_end": 2000,
        },
        "score": 0.95,
    }


@pytest.mark.integration
def test_retrieval_pipeline_returns_1000bp_flanking_context():
    """Pre-populate a FakeQdrantClient with 5 Windows and verify ≥ 1000 bp context.

    The Retrieval Pipeline must expand the flanking context supplied to the
    LLM prompt from the immediate sequence slice up to 1000 bp by concatenating
    retrieved Window sequences.

    Setup:
    - 5 living-species windows, each 2000 bp (all ACGT, no Ns), injected via
      a ``_FakeQdrantClient`` so no live Qdrant instance is required.
    - Each 2000 bp window contributes 1000 bp to the left pool and 1000 bp to
      the right pool (split at the midpoint).
    - With 5 windows the raw pools are 5000 bp each; the pipeline caps them at
      1000 bp, so both returned contexts must be exactly 1000 bp.

    Assertions:
    - ``len(left_context) >= 1000``
    - ``len(right_context) >= 1000``

    Validates: Requirements 11.5, 9.3
    """
    from backend.agents.reconstruction_agent.retrieval_pipeline import RetrievalPipeline

    SPECIES_ID = "loxodonta_africana"
    WINDOW_LEN = 2000  # each window is exactly 2000 bp
    NUM_WINDOWS = 5    # five windows → raw pool of 5 × 1000 = 5000 bp each side

    # Build 5 deterministic ACGT sequences so test output is reproducible.
    # Each window sequence is 2000 bp; the pipeline takes the first 1000 bp as
    # the left contribution and the last 1000 bp as the right contribution.
    window_sequences = [
        ("ACGT" * 250 + "TGCA" * 250)[:WINDOW_LEN],  # window 0
        ("GGCC" * 250 + "CCGG" * 250)[:WINDOW_LEN],  # window 1
        ("ATAT" * 250 + "TATA" * 250)[:WINDOW_LEN],  # window 2
        ("CCAA" * 250 + "TTGG" * 250)[:WINDOW_LEN],  # window 3
        ("GCGC" * 250 + "CGCG" * 250)[:WINDOW_LEN],  # window 4
    ]
    assert len(window_sequences) == NUM_WINDOWS
    assert all(len(s) == WINDOW_LEN for s in window_sequences)

    records = [
        _make_window_record(SPECIES_ID, seq, is_extinct=False)
        for seq in window_sequences
    ]

    fake_client = _FakeQdrantClient(records)

    # Build the cleaned sequence that provides the gap context.
    # The gap sits at positions [500, 550) — well within the sequence.
    LEFT_FLANK = "A" * 500
    GAP = "N" * 50
    RIGHT_FLANK = "T" * 500
    cleaned_sequence = LEFT_FLANK + GAP + RIGHT_FLANK
    gap_start = len(LEFT_FLANK)        # 500
    gap_end = gap_start + len(GAP)     # 550

    # Instantiate the pipeline and inject the fake client.
    pipeline = RetrievalPipeline(
        qdrant_url="http://fake-qdrant:6333",
        collection_name="reconstruction_sequence_windows",
        top_k=NUM_WINDOWS,
    )
    pipeline.client = fake_client  # bypass _get_client()

    left_context, right_context = pipeline.get_flanking_context(
        species_id=SPECIES_ID,
        gap_start=gap_start,
        gap_end=gap_end,
        cleaned_sequence=cleaned_sequence,
        species_metadata={"species_id": SPECIES_ID, "is_extinct": False},
    )

    # --- Core assertions (Req 11.5, 9.3) ---
    assert len(left_context) >= 1000, (
        f"Expected left_context length >= 1000 bp, "
        f"got {len(left_context)} bp. "
        f"left_context={left_context[:60]!r}..."
    )
    assert len(right_context) >= 1000, (
        f"Expected right_context length >= 1000 bp, "
        f"got {len(right_context)} bp. "
        f"right_context={right_context[:60]!r}..."
    )

    # Sanity check: contexts must consist only of valid nucleotide characters.
    valid_bases = set("ACGTNacgtn")
    assert all(c in valid_bases for c in left_context), (
        "left_context contains unexpected characters."
    )
    assert all(c in valid_bases for c in right_context), (
        "right_context contains unexpected characters."
    )


@pytest.mark.integration
def test_retrieval_pipeline_near_sequence_boundary():
    """Verify flanking context at a gap near the sequence edge is tolerated.

    When the gap is positioned near the start of the sequence the direct-slice
    fallback and/or the retrieved context will be shorter than 1000 bp.  The
    test confirms the pipeline does not raise an exception in this case and that
    any returned context is non-negative in length (i.e. empty strings are
    acceptable when the sequence boundary is reached).

    This covers the '(or sequence boundary if near edge)' clause from the task
    description.

    Validates: Requirements 11.5, 9.3
    """
    from backend.agents.reconstruction_agent.retrieval_pipeline import RetrievalPipeline

    SPECIES_ID = "loxodonta_africana"
    WINDOW_LEN = 2000
    NUM_WINDOWS = 5

    window_sequences = [("ACGT" * 500)[:WINDOW_LEN] for _ in range(NUM_WINDOWS)]
    records = [
        _make_window_record(SPECIES_ID, seq, is_extinct=False)
        for seq in window_sequences
    ]
    fake_client = _FakeQdrantClient(records)

    # Gap at position 10 — very close to the start, so left context is short.
    cleaned_sequence = "A" * 10 + "N" * 30 + "T" * 500
    gap_start = 10
    gap_end = 40

    pipeline = RetrievalPipeline(
        qdrant_url="http://fake-qdrant:6333",
        collection_name="reconstruction_sequence_windows",
        top_k=NUM_WINDOWS,
    )
    pipeline.client = fake_client

    # Must not raise.
    left_context, right_context = pipeline.get_flanking_context(
        species_id=SPECIES_ID,
        gap_start=gap_start,
        gap_end=gap_end,
        cleaned_sequence=cleaned_sequence,
        species_metadata={"species_id": SPECIES_ID, "is_extinct": False},
    )

    # Both contexts must be strings (possibly empty at boundaries).
    assert isinstance(left_context, str), "left_context must be a str"
    assert isinstance(right_context, str), "right_context must be a str"
    assert len(left_context) >= 0, "left_context length must be non-negative"
    assert len(right_context) >= 0, "right_context length must be non-negative"

    # Right context should be ≥ 1000 bp because the right flank is long enough
    # and we have 5 windows contributing.
    assert len(right_context) >= 1000, (
        f"Expected right_context >= 1000 bp even near left edge, "
        f"got {len(right_context)} bp."
    )


# ---------------------------------------------------------------------------
# Test 5 — No mock evolution_analysis injection
# Validates: Requirements 4.5, 3.6
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_no_mock_evolution_analysis_injection():
    """POST an extinct-species request and verify no mock evolution_analysis is injected.

    Requirement 4.5 states the Reconstruction_Agent SHALL NOT inject a mock
    ``evolution_analysis`` string into the context from ``validation_engine``
    or any node that precedes ``validation_engine``.
    Requirement 3.6 states the Validation_Engine SHALL NOT invoke the Evolution
    Agent handoff from ``evolution_agent_delegate`` or any node that executes before
    ``model_selector_predictor``.

    This test confirms both invariants by:
    1. Posting a request with ``species_metadata["is_extinct"]=True`` and NO
       ``evolution_analysis`` in context.
    2. Intercepting the state after ``input_manager`` and ``gap_locator`` complete to
       assert that neither node has silently injected ``evolution_analysis`` into
       ``state["request"].context``.
    3. Asserting the final result is ``NEEDS_AGENT`` (not ``COMPLETED`` with a mocked
       string) — proving no mock bypass occurred.

    Validates: Requirements 4.5, 3.6
    """
    from backend.agents.reconstruction_agent import agent as _agent_module
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    import backend.agents.reconstruction_agent.api as _api_module

    # Spy bookkeeping — populated inside the node wrappers below.
    nodes_visited: list[str] = []
    evolution_analysis_seen_before_validation: list[tuple[str, bool]] = []

    # Capture real node implementations so wrappers can delegate to them.
    real_input_manager = _agent_module.input_manager
    real_gap_locator = _agent_module.gap_locator
    real_model_selector_predictor = _agent_module.model_selector_predictor
    real_validation_engine = _agent_module.validation_engine
    real_prediction_assembler = _agent_module.prediction_assembler
    real_output_formatter = _agent_module.output_formatter

    def spy_input_manager(state):
        nodes_visited.append("input_manager")
        result = real_input_manager(state)
        req = result.get("request") or state.get("request")
        evo = req.context.get("evolution_analysis") if req else None
        evolution_analysis_seen_before_validation.append(("input_manager", evo is not None))
        return result

    def spy_gap_locator(state):
        nodes_visited.append("gap_locator")
        result = real_gap_locator(state)
        req = result.get("request") or state.get("request")
        evo = req.context.get("evolution_analysis") if req else None
        evolution_analysis_seen_before_validation.append(("gap_locator", evo is not None))
        return result

    def spy_model_selector_predictor(state):
        nodes_visited.append("model_selector_predictor")
        return real_model_selector_predictor(state)

    def spy_validation_engine(state):
        nodes_visited.append("validation_engine")
        return real_validation_engine(state)

    def spy_prediction_assembler(state):
        nodes_visited.append("prediction_assembler")
        return real_prediction_assembler(state)

    def spy_output_formatter(state):
        nodes_visited.append("output_formatter")
        return real_output_formatter(state)

    # Build a fresh graph wired with spy functions at compile time.
    # The spy graph must replicate the EXACT topology of the production graph
    # (reasoning_core hub) so routing behaviour is identical.
    # api.py does ``from .agent import graph`` — we patch the name in api.py's
    # own namespace so the /execute handler calls our spy graph.
    def build_spy_graph():
        g = StateGraph(_agent_module.ReconstructionState)

        # --- Spy tool nodes ---
        g.add_node("input_manager", spy_input_manager)
        g.add_node("gap_locator", spy_gap_locator)
        g.add_node("model_selector_predictor", spy_model_selector_predictor)
        g.add_node("validation_engine", spy_validation_engine)
        g.add_node("prediction_assembler", spy_prediction_assembler)
        g.add_node("output_formatter", spy_output_formatter)

        # --- Real hub and delegate nodes (no spy needed here) ---
        g.add_node("reasoning_core", _agent_module.reasoning_core_node)
        g.add_node("evolution_agent_delegate", _agent_module.evolution_agent_delegate)

        # Entry → hub
        g.set_entry_point("reasoning_core")

        # Hub routes conditionally to all tool nodes
        g.add_conditional_edges(
            "reasoning_core",
            _agent_module.reasoning_core,
            {
                "input_manager": "input_manager",
                "gap_locator": "gap_locator",
                "model_selector_predictor": "model_selector_predictor",
                "validation_engine": "validation_engine",
                "prediction_assembler": "prediction_assembler",
                "output_formatter": "output_formatter",
                END: END,
            },
        )

        # Tool nodes return to hub, except validation_engine which checks delegation
        g.add_edge("input_manager", "reasoning_core")
        g.add_edge("gap_locator", "reasoning_core")
        g.add_edge("model_selector_predictor", "reasoning_core")
        g.add_conditional_edges(
            "validation_engine",
            _agent_module.check_evolution_needs,
            {
                "evolution_agent_delegate": "evolution_agent_delegate",
                "reasoning_core": "reasoning_core",
            },
        )
        g.add_edge("evolution_agent_delegate", "reasoning_core")
        g.add_edge("prediction_assembler", "reasoning_core")
        g.add_edge("output_formatter", "reasoning_core")

        return g.compile(checkpointer=MemorySaver())

    extinct_request = {
        "instruction": "Reconstruct the missing genome region for an extinct species.",
        "context": {
            "session_id": "test-session-integration-005-no-mock",
            # In-scope gap: 20 N characters (10 ≤ 20 ≤ 500)
            "genome": "ACGTACGTACGT" + "N" * 20 + "ACGTACGTACGT",
            "species": "mammoth",
            "species_metadata": {
                "species_id": "mammuthus_primigenius",
                "is_extinct": True,
            },
            "sequence_type": "nuclear",
            # Deliberately omit "evolution_analysis" — must NOT be injected by any node
            # before validation_engine.
        },
    }

    spy_graph = build_spy_graph()

    with _make_router_mock("ACGTACGTACGTACGTACGT", 0.95):
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            # Patch the ``graph`` name in api.py's own namespace so the /execute
            # handler actually calls our spy graph.
            with patch.object(_api_module, "graph", spy_graph):
                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/execute", json=extinct_request)

    assert response.status_code == 200, (
        f"Expected HTTP 200 from /execute, got {response.status_code}. "
        f"Response body: {response.text}"
    )

    body = response.json()

    # --- Assertion 1: no mock evolution_analysis injected before validation_engine ---
    # Neither input_manager nor gap_locator should have set evolution_analysis on the context.
    for node_name, was_present in evolution_analysis_seen_before_validation:
        assert not was_present, (
            f"Requirement 4.5 / 3.6 violated: '{node_name}' injected 'evolution_analysis' "
            f"into state[\"request\"].context before validation_engine ran. "
            f"No mock evolution_analysis may be injected by any node that precedes "
            f"validation_engine."
        )

    # --- Assertion 2: final result is NEEDS_AGENT, not COMPLETED with a mock ---
    assert body.get("status") == "needs_agent", (
        f"Requirement 4.5 / 3.6 violated: expected status='needs_agent' because no "
        f"'evolution_analysis' was provided and the agent must NOT complete using a "
        f"mocked value. Got status='{body.get('status')}'. Full response: {body}"
    )

    # status must specifically NOT be 'completed' (which would indicate a mock bypass).
    assert body.get("status") != "completed", (
        f"Requirement 4.5 violated: agent returned 'completed' despite no "
        f"'evolution_analysis' in context. A mock value must NOT have been injected. "
        f"Full response: {body}"
    )

    # target_agent must be Evolution (confirms the handoff is wired correctly).
    assert body.get("target_agent") == "Evolution", (
        f"Expected target_agent='Evolution', got '{body.get('target_agent')}'. "
        f"Full response: {body}"
    )

    # Sanity: validation_engine must have been reached (i.e. the graph didn't
    # short-circuit before reaching the node that owns the NEEDS_AGENT check,
    # which would violate Req 3.6).
    assert "validation_engine" in nodes_visited, (
        f"validation_engine was never reached — the NEEDS_AGENT check may have fired "
        f"too early (in a pre-validation node), violating Req 3.6. "
        f"Nodes visited: {nodes_visited}"
    )


# ---------------------------------------------------------------------------
# Test 6 — gap_locator is invoked on a genuine fresh POST (Stage 1 verification)
# Validates: Stage 1 fix correctness — api.py request_to_state produces a
# minimal state (only "request" key), so gap_locator MUST always run for a
# brand-new request, never be bypassed due to a pre-seeded "gaps" key.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_gap_locator_invoked_on_fresh_post_request():
    """Confirm gap_locator runs before any gap-processing node on a fresh POST.

    This test exercises the REAL request path:
        POST /execute  →  request_to_state  →  graph.invoke
    with a state that contains ONLY the "request" key (exactly what
    ``request_to_state`` produces).  It uses a spy graph to record which nodes
    are visited and asserts that:

    1. ``gap_locator`` is visited.
    2. ``gap_locator`` is visited BEFORE ``model_selector_predictor``.
    3. The initial state passed to ``graph.invoke`` does NOT contain a ``"gaps"``
       key — confirming that ``request_to_state`` never pre-seeds it.

    Existing tests in ``test_agent5_graph.py`` bypass ``request_to_state`` by
    constructing a ``GraphState`` dict directly; this test does not — it goes
    through the full HTTP boundary.

    Validates: Stage 1 execution bug fix.
    """
    import backend.agents.reconstruction_agent.api as _api_module
    from backend.agents.reconstruction_agent import agent as _agent_module
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver

    nodes_visited: list[str] = []
    initial_state_snapshot: dict = {}

    real_gap_locator = _agent_module.gap_locator
    real_input_manager = _agent_module.input_manager
    real_model_selector_predictor = _agent_module.model_selector_predictor
    real_validation_engine = _agent_module.validation_engine
    real_prediction_assembler = _agent_module.prediction_assembler
    real_output_formatter = _agent_module.output_formatter

    def spy_input_manager(state):
        nodes_visited.append("input_manager")
        # Capture the state keys present at the very first node execution —
        # this is the state that graph.invoke received from request_to_state.
        if not initial_state_snapshot:
            initial_state_snapshot.update({k: v for k, v in state.items()})
        return real_input_manager(state)

    def spy_gap_locator(state):
        nodes_visited.append("gap_locator")
        return real_gap_locator(state)

    def spy_model_selector_predictor(state):
        nodes_visited.append("model_selector_predictor")
        return real_model_selector_predictor(state)

    def spy_validation_engine(state):
        nodes_visited.append("validation_engine")
        return real_validation_engine(state)

    def spy_prediction_assembler(state):
        nodes_visited.append("prediction_assembler")
        return real_prediction_assembler(state)

    def spy_output_formatter(state):
        nodes_visited.append("output_formatter")
        return real_output_formatter(state)

    def build_spy_graph():
        g = StateGraph(_agent_module.ReconstructionState)
        g.add_node("input_manager", spy_input_manager)
        g.add_node("gap_locator", spy_gap_locator)
        g.add_node("model_selector_predictor", spy_model_selector_predictor)
        g.add_node("validation_engine", spy_validation_engine)
        g.add_node("prediction_assembler", spy_prediction_assembler)
        g.add_node("output_formatter", spy_output_formatter)
        g.add_node("reasoning_core", _agent_module.reasoning_core_node)
        g.add_node("evolution_agent_delegate", _agent_module.evolution_agent_delegate)
        g.set_entry_point("reasoning_core")
        g.add_conditional_edges(
            "reasoning_core",
            _agent_module.reasoning_core,
            {
                "input_manager": "input_manager",
                "gap_locator": "gap_locator",
                "model_selector_predictor": "model_selector_predictor",
                "validation_engine": "validation_engine",
                "prediction_assembler": "prediction_assembler",
                "output_formatter": "output_formatter",
                END: END,
            },
        )
        g.add_edge("input_manager", "reasoning_core")
        g.add_edge("gap_locator", "reasoning_core")
        g.add_edge("model_selector_predictor", "reasoning_core")
        g.add_conditional_edges(
            "validation_engine",
            _agent_module.check_evolution_needs,
            {
                "evolution_agent_delegate": "evolution_agent_delegate",
                "reasoning_core": "reasoning_core",
            },
        )
        g.add_edge("evolution_agent_delegate", "reasoning_core")
        g.add_edge("prediction_assembler", "reasoning_core")
        g.add_edge("output_formatter", "reasoning_core")
        return g.compile(checkpointer=MemorySaver())

    # Living-species genome with one in-scope gap (15 Ns) — no gaps pre-seeded.
    request_payload = {
        "instruction": "Reconstruct missing region.",
        "context": {
            "session_id": "test-session-stage1-verification",
            "genome": "ACGTACGT" + "N" * 15 + "ACGTACGT",
            "species": "loxodonta_africana",
            "species_metadata": {
                "species_id": "loxodonta_africana",
                "is_extinct": False,
            },
            "sequence_type": "nuclear",
        },
    }

    spy_graph = build_spy_graph()

    with _make_router_mock("ACGTACGTACGTACG", 0.9):
        with patch(
            "backend.agents.reconstruction_agent.agent.RetrievalPipeline"
        ) as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.get_flanking_context.return_value = ("A" * 50, "T" * 50)
            MockPipeline.return_value = mock_pipeline_instance

            with patch.object(_api_module, "graph", spy_graph):
                client = TestClient(app, raise_server_exceptions=False)
                response = client.post("/execute", json=request_payload)

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )

    # --- Assertion 1: initial state has NO "gaps" key ---
    # request_to_state must only set {"request": ...}.  If "gaps" is present
    # it means the HTTP boundary is pre-seeding state, which is the Stage 1 bug.
    assert "gaps" not in initial_state_snapshot, (
        f"Stage 1 bug detected: the initial state passed to graph.invoke contains "
        f"a 'gaps' key ({initial_state_snapshot.get('gaps')!r}).  "
        f"request_to_state must NOT pre-seed 'gaps'. "
        f"Initial state keys: {list(initial_state_snapshot.keys())}"
    )

    # --- Assertion 2: gap_locator was visited ---
    assert "gap_locator" in nodes_visited, (
        f"gap_locator was never visited on a fresh POST request. "
        f"Nodes visited in order: {nodes_visited}"
    )

    # --- Assertion 3: gap_locator ran before model_selector_predictor ---
    if "model_selector_predictor" in nodes_visited:
        gl_idx = nodes_visited.index("gap_locator")
        msp_idx = nodes_visited.index("model_selector_predictor")
        assert gl_idx < msp_idx, (
            f"gap_locator ({gl_idx}) must run before model_selector_predictor ({msp_idx}). "
            f"Nodes visited in order: {nodes_visited}"
        )

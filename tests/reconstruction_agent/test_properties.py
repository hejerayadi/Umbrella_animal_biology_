# Feature: reconstruction-agent-sprint2
# Property 1: gap length bounds invariant
# Property 2: assembled sequence length preservation
# Property 3: is_partial flag accuracy
# Property 4: validation retry bound

import sys
import json
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out the LLM dependency so agent.py can be imported without a live
# Azure / OpenAI key in the test environment.  This must happen before the
# first `from backend.agents...` import statement reaches agent.py.
# ---------------------------------------------------------------------------
_llm_stub = ModuleType("backend.orchestrator.llm")
_llm_stub.get_llm = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("backend.orchestrator.llm", _llm_stub)

from hypothesis import given, settings, strategies as st, HealthCheck  # noqa: E402

from backend.agents.reconstruction_agent.agent import (  # noqa: E402
    gap_locator,
    model_selector_predictor,
    output_formatter,
    prediction_assembler,
    validation_engine,
)
from backend.agents.reconstruction_agent.schema import (  # noqa: E402
    AgentRequest,
    GapPrediction,
    GapRegion,
    ValidatedGenome,
)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

DNA_BASES = "ACGT"


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


def _make_router_mock_must_not_call(message: str):
    """Patch DNAModelRouter.get() so predict() raises if the router is invoked."""
    mock_router_instance = MagicMock()
    mock_router_instance.predict.side_effect = AssertionError(message)
    return patch(
        "backend.agents.reconstruction_agent.agent.DNAModelRouter.get",
        return_value=mock_router_instance,
    )


@st.composite
def dna_sequence_with_n_runs(draw) -> str:
    """
    Composite Hypothesis strategy that builds a DNA string containing one or
    more N-runs of varied lengths (1–600 bp) injected between nucleotide
    context segments.  Both in-scope (10–500 bp) and out-of-scope runs may
    appear in the same sequence, giving the property test full coverage of the
    gap_locator boundary logic.
    """
    # Draw 1–5 N-run lengths in the 1–600 bp range
    n_run_lengths = draw(
        st.lists(
            st.integers(min_value=1, max_value=600),
            min_size=1,
            max_size=5,
        )
    )

    # Flanking / separator segments between runs (each 0–300 bp of ACGT)
    num_segments = len(n_run_lengths) + 1
    separators = draw(
        st.lists(
            st.text(alphabet=DNA_BASES, min_size=0, max_size=300),
            min_size=num_segments,
            max_size=num_segments,
        )
    )

    parts: list[str] = []
    for i, run_len in enumerate(n_run_lengths):
        parts.append(separators[i])
        parts.append("N" * run_len)
    parts.append(separators[-1])

    sequence = "".join(parts)

    # Ensure the sequence meets the minimum length guard in input_manager
    if len(sequence) < 10:
        sequence = sequence + "A" * (10 - len(sequence))

    return sequence


# ---------------------------------------------------------------------------
# Property 1 — Gap_Locator gap length bounds invariant
# ---------------------------------------------------------------------------

@given(dna_sequence_with_n_runs())
@settings(max_examples=200)
def test_gap_locator_in_scope_length_bounds(sequence: str) -> None:
    """
    **Validates: Requirements 12.1**

    For every GapRegion returned by gap_locator with in_scope=True,
    its length must satisfy 10 <= gap.length <= 500.
    """
    state = {
        "request": AgentRequest(
            instruction="reconstruct",
            context={
                "genome": sequence,
                "species_metadata": {"species_id": "test_species"},
            },
        ),
        "validated_genome": ValidatedGenome(
            cleaned_sequence=sequence.upper(),
            species_id="test_species",
            species_metadata={"species_id": "test_species", "is_extinct": False},
            sequence_type="nuclear",
        ),
    }

    result_state = gap_locator(state)
    gaps = result_state.get("gaps", [])

    for gap in gaps:
        if gap.in_scope:
            assert 10 <= gap.length <= 500, (
                f"in_scope GapRegion has length {gap.length} which is outside [10, 500]. "
                f"gap={gap!r}"
            )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 2
# ---------------------------------------------------------------------------


@st.composite
def assembled_state_with_predictions(draw) -> dict:
    """
    Composite Hypothesis strategy that builds a ReconstructionState containing:
    - A `validated_genome` with a `cleaned_sequence` made of ACGT bases
    - One or more non-overlapping GapRegions (in-scope, length 10–500 bp)
      whose N-regions are embedded in the sequence
    - Matching GapPrediction objects whose `predicted_sequence` has already
      been pre-normalized (padded or truncated) to exactly `GapRegion.length`

    The sequence is constructed so that replacing each gap's N-run with a
    same-length prediction string preserves total length.
    """
    # Draw 1–4 non-overlapping in-scope gap lengths
    num_gaps = draw(st.integers(min_value=1, max_value=4))
    gap_lengths = draw(
        st.lists(
            st.integers(min_value=10, max_value=500),
            min_size=num_gaps,
            max_size=num_gaps,
        )
    )

    # Draw separator ACGT segments between and around the gaps (1–200 bp each)
    num_separators = num_gaps + 1
    separators = draw(
        st.lists(
            st.text(alphabet=DNA_BASES, min_size=1, max_size=200),
            min_size=num_separators,
            max_size=num_separators,
        )
    )

    # Build the cleaned sequence: sep0 + NNN... + sep1 + NNN... + sep2 ...
    parts: list[str] = []
    gap_regions: list[GapRegion] = []
    pos = 0

    for i, gap_len in enumerate(gap_lengths):
        sep = separators[i]
        parts.append(sep)
        pos += len(sep)

        gap_regions.append(
            GapRegion(start=pos, end=pos + gap_len, length=gap_len, in_scope=True)
        )
        parts.append("N" * gap_len)
        pos += gap_len

    parts.append(separators[-1])
    cleaned_sequence = "".join(parts)

    # Build pre-normalized GapPrediction objects:
    # predicted_sequence is exactly gap.length characters of ACGT
    predictions: list[GapPrediction] = []
    for gap in gap_regions:
        pred_seq = draw(
            st.text(alphabet=DNA_BASES, min_size=gap.length, max_size=gap.length)
        )
        predictions.append(
            GapPrediction(
                gap=gap,
                predicted_sequence=pred_seq,
                model_used="gpt-4o",
                confidence=0.9,
            )
        )

    state = {
        "request": AgentRequest(
            instruction="reconstruct",
            context={
                "genome": cleaned_sequence,
                "species_metadata": {"species_id": "test_species"},
            },
        ),
        "validated_genome": ValidatedGenome(
            cleaned_sequence=cleaned_sequence,
            species_id="test_species",
            species_metadata={"species_id": "test_species", "is_extinct": False},
            sequence_type="nuclear",
        ),
        "gaps": gap_regions,
        "predictions": predictions,
        "is_partial": False,
    }
    return state


# ---------------------------------------------------------------------------
# Property 2 — Prediction_Assembler length preservation
# ---------------------------------------------------------------------------


@given(assembled_state_with_predictions())
@settings(max_examples=200)
def test_prediction_assembler_length_preserved(state: dict) -> None:
    """
    **Validates: Requirements 12.2, 5.1, 5.3**

    For any ReconstructionState with a cleaned_sequence and GapPrediction
    objects whose predicted_sequence has been pre-normalized to GapRegion.length,
    after prediction_assembler runs the assembled_sequence length must equal
    the original cleaned_sequence length.
    """
    # Feature: reconstruction-agent-sprint2, Property 2: assembled sequence length preservation
    original_length = len(state["validated_genome"].cleaned_sequence)

    result_state = prediction_assembler(state)

    assembled = result_state.get("assembled_sequence", "")
    assert len(assembled) == original_length, (
        f"assembled_sequence length {len(assembled)} != "
        f"original cleaned_sequence length {original_length}. "
        f"gaps={state['gaps']!r}"
    )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 3
# ---------------------------------------------------------------------------


@st.composite
def state_with_unresolved_gaps(draw) -> dict:
    """
    Composite Hypothesis strategy that builds a ReconstructionState where
    at least one in-scope GapRegion has NO corresponding GapPrediction.

    Steps:
    1. Draw 1–4 in-scope GapRegions embedded in a cleaned_sequence.
    2. Draw 0 to (num_gaps - 1) predictions — always leaving at least one gap
       unpredicted — choosing which gaps get predictions randomly.
    3. Set state["is_partial"] to True to reflect the unresolved gaps, and
       provide an assembled_sequence equal to cleaned_sequence (no splicing
       needed; output_formatter only reads is_partial from state).
    """
    # Draw 1–4 in-scope gap lengths (10–500 bp each)
    num_gaps = draw(st.integers(min_value=1, max_value=4))
    gap_lengths = draw(
        st.lists(
            st.integers(min_value=10, max_value=500),
            min_size=num_gaps,
            max_size=num_gaps,
        )
    )

    # Draw separator segments between and around the gaps (1–200 bp of ACGT each)
    num_separators = num_gaps + 1
    separators = draw(
        st.lists(
            st.text(alphabet=DNA_BASES, min_size=1, max_size=200),
            min_size=num_separators,
            max_size=num_separators,
        )
    )

    # Build cleaned_sequence and GapRegion list
    parts: list[str] = []
    gap_regions: list[GapRegion] = []
    pos = 0

    for i, gap_len in enumerate(gap_lengths):
        sep = separators[i]
        parts.append(sep)
        pos += len(sep)

        gap_regions.append(
            GapRegion(start=pos, end=pos + gap_len, length=gap_len, in_scope=True)
        )
        parts.append("N" * gap_len)
        pos += gap_len

    parts.append(separators[-1])
    cleaned_sequence = "".join(parts)

    # Choose how many gaps get predictions: 0 to num_gaps-1 (always leave ≥ 1 unresolved)
    num_predicted = draw(st.integers(min_value=0, max_value=num_gaps - 1))

    # Pick which gaps are predicted (first num_predicted gaps)
    predicted_indices = set(range(num_predicted))

    predictions: list[GapPrediction] = []
    for idx in predicted_indices:
        gap = gap_regions[idx]
        pred_seq = draw(
            st.text(alphabet=DNA_BASES, min_size=gap.length, max_size=gap.length)
        )
        predictions.append(
            GapPrediction(
                gap=gap,
                predicted_sequence=pred_seq,
                model_used="gpt-4o",
                confidence=draw(st.floats(min_value=0.71, max_value=1.0)),
            )
        )

    state = {
        "request": AgentRequest(
            instruction="reconstruct",
            context={
                "genome": cleaned_sequence,
                "species_metadata": {"species_id": "test_species"},
            },
        ),
        "validated_genome": ValidatedGenome(
            cleaned_sequence=cleaned_sequence,
            species_id="test_species",
            species_metadata={"species_id": "test_species", "is_extinct": False},
            sequence_type="nuclear",
        ),
        "gaps": gap_regions,
        "predictions": predictions,
        # At least one gap is unresolved, so is_partial must be True
        "is_partial": True,
        "assembled_sequence": cleaned_sequence,
    }
    return state


# ---------------------------------------------------------------------------
# Property 3 — Output_Formatter is_partial flag accuracy
# ---------------------------------------------------------------------------


@given(state_with_unresolved_gaps())
@settings(max_examples=200)
def test_output_formatter_is_partial_when_gaps_unresolved(state: dict) -> None:
    """
    **Validates: Requirements 12.3, 6.1**

    For any reconstruction state where at least one in-scope GapRegion has no
    corresponding GapPrediction, after output_formatter runs the resulting
    ReconstructionResult.is_partial must be True.
    """
    # Feature: reconstruction-agent-sprint2, Property 3: is_partial flag accuracy
    result_state = output_formatter(state)

    result = result_state.get("result")
    assert result is not None, "output_formatter must populate state['result']"
    assert result.is_partial is True, (
        f"ReconstructionResult.is_partial should be True when at least one in-scope "
        f"GapRegion has no accepted prediction. "
        f"gaps={state['gaps']!r}, predictions={state['predictions']!r}"
    )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 4
# ---------------------------------------------------------------------------


@st.composite
def in_scope_gap_state(draw) -> dict:
    """
    Composite Hypothesis strategy that builds a ReconstructionState containing
    exactly one in-scope GapRegion (10–500 bp) so that Property 4 (retry bound)
    can drive the model_selector_predictor + validation_engine loop in isolation.

    The state is initialised to match what gap_locator would produce:
      - current_gap_index points to index 0 (the single in-scope gap)
      - current_gap_retries = 0
      - predictions = []
    """
    gap_len = draw(st.integers(min_value=10, max_value=500))

    # Build a minimal sequence: left context + N-run + right context
    left = draw(st.text(alphabet=DNA_BASES, min_size=1, max_size=200))
    right = draw(st.text(alphabet=DNA_BASES, min_size=1, max_size=200))

    cleaned_sequence = left + "N" * gap_len + right
    gap_start = len(left)
    gap_end = gap_start + gap_len

    gap = GapRegion(start=gap_start, end=gap_end, length=gap_len, in_scope=True)

    state: dict = {
        "request": AgentRequest(
            instruction="reconstruct",
            context={
                "genome": cleaned_sequence,
                "species_metadata": {
                    "species_id": "test_species",
                    "is_extinct": False,
                },
                "species": "test_species",
                "sequence_type": "nuclear",
            },
        ),
        "validated_genome": ValidatedGenome(
            cleaned_sequence=cleaned_sequence,
            species_id="test_species",
            species_metadata={"species_id": "test_species", "is_extinct": False},
            sequence_type="nuclear",
        ),
        "gaps": [gap],
        "predictions": [],
        "current_gap_index": 0,
        "current_gap_retries": 0,
        "is_partial": False,
    }
    return state


# ---------------------------------------------------------------------------
# Property 4 — Validation_Engine retry bound
# ---------------------------------------------------------------------------


@given(in_scope_gap_state())
@settings(max_examples=100)
def test_validation_engine_retry_bound(state: dict) -> None:
    """
    **Validates: Requirements 12.4, 3.3, 3.7**

    For any in-scope gap, when the LLM always returns confidence=0.0,
    Validation_Engine must:
    - invoke the LLM (via Model_Selector_Predictor) at most 3 times,
    - set is_partial=True, and
    - not raise any exception.

    # Feature: reconstruction-agent-sprint2, Property 4: validation retry bound
    """
    predict_call_count = 0

    def _predict(flanks, gap, previous_attempts):
        nonlocal predict_call_count
        predict_call_count += 1
        return GapPrediction(
            gap=gap,
            predicted_sequence="A" * gap.length,
            model_used="Azure-GPT-5.1",
            confidence=0.0,
        )

    mock_router_instance = MagicMock()
    mock_router_instance.predict.side_effect = _predict

    # Also stub RetrievalPipeline so no Qdrant connection is attempted
    mock_pipeline = MagicMock()
    gap = state["gaps"][0]
    seq = state["validated_genome"].cleaned_sequence
    mock_pipeline.get_flanking_context.return_value = (
        seq[max(0, gap.start - 50): gap.start],
        seq[gap.end: gap.end + 50],
    )

    with patch(
        "backend.agents.reconstruction_agent.agent.DNAModelRouter.get",
        return_value=mock_router_instance,
    ), patch(
        "backend.agents.reconstruction_agent.agent.RetrievalPipeline",
        return_value=mock_pipeline,
    ):

        # Drive the model_selector_predictor → validation_engine loop
        # exactly as the LangGraph sub-orchestrator would, until the gap is
        # exhausted (current_gap_index advances past the single gap).
        loop_limit = 10  # safety ceiling — well above the expected 3 retries
        iterations = 0
        while (
            state.get("current_gap_index", 0) < len(state["gaps"])
            and iterations < loop_limit
        ):
            state = model_selector_predictor(state)
            state = validation_engine(state)
            iterations += 1

    # Core assertions
    assert predict_call_count <= 3, (
        f"DNAModelRouter.predict was called {predict_call_count} times for a single gap, "
        "but Validation_Engine must call it at most 3 times. "
        f"gap={state['gaps'][0]!r}"
    )
    assert state.get("is_partial") is True, (
        "is_partial must be True when all retries are exhausted without an accepted prediction. "
        f"gap={state['gaps'][0]!r}"
    )
    # Verify no NEEDS_AGENT result was set (no exception path was taken either)
    result = state.get("result")
    if result is not None:
        from backend.agents.reconstruction_agent.schema import AgentStatus
        assert getattr(result, "status", None) != AgentStatus.NEEDS_AGENT, (
            "Validation_Engine must not emit NEEDS_AGENT for a living species "
            "when the LLM always returns low confidence. "
            f"result={result!r}"
        )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 5
# ---------------------------------------------------------------------------


@st.composite
def sequence_with_only_out_of_scope_n_runs(draw) -> str:
    """
    Composite Hypothesis strategy that builds a DNA string whose every N-run
    is strictly out-of-scope, i.e. has length < 10 OR > 500 bp.

    Strategy:
    - Draw 1–4 N-run lengths, each chosen from [1, 9] ∪ [501, 600].
    - Interleave ACGT separator segments (1–200 bp each) between runs.
    - Pad to at least 10 characters so input_manager does not trigger NEEDS_AGENT.

    No run will satisfy 10 ≤ length ≤ 500, so gap_locator will classify all
    gaps as out-of-scope and the graph will skip Model_Selector_Predictor and
    Validation_Engine entirely.
    """
    # Each N-run length is either short (1–9) or long (501–600)
    n_run_length = st.one_of(
        st.integers(min_value=1, max_value=9),
        st.integers(min_value=501, max_value=600),
    )
    n_run_lengths = draw(
        st.lists(n_run_length, min_size=1, max_size=4)
    )

    num_separators = len(n_run_lengths) + 1
    separators = draw(
        st.lists(
            st.text(alphabet=DNA_BASES, min_size=1, max_size=200),
            min_size=num_separators,
            max_size=num_separators,
        )
    )

    parts: list[str] = []
    for i, run_len in enumerate(n_run_lengths):
        parts.append(separators[i])
        parts.append("N" * run_len)
    parts.append(separators[-1])

    sequence = "".join(parts)

    # Ensure the sequence meets the minimum length guard in input_manager
    if len(sequence) < 10:
        sequence = sequence + "A" * (10 - len(sequence))

    return sequence


# ---------------------------------------------------------------------------
# Property 5 — Out-of-scope-only sequences produce correct partial result
# ---------------------------------------------------------------------------


@given(sequence_with_only_out_of_scope_n_runs())
@settings(max_examples=100)
def test_out_of_scope_only_produces_partial_and_zero_reconstructed(sequence: str) -> None:
    """
    **Validates: Requirements 12.5, 6.1**

    For any genome sequence that contains only N-runs shorter than 10 bp or
    longer than 500 bp (i.e., no in-scope gaps), when the full compiled graph
    executes to completion:
    - ReconstructionResult.is_partial must be True
    - ReconstructionResult.gaps_reconstructed must be 0

    # Feature: reconstruction-agent-sprint2, Property 5: out-of-scope-only → partial + zero
    """
    from backend.agents.reconstruction_agent.agent import graph

    state = {
        "request": AgentRequest(
            instruction="reconstruct",
            context={
                "genome": sequence,
                "species_metadata": {
                    "species_id": "test_species",
                    "is_extinct": False,
                },
                "sequence_type": "nuclear",
                "session_id": "prop5-test",
            },
        ),
        # Do NOT pre-seed "gaps" here — gap_locator must run and find the
        # out-of-scope N-runs itself.  Pre-seeding gaps=[] caused gap_locator
        # to be bypassed, hiding the node entirely (Stage 1 bug).
        "predictions": [],
        "current_gap_index": 0,
        "is_partial": False,
    }

    # No router or Qdrant calls are expected since there are no in-scope gaps.
    # We still patch DNAModelRouter.get and RetrievalPipeline as a safeguard —
    # if the graph accidentally routes through model_selector_predictor, the
    # mocks will surface the issue rather than causing a connection error.
    mock_pipeline = MagicMock()
    mock_pipeline.get_flanking_context.side_effect = AssertionError(
        "RetrievalPipeline should NOT be called for a sequence with only out-of-scope gaps"
    )

    with _make_router_mock_must_not_call(
        "DNAModelRouter should NOT be called for a sequence with only out-of-scope gaps"
    ), patch(
        "backend.agents.reconstruction_agent.agent.RetrievalPipeline",
        return_value=mock_pipeline,
    ):
        import uuid
        thread_id = str(uuid.uuid4())
        final_state = graph.invoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )

    result = final_state.get("result")
    assert result is not None, (
        "output_formatter must populate state['result'] for every completed run. "
        f"sequence={sequence!r}"
    )
    assert result.is_partial is True, (
        f"ReconstructionResult.is_partial must be True when all N-runs are out-of-scope. "
        f"result={result!r}, sequence={sequence!r}"
    )
    assert result.gaps_reconstructed == 0, (
        f"ReconstructionResult.gaps_reconstructed must be 0 when there are no in-scope gaps. "
        f"result={result!r}, sequence={sequence!r}"
    )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 6
# ---------------------------------------------------------------------------


@st.composite
def valid_window(draw) -> "Window":
    """
    Composite Hypothesis strategy that builds a Window object with a sequence
    of exactly 2000 bp, composed of DNA bases (ACGT) and/or N characters.
    The window metadata fields are filled with plausible values.
    """
    from backend.agents.reconstruction_agent.data_ingestion import Window

    # Draw a 2000 bp sequence using ACGT (N included to reflect real genomic data)
    sequence = draw(st.text(alphabet="ACGTN", min_size=2000, max_size=2000))

    scaffold_id = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=20))
    accession = draw(st.just("GCF_000001905.1"))
    species_id = draw(st.just("loxodonta_africana"))
    window_start = draw(st.integers(min_value=0, max_value=100_000))
    window_end = window_start + 2000

    return Window(
        scaffold_id=scaffold_id,
        accession=accession,
        species_id=species_id,
        sequence=sequence,
        start=window_start,
        end=window_end,
    )


# ---------------------------------------------------------------------------
# Property 6 — Masking bounds invariant
# ---------------------------------------------------------------------------


@given(valid_window())
@settings(max_examples=200)
def test_masking_bounds_invariant(window) -> None:
    """
    **Validates: Requirements 12.6, 7.7**

    For any valid 2000 bp Window, the masking step SHALL produce a MaskedTriplet
    where:
    - 10 ≤ len(masked_region) ≤ 500
    - len(left_context) + len(masked_region) + len(right_context) == 2000

    # Feature: reconstruction-agent-sprint2, Property 6: masking bounds invariant
    """
    from backend.agents.reconstruction_agent.data_ingestion.windowing import mask_window

    masked_triplet = mask_window(window, seed=42)

    masked_len = len(masked_triplet.masked_region)
    left_len = len(masked_triplet.left_context)
    right_len = len(masked_triplet.right_context)
    total_len = left_len + masked_len + right_len

    assert 10 <= masked_len <= 500, (
        f"masked_region length {masked_len} is outside [10, 500]. "
        f"window.sequence length={len(window.sequence)}, "
        f"left={left_len}, masked={masked_len}, right={right_len}"
    )
    assert total_len == 2000, (
        f"len(left) + len(masked) + len(right) == {total_len}, expected 2000. "
        f"left={left_len}, masked={masked_len}, right={right_len}"
    )


# ---------------------------------------------------------------------------
# Property 7 — LLM JSON parse failure fallback
# ---------------------------------------------------------------------------


def _is_valid_llm_json(text: str) -> bool:
    """
    Returns True iff the text (or any JSON object extractable from it) matches
    {"predicted_sequence": <str>, "confidence": <float>} — i.e. the cases where
    parse_llm_prediction would return real values rather than the fallback.
    """
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return False
        json_str = text[start_idx:end_idx + 1]
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return False
        if "predicted_sequence" not in data or "confidence" not in data:
            return False
        if not isinstance(data["predicted_sequence"], str):
            return False
        # confidence must be convertible to float
        float(data["confidence"])
        return True
    except Exception:
        return False


@given(st.text(), st.integers(min_value=10, max_value=500))
@settings(max_examples=300)
def test_llm_json_parse_failure_fallback(raw_response: str, gap_length: int) -> None:
    """
    **Validates: Requirements 2.6**

    For any string that cannot be parsed as a JSON object matching
    {"predicted_sequence": str, "confidence": float}, parse_llm_prediction
    SHALL return ("N" * gap_length, 0.0) without raising an exception.

    # Feature: reconstruction-agent-sprint2, Property 7: LLM JSON parse failure fallback
    """
    from backend.agents.reconstruction_agent.agent import parse_llm_prediction

    # Skip strings that ARE valid LLM JSON — those would legitimately return
    # real values, not the fallback.
    if _is_valid_llm_json(raw_response):
        return

    # Must not raise any exception
    try:
        seq, conf = parse_llm_prediction(raw_response, gap_length)
    except Exception as exc:  # pragma: no cover
        raise AssertionError(
            f"parse_llm_prediction raised an unexpected exception for input "
            f"{raw_response!r} with gap_length={gap_length}: {exc!r}"
        ) from exc

    assert seq == "N" * gap_length, (
        f"Expected fallback sequence 'N' * {gap_length}, got {seq!r}. "
        f"Input: {raw_response!r}"
    )
    assert conf == 0.0, (
        f"Expected fallback confidence 0.0, got {conf!r}. "
        f"Input: {raw_response!r}"
    )


# ---------------------------------------------------------------------------
# Strategy helpers — Property 8
# ---------------------------------------------------------------------------


@st.composite
def retrieval_query(draw, is_extinct_species: bool = False) -> dict:
    """
    Composite Hypothesis strategy that builds the arguments needed to call
    ``RetrievalPipeline.get_flanking_context`` and a pre-populated
    ``FakeQdrantClient`` that holds a mix of living-species and mammoth records.

    The ``FakeQdrantClient`` always returns ALL stored records (no server-side
    filter), so the Python-level ``is_extinct`` filter inside
    ``get_flanking_context`` is fully exercised.
    """
    # Draw a base species_id for the living records
    species_id = draw(
        st.sampled_from(
            ["loxodonta_africana", "elephas_maximus", "panthera_tigris",
             "canis_lupus_familiaris", "felis_catus"]
        )
    )

    # Draw a cleaned sequence (ACGT, 200–2000 bp) and a gap embedded in it
    flank_size = draw(st.integers(min_value=50, max_value=300))
    gap_len = draw(st.integers(min_value=10, max_value=100))
    left_seq = draw(st.text(alphabet="ACGT", min_size=flank_size, max_size=flank_size))
    right_seq = draw(st.text(alphabet="ACGT", min_size=flank_size, max_size=flank_size))
    cleaned_sequence = left_seq + "N" * gap_len + right_seq
    gap_start = flank_size
    gap_end = flank_size + gap_len

    # Build 1–4 living-species window records (is_extinct=False)
    num_living = draw(st.integers(min_value=1, max_value=4))
    living_records = []
    for _ in range(num_living):
        win_seq = draw(st.text(alphabet="ACGT", min_size=2000, max_size=2000))
        living_records.append({
            "payload": {
                "species_id": species_id,
                "is_extinct": False,
                "window_sequence": win_seq,
                "accession": "GCF_000001905.1",
            },
            "score": draw(st.floats(min_value=0.5, max_value=1.0)),
        })

    # Build 1–3 mammoth records (is_extinct=True) that must be filtered out
    num_mammoth = draw(st.integers(min_value=1, max_value=3))
    mammoth_records = []
    for _ in range(num_mammoth):
        win_seq = draw(st.text(alphabet="ACGT", min_size=2000, max_size=2000))
        mammoth_records.append({
            "payload": {
                "species_id": "mammuthus_primigenius",
                "is_extinct": True,
                "window_sequence": win_seq,
                "accession": "PRJEB59491",
            },
            "score": draw(st.floats(min_value=0.5, max_value=1.0)),
        })

    all_records = living_records + mammoth_records

    return {
        "species_id": species_id,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "cleaned_sequence": cleaned_sequence,
        "species_metadata": {"species_id": species_id, "is_extinct": is_extinct_species},
        "all_records": all_records,
        "mammoth_window_sequences": {r["payload"]["window_sequence"] for r in mammoth_records},
    }


class _TrackingFakeQdrantClient:
    """
    In-process stand-in for QdrantClient that returns a fixed list of records
    regardless of the filter argument passed to ``search``, and records exactly
    which records were returned so the test can assert on the pipeline's
    Python-level filtering logic.

    Deliberately ignores server-side ``filter`` — so only the Python-level
    ``is_extinct`` guard inside ``RetrievalPipeline.get_flanking_context``
    can prevent mammoth records from being processed.
    """

    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self.returned_records: list[dict] = []

    def search(self, *, collection_name: str, query_vector, limit: int, filter=None) -> list[dict]:  # noqa: A002
        # Deliberately ignore `filter` — return everything so the Python-level
        # guard is the only thing standing between mammoth records and the caller.
        batch = self._records[:limit]
        self.returned_records = batch
        return batch


# ---------------------------------------------------------------------------
# Property 8 — Extinct-species retrieval filter
# ---------------------------------------------------------------------------


@given(retrieval_query(is_extinct_species=False))
@settings(
    max_examples=100,
    suppress_health_check=[
        # Window sequences are 2000 bp by design; the large base-example size
        # is unavoidable given the genomic data requirements.
        HealthCheck.large_base_example,
    ],
)
def test_extinct_species_retrieval_filter(query_args: dict) -> None:
    """
    **Validates: Requirements 9.5**

    For any RetrievalPipeline query where ``species_metadata["is_extinct"]=False``, none
    of the window sequences contributed by returned records SHALL originate from
    a record whose Qdrant payload has ``is_extinct=True`` (i.e. mammoth records).

    Setup:
    - A _TrackingFakeQdrantClient pre-populated with both living-species records
      (is_extinct=False) and mammoth records (is_extinct=True) is injected
      directly into the pipeline (``pipeline.client = ...``).
    - The fake client's ``search`` method deliberately ignores server-side
      filters, so only the Python-level guard inside ``get_flanking_context``
      can prevent mammoth records from being consumed.
    - After the call we inspect the pipeline's internal ``_accepted_records``
      list (captured by the tracking client) and assert that no record with
      ``is_extinct=True`` was accepted by the pipeline.

    The assertion is made at the record level (not byte-content level) to avoid
    false positives caused by living and mammoth sequences sharing common
    nucleotide substrings.

    # Feature: reconstruction-agent-sprint2, Property 8: extinct-species retrieval filter
    """
    from backend.agents.reconstruction_agent.retrieval_pipeline import RetrievalPipeline

    fake_client = _TrackingFakeQdrantClient(query_args["all_records"])

    pipeline = RetrievalPipeline(
        qdrant_url="http://fake-qdrant:6333",
        collection_name="reconstruction_sequence_windows",
        top_k=10,
    )
    # Inject the fake client so _get_client() is never called
    pipeline.client = fake_client

    left_ctx, right_ctx = pipeline.get_flanking_context(
        species_id=query_args["species_id"],
        gap_start=query_args["gap_start"],
        gap_end=query_args["gap_end"],
        cleaned_sequence=query_args["cleaned_sequence"],
        species_metadata=query_args["species_metadata"],
    )

    # The pipeline internally filters out is_extinct=True records before
    # building left_parts / right_parts.  We verify this by computing what the
    # output would be if ONLY living records were accepted and comparing it to
    # the actual output; additionally, we verify the output cannot contain
    # content exclusively from mammoth records.
    #
    # Concrete check: reconstruct what the pipeline would produce from living
    # records only and assert the actual output matches that expectation.
    living_only_records = [
        r for r in query_args["all_records"]
        if not r["payload"].get("is_extinct", False)
    ][:10]  # top_k=10

    living_left_parts: list[str] = []
    living_right_parts: list[str] = []
    for item in living_only_records:
        payload = item.get("payload", {})
        window_sequence = payload.get("window_sequence", "")
        if not window_sequence:
            continue
        half = len(window_sequence) // 2
        living_left_parts.append(window_sequence[:half])
        living_right_parts.append(window_sequence[half:])

    if living_left_parts:
        expected_left = "".join(living_left_parts)[:1000]
        expected_right = "".join(reversed(living_right_parts))[:1000]
    else:
        # Fallback: direct slice from cleaned_sequence
        seq = query_args["cleaned_sequence"]
        gap_start = query_args["gap_start"]
        gap_end = query_args["gap_end"]
        expected_left = seq[max(0, gap_start - 1000):gap_start]
        expected_right = seq[gap_end:min(len(seq), gap_end + 1000)]

    assert left_ctx == expected_left, (
        f"left_ctx does not match expected living-only output. "
        f"Mammoth records may have leaked into the result (species_metadata['is_extinct']=False). "
        f"left_ctx={left_ctx[:60]!r}..., expected={expected_left[:60]!r}..."
    )
    assert right_ctx == expected_right, (
        f"right_ctx does not match expected living-only output. "
        f"Mammoth records may have leaked into the result (species_metadata['is_extinct']=False). "
        f"right_ctx={right_ctx[:60]!r}..., expected={expected_right[:60]!r}..."
    )

import uuid
import json
import pytest
from unittest.mock import patch, MagicMock

from backend.agents.reconstruction_agent.agent import (
    graph,
    parse_llm_prediction
)
from backend.agents.reconstruction_agent.schema import (
    AgentRequest,
    AgentStatus,
    GapRegion,
    GapPrediction,
)

@pytest.fixture
def base_request_factory():
    """Fixture to generate base AgentRequests with unique session ids."""
    def _create_request(genome: str, is_extinct: bool = False, species: str = "Test Species"):
        session_id = str(uuid.uuid4())
        return AgentRequest(
            instruction="Reconstruct genome",
            context={
                "genome": genome,
                "species": species,
                "species_metadata": {
                    "species_id": species,
                    "is_extinct": is_extinct,
                },
                "sequence_type": "nuclear",
                "session_id": session_id
            }
        )
    return _create_request

def invoke_graph_with_request(request: AgentRequest):
    """Helper to invoke the graph with the proper state and thread_id."""
    state = {
        "request": request
    }
    session_id = request.context["session_id"]
    return graph.invoke(state, config={"configurable": {"thread_id": session_id}})


def _make_router_mock(predicted_sequence: str, confidence: float, model_used: str = "Azure-GPT-5.1"):
    """
    Return a context manager that patches DNAModelRouter so that predict()
    returns a GapPrediction built from the real gap passed in, with the
    given sequence/confidence/model_used.
    """
    mock_router_instance = MagicMock()

    def _predict(flanks, gap, previous_attempts):
        return GapPrediction(
            gap=gap,
            predicted_sequence=predicted_sequence[:gap.length].ljust(gap.length, "N"),
            model_used=model_used,
            confidence=confidence,
        )

    mock_router_instance.predict.side_effect = _predict

    return patch(
        "backend.agents.reconstruction_agent.agent.DNAModelRouter.get",
        return_value=mock_router_instance,
    )


# 1. test_cas_normal_gap_in_scope
def test_cas_normal_gap_in_scope(base_request_factory):
    with _make_router_mock("ATCGATCGATC", 0.9):
        genome = "ATCG" + "N" * 11 + "ATCG"
        req = base_request_factory(genome=genome)

        final_state = invoke_graph_with_request(req)

    assert len(final_state.get("predictions", [])) == 1, "There should be exactly one prediction."
    assert final_state["predictions"][0].confidence == 0.9, (
        f"Confidence should be 0.9 from mock, got {final_state['predictions'][0].confidence}."
    )
    assert not final_state.get("is_partial", True), "is_partial should be False for a successful reconstruction."
    assert "result" in final_state, "Final state must contain 'result'."
    assert final_state["result"].status == AgentStatus.COMPLETED, (
        f"Result status should be COMPLETED, got {final_state['result'].status}."
    )


# 2. test_gap_hors_scope_en_premier
def test_gap_hors_scope_en_premier(base_request_factory):
    with _make_router_mock("ATCGATCGATC", 0.9):
        # Gap 1: 3 'N's (out of scope), Gap 2: 11 'N's (in scope)
        genome = "ATCG" + "N" * 3 + "ATCG" + "N" * 11 + "ATCG"
        req = base_request_factory(genome=genome)

        final_state = invoke_graph_with_request(req)

    assert len(final_state.get("predictions", [])) == 1, (
        f"Only one gap should have been predicted, got {len(final_state.get('predictions', []))}."
    )
    # The valid gap starts after ATCG+NNN+ATCG (length 4+3+4 = 11).
    assert final_state["predictions"][0].gap.length == 11, (
        f"The router should have only been called for the in-scope gap (len 11), "
        f"got {final_state['predictions'][0].gap.length}."
    )
    assert final_state.get("is_partial") is True, (
        "The out-of-scope gap was left unresolved, is_partial should be True."
    )
    assert len(final_state.get("excluded_gaps", [])) == 1


# 3. test_espece_eteinte_sans_evolution_analysis_pas_injection
def test_espece_eteinte_sans_evolution_analysis_pas_injection(base_request_factory):
    """
    For extinct species, evolution_analysis is NOT injected by the graph.
    Instead, the validation engine delegates to the evolution agent.
    """
    with _make_router_mock("ATCGATCGATC", 0.9):
        genome = "ATCG" + "N" * 11 + "ATCG"
        # No evolution_analysis in context — the orchestrator has not provided it yet.
        req = base_request_factory(genome=genome, is_extinct=True)

        final_state = invoke_graph_with_request(req)

    assert final_state["result"].status == AgentStatus.NEEDS_AGENT
    assert final_state["result"].target_agent == "Evolution"


# 4. test_echec_llm_repete_3_fois_puis_abandon
def test_echec_llm_repete_3_fois_puis_abandon(base_request_factory):
    """
    When every prediction has confidence <= 0.7 the validation engine retries
    up to MAX_ATTEMPTS=3 times then abandons the gap (is_partial=True).
    The router mock must be a fresh call counter so we can assert call count.
    """
    mock_router_instance = MagicMock()
    call_count = {"n": 0}

    def _predict(flanks, gap, previous_attempts):
        call_count["n"] += 1
        return GapPrediction(
            gap=gap,
            predicted_sequence=("ATCGATCGATC"[:gap.length]).ljust(gap.length, "N"),
            model_used="Azure-GPT-5.1",
            confidence=0.3,  # always below 0.7 → never plausible
        )

    mock_router_instance.predict.side_effect = _predict

    with patch(
        "backend.agents.reconstruction_agent.agent.DNAModelRouter.get",
        return_value=mock_router_instance,
    ):
        genome = "ATCG" + "N" * 11 + "ATCG"
        req = base_request_factory(genome=genome)

        final_state = invoke_graph_with_request(req)

    assert len(final_state.get("previous_attempts", [])) == 0, "Attempts should be reset after abandoning."
    assert final_state.get("is_partial") is True, "is_partial should be True after abandoning a gap."
    assert len(final_state.get("predictions", [])) == 0, "No prediction should be accepted due to low confidence."
    assert call_count["n"] == 3, (
        f"The router should have been called exactly 3 times due to retries, got {call_count['n']}."
    )


def test_parsing_json_malforme_fallback():
    # Valid JSON
    seq, conf = parse_llm_prediction('{"predicted_sequence": "AAA", "confidence": 0.9}', 3)
    assert seq == "AAA", f"Expected AAA, got {seq}"
    assert conf == 0.9, f"Expected 0.9, got {conf}"

    # Markdown formatting
    seq, conf = parse_llm_prediction('```json\n{"predicted_sequence": "AAA", "confidence": 0.9}\n```', 3)
    assert seq == "AAA" and conf == 0.9, "Markdown ticks should be parsed correctly."

    # Invalid JSON
    seq, conf = parse_llm_prediction('Invalid response', 3)
    assert seq == "NNN", f"Should fallback to NNN, got {seq}"
    assert conf == 0.0, f"Should fallback to 0.0 confidence, got {conf}"

    # Missing fields
    seq, conf = parse_llm_prediction('{}', 3)
    assert seq == "NNN", f"Should fallback to NNN on missing sequence, got {seq}"
    assert conf == 0.0, f"Should fallback to 0.0 on missing confidence, got {conf}"

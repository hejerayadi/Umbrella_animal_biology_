"""The scientific quality gates the Responder must not silently drop.

A worker agent can answer `completed` at the routing level while its own
scientific verdict is PARTIAL or REVISE - the Protein agent does exactly that
when a structure covers half the sequence. Those states live inside the
finding, where free-form prose renders them as just more text for the model to
paraphrase away. `_quality_constraints` lifts them into their own prompt
section so the answer has to carry the caveat.
"""

from backend.orchestrator.responder import _SYNTHESIS_PROMPT, _quality_constraints


def test_partial_protein_result_is_an_explicit_responder_constraint() -> None:
    constraints = _quality_constraints(
        {
            "protein_structure": {
                "status": "PARTIAL",
                "validation_status": "REVISE",
                "warnings": ["CRITIC_REVISE: selected structure covers only 50%."],
            }
        }
    )

    assert "status=PARTIAL" in constraints
    assert "validation_status=REVISE" in constraints
    assert "CRITIC_REVISE" in constraints


def test_clean_result_adds_no_quality_constraint() -> None:
    constraints = _quality_constraints(
        {"protein_structure": {"status": "COMPLETED", "validation_status": "ACCEPT", "warnings": []}}
    )

    assert constraints == "(none reported)"


def test_warnings_alone_are_reported_on_an_otherwise_clean_result() -> None:
    """A degraded provider is worth flagging even when the verdict is ACCEPT."""
    constraints = _quality_constraints(
        {
            "protein_structure": {
                "status": "COMPLETED",
                "validation_status": "ACCEPT",
                "warnings": ["RETRIEVAL_UNAVAILABLE: the knowledge base did not answer."],
            }
        }
    )

    assert "RETRIEVAL_UNAVAILABLE" in constraints


def test_plain_findings_are_not_mistaken_for_quality_metadata() -> None:
    """Most agents return a bare string; only a dict can carry a verdict."""
    assert _quality_constraints({"genome": "ACGT...", "traits": ["fur growth"]}) == "(none reported)"


def test_an_unstated_status_is_not_reported_as_a_problem() -> None:
    """`status: None` means "not stated", not a status literally named NONE.

    Reported as a constraint it would make the model hedge an answer that has
    nothing wrong with it.
    """
    assert _quality_constraints({"result": {"status": None}}) == "(none reported)"
    assert _quality_constraints({"result": {"validation_status": None}}) == "(none reported)"


def test_every_finding_that_is_degraded_is_listed() -> None:
    constraints = _quality_constraints(
        {
            "protein_structure": {"status": "PARTIAL"},
            "genome": "ACGT...",
            "literature": {"status": "NO_RESULTS", "warnings": ["Nothing indexed for this species."]},
        }
    )

    assert "protein_structure" in constraints
    assert "literature" in constraints
    assert "genome" not in constraints


def test_the_synthesis_prompt_actually_asks_for_the_constraints() -> None:
    """Guards the wiring, not the formatting.

    `_quality_constraints` returning the right text buys nothing if the prompt
    stops interpolating it: LangChain drops an input the template does not
    declare without raising, so the caveats would vanish silently.
    """
    assert "quality_constraints" in _SYNTHESIS_PROMPT.input_variables

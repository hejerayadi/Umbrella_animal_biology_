from uuid import uuid4

from app.domain.enums import AnalysisStatus, ValidationStatus
from app.orchestrators.protein import routers
from app.orchestrators.protein.nodes.names import (
    BUILD_EVIDENCE,
    COMPLETE,
    GENERATE_EXPLANATION,
    MAP_RESIDUES,
    NEEDS_CLARIFICATION,
    RESOLVE_IDENTITY,
    RETURN_PARTIAL,
    RUN_CRITIC,
    SCIENTIFIC_ABSTAIN,
    SEARCH_ALPHAFOLD,
    SELECT_STRUCTURE,
)
from app.orchestrators.protein.state import initial_state
from tests.factories import P53, alphafold_candidate, pdb_candidate, structure_request


def state(**overrides):  # type: ignore[no-untyped-def]
    base = initial_state(overrides.pop("task", structure_request()), uuid4())
    base.update(overrides)
    return base


def test_invalid_input_asks_for_clarification() -> None:
    assert (
        routers.route_after_validation(state(current_status=AnalysisStatus.needs_clarification))
        == NEEDS_CLARIFICATION
    )
    assert routers.route_after_validation(state(current_status=AnalysisStatus.validating)) == RESOLVE_IDENTITY


def test_unresolved_identity_abstains_instead_of_guessing() -> None:
    assert routers.route_after_identity(state(resolved_protein=None)) == SCIENTIFIC_ABSTAIN


def test_confirmed_identity_fans_out_to_the_three_sources() -> None:
    assert routers.route_after_identity(state(resolved_protein=P53)) == routers.PARALLEL_RETRIEVAL


def test_alphafold_runs_only_when_no_pdb_candidate_qualifies() -> None:
    assert (
        routers.route_after_pdb_evaluation(state(valid_pdb_candidates=[pdb_candidate()])) == SELECT_STRUCTURE
    )
    assert routers.route_after_pdb_evaluation(state(valid_pdb_candidates=[])) == SEARCH_ALPHAFOLD


def test_preferred_source_overrides_the_default_order() -> None:
    pdb_only = structure_request(preferred_source="PDB")
    assert (
        routers.route_after_pdb_evaluation(state(task=pdb_only, valid_pdb_candidates=[])) == SELECT_STRUCTURE
    )

    predicted = structure_request(preferred_source="ALPHAFOLD")
    assert (
        routers.route_after_pdb_evaluation(state(task=predicted, valid_pdb_candidates=[pdb_candidate()]))
        == SEARCH_ALPHAFOLD
    )


def test_sifts_is_requested_only_for_positional_questions() -> None:
    plain = state(selected_structure=pdb_candidate())
    assert routers.route_mapping_required(plain) == BUILD_EVIDENCE

    with_residue = state(task=structure_request(residue_position=273), selected_structure=pdb_candidate())
    assert routers.route_mapping_required(with_residue) == MAP_RESIDUES

    with_mutation = state(task=structure_request(mutation="R273H"), selected_structure=pdb_candidate())
    assert routers.route_mapping_required(with_mutation) == MAP_RESIDUES

    with_region = state(
        task=structure_request(requested_regions=("DNA-binding domain",)),
        selected_structure=alphafold_candidate(),
    )
    assert routers.route_mapping_required(with_region) == MAP_RESIDUES


def test_mapping_is_skipped_without_a_structure() -> None:
    no_structure = state(task=structure_request(residue_position=273), selected_structure=None)
    assert routers.route_mapping_required(no_structure) == BUILD_EVIDENCE


def test_explanation_is_optional() -> None:
    assert routers.route_explanation(state()) == GENERATE_EXPLANATION
    assert routers.route_explanation(state(task=structure_request(include_explanation=False))) == RUN_CRITIC


def test_critic_verdict_decides_the_terminal_node() -> None:
    assert routers.route_after_critic(state(validation_status=ValidationStatus.accept)) == COMPLETE
    assert routers.route_after_critic(state(validation_status=ValidationStatus.revise)) == RETURN_PARTIAL
    assert routers.route_after_critic(state(validation_status=ValidationStatus.abstain)) == SCIENTIFIC_ABSTAIN


def test_warnings_downgrade_an_accepted_run_to_partial() -> None:
    warned = state(validation_status=ValidationStatus.accept, warnings=["RETRIEVAL_UNAVAILABLE: down"])
    assert routers.route_after_critic(warned) == RETURN_PARTIAL

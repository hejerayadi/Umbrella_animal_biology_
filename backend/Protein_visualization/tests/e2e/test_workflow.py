"""End-to-end workflow scenarios from the implementation document (§20.3).

Every biological source is a stub, so these run offline and deterministically.
E2E-08 (idempotence) is deferred with the PostgreSQL persistence layer.
"""

from typing import Any

import pytest

from app.domain.enums import AnalysisStatus, ValidationStatus
from app.domain.exceptions import UpstreamServiceError
from app.domain.models import KnowledgeHit
from tests.factories import agent_task
from tests.fakes import (
    FakeAlphaFold,
    FakeRCSB,
    FakeRetriever,
    FakeSifts,
    FakeUniProt,
    build_orchestrator,
    fixture,
)


async def test_e2e_01_experimental_structure_is_selected() -> None:
    response = await build_orchestrator(
        retriever=FakeRetriever([KnowledgeHit(id="doc-1", text="p53 binds DNA.", score=0.8)])
    ).analyze(agent_task())

    assert response.status is AnalysisStatus.completed
    assert response.validation_status is ValidationStatus.accept
    assert response.selected_structure is not None
    assert response.selected_structure.source == "RCSB_PDB"
    assert response.selected_structure.structure_type == "EXPERIMENTAL"
    assert response.protein is not None and response.protein.uniprot_accession == "P04637"
    assert response.molstar_config["structure"]["id"] == "1TUP"
    assert response.warnings == []
    assert response.explanation is not None


async def test_e2e_02_alphafold_fallback_is_marked_predicted() -> None:
    response = await build_orchestrator(rcsb=FakeRCSB(ids=[])).analyze(agent_task())

    assert response.selected_structure is not None
    assert response.selected_structure.source == "ALPHAFOLD_DB"
    assert response.selected_structure.structure_type == "PREDICTED"
    assert response.validation_status is ValidationStatus.revise
    assert any(warning.startswith("ALPHAFOLD_FALLBACK") for warning in response.warnings)


async def test_e2e_03_ambiguous_identity_abstains() -> None:
    entry = fixture("uniprot_P04637.json")
    ambiguous = FakeUniProt(matches=[entry, {**entry, "primaryAccession": "P04638"}])
    task = agent_task(uniprot_accession=None)

    response = await build_orchestrator(uniprot=ambiguous).analyze(task)

    assert response.status is AnalysisStatus.failed
    assert response.validation_status is ValidationStatus.abstain
    assert response.selected_structure is None
    assert any("IDENTITY_UNRESOLVED" in warning for warning in response.warnings)


async def test_e2e_04_pdb_timeout_still_returns_a_usable_result() -> None:
    timed_out = FakeRCSB(error=UpstreamServiceError("RCSB PDB", "request timed out"))

    response = await build_orchestrator(rcsb=timed_out).analyze(agent_task())

    assert response.status is AnalysisStatus.partial
    assert response.selected_structure is not None
    assert response.selected_structure.structure_type == "PREDICTED"
    assert any(warning.startswith("PDB_TIMEOUT") for warning in response.warnings)


async def test_e2e_05_no_structure_anywhere() -> None:
    response = await build_orchestrator(rcsb=FakeRCSB(ids=[]), alphafold=FakeAlphaFold([])).analyze(
        agent_task()
    )

    assert response.status is AnalysisStatus.no_structure_found
    assert response.selected_structure is None
    assert response.validation_status is ValidationStatus.abstain
    assert response.molstar_config == {}


async def test_e2e_06_unmappable_residue_is_not_highlighted() -> None:
    empty_mapping = FakeSifts(payload={"1tup": {"UniProt": {"P04637": {"mappings": []}}}})

    response = await build_orchestrator(sifts=empty_mapping).analyze(agent_task(residue_position=273))

    assert response.residue_mappings == []
    assert response.molstar_config["selections"] == []
    assert response.validation_status in {ValidationStatus.revise, ValidationStatus.abstain}
    assert any("RESIDUE_NOT_OBSERVED" in warning for warning in response.warnings)


async def test_e2e_06b_mapped_residue_is_highlighted() -> None:
    response = await build_orchestrator().analyze(agent_task(residue_position=273))

    assert response.residue_mappings[0]["pdb_residue_number"] == "273"
    assert response.molstar_config["selections"][0]["residue_number"] == "273"
    assert response.validation_status is ValidationStatus.accept


async def test_e2e_07_knowledge_base_down_degrades_the_explanation_only() -> None:
    response = await build_orchestrator(
        retriever=FakeRetriever(error=RuntimeError("connection refused"))
    ).analyze(agent_task())

    assert response.selected_structure is not None
    assert response.status is AnalysisStatus.partial
    assert any(warning.startswith("RETRIEVAL_UNAVAILABLE") for warning in response.warnings)
    assert response.explanation is not None


async def test_invalid_input_asks_for_clarification_without_calling_a_source() -> None:
    exploding = FakeUniProt()

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("UniProt must not be called for an invalid task")

    exploding.get_entry = fail  # type: ignore[method-assign]
    response = await build_orchestrator(uniprot=exploding).analyze(agent_task(mutation="not-a-mutation"))

    assert response.status is AnalysisStatus.needs_clarification
    assert response.validation_status is ValidationStatus.abstain
    assert any("NEEDS_CLARIFICATION" in warning for warning in response.warnings)


async def test_mutation_notation_drives_the_sifts_lookup() -> None:
    response = await build_orchestrator().analyze(agent_task(mutation="R273H"))

    assert response.residue_mappings[0]["uniprot_position"] == 273


@pytest.mark.parametrize("include_explanation", [True, False])
async def test_explanation_is_optional(include_explanation: bool) -> None:
    response = await build_orchestrator().analyze(agent_task(include_explanation=include_explanation))

    assert (response.explanation is not None) is include_explanation

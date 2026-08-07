from app.capabilities.critic import CriticCapability
from app.capabilities.evidence import EvidenceCapability
from app.capabilities.explanation import ExplanationCapability
from app.capabilities.structures import StructureCapability
from app.capabilities.visualization import VisualizationCapability
from app.domain.enums import ValidationStatus
from app.domain.models import CriticReport, EvidencePack, EvidenceRef, KnowledgeHit
from app.llm.schemas import CriticOutput, ExplanationOutput
from tests.factories import (
    P53,
    alphafold_candidate,
    annotation,
    pdb_candidate,
    residue_mapping,
    structure_request,
)
from tests.fakes import FakeAlphaFold, FakeRCSB


class StubModel:
    def __init__(self, explanation: ExplanationOutput | None = None, critic: CriticOutput | None = None):
        self._explanation = explanation
        self._critic = critic
        self.seen: list[dict[str, object]] = []

    @property
    def enabled(self) -> bool:
        return True

    async def explain(self, context: dict[str, object]) -> ExplanationOutput:
        self.seen.append(context)
        if self._explanation is None:
            raise RuntimeError("model unavailable")
        return self._explanation

    async def critique(self, context: dict[str, object]) -> CriticOutput:
        self.seen.append(context)
        if self._critic is None:
            raise RuntimeError("model unavailable")
        return self._critic


# --- selection ---------------------------------------------------------------


def test_experimental_scores_reward_coverage_and_resolution() -> None:
    assert StructureCapability.score_pdb(0.9, 1.5) > StructureCapability.score_pdb(0.9, 3.5)
    assert StructureCapability.score_pdb(0.9, 2.0) > StructureCapability.score_pdb(0.4, 2.0)


async def test_experimental_structure_uses_real_rcsb_entity_fields() -> None:
    capability = StructureCapability(FakeRCSB(), FakeAlphaFold())  # type: ignore[arg-type]

    candidates = await capability.experimental(P53)

    assert len(candidates) == 1
    assert candidates[0].external_id == "1TUP"
    assert candidates[0].chain_id == "A"
    assert candidates[0].sequence_coverage == 0.72


def test_experimental_structures_outrank_predictions_by_default() -> None:
    selected, alternatives = StructureCapability.select(
        structure_request(), [pdb_candidate()], [alphafold_candidate()]
    )
    assert selected is not None and selected.external_id == "1TUP"
    assert [item.external_id for item in alternatives] == ["AF-P04637-F1"]


def test_no_candidate_yields_no_selection() -> None:
    assert StructureCapability.select(structure_request(), [], []) == (None, [])


# --- visualization -----------------------------------------------------------


def test_only_observed_residues_are_highlighted() -> None:
    capability = VisualizationCapability()
    spec = capability.build(
        pdb_candidate(),
        [
            residue_mapping(),
            residue_mapping(uniprot_position=175, is_observed=False, pdb_residue_number=None),
        ],
    )
    assert spec is not None
    assert [item["residue_number"] for item in spec.selections] == ["273"]


def test_domain_spans_are_not_applied_to_experimental_numbering() -> None:
    capability = VisualizationCapability()
    structure = pdb_candidate()
    config = capability.to_config(capability.build(structure, []), structure, [annotation()])
    assert config["domains"][0]["coordinate_space"] == "uniprot"
    assert config["domains"][0]["applicable"] is False


def test_domain_spans_apply_to_alphafold_numbering() -> None:
    capability = VisualizationCapability()
    structure = alphafold_candidate()
    config = capability.to_config(capability.build(structure, []), structure, [annotation()])
    assert config["domains"][0]["applicable"] is True
    assert config["structure"]["structure_type"] == "PREDICTED"


# --- evidence ----------------------------------------------------------------


def test_evidence_pack_marks_a_prediction_as_predicted() -> None:
    pack = EvidenceCapability().build(
        request=structure_request(),
        protein=P53,
        structure=alphafold_candidate(),
        annotations=[],
        mappings=[],
        knowledge=[],
        evidence=[],
        warnings=[],
    )
    assert any("predicted AlphaFold model" in fact for fact in pack.facts)
    assert any("must not be described as experimental" in item for item in pack.limitations)


def test_evidence_pack_reports_an_unmapped_residue_as_a_limitation() -> None:
    pack = EvidenceCapability().build(
        request=structure_request(residue_position=273),
        protein=P53,
        structure=pdb_candidate(),
        annotations=[],
        mappings=[],
        knowledge=[],
        evidence=[],
        warnings=[],
    )
    assert any("No SIFTS mapping" in item for item in pack.limitations)


def test_evidence_pack_quotes_indexed_documents() -> None:
    hit = KnowledgeHit(
        id="doc-1", text="p53 binds DNA as a tetramer.", score=0.9, metadata={"source": "UniProt"}
    )
    pack = EvidenceCapability().build(
        request=structure_request(),
        protein=P53,
        structure=pdb_candidate(),
        annotations=[annotation()],
        mappings=[residue_mapping()],
        knowledge=[hit],
        evidence=[EvidenceRef("UniProt", "P04637", "2026-08-06T00:00:00Z")],
        warnings=[],
    )
    assert any("doc-1" in fact for fact in pack.facts)
    assert any("IPR011615" in fact for fact in pack.facts)
    assert any("SIFTS maps UniProt position 273" in fact for fact in pack.facts)
    assert len(pack.evidence) == 1


# --- explanation -------------------------------------------------------------


async def test_explanation_falls_back_to_the_facts_when_the_model_fails() -> None:
    pack = EvidencePack(
        facts=("UniProt P04637 is the canonical entry.",), limitations=("Coverage is partial.",)
    )
    explanation = await ExplanationCapability(StubModel()).explain(pack)
    assert explanation.generated is False
    assert "canonical entry" in explanation.summary
    assert explanation.limitations == ("Coverage is partial.",)


async def test_the_model_only_ever_sees_the_evidence_pack() -> None:
    model = StubModel(explanation=ExplanationOutput(summary="Grounded summary.", limitations=[]))
    pack = EvidencePack(
        facts=("fact",), limitations=("limit",), evidence=(EvidenceRef("UniProt", "P04637", "now"),)
    )
    explanation = await ExplanationCapability(model).explain(pack)
    assert explanation.generated is True
    assert set(model.seen[0]) == {"facts", "limitations", "sources"}


# --- critic ------------------------------------------------------------------


def test_missing_structure_abstains() -> None:
    report = CriticCapability().review(structure_request(), P53, None, [])
    assert report.verdict == ValidationStatus.abstain.value


def test_unmapped_requested_residue_forces_revise() -> None:
    report = CriticCapability().review(structure_request(residue_position=273), P53, pdb_candidate(), [])
    assert report.verdict == ValidationStatus.revise.value


def test_prediction_forces_revise() -> None:
    report = CriticCapability().review(structure_request(), P53, alphafold_candidate(), [])
    assert report.verdict == ValidationStatus.revise.value


def test_complete_evidence_is_accepted() -> None:
    report = CriticCapability().review(
        structure_request(residue_position=273), P53, pdb_candidate(), [residue_mapping()]
    )
    assert report.verdict == ValidationStatus.accept.value


async def test_the_model_may_tighten_the_verdict() -> None:
    model = StubModel(
        critic=CriticOutput(verdict="ABSTAIN", reasons=["Coverage does not include the residue."])
    )
    audited = await CriticCapability().audit(
        CriticReport(verdict="REVISE", reasons=("deterministic",)), EvidencePack(), model
    )
    assert audited.verdict == "ABSTAIN"
    assert "deterministic" in audited.reasons


async def test_the_model_may_not_loosen_the_verdict() -> None:
    model = StubModel(critic=CriticOutput(verdict="ACCEPT", reasons=["Looks fine to me."]))
    audited = await CriticCapability().audit(
        CriticReport(verdict="REVISE", reasons=("deterministic",)), EvidencePack(), model
    )
    assert audited.verdict == "REVISE"
    assert audited.reasons == ("deterministic",)


async def test_a_failing_model_leaves_the_deterministic_verdict_intact() -> None:
    audited = await CriticCapability().audit(
        CriticReport(verdict="ACCEPT", reasons=("deterministic",)), EvidencePack(), StubModel()
    )
    assert audited.verdict == "ACCEPT"

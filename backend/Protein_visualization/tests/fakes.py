"""Offline stand-ins for the biological sources, plus a wired-up orchestrator."""

import json
from pathlib import Path
from typing import Any

from app.capabilities.annotations import AnnotationCapability
from app.capabilities.critic import CriticCapability
from app.capabilities.evidence import EvidenceCapability
from app.capabilities.explanation import ExplanationCapability
from app.capabilities.identity import IdentityCapability
from app.capabilities.residue_mapping import ResidueMappingCapability
from app.capabilities.retrieval import RetrievalCapability
from app.capabilities.structures import StructureCapability
from app.capabilities.visualization import VisualizationCapability
from app.domain.models import KnowledgeHit
from app.orchestrators.protein.graph import build_graph
from app.orchestrators.protein.nodes import ProteinNodes
from app.orchestrators.protein.orchestrator import ProteinOrchestrator

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeUniProt:
    def __init__(self, entry: dict[str, Any] | None = None, matches: list[dict[str, Any]] | None = None):
        self.entry = entry if entry is not None else fixture("uniprot_P04637.json")
        self.matches = matches

    async def get_entry(self, accession: str) -> dict[str, Any]:
        return self.entry

    async def search(self, query: str, organism: str | None = None) -> list[dict[str, Any]]:
        return self.matches if self.matches is not None else [self.entry]


class FakeRCSB:
    def __init__(self, ids: list[str] | None = None, error: Exception | None = None, entry: Any = None):
        self.ids = ids if ids is not None else ["1TUP_1"]
        self.error = error
        payload = fixture("rcsb_1TUP.json")
        self.entry_payload = entry if entry is not None else payload["entry"]
        self.entity_payload = payload["polymer_entity"]

    async def find_by_uniprot(self, accession: str, limit: int = 10) -> list[str]:
        if self.error:
            raise self.error
        return self.ids

    async def entry(self, pdb_id: str) -> dict[str, Any]:
        return self.entry_payload

    async def polymer_entity(self, pdb_id: str, entity_id: str) -> dict[str, Any]:
        return self.entity_payload

    def structure_urls(self, pdb_id: str) -> dict[str, str]:
        return {"data": f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"}


class FakeAlphaFold:
    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records = records if records is not None else fixture("alphafold_P04637.json")

    async def predictions(self, accession: str) -> list[dict[str, Any]]:
        return self.records


class FakeInterPro:
    def __init__(self, error: Exception | None = None):
        self.error = error

    async def annotations(self, accession: str) -> list[dict[str, Any]]:
        if self.error:
            raise self.error
        return list(fixture("interpro_P04637.json")["results"])


class FakeSifts:
    def __init__(self, payload: Any = None, error: Exception | None = None):
        self.payload = payload if payload is not None else fixture("sifts_1TUP.json")
        self.error = error

    async def mappings(self, pdb_id: str) -> dict[str, Any]:
        if self.error:
            raise self.error
        return self.payload


class FakeRetriever:
    def __init__(self, hits: list[KnowledgeHit] | None = None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error

    async def search(self, query: str, limit: int, **filters: Any) -> list[KnowledgeHit]:
        if self.error:
            raise self.error
        return self.hits


def build_orchestrator(
    uniprot: FakeUniProt | None = None,
    rcsb: FakeRCSB | None = None,
    alphafold: FakeAlphaFold | None = None,
    interpro: FakeInterPro | None = None,
    sifts: FakeSifts | None = None,
    retriever: FakeRetriever | None = None,
) -> ProteinOrchestrator:
    nodes = ProteinNodes.build(
        identity=IdentityCapability(uniprot or FakeUniProt()),  # type: ignore[arg-type]
        structures=StructureCapability(rcsb or FakeRCSB(), alphafold or FakeAlphaFold()),  # type: ignore[arg-type]
        annotations=AnnotationCapability(interpro or FakeInterPro()),  # type: ignore[arg-type]
        retrieval=RetrievalCapability(retriever or FakeRetriever()),  # type: ignore[arg-type]
        residue_mapping=ResidueMappingCapability(sifts or FakeSifts()),  # type: ignore[arg-type]
        evidence=EvidenceCapability(),
        visualization=VisualizationCapability(),
        explanation=ExplanationCapability(None),
        critic=CriticCapability(),
        min_sequence_coverage=0.3,
    )
    return ProteinOrchestrator(build_graph(nodes))

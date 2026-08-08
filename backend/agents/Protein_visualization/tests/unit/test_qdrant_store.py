from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from backend.agents.Protein_visualization.app.knowledge_base.qdrant import QdrantStore
from backend.agents.Protein_visualization.app.knowledge_base.schemas import KnowledgeDocument


def _document() -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id="uniprot:P04637:function:0",
        text="TP53 acts as a tumor suppressor.",
        source="UniProt",
        source_record_id="P04637",
        protein_id="P04637",
        gene_symbol="TP53",
        species_name="Homo sapiens",
        taxonomy_id=9606,
        document_type="protein_function",
        section="function",
    )


async def test_upsert_maps_domain_document_id_to_stable_qdrant_uuid() -> None:
    store = QdrantStore.__new__(QdrantStore)
    store.collection = "protein_knowledge"
    store.client = AsyncMock()
    document = _document()

    assert await store.upsert([document], [[0.1, 0.2]]) == 1

    point = store.client.upsert.await_args.kwargs["points"][0]
    UUID(str(point.id))
    assert point.payload["document_id"] == document.document_id


async def test_search_returns_the_domain_document_id_from_payload() -> None:
    store = QdrantStore.__new__(QdrantStore)
    store.collection = "protein_knowledge"
    store.client = AsyncMock()
    document = _document()
    store.client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="91fd874e-50de-4f11-8746-63d55c976a2a", score=0.8, payload=document.model_dump(mode="json")
            )
        ]
    )

    hits = await store.search([0.1, 0.2], 5, "P04637", 9606)

    assert [hit.id for hit in hits] == [document.document_id]

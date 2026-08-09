from app.domain.models import KnowledgeHit
from app.knowledge_base.embeddings import EmbeddingProvider, HashEmbedding
from app.knowledge_base.qdrant import QdrantStore
from app.knowledge_base.schemas import KnowledgeDocument


class KnowledgeBase:
    def __init__(
        self,
        embedding: EmbeddingProvider | None = None,
        store: QdrantStore | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        self.embedding = embedding or HashEmbedding()
        self.store = store
        self.unavailable_reason = unavailable_reason
        self._documents: dict[str, tuple[KnowledgeDocument, list[float]]] = {}

    async def ingest(self, documents: list[KnowledgeDocument]) -> int:
        vectors = [await self.embedding.embed(document.text) for document in documents]
        if self.store:
            await self.store.ensure_collection(self.embedding.dimensions)
            return await self.store.upsert(documents, vectors)
        self._documents.update(
            {
                document.document_id: (document, vector)
                for document, vector in zip(documents, vectors, strict=True)
            }
        )
        return len(documents)

    async def search(
        self,
        query: str,
        limit: int,
        protein_id: str | None = None,
        taxonomy_id: int | None = None,
        document_types: list[str] | None = None,
    ) -> list[KnowledgeHit]:
        if self.unavailable_reason:
            return []
        if not protein_id or taxonomy_id is None:
            return []
        vector = await self.embedding.embed(query)
        if self.store:
            return await self.store.search(vector, limit, protein_id, taxonomy_id, document_types)
        scored = []
        for document, candidate in self._documents.values():
            if document.protein_id != protein_id or document.taxonomy_id != taxonomy_id:
                continue
            if document_types and document.document_type not in document_types:
                continue
            score = sum(left * right for left, right in zip(vector, candidate, strict=True))
            scored.append(
                KnowledgeHit(
                    id=document.document_id,
                    text=document.text,
                    score=max(0.0, min(1.0, (score + 1.0) / 2.0)),
                    metadata=document.model_dump(mode="json"),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

from backend.agents.Protein_visualization.app.domain.models import KnowledgeHit
from backend.agents.Protein_visualization.app.knowledge_base.embeddings import EmbeddingProvider
from backend.agents.Protein_visualization.app.knowledge_base.qdrant import QdrantStore
from backend.agents.Protein_visualization.app.knowledge_base.schemas import KnowledgeDocument


class KnowledgeBaseUnavailableError(RuntimeError):
    """Raised when production retrieval has no usable vector-store configuration."""


class KnowledgeBase:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        store: QdrantStore | None = None,
        unavailable_reason: str | None = None,
    ) -> None:
        # Required, with no default: the only other provider in this package is
        # HashEmbedding, whose vectors carry no semantics. Defaulting to it would
        # let a misconfiguration return confident-looking nonsense from a
        # production search instead of failing.
        self.embedding = embedding
        self.store = store
        self.unavailable_reason = unavailable_reason
        self._documents: dict[str, tuple[KnowledgeDocument, list[float]]] = {}

    def _require_available(self) -> None:
        if self.unavailable_reason:
            raise KnowledgeBaseUnavailableError(self.unavailable_reason)

    async def ingest(self, documents: list[KnowledgeDocument]) -> int:
        self._require_available()
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
        # Fail before embedding so a missing Qdrant configuration neither
        # downloads BGE-M3 nor looks like a valid search with zero matches.
        self._require_available()
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

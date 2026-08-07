from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from app.domain.models import KnowledgeHit
from app.knowledge_base.schemas import KnowledgeDocument


class QdrantStore:
    def __init__(self, url: str, collection: str, api_key: str | None = None, timeout: float = 15.0) -> None:
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=max(1, int(timeout)),
            check_compatibility=False,
        )
        self.collection = collection

    async def healthy(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def ensure_collection(self, dimensions: int) -> None:
        """Create and validate the collection used by production retrieval.

        Existing collections are never recreated automatically: deleting one to
        repair a vector-size mismatch would destroy indexed scientific evidence.
        """
        if await self.client.collection_exists(self.collection):
            info = await self.client.get_collection(self.collection)
            configured_vectors = info.config.params.vectors
            actual_dimensions = getattr(configured_vectors, "size", None)
            if actual_dimensions != dimensions:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' has vector size "
                    f"{actual_dimensions}; expected {dimensions}"
                )
        else:
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
            )

        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        indexes = {
            "domain": models.PayloadSchemaType.KEYWORD,
            "protein_id": models.PayloadSchemaType.KEYWORD,
            "taxonomy_id": models.PayloadSchemaType.INTEGER,
            "document_type": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, field_schema in indexes.items():
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    async def readiness(self, dimensions: int) -> tuple[bool, str]:
        """Return collection-level readiness without exposing credentials."""
        try:
            if not await self.client.collection_exists(self.collection):
                return False, f"collection={self.collection} missing"
            info = await self.client.get_collection(self.collection)
            configured_vectors = info.config.params.vectors
            actual_dimensions = getattr(configured_vectors, "size", None)
            if actual_dimensions != dimensions:
                return (
                    False,
                    f"collection={self.collection} vector_size={actual_dimensions} expected={dimensions}",
                )
            count = await self.client.count(self.collection, exact=True)
            return True, f"collection={self.collection} vector_size={dimensions} points={count.count}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: Qdrant collection check failed"

    async def upsert(self, documents: list[KnowledgeDocument], vectors: list[list[float]]) -> int:
        await self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"umbrella:protein-knowledge:{document.document_id}")),
                    vector=vector,
                    payload=document.model_dump(mode="json"),
                )
                for document, vector in zip(documents, vectors, strict=True)
            ],
            wait=True,
        )
        return len(documents)

    async def search(
        self,
        vector: list[float],
        limit: int,
        protein_id: str,
        taxonomy_id: int,
        document_types: list[str] | None = None,
    ) -> list[KnowledgeHit]:
        must = [
            models.FieldCondition(key="domain", match=models.MatchValue(value="protein")),
            models.FieldCondition(key="protein_id", match=models.MatchValue(value=protein_id)),
            models.FieldCondition(key="taxonomy_id", match=models.MatchValue(value=taxonomy_id)),
        ]
        if document_types:
            must.append(models.FieldCondition(key="document_type", match=models.MatchAny(any=document_types)))
        response = await self.client.query_points(
            self.collection,
            query=vector,
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            KnowledgeHit(
                id=str((point.payload or {}).get("document_id", point.id)),
                text=str((point.payload or {}).get("text", "")),
                score=max(0.0, min(1.0, point.score)),
                metadata=dict(point.payload or {}),
            )
            for point in response.points
            if (point.payload or {}).get("source")
        ]

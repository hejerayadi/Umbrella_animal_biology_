from importlib import import_module
from types import ModuleType
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.domain.models import KnowledgeHit
from app.knowledge_base.schemas import KnowledgeDocument


class QdrantDependencyError(RuntimeError):
    """Raised when the optional Qdrant client cannot be loaded locally."""


def _load_qdrant_client() -> tuple[type[Any], ModuleType]:
    """Load Qdrant only when it is configured.

    qdrant-client imports its local SQLite backend from ``__init__`` even when
    the application only uses a remote cluster. Some managed Windows machines
    block that SQLite DLL. Keeping the import lazy lets the rest of the protein
    agent start and report Qdrant as degraded instead of crashing at import time.
    """
    try:
        qdrant_client = import_module("qdrant_client")
    except (ImportError, OSError) as exc:
        raise QdrantDependencyError(
            "qdrant-client could not be loaded; knowledge retrieval is disabled"
        ) from exc
    return qdrant_client.AsyncQdrantClient, qdrant_client.models


class QdrantStore:
    client: Any
    _models: ModuleType | None

    def __init__(self, url: str, collection: str, api_key: str | None = None, timeout: float = 15.0) -> None:
        async_qdrant_client, self._models = _load_qdrant_client()
        self.client = async_qdrant_client(
            url=url,
            api_key=api_key,
            timeout=max(1, int(timeout)),
            check_compatibility=False,
        )
        self.collection = collection

    @property
    def models(self) -> ModuleType:
        """Expose Qdrant models, including for lightweight test doubles."""
        models = getattr(self, "_models", None)
        if models is None:
            _, models = _load_qdrant_client()
            self._models = models
        return models

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
                vectors_config=self.models.VectorParams(
                    size=dimensions,
                    distance=self.models.Distance.COSINE,
                ),
            )

        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        indexes = {
            "domain": self.models.PayloadSchemaType.KEYWORD,
            "protein_id": self.models.PayloadSchemaType.KEYWORD,
            "taxonomy_id": self.models.PayloadSchemaType.INTEGER,
            "document_type": self.models.PayloadSchemaType.KEYWORD,
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
                self.models.PointStruct(
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
            self.models.FieldCondition(key="domain", match=self.models.MatchValue(value="protein")),
            self.models.FieldCondition(key="protein_id", match=self.models.MatchValue(value=protein_id)),
            self.models.FieldCondition(key="taxonomy_id", match=self.models.MatchValue(value=taxonomy_id)),
        ]
        if document_types:
            must.append(
                self.models.FieldCondition(
                    key="document_type",
                    match=self.models.MatchAny(any=document_types),
                )
            )
        response = await self.client.query_points(
            self.collection,
            query=vector,
            query_filter=self.models.Filter(must=must),
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

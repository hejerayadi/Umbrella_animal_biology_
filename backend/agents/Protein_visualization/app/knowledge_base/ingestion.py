import hashlib
from datetime import UTC, datetime

from backend.agents.Protein_visualization.app.knowledge_base.retrieval import KnowledgeBase
from backend.agents.Protein_visualization.app.knowledge_base.schemas import IngestionResult, KnowledgeDocument


class KnowledgeIngestionService:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self._hashes: dict[str, str] = {}

    async def ingest(self, documents: list[KnowledgeDocument]) -> IngestionResult:
        prepared = []
        skipped = 0
        for document in documents:
            normalized = " ".join(document.text.split())
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if self._hashes.get(document.document_id) == digest:
                skipped += 1
                continue
            self._hashes[document.document_id] = digest
            prepared.append(
                document.model_copy(
                    update={"text": normalized, "content_sha256": digest, "indexed_at": datetime.now(UTC)}
                )
            )
        inserted = await self.knowledge_base.ingest(prepared) if prepared else 0
        return IngestionResult(inserted=inserted, skipped=skipped)

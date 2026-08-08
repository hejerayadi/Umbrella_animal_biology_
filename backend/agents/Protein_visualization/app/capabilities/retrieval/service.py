from typing import Protocol

from backend.agents.Protein_visualization.app.domain.models import KnowledgeHit, ResolvedProtein


class KnowledgeRetriever(Protocol):
    async def search(
        self, query: str, limit: int, protein_id: str | None = None, taxonomy_id: int | None = None
    ) -> list[KnowledgeHit]: ...


class RetrievalCapability:
    def __init__(self, retriever: KnowledgeRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    async def retrieve(self, protein: ResolvedProtein) -> list[KnowledgeHit]:
        query = f"{protein.gene_symbol} {protein.protein_name or ''} structure function"
        return await self.retriever.search(
            query, self.top_k, protein_id=protein.uniprot_accession, taxonomy_id=protein.taxon_id
        )

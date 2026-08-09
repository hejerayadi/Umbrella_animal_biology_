from functools import lru_cache
from typing import TYPE_CHECKING

from backend.agents.Protein_visualization.app.capabilities.annotations import AnnotationCapability
from backend.agents.Protein_visualization.app.capabilities.critic import CriticCapability
from backend.agents.Protein_visualization.app.capabilities.evidence import EvidenceCapability
from backend.agents.Protein_visualization.app.capabilities.explanation import ExplanationCapability
from backend.agents.Protein_visualization.app.capabilities.identity import IdentityCapability
from backend.agents.Protein_visualization.app.capabilities.residue_mapping import ResidueMappingCapability
from backend.agents.Protein_visualization.app.capabilities.retrieval import RetrievalCapability
from backend.agents.Protein_visualization.app.capabilities.structures import StructureCapability
from backend.agents.Protein_visualization.app.capabilities.visualization import VisualizationCapability
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.knowledge_base.embeddings import (
    BgeM3Embedding,
    EmbeddingProvider,
)
from backend.agents.Protein_visualization.app.knowledge_base.ingestion import KnowledgeIngestionService
from backend.agents.Protein_visualization.app.knowledge_base.qdrant import QdrantDependencyError, QdrantStore
from backend.agents.Protein_visualization.app.knowledge_base.retrieval import KnowledgeBase
from backend.agents.Protein_visualization.app.llm.azure_foundry import AzureFoundryClient
from backend.agents.Protein_visualization.app.orchestrators.protein.graph import build_graph
from backend.agents.Protein_visualization.app.orchestrators.protein.nodes import ProteinNodes
from backend.agents.Protein_visualization.app.orchestrators.protein.orchestrator import ProteinOrchestrator
from backend.agents.Protein_visualization.app.tools import (
    AlphaFoldClient,
    InterProClient,
    RCSBClient,
    SiftsClient,
    UniProtClient,
)

if TYPE_CHECKING:  # SQLAlchemy is only imported when persistence is enabled
    from app.persistence.repositories import AnalysisRepository


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured production embedding provider."""
    settings = get_settings()
    wants_bge = settings.embedding_provider.lower() in {"bge-m3", "bge_m3", "bgem3"}
    if wants_bge:
        return BgeM3Embedding(settings.embedding_model, settings.embedding_dimensions)
    raise ValueError(f"Unsupported production embedding provider: {settings.embedding_provider}")


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    """Knowledge base backed by the managed Qdrant cluster when ``QDRANT_URL`` is set."""
    settings = get_settings()
    if not settings.qdrant_url:
        return KnowledgeBase(embedding=get_embedding_provider())
    try:
        store = QdrantStore(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.qdrant_api_key,
            settings.http_timeout_seconds,
        )
    except (QdrantDependencyError, ValueError) as exc:
        return KnowledgeBase(
            embedding=get_embedding_provider(),
            unavailable_reason=f"{type(exc).__name__}: {exc}",
        )
    return KnowledgeBase(embedding=get_embedding_provider(), store=store)


@lru_cache
def get_ingestion_service() -> KnowledgeIngestionService:
    return KnowledgeIngestionService(get_knowledge_base())


@lru_cache
def get_llm_client() -> AzureFoundryClient:
    return AzureFoundryClient(get_settings())


@lru_cache
def get_orchestrator() -> ProteinOrchestrator:
    settings = get_settings()
    llm = get_llm_client()
    nodes = ProteinNodes.build(
        identity=IdentityCapability(UniProtClient(settings)),
        structures=StructureCapability(
            RCSBClient(settings), AlphaFoldClient(settings), settings.max_structure_candidates
        ),
        annotations=AnnotationCapability(InterProClient(settings)),
        retrieval=RetrievalCapability(get_knowledge_base(), settings.retrieval_top_k),
        residue_mapping=ResidueMappingCapability(SiftsClient(settings)),
        evidence=EvidenceCapability(),
        visualization=VisualizationCapability(),
        explanation=ExplanationCapability(llm),
        critic=CriticCapability(),
        min_sequence_coverage=settings.min_sequence_coverage,
        llm=llm,
    )
    return ProteinOrchestrator(build_graph(nodes))


@lru_cache
def get_analysis_repository() -> "AnalysisRepository | None":
    """Returns ``None`` while ``PERSISTENCE_ENABLED`` is off (PostgreSQL is out of sprint scope)."""
    settings = get_settings()
    if not settings.persistence_enabled:
        return None
    from app.persistence.db import Database
    from app.persistence.repositories import AnalysisRepository

    return AnalysisRepository(Database(settings.database_url))

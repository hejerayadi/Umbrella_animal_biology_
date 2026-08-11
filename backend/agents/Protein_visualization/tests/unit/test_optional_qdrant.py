import pytest

from backend.agents.Protein_visualization.app.api.v1 import dependencies
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.knowledge_base import qdrant
from backend.agents.Protein_visualization.app.knowledge_base.retrieval import (
    KnowledgeBaseUnavailableError,
)


async def test_qdrant_import_failure_does_not_prevent_application_startup(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def blocked_import(_: str) -> None:
        raise ImportError("blocked by application control")

    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example.invalid")
    monkeypatch.setattr(qdrant, "import_module", blocked_import)
    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_knowledge_base.cache_clear()
    try:
        knowledge_base = dependencies.get_knowledge_base()

        assert knowledge_base.store is None
        assert knowledge_base.unavailable_reason is not None
        assert "knowledge retrieval is disabled" in knowledge_base.unavailable_reason
        with pytest.raises(KnowledgeBaseUnavailableError, match="knowledge retrieval is disabled"):
            await knowledge_base.search("TP53 structure", 5, "P04637", 9606)
    finally:
        dependencies.get_knowledge_base.cache_clear()
        dependencies.get_embedding_provider.cache_clear()
        get_settings.cache_clear()


async def test_missing_qdrant_configuration_is_reported_before_embedding(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("QDRANT_URL", "")
    get_settings.cache_clear()
    dependencies.get_embedding_provider.cache_clear()
    dependencies.get_knowledge_base.cache_clear()
    try:
        knowledge_base = dependencies.get_knowledge_base()

        assert knowledge_base.store is None
        assert knowledge_base.unavailable_reason is not None

        async def unexpected_embedding(_: str) -> list[float]:
            raise AssertionError("BGE-M3 must not be loaded without Qdrant")

        monkeypatch.setattr(knowledge_base.embedding, "embed", unexpected_embedding)
        with pytest.raises(KnowledgeBaseUnavailableError, match="QDRANT_URL is not configured"):
            await knowledge_base.search("TP53 structure", 5, "P04637", 9606)
    finally:
        dependencies.get_knowledge_base.cache_clear()
        dependencies.get_embedding_provider.cache_clear()
        get_settings.cache_clear()

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.v1.dependencies import get_ingestion_service, get_knowledge_base
from app.configuration.settings import get_settings
from app.contracts.envelope import ApiResponse, success
from app.domain.models import KnowledgeHit
from app.knowledge_base.ingestion import KnowledgeIngestionService
from app.knowledge_base.retrieval import KnowledgeBase
from app.knowledge_base.schemas import IngestionResult, KnowledgeDocument, KnowledgeSearchRequest
from app.observability.logging import log_stage

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

logger = logging.getLogger("app.knowledge")


@router.post("/search", response_model=ApiResponse[list[KnowledgeHit]])
async def search_knowledge(
    request: KnowledgeSearchRequest,
    knowledge_base: Annotated[KnowledgeBase, Depends(get_knowledge_base)],
) -> ApiResponse[list[KnowledgeHit]]:
    with log_stage(logger, "knowledge_search", capability="retrieval") as outcome:
        hits = await knowledge_base.search(
            request.query,
            request.limit,
            protein_id=request.protein_id,
            taxonomy_id=request.taxonomy_id,
            document_types=list(request.document_types) or None,
        )
        outcome["hits"] = len(hits)
    return success(hits)


@router.post(
    "/protein/ingestions",
    response_model=ApiResponse[IngestionResult],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_documents(
    documents: list[KnowledgeDocument],
    ingestion: Annotated[KnowledgeIngestionService, Depends(get_ingestion_service)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> ApiResponse[IngestionResult]:
    expected = get_settings().internal_ingestion_api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid ingestion API key")
    with log_stage(logger, "knowledge_ingestion", capability="ingestion") as outcome:
        result = await ingestion.ingest(documents)
        outcome["documents"] = len(documents)
        outcome["inserted"] = result.inserted
        outcome["skipped"] = result.skipped
    return success(result)

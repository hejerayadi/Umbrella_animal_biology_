
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

try:
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    _QDRANT_AVAILABLE = True
except ModuleNotFoundError:
    FieldCondition = Filter = MatchAny = MatchValue = Range = Any  # type: ignore[assignment]
    _QDRANT_AVAILABLE = False

try:
    from langsmith import traceable
except ModuleNotFoundError:
    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        """No-op fallback so retrieval works even if langsmith isn't installed
        (e.g. a slim deploy environment) -- tracing is purely observability,
        never a hard dependency for retrieval to function."""
        def _decorator(fn):
            return fn
        return _decorator

from kb.embeddings import embed_text
from kb.qdrant_store import COLLECTIONS, get_client

logger = logging.getLogger(__name__)

# The payload fields each collection's upsert_point() call always writes plus the
# text_hash/dedup_key bookkeeping fields upsert_point() itself adds. validate_document()
# checks retrieved payloads against this contract so a schema drift in a writer shows up
# immediately at retrieval time instead of silently reaching the LLM.
REQUIRED_FIELDS: dict[str, set[str]] = {
    "go_annotations": {
        "gene_symbol", "go_id", "go_name", "source", "ingested_at", "schema_version",
    },
    "kegg_pathways": {
        "gene_symbol", "pathway_id", "pathway_name", "source", "ingested_at", "schema_version",
    },
    "uniprot_proteins": {
        "gene_symbol", "protein_name", "function_summary", "species_tax_id",
        "source", "source_accession", "ingested_at", "schema_version",
    },
}
_BOOKKEEPING_FIELDS = {"text_hash", "dedup_key"}


@dataclass
class RetrievedDocument:
    id: str
    score: float
    payload: dict[str, Any]


@dataclass
class ValidationResult:
    is_valid: bool
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _qdrant_unavailable_warning() -> None:
    if not _QDRANT_AVAILABLE:
        logger.warning("qdrant-client is not installed; retrieval is disabled")


def build_filter(**conditions: Any) -> Optional["Filter"]:

    if not _QDRANT_AVAILABLE:
        _qdrant_unavailable_warning()
        return None

    must: list[FieldCondition] = []
    for key, value in conditions.items():
        if value is None:
            continue
        if isinstance(value, dict):
            must.append(FieldCondition(key=key, range=Range(**value)))
        elif isinstance(value, (list, tuple, set)):
            must.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=must) if must else None


def _summarize_semantic_search_output(results: list["RetrievedDocument"]) -> dict[str, Any]:
    """Shapes what LangSmith logs for this span's outputs (Sprint 4 Part C,
    step 4.3): the collection name is already visible in the span's inputs
    since it's a plain function argument, but the raw list[RetrievedDocument]
    return value doesn't show the chunk count or per-doc gene_symbol/score at
    a glance without opening every item, which is exactly what the
    trace-based spot-check in step 5.1 needs to be fast. This only affects
    what's recorded in the trace -- the function's actual return value to
    its caller is untouched.
    """
    return {
        "num_results": len(results),
        "results": [
            {
                "id": doc.id,
                "score": doc.score,
                "gene_symbol": doc.payload.get("gene_symbol"),
            }
            for doc in results
        ],
    }


@traceable(run_type="retriever", name="qdrant_semantic_search", process_outputs=_summarize_semantic_search_output)
async def semantic_search(
    collection: str,
    query_text: str,
    top_k: int = 5,
    filters: Optional[dict[str, Any]] = None,
    score_threshold: Optional[float] = None,
) -> list[RetrievedDocument]:

    if collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}; expected one of {COLLECTIONS}")

    if not _QDRANT_AVAILABLE:
        _qdrant_unavailable_warning()
        return []

    client = get_client()
    query_vector = await embed_text(query_text)
    query_filter = build_filter(**filters) if filters else None

    response = await client.query_points(           
        collection_name=collection,
        query=query_vector,                         
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [
        RetrievedDocument(id=str(point.id), score=point.score, payload=point.payload or {})
        for point in response.points                
    ]


def validate_document(collection: str, payload: dict[str, Any]) -> ValidationResult:

    required = REQUIRED_FIELDS.get(collection, set())
    missing = sorted(required - payload.keys())
    warnings: list[str] = []

    for field_name in sorted(_BOOKKEEPING_FIELDS):
        if not payload.get(field_name):
            warnings.append(f"missing bookkeeping field '{field_name}'")

    schema_version = payload.get("schema_version")
    if schema_version is not None and not isinstance(schema_version, int):
        warnings.append(f"schema_version is not an int: {schema_version!r}")

    ingested_at = payload.get("ingested_at")
    if ingested_at is not None:
        try:
            datetime.fromisoformat(ingested_at)
        except (TypeError, ValueError):
            warnings.append(f"ingested_at is not ISO-8601: {ingested_at!r}")

    return ValidationResult(is_valid=not missing, missing_fields=missing, warnings=warnings)

import pytest
from datetime import datetime, timezone

from kb.qdrant_store import upsert_point
from kb.retrieval import semantic_search, validate_document

SCHEMA_VERSION = 1


async def _seed(collection: str, dedup_key: str, text: str, extra: dict) -> None:
    await upsert_point(
        collection,
        dedup_key,
        text_to_embed=text,
        payload={
            **extra,
            "source": "pytest fixture",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        },
    )


@pytest.mark.asyncio
async def test_semantic_search_finds_paraphrased_match():
    await _seed(
        "go_annotations", "pytest_rq:go:hair_follicle:FGF5", "hair follicle development",
        {"gene_symbol": "FGF5", "go_id": "GO:0031069", "go_name": "hair follicle development"},
    )

    hits = await semantic_search(
        "go_annotations", "skin structure responsible for growing hair", top_k=3,
    )

    assert any(h.payload.get("dedup_key") == "pytest_rq:go:hair_follicle:FGF5" for h in hits)


@pytest.mark.asyncio
async def test_metadata_filter_restricts_to_matching_gene():
    await _seed(
        "go_annotations", "pytest_rq:go:hair_follicle:FGF5", "hair follicle development",
        {"gene_symbol": "FGF5", "go_id": "GO:0031069", "go_name": "hair follicle development"},
    )
    await _seed(
        "go_annotations", "pytest_rq:go:hair_cycle:KRT71", "regulation of the hair growth cycle",
        {"gene_symbol": "KRT71", "go_id": "GO:0042633", "go_name": "hair cycle"},
    )

    hits = await semantic_search(
        "go_annotations", "hair related gene ontology term", top_k=5,
        filters={"gene_symbol": "KRT71"},
    )

    assert hits, "expected at least one hit"
    assert all(h.payload.get("gene_symbol") == "KRT71" for h in hits)


@pytest.mark.asyncio
async def test_semantic_search_rejects_unknown_collection():
    with pytest.raises(ValueError):
        await semantic_search("not_a_real_collection", "anything", top_k=3)


def test_validate_document_accepts_well_formed_payload():
    payload = {
        "gene_symbol": "FGF5", "go_id": "GO:0031069", "go_name": "hair follicle development",
        "source": "GO REST API (QuickGO)", "ingested_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1, "text_hash": "abc123", "dedup_key": "go:GO:0031069:FGF5",
    }

    result = validate_document("go_annotations", payload)

    assert result.is_valid
    assert result.missing_fields == []
    assert result.warnings == []


def test_validate_document_flags_missing_fields():
    payload = {"gene_symbol": "FGF5", "go_id": "GO:0031069"}  # missing go_name, source, etc.

    result = validate_document("go_annotations", payload)

    assert not result.is_valid
    assert "go_name" in result.missing_fields
    assert "source" in result.missing_fields


def test_validate_document_warns_on_malformed_ingested_at():
    payload = {
        "gene_symbol": "FGF5", "go_id": "GO:0031069", "go_name": "hair follicle development",
        "source": "GO REST API (QuickGO)", "ingested_at": "not-a-timestamp",
        "schema_version": 1, "text_hash": "abc123", "dedup_key": "go:GO:0031069:FGF5",
    }

    result = validate_document("go_annotations", payload)

    assert result.is_valid  # malformed timestamp is a warning, not a hard failure
    assert any("ingested_at" in w for w in result.warnings)
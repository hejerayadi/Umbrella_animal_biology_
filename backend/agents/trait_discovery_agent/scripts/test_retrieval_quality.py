import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.qdrant_store import ensure_collections, get_client, upsert_point
from kb.retrieval import semantic_search, validate_document

SCHEMA_VERSION = 1

FIXTURES = [
    (
        "go_annotations", "rq_test:go:hair_follicle:FGF5", "hair follicle development",
        {"gene_symbol": "FGF5", "go_id": "GO:0031069", "go_name": "hair follicle development"},
    ),
    (
        "go_annotations", "rq_test:go:hair_cycle:KRT71", "regulation of the hair growth cycle",
        {"gene_symbol": "KRT71", "go_id": "GO:0042633", "go_name": "hair cycle"},
    ),
    (
        "go_annotations", "rq_test:go:cold_response:UCP1", "cellular response to cold stimulus",
        {"gene_symbol": "UCP1", "go_id": "GO:0009408", "go_name": "response to heat"},
    ),
    (
        "kegg_pathways", "rq_test:kegg:thermogenesis:PRDM16", "thermogenesis",
        {"gene_symbol": "PRDM16", "pathway_id": "ko04928", "pathway_name": "Thermogenesis"},
    ),
    (
        "uniprot_proteins", "rq_test:uniprot:UCP1:9606",
        "mitochondrial inner membrane proton channel that dissipates energy as heat",
        {
            "gene_symbol": "UCP1", "protein_name": "Uncoupling protein 1",
            "function_summary": "Mitochondrial proton channel, generates heat.",
            "species_tax_id": 9606, "source_accession": "P25874",
        },
    ),
]


SEMANTIC_CASES = [
    (
        "go_annotations", "skin structure that grows body hair", "rq_test:go:hair_follicle:FGF5",
        None, 3,
    ),
    (
        "go_annotations", "process controlling how fur grows over time", "rq_test:go:hair_cycle:KRT71",
        None, 3,
    ),
    (
        "go_annotations", "how a cell reacts to being chilled", "rq_test:go:cold_response:UCP1",
        None, 3,
    ),
    (
        "kegg_pathways", "generating body heat metabolically", "rq_test:kegg:thermogenesis:PRDM16",
        None, 3,
    ),
    (
        "uniprot_proteins", "a channel protein that burns energy as warmth",
        "rq_test:uniprot:UCP1:9606", None, 3,
    ),
]

# Same query as the FGF5/KRT71 pair above, but this time scoped with a metadata filter —
# checks that every hit respects the filter, regardless of vector score.
FILTER_CASES = [
    (
        "go_annotations", "hair related gene ontology term", {"gene_symbol": "KRT71"}, 5,
    ),
    (
        "uniprot_proteins", "protein function", {"species_tax_id": 9606}, 5,
    ),
]


def _payload_for(collection: str, dedup_key: str, text: str, extra: dict) -> dict:
    return {
        **extra,
        "source": f"fixture ({collection})",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }


async def seed_fixtures() -> None:
    print("--- seeding fixtures ---")
    for collection, dedup_key, text, extra in FIXTURES:
        payload = _payload_for(collection, dedup_key, text, extra)
        await upsert_point(collection, dedup_key, text_to_embed=text, payload=payload)
        print(f"  upserted {dedup_key} into {collection}")


async def check_semantic_quality() -> list[str]:
    print("\n--- semantic retrieval quality ---")
    failures = []
    for collection, query_text, expected_key, filters, top_k in SEMANTIC_CASES:
        hits = await semantic_search(collection, query_text, top_k=top_k, filters=filters)
        hit_keys = [h.payload.get("dedup_key") for h in hits]
        ok = expected_key in hit_keys
        rank = hit_keys.index(expected_key) + 1 if ok else None
        top_score = hits[0].score if hits else None
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {collection!r} query={query_text!r} "
            f"expected={expected_key!r} rank={rank} top_score={top_score}"
        )
        if not ok:
            failures.append(f"{collection}: {query_text!r} did not surface {expected_key!r} "
                             f"in top {top_k} (got {hit_keys})")
    return failures


async def check_metadata_filtering() -> list[str]:
    print("\n--- metadata filtering ---")
    failures = []
    for collection, query_text, filters, top_k in FILTER_CASES:
        hits = await semantic_search(collection, query_text, top_k=top_k, filters=filters)
        bad = [
            h for h in hits
            if any(h.payload.get(k) != v for k, v in filters.items())
        ]
        ok = len(hits) > 0 and not bad
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {collection!r} filters={filters} -> {len(hits)} hits, "
              f"{len(bad)} violate the filter")
        if not ok:
            failures.append(f"{collection}: filter {filters} returned {len(hits)} hits, "
                             f"{len(bad)} of which don't match the filter")
    return failures


async def check_validation() -> list[str]:
    print("\n--- retrieved-document validation ---")
    failures = []
    for collection, query_text, _, filters, top_k in SEMANTIC_CASES:
        hits = await semantic_search(collection, query_text, top_k=top_k, filters=filters)
        for hit in hits:
            result = validate_document(collection, hit.payload)
            status = "PASS" if result.is_valid else "FAIL"
            print(f"  [{status}] {collection!r} id={hit.id} "
                  f"missing={result.missing_fields} warnings={result.warnings}")
            if not result.is_valid:
                failures.append(f"{collection} id={hit.id} missing fields: {result.missing_fields}")
    return failures


async def cleanup_fixtures() -> None:
    print("\n--- cleaning up fixtures ---")
    from kb.qdrant_store import _point_id  # local import: internal helper, cleanup-only use

    client = get_client()
    for collection, dedup_key, _, _ in FIXTURES:
        await client.delete(collection_name=collection, points_selector=[_point_id(dedup_key)])
        print(f"  deleted {dedup_key} from {collection}")


async def main() -> int:
    await ensure_collections()
    await seed_fixtures()

    failures = []
    failures += await check_semantic_quality()
    failures += await check_metadata_filtering()
    failures += await check_validation()

    if "--cleanup" in sys.argv:
        await cleanup_fixtures()

    print("\n--- summary ---")
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("all retrieval pipeline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
"""
Live Neo4j integration test — requires the docker-compose `neo4j` service
(run via `docker compose run --rm test-neo4j-store`, which depends_on it).
Not part of the hermetic unit suite in tests/test_neo4j_store.py.
"""
import uuid

import pytest

from kb.neo4j_store import get_driver, upsert_trait_gene_relationship


@pytest.mark.asyncio
async def test_write_then_read_back_via_direct_cypher():
    # Unique trait/gene names per run so repeated CI runs don't collide.
    trait = f"test-trait-{uuid.uuid4().hex[:8]}"
    gene = f"TESTGENE{uuid.uuid4().hex[:6].upper()}"

    ok = await upsert_trait_gene_relationship(trait, gene, "12345678")
    assert ok is True

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (t:Trait {name: $trait})-[r:SUPPORTED_BY]->(g:Gene {symbol: $gene}) "
            "RETURN r.pmids AS pmids",
            trait=trait, gene=gene,
        )
        record = await result.single()

    assert record is not None
    assert record["pmids"] == ["12345678"]


@pytest.mark.asyncio
async def test_repeat_write_merges_not_duplicates_and_accumulates_pmids():
    trait = f"test-trait-{uuid.uuid4().hex[:8]}"
    gene = f"TESTGENE{uuid.uuid4().hex[:6].upper()}"

    await upsert_trait_gene_relationship(trait, gene, "11111111")
    await upsert_trait_gene_relationship(trait, gene, "22222222")
    await upsert_trait_gene_relationship(trait, gene, "11111111")  # exact repeat

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (t:Trait {name: $trait})-[r:SUPPORTED_BY]->(g:Gene {symbol: $gene}) "
            "RETURN r.pmids AS pmids, count{(t)-[:SUPPORTED_BY]->(g)} AS edge_count",
            trait=trait, gene=gene,
        )
        record = await result.single()

    assert record is not None
    assert record["edge_count"] == 1  # MERGE, not a fresh relationship each call
    assert sorted(record["pmids"]) == ["11111111", "22222222"]  # no duplicate pmid

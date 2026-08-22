"""
Neo4j knowledge-graph access layer — the trait->gene relationship write
(design guide §1/§6).

Only the Literature Support agent writes here, and only through
upsert_trait_gene_relationship(trait_name, gene_symbol, pmid). That
signature is itself the enforcement mechanism for "never cache evidence
content": there is no parameter here a caller could use to smuggle a title
or short_summary into the graph, so the rule can't be silently violated by
a future edit the way a permissive dict-payload function could be.

Mirrors kb/qdrant_store.py's degrade-gracefully-if-driver-missing pattern:
if the neo4j package isn't installed, every call here is a logged no-op
rather than an import-time crash, since Literature Support must keep
returning evidence even if the graph write path is unavailable (§9).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    from neo4j import AsyncGraphDatabase

    _NEO4J_AVAILABLE = True
except ModuleNotFoundError:
    AsyncGraphDatabase = Any  # type: ignore[assignment]
    _NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

_driver: Optional[Any] = None

# MERGE (not CREATE) on both nodes and the edge: repeated writes for the same
# trait/gene pair must not create duplicate nodes/relationships. The pmid
# list on the relationship accumulates distinct supporting pmids across
# separate literature_support_agent runs rather than overwriting.
_MERGE_TRAIT_GENE_RELATIONSHIP = """
MERGE (t:Trait {name: $trait_name})
MERGE (g:Gene {symbol: $gene_symbol})
MERGE (t)-[r:SUPPORTED_BY]->(g)
ON CREATE SET r.pmids = [$pmid]
ON MATCH SET r.pmids = CASE
    WHEN $pmid IN r.pmids THEN r.pmids
    ELSE r.pmids + $pmid
END
RETURN r.pmids AS pmids
"""


def _neo4j_unavailable_warning() -> None:
    if not _NEO4J_AVAILABLE:
        logger.warning("neo4j driver is not installed; trait-gene graph writes are no-ops")


def get_driver() -> Any:
    if not _NEO4J_AVAILABLE:
        _neo4j_unavailable_warning()
        raise RuntimeError("neo4j driver is not installed")

    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


async def upsert_trait_gene_relationship(
    trait_name: str, gene_symbol: str, pmid: str
) -> bool:
    """
    Writes/merges a Trait-[:SUPPORTED_BY]->Gene edge backed by a single pmid.

    Scalar-only args by design (§6) — no evidence content (title,
    short_summary) is ever accepted here. Fails soft (§9): any driver or
    query error is logged and returns False rather than raising, since a
    graph-write failure must never downgrade the calling subagent's status
    or drop the evidence it already has.
    """
    if not trait_name or not gene_symbol or not pmid:
        logger.warning(
            "refusing to write incomplete trait-gene relationship "
            "(trait_name=%r, gene_symbol=%r, pmid=%r)",
            trait_name, gene_symbol, pmid,
        )
        return False

    if not _NEO4J_AVAILABLE:
        _neo4j_unavailable_warning()
        return False

    try:
        driver = get_driver()
    except Exception as exc:
        logger.warning("Neo4j driver unavailable, skipping graph write: %s", exc)
        return False

    try:
        async with driver.session() as session:
            await session.run(
                _MERGE_TRAIT_GENE_RELATIONSHIP,
                trait_name=trait_name,
                gene_symbol=gene_symbol,
                pmid=str(pmid),
            )
        logger.info(
            "wrote (%s)-[:SUPPORTED_BY {pmid=%s}]->(%s) to Neo4j",
            trait_name, pmid, gene_symbol,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Neo4j write failed for (%s, %s, %s): %s", trait_name, gene_symbol, pmid, exc
        )
        return False


async def close_driver() -> None:
    """Call at API shutdown (mirrors qdrant_store's client lifecycle)."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None

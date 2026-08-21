import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for candidate in [ROOT, os.path.join(ROOT, "backend", "agents", "trait_discovery_agent")]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

import kb.qdrant_store as qdrant_store


async def _purge_all_collections() -> None:
    """Delete every point from every KB collection.

    The collections live in a real, persistent Qdrant instance that is shared
    across the whole test session. Without this, a point written by one
    integration test (e.g. a pathways/protein-data run against the standard
    FGF5/UCP1 fixtures) stays cached for every later test that happens to use
    the same gene/tax_id — including tests that specifically assert on
    cache-miss-then-cache-hit behavior. Those tests would find the point
    already cached on their very first call, so `upsert_point` (and therefore
    `embed_text`) never runs at all.
    """
    if not qdrant_store._QDRANT_AVAILABLE or not os.environ.get("QDRANT_URL"):
        return

    from qdrant_client.models import Filter, FilterSelector

    client = qdrant_store.get_client()
    for collection in qdrant_store.COLLECTIONS:
        try:
            await client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=Filter()),
            )
        except Exception:
            # Collection may not exist yet on a fresh cluster; nothing to purge.
            pass


@pytest.fixture(autouse=True)
async def _reset_qdrant_client():
    await _purge_all_collections()
    yield
    if qdrant_store._client is not None:
        await qdrant_store._client.close()
        qdrant_store._client = None
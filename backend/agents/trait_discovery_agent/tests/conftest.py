import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for candidate in [ROOT, os.path.join(ROOT, "backend", "agents", "trait_discovery_agent")]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

import kb.qdrant_store as qdrant_store


@pytest.fixture(autouse=True)
async def _reset_qdrant_client():
    yield
    if qdrant_store._client is not None:
        await qdrant_store._client.close()
        qdrant_store._client = None

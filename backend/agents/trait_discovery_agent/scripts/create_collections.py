import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb.qdrant_store import ensure_collections



asyncio.run(ensure_collections())
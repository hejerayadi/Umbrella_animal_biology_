"""Index a controlled document set into the managed Qdrant collection.

Runs the same ingestion service as the HTTP endpoint, so normalisation, SHA-256
hashing and the skip-if-unchanged rule apply here too.
"""

import argparse
import asyncio
import json
from pathlib import Path

from backend.agents.Protein_visualization.app.api.v1.dependencies import get_ingestion_service
from backend.agents.Protein_visualization.app.configuration.settings import get_settings
from backend.agents.Protein_visualization.app.knowledge_base.schemas import KnowledgeDocument


async def main(path: Path) -> None:
    settings = get_settings()
    contents = await asyncio.to_thread(path.read_text, encoding="utf-8")
    documents = [KnowledgeDocument.model_validate(item) for item in json.loads(contents)]
    result = await get_ingestion_service().ingest(documents)
    target = settings.qdrant_url or "the in-memory store (QDRANT_URL is not set)"
    print(f"Indexed {result.inserted} document(s), skipped {result.skipped} unchanged, into {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="JSON array of knowledge documents")
    args = parser.parse_args()
    asyncio.run(main(args.path))
